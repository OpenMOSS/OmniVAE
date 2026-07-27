"""Shared T2AV inference pipeline for OmniVAE joint-AV checkpoints.

Used by both ``infer/t2av/streamlit_app.py`` (interactive)
and ``infer/t2av/infer_t2av.py`` (CLI batch). Wraps the
**same** denoising / decoding / muxing helpers that
``omnivae_generation.trainer.joint_av.validation.run_joint_av_validation`` uses, so videos
generated here are numerically identical to a one-prompt validation
pass with the same prompt / seed / steps / cfg / cfg_mode.

Public surface
--------------
* :func:`load_joint_av_pipeline` -- build the full pipeline (text
  encoder + scheduler + video VAE + audio VAE + BridgedZImageJointModel
  with both branches and bridges restored from a checkpoint directory).

* :func:`generate_one_av` -- run one prompt through the joint denoising
  loop, decode, pad/trim the audio so it matches ``num_frames / fps``,
  and (in ``joint_av`` mode) mux video + audio into a single ``.av.mp4``
  whose video stream is bit-for-bit copied from the libx264 output
  (i.e. no frames dropped at the tail).

Layout assumption for ``checkpoint_dir``
----------------------------------------
The trainer's ``omnivae_generation.trainer.joint_av.save_split_branches`` writes the
following structure (mirrored here)::

    checkpoint-XXXXXXXX/
      transformer_video/   diffusers ZImageTransformer2DModel (video branch)
      transformer_audio/   diffusers ZImageTransformer2DModel (audio branch)
      bridges/bridges.safetensors + bridge_config.json
      tokenizer/, scheduler/, metadata.json

A run-config file is expected two levels up from the snapshot
(``run_dir/checkpoints/snapshots/checkpoint-XXXXXXXX``) and supplies
text encoder / video VAE / audio VAE / transformer-branch configs.
``resolved_config.json`` (the canonical post-override form written by
the trainer at run-dir creation) is preferred when present;
``resolved_config.yaml`` is used as a fall-back for older runs that
predate the json sidecar.

The text encoder and the two VAEs are *not* in the snapshot (frozen at
training time), they are reloaded from the paths in the resolved
config -- which can themselves be overridden at load time via the
``*_override`` arguments.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
import torchaudio


logger = logging.getLogger(__name__)


def _can_use_transformers_device_map() -> bool:
    try:
        from transformers.utils import is_accelerate_available

        return bool(is_accelerate_available())
    except Exception:
        return False


# Module-level lock that serializes the parts of ``load_joint_av_pipeline``
# which are not thread-safe under "single process, N threads, N CUDA
# devices". Two distinct failure modes have been observed:
#
# 1. Concurrent ``Module.to(cuda:N)`` for the ~6.5 GB transformer
#    branches across 3 devices: PyTorch's per-device caching allocator
#    is thread-safe per-call but its first-time bring-up (cuBLAS /
#    cuDNN handles, memory pool init) is not fully reentrant, and the
#    corruption surfaces later as an ``illegal memory access`` on the
#    very first inference kernel.
#
# 2. Concurrent ``AutoModel.from_pretrained(..., low_cpu_mem_usage=True)``
#    of the *same* Qwen text encoder from the HF cache: transformers'
#    meta-tensor materialization races on shared module-state and one
#    of the loaders is left with empty meta parameters
#    (``Cannot copy out of meta tensor; no data!``).
#
# A single lock is held both around the parallel component-loading
# block AND the subsequent ``.to(device)`` block. The trade-off: per-
# slot loading becomes effectively serial across slots (3 ckpts = 3x
# single-load time) but never crashes. Each slot's intra-load
# components can still pipeline freely behind this lock (CPU init +
# disk read while previous slot's GPU move drains).
_PIPELINE_LOAD_LOCK = threading.Lock()
# Backwards-compat alias for older callers; kept as an alias of the
# same underlying lock so behaviour is unchanged.
_GPU_INIT_LOCK = _PIPELINE_LOAD_LOCK


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _release_root() -> Path:
    explicit = os.environ.get("OMNIVAE_RELEASE_ROOT") or os.environ.get("OPEN_SOURCE_ROOT")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root / "open_source",
        repo_root.parent / "open_source",
        repo_root.parent.parent / "open_source",
        repo_root / "open_source" / "open_source",
        repo_root.parent / "open_source" / "open_source",
        repo_root.parent.parent / "open_source" / "open_source",
    ])
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "models").is_dir() and (candidate / "eval").is_dir():
            return candidate.resolve()
    return (repo_root / "open_source").resolve()


def _resolve_release_path(value: str | os.PathLike | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(("models/", "eval/")):
        return str((_release_root() / text).resolve())
    return text


def _resolve_release_model_paths(config: dict[str, Any]) -> None:
    text_cfg = config.get("text_encoder")
    if isinstance(text_cfg, dict):
        resolved = _resolve_release_path(text_cfg.get("model_name_or_path"))
        if resolved:
            text_cfg["model_name_or_path"] = resolved
            if resolved.startswith("/"):
                text_cfg.setdefault("local_files_only", True)

    vae_cfg = config.get("vae")
    if isinstance(vae_cfg, dict):
        resolved = _resolve_release_path(vae_cfg.get("model_name_or_path"))
        if resolved:
            vae_cfg["model_name_or_path"] = resolved

    audio_vae_cfg = config.get("audio_vae")
    if isinstance(audio_vae_cfg, dict):
        key = "model_path" if "model_path" in audio_vae_cfg else "model_name_or_path"
        resolved = _resolve_release_path(audio_vae_cfg.get(key))
        if resolved:
            audio_vae_cfg[key] = resolved

    scheduler_cfg = config.get("scheduler")
    if isinstance(scheduler_cfg, dict):
        resolved = _resolve_release_path(scheduler_cfg.get("model_name_or_path"))
        if resolved:
            scheduler_cfg["model_name_or_path"] = resolved
            if resolved.startswith("/"):
                scheduler_cfg.setdefault("local_files_only", True)


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
@dataclass
class T2AVPipeline:
    """Container for an instantiated T2AV inference pipeline.

    Used as a ``SimpleNamespace``-style record; accessed by attribute
    name from both UIs to keep call sites readable.
    """

    joint_model: Any                        # BridgedZImageJointModel (on device, eval())
    tokenizer: Any
    text_encoder: Any                       # on device, eval()
    video_vae: Any                          # on device, eval()
    audio_vae: Any                          # on device, eval()
    scheduler: Any                          # FlowMatchEulerDiscreteScheduler
    run_config: dict                        # resolved_config.yaml (deep-copied)
    run_dir: Path
    checkpoint_dir: Path
    checkpoint_step: int
    device: torch.device
    train_patch_size: int
    train_f_patch_size: int
    shift_v: float
    shift_a: float
    predict_target: str


def _apply_runtime_patches() -> None:
    """Mirror the patches that the training entry / t2v eval loader apply
    before instantiating diffusers' Z-Image transformer. Idempotent.
    """
    from omnivae_generation.trainer.runtime_patches import (
        patch_diffusers_zimage_forward_block_stacks,
        patch_diffusers_zimage_real_rope,
        patch_transformers_qwen3_5_disable_fast_path,
    )

    patch_diffusers_zimage_real_rope()
    patch_diffusers_zimage_forward_block_stacks()
    # The training config sets ``disable_qwen3_5_fast_path: true`` for the
    # frozen text encoder. The patch is idempotent, so applying it here
    # unconditionally is safe even if the underlying yaml disagrees.
    patch_transformers_qwen3_5_disable_fast_path()


def _read_bridge_config(checkpoint_dir: Path) -> dict[str, Any]:
    """Prefer the snapshot-local ``bridge_config.json`` because the
    bridges' weight shapes are pinned by training-time config; falling
    back to the yaml would silently re-init at a different
    ``bridge_interval`` and break the strict bridge load.
    """
    descriptor_path = checkpoint_dir / "bridges" / "bridge_config.json"
    if descriptor_path.is_file():
        return json.loads(descriptor_path.read_text(encoding="utf-8"))
    return {}


def _load_branch_from_pretrained(
    branch_dir: Path,
    branch_cfg: dict,
    *,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> Any:
    """Load one Z-Image transformer branch directly via diffusers'
    ``from_pretrained``.

    When ``device`` is provided (and is a CUDA device), the branch is
    loaded **directly onto that GPU** via ``device_map={"": device}``;
    no CPU staging, no meta-tensor materialization, no follow-up
    ``.to(device)`` move. This is essential for parallel multi-GPU slot
    loading because the "meta -> CPU -> GPU" path that ``low_cpu_mem_
    usage=True`` previously triggered is not thread-safe under
    concurrent ``from_pretrained`` calls for the same source (the
    transformers / accelerate internals race on materialisation
    bookkeeping and one of the callers ends up with empty meta
    parameters that raise ``Cannot copy out of meta tensor`` later).
    Loading direct-to-GPU also saves the CPU->GPU memcpy time on the
    happy path.

    Compared to the previous "build with random init + state_dict copy"
    path (``omnivae_generation.trainer.modeling.build_transformer`` followed by
    ``omnivae_generation.trainer.joint_av.load_pretrained_branches``), this saves the
    random-init allocation pass for ~6.5 GB of parameters per branch
    and uses safetensors' mmap-backed reader to populate the model in
    place.

    The CALLER is responsible for setting the class-level
    ``ZImageTransformerBlock._laion_force_disable_modulation`` /
    ``FinalLayer._laion_default_modulation`` flags to match
    ``branch_cfg['use_timestep']`` BEFORE invoking this function; these
    flags affect parameter allocation in ``ZImageTransformer2DModel
    .__init__`` and are global, so they cannot be set safely inside a
    worker thread when both branches load concurrently.
    """
    from diffusers import ZImageTransformer2DModel

    from omnivae_generation.trainer.modeling import (
        configure_transformer_prediction_target,
        configure_transformer_timestep_usage,
    )

    from_pretrained_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
    }
    use_device_map = device is not None and device.type == "cuda" and _can_use_transformers_device_map()
    if use_device_map:
        # Pin the entire model on this slot's CUDA device. ``device_map``
        # bypasses ``low_cpu_mem_usage``'s meta-tensor path and is
        # internally re-entrant under accelerate.dispatch_model_with_state,
        # so 3 worker threads loading 3 distinct branch paths to 3 distinct
        # CUDA devices don't trip over each other.
        from_pretrained_kwargs["device_map"] = {"": str(device)}
    elif device is None or device.type != "cuda":
        # CPU fallback for ``device=None`` or ``cpu``. Keep the previous
        # low_cpu_mem_usage path here because there's no GPU pinning to
        # talk about and the mmap loader is still strictly better than
        # the random-init route.
        from_pretrained_kwargs["low_cpu_mem_usage"] = True

    transformer = ZImageTransformer2DModel.from_pretrained(
        str(branch_dir),
        **from_pretrained_kwargs,
    )
    if device is not None and not use_device_map:
        transformer.to(device)

    use_timestep = bool(branch_cfg.get("use_timestep", True))
    configure_transformer_timestep_usage(transformer, use_timestep)
    configure_transformer_prediction_target(transformer, branch_cfg.get("predict_target", "v"))

    # Pad tokens are nn.Parameters and round-trip via safetensors. Older
    # snapshots that pre-date the zero-init fix can leave them at
    # ``torch.empty`` values; replace any non-finite contents with zeros
    # so attention masks don't propagate NaNs.
    for name in ("x_pad_token", "cap_pad_token", "siglip_pad_token"):
        tensor = getattr(transformer, name, None)
        if isinstance(tensor, torch.nn.Parameter):
            with torch.no_grad():
                if not torch.isfinite(tensor).all():
                    tensor.data.zero_()
            tensor.requires_grad_(False)
    return transformer


def _emit_progress(message: str) -> None:
    """Single-line per-stage progress emitter.

    Goes to both python logging and stdout (with explicit flush) so that
    when this loader runs under streamlit the user sees the progress live
    in the terminal that launched the server. We intentionally do NOT
    touch any streamlit APIs here because this function is also called
    from worker threads (parallel mode), and streamlit only supports
    being called from the main script thread.
    """
    logger.info("%s", message)
    print(message, flush=True)


def _load_text_components_on_device(
    text_cfg: dict,
    dtype: torch.dtype,
    *,
    device: Optional[torch.device] = None,
) -> tuple[Any, Any, int]:
    """Drop-in replacement for ``omnivae_generation.trainer.modeling.load_text_components``
    that pins the text encoder to a specific GPU at ``from_pretrained``
    time instead of going through the meta-tensor + ``.to(device)``
    path baked into the trainer helper.

    Why we cannot reuse the trainer helper here: it hardcodes
    ``low_cpu_mem_usage=True`` for every call. Under concurrent slot
    loading (3 threads each calling ``AutoModel.from_pretrained`` for
    the SAME Qwen text encoder), transformers' meta-tensor
    materialisation races on shared module state and one of the
    callers ends up with empty meta parameters that raise
    ``Cannot copy out of meta tensor; no data!`` on the follow-up
    ``.to(device)``. Passing ``device_map={"": device}`` instead loads
    direct-to-GPU via ``accelerate.dispatch_model``, which IS thread-
    safe across distinct device targets, and avoids the meta detour
    entirely.
    """
    from transformers import AutoModel, AutoTokenizer

    if text_cfg.get("disable_qwen3_5_fast_path", False):
        from omnivae_generation.trainer.runtime_patches import patch_transformers_qwen3_5_disable_fast_path

        patch_transformers_qwen3_5_disable_fast_path()

    tokenizer = AutoTokenizer.from_pretrained(
        text_cfg["model_name_or_path"],
        trust_remote_code=text_cfg.get("trust_remote_code", False),
        local_files_only=text_cfg.get("local_files_only", False),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": text_cfg.get("trust_remote_code", False),
        "torch_dtype": dtype,
        "local_files_only": text_cfg.get("local_files_only", False),
    }
    attn_implementation = text_cfg.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    use_device_map = device is not None and device.type == "cuda" and _can_use_transformers_device_map()
    if use_device_map:
        model_kwargs["device_map"] = {"": str(device)}
    elif device is None or device.type != "cuda":
        model_kwargs["low_cpu_mem_usage"] = True

    text_encoder = AutoModel.from_pretrained(
        text_cfg["model_name_or_path"], **model_kwargs
    )
    if device is not None and not use_device_map:
        text_encoder.to(device)
    text_config = text_encoder.config.get_text_config()
    hidden_size = getattr(text_config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(
            "Could not determine the text hidden size from the text encoder config returned by "
            "`config.get_text_config()`."
        )
    return tokenizer, text_encoder, int(hidden_size)


def _run_components(
    tasks: list[tuple[str, Callable[[], Any]]],
    *,
    max_workers: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run a list of named ``(label, callable)`` loaders.

    Returns ``(results, timings)`` keyed by label. When ``max_workers ==
    1`` the tasks run sequentially in the caller's thread, in the
    listed order (so the user sees the stages tick through one by one
    on stdout). When ``max_workers > 1`` the tasks are dispatched in a
    :class:`ThreadPoolExecutor`; each task emits "started" / "done"
    messages so progress is still visible even with N concurrent
    workers.

    The first exception is re-raised once all submitted futures have
    been collected, so partially-completed pipelines never leak into
    the caller's namespace.
    """
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}
    errors: list[tuple[str, BaseException]] = []
    n = len(tasks)
    if n == 0:
        return results, timings

    mode_str = "serial" if max_workers <= 1 else f"parallel({max_workers})"
    _emit_progress(
        f"[t2av_pipeline] loading {n} components ({mode_str}): "
        + ", ".join(label for label, _ in tasks)
    )

    completed_counter = [0]
    progress_lock = threading.Lock()

    def _wrap(label: str, fn: Callable[[], Any], index: int):
        _emit_progress(f"[t2av_pipeline]   ({index}/{n}) starting {label} ...")
        t0 = time.time()
        try:
            out = fn()
        except BaseException as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            with progress_lock:
                timings[label] = elapsed
                completed_counter[0] += 1
                done = completed_counter[0]
            _emit_progress(
                f"[t2av_pipeline]   ({done}/{n}) FAILED  {label} after "
                f"{elapsed:.1f}s: {exc!r}"
            )
            raise
        elapsed = time.time() - t0
        with progress_lock:
            timings[label] = elapsed
            completed_counter[0] += 1
            done = completed_counter[0]
        _emit_progress(
            f"[t2av_pipeline]   ({done}/{n}) done    {label} in {elapsed:.1f}s"
        )
        return out

    if max_workers <= 1:
        # Run in deterministic listed order so the terminal output reads
        # like a sequential script (text_encoder -> video_vae -> ... ->
        # video_branch -> audio_branch). When something hangs, the last
        # "starting X" line on stdout pinpoints which loader stalled.
        for i, (label, fn) in enumerate(tasks, start=1):
            try:
                results[label] = _wrap(label, fn, i)
            except BaseException as exc:  # noqa: BLE001
                errors.append((label, exc))
                break
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_label = {
                executor.submit(_wrap, label, fn, i): label
                for i, (label, fn) in enumerate(tasks, start=1)
            }
            for future in future_to_label:
                label = future_to_label[future]
                try:
                    results[label] = future.result()
                except BaseException as exc:  # noqa: BLE001
                    errors.append((label, exc))

    if errors:
        # Surface the first error, attaching a summary of which other
        # components failed for easier debugging when several fail at once.
        label, exc = errors[0]
        if len(errors) > 1:
            extra = ", ".join(f"{lab}={type(e).__name__}" for lab, e in errors[1:])
            raise RuntimeError(
                f"{label} failed during component load ({exc!r}); also failed: {extra}"
            ) from exc
        raise exc
    return results, timings


def load_joint_av_pipeline(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    run_dir: Optional[str | Path] = None,
    vae_type_override: Optional[str] = None,
    vae_path_override: Optional[str | Path] = None,
    audio_vae_type_override: Optional[str] = None,
    audio_vae_path_override: Optional[str | Path] = None,
) -> T2AVPipeline:
    """Build the full T2AV inference pipeline from a saved checkpoint dir.

    Parameters
    ----------
    checkpoint_dir
        ``.../checkpoints/snapshots/checkpoint-XXXXXXXX`` produced by
        :func:`omnivae_generation.trainer.joint_av.save_split_branches`.
    device
        Torch device for both branches, VAEs, and the text encoder.
    vae_type_override, vae_path_override
        Override the video VAE ``type`` / ``model_name_or_path`` from
        ``resolved_config.yaml`` (handy when the yaml's vae block does
        not match the checkpoint your snapshot was actually trained
        against; same semantics as ``infer/t2v/streamlit_app.py``).
    audio_vae_type_override, audio_vae_path_override
        Override the audio VAE block. ``audio_vae`` uses ``model_path``
        (not ``model_name_or_path``) so ``audio_vae_path_override`` is
        written into ``model_path``.

    Performance
    -----------
    Six components are loaded one after another (text encoder +
    tokenizer, video VAE, audio VAE, scheduler, video branch, audio
    branch) by default. Each stage emits a ``starting <label>`` /
    ``done <label> in N.Ns`` line to stdout so the terminal that
    launched the server can pinpoint a stall.

    Both transformer branches use diffusers' ``from_pretrained(...,
    low_cpu_mem_usage=True)`` which skips the ~20-40 s random-init
    allocation pass and uses safetensors' mmap reader.

    Set ``T2AV_LOAD_PARALLEL=1`` (or ``=N`` for an explicit thread
    pool size) to load the six components concurrently. This helps on
    SSD / tmpfs / NVMe but typically hurts on rotational HDD because
    concurrent readers cause head-seek thrashing; serial is the safer
    default.
    """
    overall_t0 = time.time()
    _apply_runtime_patches()

    from omnivae_generation.trainer.eval.guided_diffusion import (
        extract_checkpoint_step,
        load_run_config_for_eval,
        resolve_run_dir,
    )
    from omnivae_generation.trainer.joint_av import (
        BridgedZImageJointModel,
        load_bridges_from_dir,
    )
    from omnivae_generation.trainer.modeling import (
        load_audio_vae,
        load_scheduler,
        load_vae,
        resolve_dtype,
    )

    cdir = Path(checkpoint_dir).expanduser().resolve()
    if not cdir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {cdir}")
    rdir = resolve_run_dir(run_dir, cdir)

    # Prefer ``resolved_config.json`` over ``resolved_config.yaml`` when
    # both exist. The trainer writes both at run-dir creation time, but
    # the json is the canonical resolved form (yaml in the same dir is
    # the un-resolved template before CLI / programmatic overrides are
    # merged). For runs where the two disagree on VAE paths
    # (e.g. yaml has ``vae: wan2_2_vae``, json has the actual
    # ``vae: omnivae`` + local state_dict.pt the run trained against),
    # reading the json is the only way to recover what was actually
    # used. Fall back to yaml when the json is absent so older runs
    # without a saved json still load.
    json_cfg_path = rdir / "resolved_config.json"
    yaml_cfg_path = rdir / "resolved_config.yaml"
    if json_cfg_path.is_file():
        rcfg_raw = json.loads(json_cfg_path.read_text(encoding="utf-8"))
        _emit_progress(f"[t2av_pipeline] config source: {json_cfg_path}")
    elif yaml_cfg_path.is_file():
        rcfg_raw = load_run_config_for_eval(rdir)
        _emit_progress(f"[t2av_pipeline] config source: {yaml_cfg_path}")
    else:
        raise FileNotFoundError(
            f"Neither resolved_config.json nor resolved_config.yaml found under {rdir}"
        )
    rcfg = copy.deepcopy(rcfg_raw)

    if not isinstance(rcfg.get("transformer_video"), dict) or not isinstance(
        rcfg.get("transformer_audio"), dict
    ):
        raise ValueError(
            "T2AV pipeline requires resolved_config.yaml with both "
            "'transformer_video' and 'transformer_audio' blocks (the t2av "
            "trainer's split-branch config)."
        )

    if vae_type_override:
        rcfg.setdefault("vae", {})["type"] = str(vae_type_override).strip()
    if vae_path_override:
        rcfg.setdefault("vae", {})["model_name_or_path"] = str(vae_path_override)
    if audio_vae_type_override:
        rcfg.setdefault("audio_vae", {})["type"] = str(audio_vae_type_override).strip()
    if audio_vae_path_override:
        rcfg.setdefault("audio_vae", {})["model_path"] = str(audio_vae_path_override)
    _resolve_release_model_paths(rcfg)

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {dev} but cuda is not available.")
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())

    text_dtype = resolve_dtype(rcfg["text_encoder"].get("torch_dtype"), fallback=torch.bfloat16)
    transformer_dtype = resolve_dtype(
        rcfg["transformer_video"].get("torch_dtype"), fallback=torch.bfloat16
    )

    # `ZImageTransformerBlock._laion_force_disable_modulation` and
    # `FinalLayer._laion_default_modulation` are CLASS-level flags
    # consulted inside `__init__` to decide whether to allocate
    # modulation parameters. They MUST be set before
    # `ZImageTransformer2DModel.from_pretrained` calls `__init__`. Since
    # the flags are global we set them once for both branches; the t2av
    # config asserts both branches share `use_timestep`, so this is
    # always safe in practice. If they ever disagree we fall back to
    # the trainer's per-branch context manager (serial loading).
    use_timestep_v = bool(rcfg["transformer_video"].get("use_timestep", True))
    use_timestep_a = bool(rcfg["transformer_audio"].get("use_timestep", True))
    if use_timestep_v != use_timestep_a:
        raise ValueError(
            "transformer_video.use_timestep and transformer_audio.use_timestep "
            f"must agree for parallel loading; got {use_timestep_v} vs "
            f"{use_timestep_a}. Update the yaml or load branches serially."
        )

    from diffusers.models.transformers.transformer_z_image import (  # noqa: E402
        FinalLayer,
        ZImageTransformerBlock,
    )

    prev_force_disable = getattr(ZImageTransformerBlock, "_laion_force_disable_modulation", False)
    prev_default_modulation = getattr(FinalLayer, "_laion_default_modulation", True)
    ZImageTransformerBlock._laion_force_disable_modulation = not use_timestep_v
    FinalLayer._laion_default_modulation = use_timestep_v

    text_cfg = rcfg["text_encoder"]
    vae_cfg = rcfg["vae"]
    audio_vae_cfg = rcfg["audio_vae"]
    scheduler_cfg = rcfg["scheduler"]
    transformer_video_cfg = rcfg["transformer_video"]
    transformer_audio_cfg = rcfg["transformer_audio"]

    # Task list ordered cheap -> expensive when running serially so the
    # terminal output reads as a clear progress trail (the user sees
    # each cheap component tick by, then the two ~6.5 GB transformer
    # branches as the dominant tail). For parallel mode the order
    # doesn't matter but we keep it identical for determinism.
    #
    # ``text_encoder`` + the two transformer branches are loaded
    # **direct-to-GPU** via ``device_map={"": dev}`` (see
    # ``_load_text_components_on_device`` / ``_load_branch_from_pretrained``):
    # this avoids the meta-tensor + ``.to(device)`` path inside
    # ``transformers`` / ``accelerate`` that is not thread-safe under
    # concurrent loads of the same model from different slot threads.
    # VAEs / scheduler stay on the trainer helpers because they don't
    # go through the meta-tensor path.
    tasks: list[tuple[str, Callable[[], Any]]] = [
        ("scheduler", lambda: load_scheduler(scheduler_cfg)),
        ("audio_vae", lambda: load_audio_vae(audio_vae_cfg)),
        (
            "text_encoder",
            lambda: _load_text_components_on_device(
                text_cfg, text_dtype, device=dev
            ),
        ),
        ("video_vae", lambda: load_vae(vae_cfg)),
        (
            "video_branch",
            lambda: _load_branch_from_pretrained(
                cdir / "transformer_video",
                transformer_video_cfg,
                dtype=transformer_dtype,
                device=dev,
            ),
        ),
        (
            "audio_branch",
            lambda: _load_branch_from_pretrained(
                cdir / "transformer_audio",
                transformer_audio_cfg,
                dtype=transformer_dtype,
                device=dev,
            ),
        ),
    ]

    # Serial by default on HDD-backed storage: 6 concurrent safetensors
    # readers on a single spindle can thrash worse than fully serialized
    # reads (head seek storms negate any CPU-init overlap). Users on
    # SSD / tmpfs can opt in with ``T2AV_LOAD_PARALLEL=1`` (or set
    # ``T2AV_LOAD_PARALLEL=N`` for an explicit worker count).
    parallel_raw = os.environ.get("T2AV_LOAD_PARALLEL", "0").strip()
    try:
        parallel_n = int(parallel_raw)
    except ValueError:
        parallel_n = 1 if _env_truthy("T2AV_LOAD_PARALLEL") else 0
    if parallel_n <= 0:
        max_workers = 1
    elif parallel_n == 1:
        max_workers = len(tasks)
    else:
        max_workers = min(parallel_n, len(tasks))

    # Hold ``_PIPELINE_LOAD_LOCK`` for the whole "load + compose + move
    # to GPU" critical section so concurrent slot loaders serialize on
    # the parts of HF / diffusers / PyTorch that are NOT thread-safe:
    # ``from_pretrained(low_cpu_mem_usage=True)`` meta-tensor
    # materialization, class-level ZImageTransformerBlock flags, and
    # first-time CUDA allocator bring-up per device. Outside this lock
    # we only do thread-safe metadata bookkeeping.
    t_wait_lock_0 = time.time()
    with _PIPELINE_LOAD_LOCK:
        timings_wait_lock = time.time() - t_wait_lock_0
        # Class-level flag flip must happen INSIDE the lock so a
        # second slot loading a different ``use_timestep`` mid-stream
        # can't corrupt this slot's transformer __init__.
        ZImageTransformerBlock._laion_force_disable_modulation = not use_timestep_v
        FinalLayer._laion_default_modulation = use_timestep_v
        try:
            components, timings = _run_components(tasks, max_workers=max_workers)
        finally:
            ZImageTransformerBlock._laion_force_disable_modulation = prev_force_disable
            FinalLayer._laion_default_modulation = prev_default_modulation
        timings["wait_pipeline_lock"] = timings_wait_lock

        tokenizer, text_encoder, _cap_feat_dim = components["text_encoder"]
        video_vae = components["video_vae"]
        audio_vae = components["audio_vae"]
        scheduler = components["scheduler"]
        video_transformer = components["video_branch"]
        audio_transformer = components["audio_branch"]

        bridge_descriptor = _read_bridge_config(cdir)
        transformer_cfg = rcfg.get("transformer", {}) or {}
        bridge_interval = int(
            bridge_descriptor.get("bridge_interval", transformer_cfg.get("bridge_interval", 2))
        )
        use_asymmetric_ati = bool(
            bridge_descriptor.get(
                "use_asymmetric_ati", transformer_cfg.get("use_asymmetric_ati", False)
            )
        )
        a2v_window_size = int(
            bridge_descriptor.get(
                "a2v_window_size", transformer_cfg.get("a2v_window_size", 1)
            )
        )
        qk_norm = bool(
            transformer_video_cfg.get("qk_norm", True)
            and transformer_audio_cfg.get("qk_norm", True)
        )
        norm_eps = float(
            min(
                transformer_video_cfg.get("norm_eps", 1e-5),
                transformer_audio_cfg.get("norm_eps", 1e-5),
            )
        )

        t_compose_0 = time.time()
        joint = BridgedZImageJointModel(
            video_transformer=video_transformer,
            audio_transformer=audio_transformer,
            bridge_interval=bridge_interval,
            bridge_enabled=True,
            use_asymmetric_ati=use_asymmetric_ati,
            a2v_window_size=a2v_window_size,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        load_bridges_from_dir(joint, cdir / "bridges")
        timings["compose_joint"] = time.time() - t_compose_0

        t_move_0 = time.time()
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        # The text encoder and both transformer branches were loaded
        # direct-to-``dev`` via ``device_map``; ``.to(dev)`` here is a
        # no-op for them but still needs to run on the joint module so
        # the freshly-constructed bridges (created on CPU in fp32 by
        # ``BridgedZImageJointModel.__init__``) move + cast to the
        # branches' device / dtype. VAEs are still loaded on CPU by the
        # trainer helpers, so the ``.to(dev)`` actually moves them.
        text_encoder.eval()
        video_vae.to(dev).eval()
        audio_vae.to(dev).eval()
        joint.to(device=dev, dtype=transformer_dtype).eval()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        timings["to_device"] = time.time() - t_move_0

    # Surface the exact VAE paths consulted so users never have to grep
    # the resolved yaml to know what was loaded. ``audio_vae`` uses
    # ``model_path`` whereas video VAE uses ``model_name_or_path`` --
    # respect both naming conventions when reporting.
    vae_path_used = rcfg.get("vae", {}).get("model_name_or_path", "<missing>")
    vae_type_used = rcfg.get("vae", {}).get("type", "<missing>")
    audio_vae_path_used = (
        rcfg.get("audio_vae", {}).get("model_path")
        or rcfg.get("audio_vae", {}).get("model_name_or_path")
        or "<missing>"
    )
    audio_vae_type_used = rcfg.get("audio_vae", {}).get("type", "<missing>")
    _emit_progress(
        f"[t2av_pipeline]   video_vae : type={vae_type_used} path={vae_path_used}"
    )
    _emit_progress(
        f"[t2av_pipeline]   audio_vae : type={audio_vae_type_used} path={audio_vae_path_used}"
    )

    train_patch_size = int(transformer_video_cfg["all_patch_size"][0])
    train_f_patch_size = int(transformer_video_cfg["all_f_patch_size"][0])

    train_cfg = rcfg.get("train", {}) or {}
    shift_v = float(train_cfg.get("shift_v", 5.0))
    shift_a = float(train_cfg.get("shift_a", 1.0))
    predict_target = str(transformer_video_cfg.get("predict_target", "v"))

    timings["total"] = time.time() - overall_t0
    _emit_progress(
        f"[t2av_pipeline] load complete in {timings['total']:.1f}s "
        f"(compose_joint={timings.get('compose_joint', 0.0):.1f}s, "
        f"to_device={timings.get('to_device', 0.0):.1f}s) "
        f"for {cdir}"
    )

    return T2AVPipeline(
        joint_model=joint,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        scheduler=scheduler,
        run_config=rcfg,
        run_dir=rdir,
        checkpoint_dir=cdir,
        checkpoint_step=int(extract_checkpoint_step(cdir)),
        device=dev,
        train_patch_size=train_patch_size,
        train_f_patch_size=train_f_patch_size,
        shift_v=shift_v,
        shift_a=shift_a,
        predict_target=predict_target,
    )


# ----------------------------------------------------------------------
# Per-prompt generation
# ----------------------------------------------------------------------
_VALID_MODES = ("joint_av", "video_only", "audio_only")
_VALID_CFG_MODES = ("simple", "dual")


def _build_run_config_for_request(
    base_config: dict,
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    audio_duration_seconds: float,
) -> dict:
    """Return a deep-copied config with the ``validation`` block overridden
    so the shared shape helpers (``_resolve_video_latent_shape`` /
    ``_resolve_audio_latent_shape`` from ``omnivae_generation.trainer.joint_av.validation``)
    pick up the per-request video / audio target dimensions.
    """
    config = copy.deepcopy(base_config)
    val_cfg = config.setdefault("validation", {})
    val_cfg["video_frame_size"] = [int(height), int(width)]
    val_cfg["video_num_frames"] = int(num_frames)
    val_cfg["video_fps"] = float(fps)
    val_cfg["audio_duration_seconds"] = float(audio_duration_seconds)
    return config


@torch.no_grad()
def generate_one_av(
    pipe: T2AVPipeline,
    *,
    prompt: str,
    negative_prompt: str = "",
    mode: str = "joint_av",
    cfg_mode: str = "simple",
    num_inference_steps: int = 50,
    video_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    cfg_normalization: bool = False,
    video_text_guidance: Optional[float] = None,
    video_modality_guidance: Optional[float] = None,
    audio_text_guidance: Optional[float] = None,
    audio_modality_guidance: Optional[float] = None,
    num_frames: int = 121,
    fps: float = 24.0,
    height: int = 256,
    width: int = 256,
    audio_duration_seconds: float = 5.04,
    seed: int = 20260508,
    output_dir: str | Path,
    file_stem: str,
    video_quality: int = 8,
    wrap_task_prefix: bool = True,
    task_prefix_kind: str = "t2av",
    append_duration_suffix: bool = True,
    duration_precision: int = 1,
) -> dict[str, Any]:
    """Run one denoising pass and write the decoded media to ``output_dir``.

    Returns a record with output paths, decoded counts, and timing
    metadata so callers can render the result and / or persist a
    JSON sidecar.

    Implementation notes
    --------------------
    * The denoising / CFG layouts (simple NFE=2 vs BridgeDiT dual NFE=3)
      are delegated to :func:`omnivae_generation.trainer.joint_av.validation._denoise_joint`
      verbatim, including the per-sample bridge mask for dual CFG.
    * In ``video_only`` / ``audio_only`` modes the inner helper auto-
      collapses dual to simple (no opposite-modality KV to gate); the
      ``cfg_mode`` argument from the caller therefore "asks for" dual
      but the bridge is structurally disabled, mirroring training-time
      behaviour.
    * The audio waveform is padded / trimmed to exactly
      ``num_frames * sample_rate / fps`` samples before being saved, and
      :func:`omnivae_generation.trainer.joint_av.validation._ffmpeg_mux_av` omits
      ``-shortest`` and uses ``-c:v copy``, which together guarantee
      ``ffprobe -count_frames`` on the muxed ``.av.mp4`` reports
      exactly ``num_frames`` (no tail B-frame drop).
    * ``wrap_task_prefix`` / ``append_duration_suffix`` mirror the
      ``task_kind`` / ``append_duration_suffix`` knobs the trainer's
      ``validation.joint_av_prompts`` block applies to every prompt
      before encoding. With both ON, the actual text fed to the text
      encoder is::

          "<one of 10 t2av templates>(<raw prompt>) duration: 5.0s"

      which matches the distribution the joint model was trained /
      validated on. Disabling these is intended for advanced users who
      already include the prefix / suffix in their raw prompt -- and
      strongly NOT recommended for distilled student models, which are
      especially sensitive to prompt-distribution drift.
    """
    from omnivae_generation.trainer.audio_task_prefix import apply_task_prefix
    from omnivae_generation.trainer.data import maybe_format_chat_prompt
    from omnivae_generation.trainer.joint_av.validation import (
        _build_inference_scheduler,
        _denoise_joint,
        _ffmpeg_mux_av,
        _resolve_audio_latent_shape,
        _resolve_video_latent_shape,
        _video_tensor_to_uint8_frames,
    )
    from omnivae_generation.trainer.modeling import decode_latents_to_images, encode_prompts

    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}.")
    if cfg_mode not in _VALID_CFG_MODES:
        raise ValueError(f"cfg_mode must be one of {_VALID_CFG_MODES}, got {cfg_mode!r}.")

    config = _build_run_config_for_request(
        pipe.run_config,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        audio_duration_seconds=audio_duration_seconds,
    )
    device = pipe.device

    (
        requested_num_frames,
        latent_frames,
        eff_height,
        eff_width,
        latent_height,
        latent_width,
        eff_fps,
    ) = _resolve_video_latent_shape(config, pipe.video_vae)
    audio_in_channels, t_latent, sample_rate, eff_audio_duration = _resolve_audio_latent_shape(
        config, pipe.audio_vae
    )
    video_in_channels = int(config["transformer_video"]["in_channels"])

    max_seq_len = int(config["text_encoder"]["max_sequence_length"])
    cache_enabled = bool(config["text_encoder"].get("cache_enabled", False))

    # Wrap the positive prompt the same way the trainer's
    # ``run_joint_av_validation`` does: random t2av template + duration
    # suffix. The template is picked deterministically from the user's
    # seed (so same prompt + same seed -> same wrapping; bumping the
    # seed rotates the template along with the noise). Negative prompt
    # is left untouched -- the trainer's validation always uses an
    # unwrapped empty string for the unconditional branch.
    wrapped_prompt = prompt
    if wrap_task_prefix and task_prefix_kind:
        wrapped_prompt = apply_task_prefix(
            str(task_prefix_kind), wrapped_prompt, rng=random.Random(int(seed))
        )
    if append_duration_suffix:
        suffix = (
            f" duration: {float(audio_duration_seconds):.{int(duration_precision)}f}s"
        )
        wrapped_prompt = f"{wrapped_prompt}{suffix}"

    formatted_pos = maybe_format_chat_prompt(wrapped_prompt, pipe.tokenizer)
    formatted_neg = maybe_format_chat_prompt(negative_prompt, pipe.tokenizer)
    prompt_embeds_pos = encode_prompts(
        [formatted_pos], pipe.tokenizer, pipe.text_encoder, device, max_seq_len, cache_enabled=cache_enabled
    )
    prompt_embeds_neg = encode_prompts(
        [formatted_neg], pipe.tokenizer, pipe.text_encoder, device, max_seq_len, cache_enabled=cache_enabled
    )

    video_scheduler = _build_inference_scheduler(
        pipe.scheduler,
        shift=pipe.shift_v,
        num_inference_steps=int(num_inference_steps),
        device=device,
        predict_target=pipe.predict_target,
    )
    audio_scheduler = _build_inference_scheduler(
        pipe.scheduler,
        shift=pipe.shift_a,
        num_inference_steps=int(num_inference_steps),
        device=device,
        predict_target=pipe.predict_target,
    )

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    video_latents_init: Optional[torch.Tensor] = None
    audio_latents_init: Optional[torch.Tensor] = None
    if mode in ("joint_av", "video_only"):
        video_latents_init = torch.randn(
            (1, video_in_channels, latent_frames, latent_height, latent_width),
            generator=generator, device=device, dtype=torch.float32,
        )
    if mode in ("joint_av", "audio_only"):
        audio_latents_init = torch.randn(
            (1, audio_in_channels, t_latent),
            generator=generator, device=device, dtype=torch.float32,
        )

    fake_accelerator = SimpleNamespace(device=device)

    t0 = time.time()
    final_video, final_audio = _denoise_joint(
        joint_model=pipe.joint_model,
        config=config,
        accelerator=fake_accelerator,
        mode=mode,
        video_latents_init=video_latents_init,
        audio_latents_init=audio_latents_init,
        video_scheduler=video_scheduler if video_latents_init is not None else None,
        audio_scheduler=audio_scheduler if audio_latents_init is not None else None,
        prompt_embeds_video_pos=prompt_embeds_pos if video_latents_init is not None else None,
        prompt_embeds_video_neg=prompt_embeds_neg if video_latents_init is not None else None,
        prompt_embeds_audio_pos=prompt_embeds_pos if audio_latents_init is not None else None,
        prompt_embeds_audio_neg=prompt_embeds_neg if audio_latents_init is not None else None,
        guidance_scale_v=float(video_guidance_scale),
        guidance_scale_a=float(audio_guidance_scale),
        cfg_normalization=bool(cfg_normalization),
        train_patch_size=pipe.train_patch_size,
        train_f_patch_size=pipe.train_f_patch_size,
        cfg_mode=cfg_mode,
        video_text_guidance=video_text_guidance,
        video_modality_guidance=video_modality_guidance,
        audio_text_guidance=audio_text_guidance,
        audio_modality_guidance=audio_modality_guidance,
    )

    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "mode": mode,
        "cfg_mode": cfg_mode,
        "prompt": prompt,
        "wrapped_prompt": wrapped_prompt,
        "wrap_task_prefix": bool(wrap_task_prefix),
        "task_prefix_kind": (str(task_prefix_kind) if (wrap_task_prefix and task_prefix_kind) else None),
        "append_duration_suffix": bool(append_duration_suffix),
        "duration_precision": int(duration_precision),
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "video_guidance_scale": float(video_guidance_scale),
        "audio_guidance_scale": float(audio_guidance_scale),
        "cfg_normalization": bool(cfg_normalization),
        "shift_v": float(pipe.shift_v),
        "shift_a": float(pipe.shift_a),
        "requested_num_frames": int(requested_num_frames),
        "latent_frames": int(latent_frames),
        "height": int(eff_height),
        "width": int(eff_width),
        "fps": float(eff_fps),
        "audio_duration_seconds": float(eff_audio_duration),
    }
    if cfg_mode == "dual":
        record.update(
            video_text_guidance=float(
                video_text_guidance if video_text_guidance is not None else video_guidance_scale
            ),
            video_modality_guidance=float(
                video_modality_guidance if video_modality_guidance is not None else video_guidance_scale
            ),
            audio_text_guidance=float(
                audio_text_guidance if audio_text_guidance is not None else audio_guidance_scale
            ),
            audio_modality_guidance=float(
                audio_modality_guidance if audio_modality_guidance is not None else audio_guidance_scale
            ),
        )

    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    av_path: Optional[Path] = None

    if final_video is not None:
        video_decoded = decode_latents_to_images(
            final_video.to(dtype=getattr(pipe.video_vae, "dtype", final_video.dtype)),
            pipe.video_vae,
        )[0]
        frames = _video_tensor_to_uint8_frames(video_decoded)
        video_path = out_root / f"{file_stem}.mp4"
        imageio.mimsave(
            str(video_path),
            list(frames),
            fps=float(eff_fps),
            codec="libx264",
            quality=int(video_quality),
            macro_block_size=None,
        )
        record["video_path"] = str(video_path)
        record["decoded_num_frames"] = int(video_decoded.shape[1])

    if final_audio is not None:
        audio_out = pipe.audio_vae.decode(
            final_audio.to(dtype=getattr(pipe.audio_vae, "dtype", final_audio.dtype))
        )
        wave = audio_out[0, 0].float().clamp(-1.0, 1.0).cpu().unsqueeze(0)

        # ---- AV duration alignment ----
        # Pad / trim the waveform to ``requested_num_frames *
        # sample_rate / fps`` samples so the muxed ``.av.mp4`` has
        # matching nominal duration. Combined with ``_ffmpeg_mux_av``
        # omitting ``-shortest``, this guarantees ``ffprobe -count_frames``
        # on the muxed mp4 reports exactly ``requested_num_frames``.
        if final_video is not None:
            target_audio_len = int(round(
                int(requested_num_frames) * int(sample_rate) / float(eff_fps)
            ))
            cur_len = int(wave.shape[-1])
            if cur_len < target_audio_len:
                wave = F.pad(wave, (0, target_audio_len - cur_len))
            elif cur_len > target_audio_len:
                wave = wave[..., :target_audio_len].contiguous()

        audio_path = out_root / f"{file_stem}.wav"
        torchaudio.save(str(audio_path), wave, sample_rate=int(sample_rate))
        record["audio_path"] = str(audio_path)
        record["sample_rate"] = int(sample_rate)

    if mode == "joint_av" and video_path is not None and audio_path is not None:
        av_path = out_root / f"{file_stem}.av.mp4"
        if _ffmpeg_mux_av(video_path, audio_path, av_path):
            record["av_path"] = str(av_path)
        else:
            logger.warning(
                "t2av_pipeline: ffmpeg mux failed for %s + %s; falling back "
                "to side-by-side .mp4 / .wav.",
                video_path, audio_path,
            )

    record["elapsed_s"] = float(time.time() - t0)

    del final_video, final_audio
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    return record


def write_record_sidecar(record: dict[str, Any], sidecar_path: str | Path) -> None:
    """Write a ``.json`` sidecar with the full request + result record."""
    Path(sidecar_path).write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def check_ffmpeg_available() -> bool:
    """Mirror :func:`omnivae_generation.trainer.joint_av.validation._has_ffmpeg` without
    importing it (so the UI / CLI can warn early if mux will fail)."""
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"], check=False, capture_output=True, text=True
        )
    except FileNotFoundError:
        return False
    return completed.returncode == 0
