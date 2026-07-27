"""Three-mode validation for the T2AV joint model.

For every step we exercise:

* ``joint_av``      -- both branches active, bridges contribute deltas
* ``video_only``    -- bridges disabled; video branch runs alone
* ``audio_only``    -- bridges disabled; audio branch runs alone

Each mode persists its samples under
``<output_dir>/samples/step-XXXXXXXX/<mode>/`` and uploads previews to
the active accelerator trackers (TensorBoard / wandb). When ``ffmpeg``
is available on PATH the joint mode also muxes audio + video into a
single MP4 (`<sample-id>.av.mp4`); otherwise the .wav and .mp4 are
written side-by-side and the failure is logged once.

Implementation notes
--------------------

* Each modality keeps its own :class:`FlowMatchEulerDiscreteScheduler`
  instance, with the scheduler ``shift`` overridden to the per-modality
  value used during training (``shift_v`` / ``shift_a``). This matches
  the dual sigma-shift recipe and lets the two branches walk
  independent sigma schedules within the same denoising loop.
* We CFG-double the *latents* and the prompt-embedding lists (``[pos,
  neg]``) so a single joint forward computes both branches. The bridge
  cross-attention thus lets ``video_cond`` interact with ``audio_cond``
  and ``video_uncond`` interact with ``audio_uncond``, which is the
  expected behaviour under independent CFG of the two modalities.
* The bridge stack is toggled on/off via
  ``joint_model.bridge_enabled`` so video-only / audio-only modes share
  the *exact* same forward path as joint mode (no separate code path,
  no risk of the modes drifting from training-time behaviour).
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from accelerate.utils import gather_object
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm

from omnivae_generation.trainer.audio_task_prefix import apply_task_prefix
from omnivae_generation.trainer.data import maybe_format_chat_prompt
from omnivae_generation.trainer.modeling import (
    configure_scheduler_prediction_target,
    decode_latents_to_images,
    encode_prompts,
)
from omnivae_generation.trainer.utils import ensure_dir, save_json
from omnivae_generation.trainer.video_validation import apply_zimage_cfg


logger = logging.getLogger(__name__)

_VALIDATION_SAMPLE_SEED_STRIDE = 100_003
_DEFAULT_MODES: tuple[str, ...] = ("joint_av", "video_only", "audio_only")
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]+")

# Fallback offset for jsonl rows whose ``index`` field is missing or
# non-numeric (e.g. versebench_expanded.jsonl set2 rows with
# ``index="clip_05f5760d"``). Without an offset the fallback uses the
# raw ``line_idx`` directly, which can collide with legitimate numeric
# ``index`` values from sibling rows of the same ``type`` -- on
# versebench this clobbered ~64 of the 600 samples on disk (two records
# racing to write the same ``sample-versebench-0309-set2.*`` file).
# Adding ~10M to the fallback puts it in a strictly higher namespace
# than any real numeric index in any of our jsonl sources, so the two
# ranges become disjoint by construction.
#
# Three-way mirror: this constant MUST equal
#   * ``infer/t2av/infer_t2av.INDEX_FALLBACK_OFFSET``
#   * ``infer/t2av/build_sample_manifest.INDEX_FALLBACK_OFFSET``
# or the (type, index) lookup at eval time will mis-match what the
# trainer wrote to disk. Keep all three in lockstep.
INDEX_FALLBACK_OFFSET = 10_000_000


# ---------------------------------------------------------- prompt loading
@dataclass
class ValPromptRecord:
    """One validation prompt with all the metadata needed to format it
    consistently with the training distribution and to label outputs in
    wandb / on disk."""
    text: str                       # raw text from source jsonl (for logging)
    formatted: str                  # task-prefix + duration-suffix applied (pre chat-template)
    type_label: str                 # entry["type"], for wandb grouping
    source_name: str                # which prompt source (e.g. "compass", "basetts_valid", "tta_general_en")
    source_index: int               # 0-based offset within the source
    entry_index: int                # int(entry["index"]) when parseable, else
                                    # INDEX_FALLBACK_OFFSET + line_idx (kept in
                                    # lockstep with infer_t2av / build_sample_manifest)
    task_kind: Optional[str]        # "tts" / "tta" / None


def _filename_safe(value: str, *, fallback: str = "x") -> str:
    """Sanitize ``value`` for use in a filesystem path."""
    cleaned = _FILENAME_SAFE_RE.sub("_", str(value)).strip("_")
    return cleaned or fallback


def _cfg_dir_name(cfg_mode: str, val_cfg: dict) -> str:
    """Directory name for one CFG variant.

    Historical validation writes ``cfg_simple`` / ``cfg_dual``. Sweeps that
    evaluate several numeric CFG values can set ``cfg_output_suffix`` (for
    example ``g4``) to avoid clobbering samples from another value while
    keeping the base variant visible in the path.
    """
    explicit = val_cfg.get("cfg_output_dir")
    if explicit:
        return _filename_safe(str(explicit), fallback=f"cfg_{cfg_mode}")
    base = f"cfg_{cfg_mode}"
    suffix = val_cfg.get("cfg_output_suffix")
    if suffix:
        return f"{base}_{_filename_safe(str(suffix), fallback='x')}"
    return base


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_sample_index(record_index: Any, row_index: int) -> int:
    """Mirror of ``infer_t2av._resolve_sample_index`` /
    ``build_sample_manifest.load_valid_jsonl`` fallback logic.

    Returns ``int(record_index)`` when it parses as an int, else
    ``INDEX_FALLBACK_OFFSET + row_index``. The offset is what keeps
    numeric and non-numeric ``index`` rows of the same ``type`` in
    disjoint filename namespaces -- see the constant's docstring for
    the versebench collision this was added to prevent.
    """
    try:
        return int(record_index)
    except (TypeError, ValueError):
        return INDEX_FALLBACK_OFFSET + int(row_index)


def _resolve_max_examples(spec: dict) -> Optional[int]:
    """Resolve the per-source cap on number of prompts.

    Reads ``max_examples`` (preferred) or legacy ``max_prompts``. Any of
    ``None`` / missing / ``0`` / negative is treated as "no cap, take
    everything".
    """
    raw = spec.get("max_examples")
    if raw is None:
        raw = spec.get("max_prompts")
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _as_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = raw.replace("\n", ",").split(",")
        return [p.strip() for p in parts if p.strip()]
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            out.extend(_as_string_list(item))
        return out
    text = str(raw).strip()
    return [text] if text else []


def _entry_matches_type_filter(
    entry: dict,
    *,
    allowed_values: Optional[set[str]],
    filter_fields: list[str],
) -> bool:
    if not allowed_values:
        return True
    for field in filter_fields:
        value = entry.get(field)
        if isinstance(value, (list, tuple, set)):
            values = value
        else:
            values = [value]
        for item in values:
            if item is not None and str(item) in allowed_values:
                return True
    return False


def _load_jsonl_prompts(
    spec: dict,
    *,
    default_text_field: str,
    default_task_kind: Optional[str] = None,
    audio_duration_seconds: Optional[float] = None,
) -> list[ValPromptRecord]:
    """Load prompts from a single jsonl spec.

    Recognised keys::

        jsonl_path / path        Required jsonl file.
        text_field / prompt_field
                                 Field to read the prompt from. If the field
                                 is a list (e.g. ``caption_en``) we pick the
                                 first non-empty item deterministically.
        type_field, index_field  Optional metadata fields used for wandb
                                 grouping and per-sample filenames.
        task_kind                "tts" / "tta" / "t2av" / null. When set we
                                 wrap the raw text with
                                 ``apply_task_prefix(...)`` so the validation
                                 prompt distribution matches the
                                 corresponding training distribution
                                 (``t2av`` mirrors AVPairedJsonlDataset's
                                 joint-modality instruction wrapper, used
                                 by the joint_av validation mode).
        append_duration_suffix   bool, default False. When true we append
                                 ``" duration: X.Xs"`` (mirrors the
                                 AVPairedJsonlDataset / AudioJsonlT2ADataset
                                 prompt format).
        duration_precision       int, default 1. Decimals in the suffix.
        max_examples             int / null. Cap the number of prompts
                                 loaded. ``null`` / missing / ``0`` /
                                 negative all mean "take everything"
                                 (legacy alias: ``max_prompts``).
        type_filter / types      Optional exact-match whitelist applied
                                 before max_examples. Values can be a
                                 comma-separated string or a list.
        type_filter_field(s)     Field(s) checked against the type filter.
                                 Defaults to type_field, then "type".
        name                     Override the source label (defaults to the
                                 file stem).
    """
    raw_path = spec.get("jsonl_path") or spec.get("path")
    if not raw_path:
        raise ValueError(f"Validation jsonl spec missing 'jsonl_path': {spec!r}")
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Validation jsonl not found: {path}")

    text_field = str(spec.get("text_field") or spec.get("prompt_field") or default_text_field)
    type_field = spec.get("type_field")
    index_field = spec.get("index_field")
    task_kind = spec.get("task_kind", default_task_kind)
    name = str(spec.get("name") or path.stem)
    max_examples = _resolve_max_examples(spec)
    append_duration = bool(spec.get("append_duration_suffix", False))
    duration_precision = max(0, int(spec.get("duration_precision", 1)))
    type_filter_values = set(
        _as_string_list(
            spec.get("type_filter")
            if spec.get("type_filter") is not None
            else spec.get("types")
        )
    )
    filter_fields = _as_string_list(
        spec.get("type_filter_fields")
        if spec.get("type_filter_fields") is not None
        else spec.get("type_filter_field")
    )
    if not filter_fields:
        filter_fields = [str(type_field or "type")]

    # Deterministic random for picking task_prefix templates / list
    # captions, so the same step gets the same wrapped prompt across
    # ranks.
    rng = random.Random(0)

    records: list[ValPromptRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _entry_matches_type_filter(
                entry,
                allowed_values=type_filter_values or None,
                filter_fields=filter_fields,
            ):
                continue
            raw = entry.get(text_field)
            if isinstance(raw, list):
                raw = next(
                    (str(x).strip() for x in raw if x and str(x).strip()), None,
                )
            if not raw:
                continue
            text = str(raw).strip()
            formatted = text
            if task_kind:
                formatted = apply_task_prefix(str(task_kind), formatted, rng=rng)
            if append_duration and audio_duration_seconds is not None:
                fmt = f"{{:.{duration_precision}f}}"
                formatted = f"{formatted} duration: {fmt.format(float(audio_duration_seconds))}s"
            # Disjoint-namespace resolver: numeric ``index`` stays as-is
            # (so the (type, index) eval lookup keeps working), but
            # non-numeric rows fall back to ``INDEX_FALLBACK_OFFSET +
            # line_idx`` rather than the raw line index. Mirrors the
            # inference / eval pipeline so all three agree on filenames.
            entry_index = _resolve_sample_index(
                entry.get(index_field) if index_field else None, line_idx,
            )
            records.append(
                ValPromptRecord(
                    text=text,
                    formatted=formatted,
                    type_label=str(entry.get(type_field)) if type_field else "",
                    source_name=name,
                    source_index=len(records),
                    entry_index=entry_index,
                    task_kind=str(task_kind) if task_kind else None,
                )
            )
            if max_examples is not None and len(records) >= max_examples:
                break
    return records


def _load_inline_prompts(spec: Any, *, name: str) -> list[ValPromptRecord]:
    """Materialise an inline list of strings.

    ``spec`` can be either a bare list of strings or a dict like
    ``{prompts: [...], max_examples: 8, name: "foo"}``. ``max_examples``
    follows the same null/0 = "all" convention as the jsonl loader.
    """
    actual_name = name
    if isinstance(spec, dict):
        items = list(spec.get("prompts") or [])
        max_examples = _resolve_max_examples(spec)
        if max_examples is not None:
            items = items[:max_examples]
        if spec.get("name"):
            actual_name = str(spec["name"])
    else:
        items = list(spec or [])
    out: list[ValPromptRecord] = []
    for i, p in enumerate(items):
        text = str(p).strip()
        if not text:
            continue
        out.append(
            ValPromptRecord(
                text=text,
                formatted=text,
                type_label="",
                source_name=actual_name,
                source_index=len(out),
                entry_index=i,
                task_kind=None,
            )
        )
    return out


def _resolve_mode_prompts(
    val_cfg: dict,
    mode: str,
    *,
    audio_duration_seconds: float,
) -> list[ValPromptRecord]:
    """Build the prompt list for one validation ``mode``.

    Resolution order (first hit wins):

      * mode-specific prompt set / spec under ``validation.<mode>_prompts``
        (or ``validation.<mode>_prompt_sets`` for audio-only multi-source);
      * legacy ``validation.prompts`` list (back-compat).

    Audio-only is special: it can take an *array* of jsonl specs (TTS +
    TTA mix) rather than a single source, so each spec gets its task
    prefix applied independently.
    """
    legacy = val_cfg.get("prompts") or []

    if mode == "joint_av":
        spec = val_cfg.get("joint_av_prompts")
        if isinstance(spec, dict) and (spec.get("jsonl_path") or spec.get("path")):
            return _load_jsonl_prompts(
                spec,
                default_text_field="av_caption",
                audio_duration_seconds=audio_duration_seconds,
            )
        if isinstance(spec, list):
            return _load_inline_prompts(spec, name="joint_av_inline")
        return _load_inline_prompts(legacy, name="joint_av_legacy")

    if mode == "video_only":
        spec = val_cfg.get("video_only_prompts")
        if isinstance(spec, dict) and (spec.get("jsonl_path") or spec.get("path")):
            return _load_jsonl_prompts(
                spec,
                default_text_field=spec.get("text_field") or "video_caption",
                audio_duration_seconds=audio_duration_seconds,
            )
        if isinstance(spec, list) or isinstance(spec, dict):
            return _load_inline_prompts(spec, name="video_only_inline")
        return _load_inline_prompts(legacy, name="video_only_legacy")

    if mode == "audio_only":
        sets = val_cfg.get("audio_only_prompt_sets")
        if isinstance(sets, list) and sets:
            records: list[ValPromptRecord] = []
            for s in sets:
                records.extend(
                    _load_jsonl_prompts(
                        s,
                        default_text_field="text",
                        audio_duration_seconds=audio_duration_seconds,
                    )
                )
            return records
        spec = val_cfg.get("audio_only_prompts")
        if isinstance(spec, dict) and (spec.get("jsonl_path") or spec.get("path")):
            return _load_jsonl_prompts(
                spec,
                default_text_field="text",
                audio_duration_seconds=audio_duration_seconds,
            )
        if isinstance(spec, list) or isinstance(spec, dict):
            return _load_inline_prompts(spec, name="audio_only_inline")
        return _load_inline_prompts(legacy, name="audio_only_legacy")

    raise ValueError(f"Unknown validation mode {mode!r}")


# ----------------------------------------------------------------- helpers
def _video_tensor_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """``[C, T, H, W]`` float in ``[-1, 1]`` -> ``[T, H, W, C]`` uint8."""
    frames = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).cpu().numpy()


def _build_inference_scheduler(
    base_scheduler,
    *,
    shift: float,
    num_inference_steps: int,
    device: torch.device,
    predict_target: str,
) -> FlowMatchEulerDiscreteScheduler:
    """Per-modality inference scheduler, with the scheduler's static
    shift overridden to the per-branch value (mirrors the dual
    sigma-shift recipe used at training time)."""
    inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler.config)
    inference_scheduler.config.shift = float(shift)
    inference_scheduler.config.use_dynamic_shifting = False
    configure_scheduler_prediction_target(inference_scheduler, predict_target)
    inference_scheduler.set_timesteps(int(num_inference_steps), device=device)
    return inference_scheduler


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_mux_av(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Best-effort ffmpeg mux. Returns False (and logs) on failure so
    the validation loop can continue.
    """
    if not _has_ffmpeg():
        logger.warning(
            "joint_av_validation: ffmpeg not found on PATH; skipping mux for %s + %s",
            video_path, audio_path,
        )
        return False
    # ``-shortest`` is intentionally omitted: the audio waveform is
    # padded / trimmed to ``num_frames * sample_rate / fps`` samples
    # before being saved (see ``run_joint_av_validation``), so both
    # streams already have identical nominal duration. With
    # ``-c:v copy`` operating at packet level, leaving ``-shortest`` on
    # was costing 1-3 trailing video frames per sample because libx264
    # places a few B-frame reorder packets just past the audio EOF
    # DTS; the muxer would drop them and ``ffprobe -count_frames``
    # would only see 117-120 of the 121 generated frames. Without
    # ``-shortest`` the muxer keeps every packet and the resulting
    # ``.av.mp4`` consistently reports 121 frames.
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        str(output_path),
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            logger.warning(
                "joint_av_validation: ffmpeg mux failed (%s): %s",
                output_path, completed.stderr.strip()[:512],
            )
            return False
    except Exception as exc:  # pragma: no cover -- best-effort
        logger.warning("joint_av_validation: ffmpeg mux exception: %r", exc)
        return False
    return True


def _prompt_file_stem(rec_meta: ValPromptRecord) -> str:
    """Return the on-disk stem used for one validation prompt."""
    source_tag = _filename_safe(rec_meta.source_name, fallback="src")
    type_tag = _filename_safe(rec_meta.type_label, fallback="") if rec_meta.type_label else ""
    file_stem = f"sample-{source_tag}-{rec_meta.entry_index:04d}"
    if type_tag:
        file_stem = f"{file_stem}-{type_tag}"
    return file_stem


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _sample_outputs_complete(mode_dir: Path, file_stem: str, mode: str) -> bool:
    """Check whether the durable files needed by downstream eval exist.

    The sidecar JSON is written by the resume-aware path added after older
    validation runs already existed. For backwards compatibility we therefore
    treat non-empty media files as authoritative even when the sidecar is
    missing. The muxed ``.av.mp4`` is optional because evaluation consumes the
    split ``.mp4`` + ``.wav`` pair and some environments intentionally run
    without ffmpeg.
    """
    if mode in ("joint_av", "video_only") and not _nonempty_file(mode_dir / f"{file_stem}.mp4"):
        return False
    if mode in ("joint_av", "audio_only") and not _nonempty_file(mode_dir / f"{file_stem}.wav"):
        return False
    return True


def _build_validation_record(
    *,
    rec_meta: ValPromptRecord,
    sample_index: int,
    mode: str,
    cfg_mode: str,
    sample_seed: Optional[int],
    mode_dir: Path,
    file_stem: str,
    fps: float,
    sample_rate: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_index": sample_index,
        "mode": mode,
        "cfg_mode": cfg_mode,
        "prompt": rec_meta.text,
        "formatted_prompt": rec_meta.formatted,
        "source_name": rec_meta.source_name,
        "source_index": rec_meta.source_index,
        "entry_index": rec_meta.entry_index,
        "type_label": rec_meta.type_label,
        "task_kind": rec_meta.task_kind,
        "sample_seed": sample_seed,
    }
    video_path = mode_dir / f"{file_stem}.mp4"
    if mode in ("joint_av", "video_only") and _nonempty_file(video_path):
        record["video_path"] = str(video_path)
        record["fps"] = float(fps)
    audio_path = mode_dir / f"{file_stem}.wav"
    if mode in ("joint_av", "audio_only") and _nonempty_file(audio_path):
        record["audio_path"] = str(audio_path)
        record["sample_rate"] = int(sample_rate)
    av_path = mode_dir / f"{file_stem}.av.mp4"
    if mode == "joint_av" and _nonempty_file(av_path):
        record["av_path"] = str(av_path)
    return record


def _write_sample_sidecar(mode_dir: Path, file_stem: str, record: dict[str, Any]) -> None:
    sidecar = mode_dir / f"{file_stem}.json"
    try:
        sidecar.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("joint_av_validation: failed to write sidecar %s: %r", sidecar, exc)


def _cfg_variants_for_mode(val_cfg: dict, mode: str) -> list[str]:
    cfg_modes = list(val_cfg.get("cfg_modes", ["simple"]))
    invalid_cfg = [c for c in cfg_modes if c not in ("simple", "dual")]
    if invalid_cfg:
        raise ValueError(
            f"Unknown validation.cfg_modes entries {invalid_cfg!r}; "
            "expected subset of ('simple', 'dual')."
        )
    if not cfg_modes:
        cfg_modes = ["simple"]
    mode_cfg_variants = [
        c for c in cfg_modes
        if mode == "joint_av" or c == "simple"
    ]
    if mode != "joint_av" and "simple" not in mode_cfg_variants:
        mode_cfg_variants = ["simple"]
    return mode_cfg_variants


def validation_outputs_complete(config: dict, step: int) -> bool:
    """Return True if every requested validation sample is already on disk."""
    val_cfg = config.get("validation") or {}
    modes = list(val_cfg.get("modes", _DEFAULT_MODES))
    invalid = [m for m in modes if m not in _DEFAULT_MODES]
    if invalid:
        raise ValueError(f"Unknown validation modes {invalid!r}; expected subset of {_DEFAULT_MODES!r}.")

    sample_root = Path(config["experiment"]["output_dir"]) / "samples" / f"step-{int(step):08d}"
    audio_duration_seconds = float(val_cfg.get("audio_duration_seconds", 8.0))
    for mode in modes:
        mode_records = _resolve_mode_prompts(
            val_cfg, mode, audio_duration_seconds=audio_duration_seconds,
        )
        if not mode_records:
            continue
        for cfg_mode in _cfg_variants_for_mode(val_cfg, mode):
            mode_dir = sample_root / mode / _cfg_dir_name(cfg_mode, val_cfg)
            if not mode_dir.is_dir():
                return False
            for rec_meta in mode_records:
                if not _sample_outputs_complete(mode_dir, _prompt_file_stem(rec_meta), mode):
                    return False
    return True


# ------------------------------------------------------------- shape utils
def _resolve_video_latent_shape(config: dict, video_vae) -> tuple[int, int, int, int, int, int, float]:
    val_cfg = config.get("validation", {})
    height, width = val_cfg.get("video_frame_size") or config["dataset"].get("frame_size") or [256, 256]
    height, width = int(height), int(width)
    num_frames = int(val_cfg.get("video_num_frames") or config["dataset"].get("num_frames") or 49)
    fps = float(val_cfg.get("video_fps") or config["dataset"].get("target_fps") or 24.0)
    spatial_scale = int(getattr(getattr(video_vae, "config", None), "scale_factor_spatial", 8))
    temporal_scale = int(getattr(getattr(video_vae, "config", None), "scale_factor_temporal", 4) or 1)
    if height % spatial_scale != 0 or width % spatial_scale != 0:
        raise ValueError(
            f"Joint AV validation video frame_size {(height, width)} must be divisible by "
            f"VAE spatial scale {spatial_scale}."
        )
    latent_frames = 1 + (num_frames - 1) // temporal_scale
    latent_height = height // spatial_scale
    latent_width = width // spatial_scale
    return num_frames, latent_frames, height, width, latent_height, latent_width, fps


def _audio_vae_hop_length(audio_vae, audio_vae_cfg: dict) -> int:
    hop = getattr(audio_vae, "hop_length", None)
    if hop is None:
        hop = audio_vae_cfg.get("hop_length")
    if hop is None:
        raise ValueError("Joint AV validation requires audio_vae.hop_length (or vae.hop_length attribute).")
    return int(hop)


def _resolve_audio_latent_shape(config: dict, audio_vae) -> tuple[int, int, int, float]:
    val_cfg = config.get("validation", {})
    duration_seconds = float(val_cfg.get("audio_duration_seconds", 8.0))
    sample_rate = int(config["dataset"].get("sample_rate", 48000))
    hop_length = _audio_vae_hop_length(audio_vae, config.get("audio_vae", {}))
    audio_in_channels = int(
        config.get("transformer_audio", {}).get("in_channels")
        or config.get("audio_transformer", {}).get("in_channels")
        or config.get("transformer", {}).get("in_channels", 128)
    )
    t_latent = max(1, int(round(duration_seconds * sample_rate / hop_length)))
    return audio_in_channels, t_latent, sample_rate, duration_seconds


# ----------------------------------------------------------- inner helpers
def _model_timesteps_for_branch(
    inference_scheduler: FlowMatchEulerDiscreteScheduler,
    timestep_value: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Mirror the trainer's normalisation: ``model_t = (N - t) / N`` with
    ``t`` the scheduler's discrete timestep index (already shifted by
    the scheduler's ``shift``)."""
    t = timestep_value.expand(batch_size).to(device=device, dtype=torch.float32)
    n_train = float(inference_scheduler.config.num_train_timesteps)
    return (n_train - t) / n_train


def _list_pos_neg(prompt_embeds_pos: list[torch.Tensor], prompt_embeds_neg: list[torch.Tensor]) -> list[torch.Tensor]:
    """Build the [pos_for_sample0, neg_for_sample0] list that the joint
    forward expects when CFG is doubled along the batch axis."""
    out: list[torch.Tensor] = []
    for pos, neg in zip(prompt_embeds_pos, prompt_embeds_neg):
        out.append(pos)
        out.append(neg)
    return out


def _list_triple_dual_cfg(
    prompt_embeds_pos: list[torch.Tensor],
    prompt_embeds_neg: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Build the per-sample triple ``[empty, empty, real]`` list that the
    joint forward expects when running BridgeDiT-style dual CFG (NFE=3).

    Layout (per source sample i):
      slot 0 -> ``v(z, empty, bridge=off)``  (uses empty prompt)
      slot 1 -> ``v(z, empty, bridge=on)``   (uses empty prompt)
      slot 2 -> ``v(z, real,  bridge=on)``   (uses real prompt)

    The bridge_mask=[F,T,T] handed to the joint forward decides which
    slots actually receive the cross-modal delta; the prompt embedding
    itself only needs to carry "empty" twice and "real" once per source
    sample.
    """
    out: list[torch.Tensor] = []
    for pos, neg in zip(prompt_embeds_pos, prompt_embeds_neg):
        out.append(neg)   # slot 0: (empty, bridge off)
        out.append(neg)   # slot 1: (empty, bridge on)
        out.append(pos)   # slot 2: (real,  bridge on)
    return out


def _build_triple_bridge_mask(
    per_sample_count: int, *, device: torch.device,
) -> torch.Tensor:
    """Per-sample bridge gate for dual CFG batch tripling.

    Produces ``[F, T, T, F, T, T, ...]`` with length
    ``3 * per_sample_count``.
    """
    base = torch.tensor([False, True, True], device=device)
    return base.repeat(per_sample_count)


def _apply_dual_cfg(
    v00: torch.Tensor,       # (z, empty, bridge=off)
    v0B: torch.Tensor,       # (z, empty, bridge=on)
    vTB: torch.Tensor,       # (z, real,  bridge=on)
    *,
    s_text: float,
    s_modality: float,
) -> torch.Tensor:
    """BridgeDiT Eq.6 dual CFG.

    pred = v00 + s_modality * (v0B - v00) + s_text * (vTB - v0B)
    """
    return (
        v00
        + float(s_modality) * (v0B - v00)
        + float(s_text) * (vTB - v0B)
    )


@torch.no_grad()
def _denoise_joint(
    *,
    joint_model,                       # BridgedZImageJointModel (unwrapped)
    config: dict,
    accelerator: Accelerator,
    mode: str,                         # "joint_av" | "video_only" | "audio_only"
    video_latents_init: Optional[torch.Tensor],
    audio_latents_init: Optional[torch.Tensor],
    video_scheduler: Optional[FlowMatchEulerDiscreteScheduler],
    audio_scheduler: Optional[FlowMatchEulerDiscreteScheduler],
    prompt_embeds_video_pos: Optional[list[torch.Tensor]],
    prompt_embeds_video_neg: Optional[list[torch.Tensor]],
    prompt_embeds_audio_pos: Optional[list[torch.Tensor]],
    prompt_embeds_audio_neg: Optional[list[torch.Tensor]],
    guidance_scale_v: float,
    guidance_scale_a: float,
    cfg_normalization: bool,
    train_patch_size: int,
    train_f_patch_size: int,
    cfg_mode: str = "simple",
    video_text_guidance: Optional[float] = None,
    video_modality_guidance: Optional[float] = None,
    audio_text_guidance: Optional[float] = None,
    audio_modality_guidance: Optional[float] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the joint denoising loop for one prompt pair.

    Returns ``(video_latent_clean, audio_latent_clean)``; either is
    ``None`` if that branch is inactive in the given mode.

    ``cfg_mode``:

    * ``"simple"`` -- legacy NFE=2 CFG. Latents are doubled along the
      batch dim and the bridge cross-attends within each cond/uncond
      leg. Equivalent to a special case of BridgeDiT's dual CFG with
      ``s_T = s_B = guidance_scale``.
    * ``"dual"``   -- BridgeDiT-style dual CFG (NFE=3, Eq.6). Latents
      are tripled, prompts are arranged as ``[empty, empty, real]``,
      and a per-sample ``bridge_mask=[F,T,T]`` gates the bridge in the
      first slot. The video branch uses
      ``(video_text_guidance, video_modality_guidance)``; the audio
      branch uses ``(audio_text_guidance, audio_modality_guidance)``.
      Only valid in ``mode == "joint_av"``; the single-modality modes
      transparently fall back to ``"simple"``.
    """
    device = accelerator.device

    cfg_mode = str(cfg_mode).lower()
    if cfg_mode not in ("simple", "dual"):
        raise ValueError(f"cfg_mode must be 'simple' or 'dual', got {cfg_mode!r}.")

    # Dual CFG only makes sense when *both* branches are active. In
    # video_only / audio_only the bridge is structurally disabled, so
    # ``v00`` and ``v0B`` would be identical and ``s_modality`` would
    # have no effect -- collapse back to simple CFG to save one model
    # call per step.
    if cfg_mode == "dual" and mode != "joint_av":
        cfg_mode = "simple"

    repeat_factor = 2 if cfg_mode == "simple" else 3

    video_latents = video_latents_init
    audio_latents = audio_latents_init
    if video_latents is not None:
        video_latents = video_latents.repeat(repeat_factor, 1, 1, 1, 1)
    if audio_latents is not None:
        if audio_latents.dim() == 5:
            audio_latents = audio_latents.repeat(repeat_factor, 1, 1, 1, 1)
        else:
            audio_latents = audio_latents.repeat(repeat_factor, 1, 1)

    if cfg_mode == "simple":
        pe_video = (
            _list_pos_neg(prompt_embeds_video_pos, prompt_embeds_video_neg)
            if prompt_embeds_video_pos is not None else None
        )
        pe_audio = (
            _list_pos_neg(prompt_embeds_audio_pos, prompt_embeds_audio_neg)
            if prompt_embeds_audio_pos is not None else None
        )
        bridge_mask: Optional[torch.Tensor] = None
    else:  # dual
        if prompt_embeds_video_pos is None or prompt_embeds_audio_pos is None:
            raise ValueError(
                "dual CFG requires both video and audio prompt embeddings "
                "(joint_av mode); single-modality fall-back should have "
                "switched to simple CFG before reaching this branch."
            )
        pe_video = _list_triple_dual_cfg(prompt_embeds_video_pos, prompt_embeds_video_neg)
        pe_audio = _list_triple_dual_cfg(prompt_embeds_audio_pos, prompt_embeds_audio_neg)
        per_sample_count = len(prompt_embeds_video_pos)
        bridge_mask = _build_triple_bridge_mask(per_sample_count, device=device)

    # Resolve dual-CFG scales (default to falling back to the single-
    # scalar values so simple CFG callers don't have to thread them).
    s_v_text = float(video_text_guidance if video_text_guidance is not None else guidance_scale_v)
    s_v_mod = float(video_modality_guidance if video_modality_guidance is not None else guidance_scale_v)
    s_a_text = float(audio_text_guidance if audio_text_guidance is not None else guidance_scale_a)
    s_a_mod = float(audio_modality_guidance if audio_modality_guidance is not None else guidance_scale_a)

    # Per-mode bridge gate. We restore the previous value at the end so
    # back-to-back validation modes don't accidentally desync from
    # training-time settings. In dual mode bridge is logically "on" for
    # the model (the per-sample bridge_mask gates individual slots).
    saved_bridge_enabled = bool(joint_model.bridge_enabled)
    joint_model.bridge_enabled = (mode == "joint_av")

    # The two schedulers walk in lockstep over inference_steps. We use
    # the longer of the two if the user ever configures different step
    # counts (currently we always feed them the same value).
    video_steps = list(video_scheduler.timesteps) if video_scheduler is not None else []
    audio_steps = list(audio_scheduler.timesteps) if audio_scheduler is not None else []
    n_steps = max(len(video_steps), len(audio_steps))

    try:
        for step_idx in range(n_steps):
            v_step = video_steps[step_idx] if step_idx < len(video_steps) else None
            a_step = audio_steps[step_idx] if step_idx < len(audio_steps) else None

            video_t = (
                _model_timesteps_for_branch(video_scheduler, v_step, device=device, batch_size=video_latents.shape[0])
                if v_step is not None and video_latents is not None else None
            )
            audio_t = (
                _model_timesteps_for_branch(audio_scheduler, a_step, device=device, batch_size=audio_latents.shape[0])
                if a_step is not None and audio_latents is not None else None
            )

            video_pred_list, audio_pred_list = joint_model(
                video_x=(
                    video_latents.to(dtype=getattr(joint_model.video, "dtype", video_latents.dtype))
                    if video_latents is not None else None
                ),
                video_t=video_t,
                audio_x=(
                    audio_latents.to(dtype=getattr(joint_model.audio, "dtype", audio_latents.dtype))
                    if audio_latents is not None else None
                ),
                audio_t=audio_t,
                prompt_embeds_video=pe_video if pe_video is not None else [],
                prompt_embeds_audio=pe_audio if pe_audio is not None else [],
                video_patch_size=train_patch_size,
                video_f_patch_size=train_f_patch_size,
                bridge_mask=bridge_mask,
            )

            if video_pred_list is not None and video_latents is not None:
                v_pred = joint_model.stack_branch_predictions(video_pred_list)
                if cfg_mode == "simple":
                    v_pos = v_pred[0::2]
                    v_neg = v_pred[1::2]
                    v_cfg = apply_zimage_cfg(v_pos, v_neg, guidance_scale_v, cfg_normalization)
                else:
                    v00 = v_pred[0::3]
                    v0B = v_pred[1::3]
                    vTB = v_pred[2::3]
                    v_cfg = _apply_dual_cfg(
                        v00, v0B, vTB, s_text=s_v_text, s_modality=s_v_mod,
                    )
                v_cfg = -v_cfg                              # match training trainer's sign convention
                # CFG output applies to the cond slot of the layout
                # (slot 0 in simple mode, slot 2 in dual mode); we then
                # re-tile after the scheduler step so the bridge keeps
                # seeing matched layout next round.
                stepped = video_scheduler.step(
                    v_cfg.to(torch.float32),
                    v_step,
                    video_latents[:1].to(torch.float32),
                    return_dict=False,
                )[0]
                video_latents = stepped.repeat(repeat_factor, 1, 1, 1, 1).to(torch.float32)

            if audio_pred_list is not None and audio_latents is not None:
                a_pred = joint_model.stack_branch_predictions(audio_pred_list)
                if cfg_mode == "simple":
                    a_pos = a_pred[0::2]
                    a_neg = a_pred[1::2]
                    a_cfg = apply_zimage_cfg(a_pos, a_neg, guidance_scale_a, cfg_normalization)
                else:
                    a00 = a_pred[0::3]
                    a0B = a_pred[1::3]
                    aTB = a_pred[2::3]
                    a_cfg = _apply_dual_cfg(
                        a00, a0B, aTB, s_text=s_a_text, s_modality=s_a_mod,
                    )
                a_cfg = -a_cfg
                # The audio branch returns 5-D ``[B, C, T, 1, 1]``; squeeze
                # back to 3-D so the inference scheduler step matches the
                # latent shape.
                if a_cfg.dim() == 5:
                    a_cfg = a_cfg.squeeze(-1).squeeze(-1)
                a_input = audio_latents
                if a_input.dim() == 5:
                    a_input = a_input.squeeze(-1).squeeze(-1)
                stepped = audio_scheduler.step(
                    a_cfg.to(torch.float32),
                    a_step,
                    a_input[:1].to(torch.float32),
                    return_dict=False,
                )[0]
                audio_latents = stepped.repeat(repeat_factor, 1, 1).to(torch.float32)
    finally:
        joint_model.bridge_enabled = saved_bridge_enabled

    final_video = video_latents[:1] if video_latents is not None else None
    final_audio = audio_latents[:1] if audio_latents is not None else None
    if final_audio is not None and final_audio.dim() == 5:
        final_audio = final_audio.squeeze(-1).squeeze(-1)
    return final_video, final_audio


# ------------------------------------------------------------ public entry
@torch.no_grad()
def run_joint_av_validation(
    *,
    accelerator: Accelerator,
    config: dict,
    step: int,
    joint_model,                              # BridgedZImageJointModel
    tokenizer,
    text_encoder,
    video_vae,
    audio_vae,
    scheduler,                                # noise scheduler used during training
    skip_completed: bool = False,
) -> None:
    val_cfg = config.get("validation") or {}
    if text_encoder is None:
        raise NotImplementedError("Joint AV validation requires a separate text encoder.")

    modes = list(val_cfg.get("modes", _DEFAULT_MODES))
    invalid = [m for m in modes if m not in _DEFAULT_MODES]
    if invalid:
        raise ValueError(f"Unknown validation modes {invalid!r}; expected subset of {_DEFAULT_MODES!r}.")

    transformer_model = accelerator.unwrap_model(joint_model, keep_torch_compile=False)
    text_encoder_model = accelerator.unwrap_model(text_encoder, keep_torch_compile=False)
    video_vae_model = accelerator.unwrap_model(video_vae, keep_torch_compile=False)
    audio_vae_model = accelerator.unwrap_model(audio_vae, keep_torch_compile=False)

    transformer_was_training = transformer_model.training
    text_encoder_was_training = text_encoder_model.training
    video_vae_was_training = video_vae_model.training
    audio_vae_was_training = audio_vae_model.training
    transformer_model.eval()
    text_encoder_model.eval()
    video_vae_model.eval()
    audio_vae_model.eval()

    try:
        train_patch_size = int(config["transformer"]["all_patch_size"][0])
        train_f_patch_size = int(config["transformer"]["all_f_patch_size"][0])

        (
            requested_num_frames,
            latent_frames,
            height,
            width,
            latent_height,
            latent_width,
            fps,
        ) = _resolve_video_latent_shape(config, video_vae_model)
        audio_in_channels, t_latent, sample_rate, audio_duration_seconds = _resolve_audio_latent_shape(
            config, audio_vae_model,
        )
        video_in_channels = int(
            config.get("transformer_video", {}).get("in_channels")
            or config.get("video_transformer", {}).get("in_channels")
            or config.get("transformer", {}).get("in_channels", 48)
        )

        num_inference_steps = int(val_cfg.get("num_inference_steps", 25))
        guidance_scale_v = float(val_cfg.get("video_guidance_scale", val_cfg.get("guidance_scale", 4.0)))
        guidance_scale_a = float(val_cfg.get("audio_guidance_scale", val_cfg.get("guidance_scale", 4.0)))
        cfg_normalization = bool(val_cfg.get("cfg_normalization", False))
        # Per-source upload cap for trackers. Keeps wandb panels readable
        # when joint_av runs hundreds of prompts. None / 0 disables.
        max_wandb_samples_per_source_raw = val_cfg.get("max_wandb_samples_per_source")
        max_wandb_samples_per_source: Optional[int] = (
            int(max_wandb_samples_per_source_raw)
            if max_wandb_samples_per_source_raw is not None
            and int(max_wandb_samples_per_source_raw) > 0
            else None
        )
        # Dual CFG knobs (BridgeDiT Eq.6); when missing, default to the
        # corresponding simple-CFG scale so callers can stay backwards-
        # compatible without introducing a behaviour change for simple.
        video_text_guidance = float(val_cfg.get("video_text_guidance", guidance_scale_v))
        video_modality_guidance = float(val_cfg.get("video_modality_guidance", guidance_scale_v))
        audio_text_guidance = float(val_cfg.get("audio_text_guidance", guidance_scale_a))
        audio_modality_guidance = float(val_cfg.get("audio_modality_guidance", guidance_scale_a))
        shift_v = float(config["train"].get("shift_v", 5.0))
        shift_a = float(config["train"].get("shift_a", 1.0))
        predict_target = getattr(transformer_model.video, "_laion_predict_target", "v")

        # Which CFG variants to evaluate. Defaults to running both simple
        # (NFE=2, current behaviour) and BridgeDiT dual (NFE=3, Eq.6) in
        # joint_av mode for a direct A/B comparison; falls back to simple
        # only in single-modality modes where dual collapses to simple
        # anyway.
        _cfg_variants_for_mode(val_cfg, "joint_av")

        max_seq_len = int(config["text_encoder"]["max_sequence_length"])
        cache_enabled = bool(config["text_encoder"].get("cache_enabled", False))

        empty_prompt = maybe_format_chat_prompt("", tokenizer)
        empty_prompt_embeds = encode_prompts(
            [empty_prompt],
            tokenizer,
            text_encoder_model,
            accelerator.device,
            max_seq_len,
            cache_enabled=cache_enabled,
        )

        base_seed = (
            None if config["train"].get("seed") is None
            else int(config["train"]["seed"]) + int(step)
        )

        run_name = (
            str(config.get("wandb", {}).get("run_name") or "")
            or str(config.get("experiment", {}).get("name") or "")
            or "default"
        )
        sample_root = ensure_dir(
            Path(config["experiment"]["output_dir"]) / "samples" / f"step-{step:08d}"
        )

        num_processes = int(accelerator.num_processes)
        process_index = int(accelerator.process_index)

        for mode in modes:
            mode_records = _resolve_mode_prompts(
                val_cfg, mode, audio_duration_seconds=audio_duration_seconds,
            )
            if not mode_records:
                if accelerator.is_local_main_process:
                    logger.info(
                        "[val] mode=%s has no prompts configured; skipping.", mode,
                    )
                continue

            # Single-modality modes have no opposite-modality bridge KV
            # to gate, so dual CFG collapses to simple. Skip the dual
            # variant for those modes to save the extra forward pass.
            mode_cfg_variants = _cfg_variants_for_mode(val_cfg, mode)

            for cfg_mode in mode_cfg_variants:
                shard_indices = [
                    i for i in range(len(mode_records)) if i % num_processes == process_index
                ]
                cfg_dir_name = _cfg_dir_name(cfg_mode, val_cfg)
                mode_dir = ensure_dir(sample_root / mode / cfg_dir_name)
                local_records: list[dict] = []
                sample_progress = tqdm(
                    shard_indices,
                    total=len(shard_indices),
                    desc=f"val[{mode}/{cfg_mode}] step={step}",
                    leave=True,
                    # Only show on rank-0 to avoid interleaved bars under torchrun.
                    disable=not accelerator.is_local_main_process,
                    mininterval=0.5,
                )
                for sample_index in sample_progress:
                    rec_meta = mode_records[sample_index]
                    prompt_raw = rec_meta.text                       # for logging / wandb
                    prompt_for_te = rec_meta.formatted               # what the model conditions on
                    sample_progress.set_postfix_str(prompt_raw[:48].replace("\n", " "))
                    sample_seed = (
                        None if base_seed is None
                        else base_seed + sample_index * _VALIDATION_SAMPLE_SEED_STRIDE
                    )
                    file_stem = _prompt_file_stem(rec_meta)
                    if skip_completed and _sample_outputs_complete(mode_dir, file_stem, mode):
                        record = _build_validation_record(
                            rec_meta=rec_meta,
                            sample_index=sample_index,
                            mode=mode,
                            cfg_mode=cfg_mode,
                            sample_seed=sample_seed,
                            mode_dir=mode_dir,
                            file_stem=file_stem,
                            fps=fps,
                            sample_rate=sample_rate,
                        )
                        _write_sample_sidecar(mode_dir, file_stem, record)
                        local_records.append(record)
                        sample_progress.set_postfix_str(f"skip {file_stem}")
                        continue

                    generator = (
                        torch.Generator(device=accelerator.device).manual_seed(int(sample_seed))
                        if sample_seed is not None else None
                    )

                    formatted_prompt = maybe_format_chat_prompt(prompt_for_te, tokenizer)
                    prompt_embeds_pos = encode_prompts(
                        [formatted_prompt],
                        tokenizer,
                        text_encoder_model,
                        accelerator.device,
                        max_seq_len,
                        cache_enabled=cache_enabled,
                    )

                    video_scheduler = _build_inference_scheduler(
                        scheduler, shift=shift_v, num_inference_steps=num_inference_steps,
                        device=accelerator.device, predict_target=predict_target,
                    )
                    audio_scheduler = _build_inference_scheduler(
                        scheduler, shift=shift_a, num_inference_steps=num_inference_steps,
                        device=accelerator.device, predict_target=predict_target,
                    )

                    video_latents_init: Optional[torch.Tensor] = None
                    audio_latents_init: Optional[torch.Tensor] = None
                    if mode in ("joint_av", "video_only"):
                        video_latents_init = torch.randn(
                            (1, video_in_channels, latent_frames, latent_height, latent_width),
                            generator=generator, device=accelerator.device, dtype=torch.float32,
                        )
                    if mode in ("joint_av", "audio_only"):
                        audio_latents_init = torch.randn(
                            (1, audio_in_channels, t_latent),
                            generator=generator, device=accelerator.device, dtype=torch.float32,
                        )

                    final_video, final_audio = _denoise_joint(
                        joint_model=transformer_model,
                        config=config,
                        accelerator=accelerator,
                        mode=mode,
                        video_latents_init=video_latents_init,
                        audio_latents_init=audio_latents_init,
                        video_scheduler=video_scheduler if video_latents_init is not None else None,
                        audio_scheduler=audio_scheduler if audio_latents_init is not None else None,
                        prompt_embeds_video_pos=prompt_embeds_pos if video_latents_init is not None else None,
                        prompt_embeds_video_neg=empty_prompt_embeds if video_latents_init is not None else None,
                        prompt_embeds_audio_pos=prompt_embeds_pos if audio_latents_init is not None else None,
                        prompt_embeds_audio_neg=empty_prompt_embeds if audio_latents_init is not None else None,
                        guidance_scale_v=guidance_scale_v,
                        guidance_scale_a=guidance_scale_a,
                        cfg_normalization=cfg_normalization,
                        train_patch_size=train_patch_size,
                        train_f_patch_size=train_f_patch_size,
                        cfg_mode=cfg_mode,
                        video_text_guidance=video_text_guidance,
                        video_modality_guidance=video_modality_guidance,
                        audio_text_guidance=audio_text_guidance,
                        audio_modality_guidance=audio_modality_guidance,
                    )

                    record = _build_validation_record(
                        rec_meta=rec_meta,
                        sample_index=sample_index,
                        mode=mode,
                        cfg_mode=cfg_mode,
                        sample_seed=sample_seed,
                        mode_dir=mode_dir,
                        file_stem=file_stem,
                        fps=fps,
                        sample_rate=sample_rate,
                    )

                    # Decode video.
                    if final_video is not None:
                        video_decoded = decode_latents_to_images(
                            final_video.to(dtype=getattr(video_vae_model, "dtype", final_video.dtype)),
                            video_vae_model,
                        )[0]
                        frames = _video_tensor_to_uint8_frames(video_decoded)
                        video_path = mode_dir / f"{file_stem}.mp4"
                        imageio.mimsave(
                            video_path, list(frames), fps=fps,
                            codec="libx264", quality=8, macro_block_size=None,
                        )
                        record["video_path"] = str(video_path)
                        record["fps"] = float(fps)

                    # Decode audio.
                    if final_audio is not None:
                        audio_out = audio_vae_model.decode(
                            final_audio.to(dtype=getattr(audio_vae_model, "dtype", final_audio.dtype))
                        )
                        wave = audio_out[0, 0].float().clamp(-1.0, 1.0).cpu().unsqueeze(0)

                        # ---- AV duration alignment ----
                        # The audio VAE's hop_length almost never divides
                        # ``num_frames * sample_rate / fps`` evenly, so the
                        # decoded waveform's length differs from the video
                        # window by a fractional frame. When the muxed
                        # ``.av.mp4`` is built with ``ffmpeg -shortest``
                        # that ~1-2 ms mismatch combines with libx264's
                        # B-frame reordering at the tail and ends up
                        # dropping 1-3 video frames per sample (visible
                        # as 117-120 instead of 121 frames in
                        # ``ffprobe -count_frames``). Pad / trim the wav
                        # to *exactly* ``num_frames / fps`` seconds before
                        # saving so the two streams agree by construction.
                        if final_video is not None:
                            target_audio_len = int(round(
                                requested_num_frames * int(sample_rate) / float(fps)
                            ))
                            cur_len = int(wave.shape[-1])
                            if cur_len < target_audio_len:
                                wave = F.pad(wave, (0, target_audio_len - cur_len))
                            elif cur_len > target_audio_len:
                                wave = wave[..., :target_audio_len].contiguous()

                        audio_path = mode_dir / f"{file_stem}.wav"
                        torchaudio.save(str(audio_path), wave, sample_rate=int(sample_rate))
                        record["audio_path"] = str(audio_path)
                        record["sample_rate"] = int(sample_rate)

                    # Mux for joint mode.
                    if mode == "joint_av" and "video_path" in record and "audio_path" in record:
                        muxed_path = mode_dir / f"{file_stem}.av.mp4"
                        if _ffmpeg_mux_av(Path(record["video_path"]), Path(record["audio_path"]), muxed_path):
                            record["av_path"] = str(muxed_path)

                    _write_sample_sidecar(mode_dir, file_stem, record)
                    local_records.append(record)

                    # Drop big intermediates between samples to keep VRAM use stable.
                    del final_video, final_audio
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                accelerator.wait_for_everyone()
                gathered = gather_object(local_records)
                if accelerator.is_main_process:
                    by_idx = {int(rec["sample_index"]): rec for rec in gathered}
                    ordered = [by_idx[k] for k in sorted(by_idx.keys())]
                    manifest_payload = {
                        "step": step,
                        "mode": mode,
                        "cfg_mode": cfg_mode,
                        "cfg_dir": cfg_dir_name,
                        "num_inference_steps": num_inference_steps,
                        "video_guidance_scale": guidance_scale_v,
                        "audio_guidance_scale": guidance_scale_a,
                        "shift_v": shift_v,
                        "shift_a": shift_a,
                        "samples": ordered,
                    }
                    if val_cfg.get("cfg_value") is not None:
                        manifest_payload["cfg_value"] = float(val_cfg["cfg_value"])
                    if cfg_mode == "dual":
                        manifest_payload.update(
                            video_text_guidance=video_text_guidance,
                            video_modality_guidance=video_modality_guidance,
                            audio_text_guidance=audio_text_guidance,
                            audio_modality_guidance=audio_modality_guidance,
                        )
                    save_json(mode_dir / "manifest.json", manifest_payload)
                    _log_mode_to_trackers(
                        accelerator,
                        mode=mode,
                        records=ordered,
                        sample_rate=sample_rate,
                        fps=fps,
                        step=step,
                        cfg_mode=cfg_mode,
                        max_wandb_samples_per_source=max_wandb_samples_per_source,
                    )
    finally:
        transformer_model.train(transformer_was_training)
        text_encoder_model.train(text_encoder_was_training)
        video_vae_model.train(video_vae_was_training)
        audio_vae_model.train(audio_vae_was_training)
        if hasattr(video_vae_model, "clear_cache"):
            video_vae_model.clear_cache()
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cap_records_per_source(
    records: list[dict],
    *,
    max_per_source: Optional[int],
) -> list[dict]:
    """Take at most ``max_per_source`` records from each source_name,
    preserving the input order (which is already sample_index-sorted).

    ``records`` may mix multiple sources for ``audio_only`` (TTS + TTA
    in the same mode), so a flat slice would leak the cap across
    sources. ``None`` / non-positive disables the cap (return as-is).
    """
    if not max_per_source or max_per_source <= 0:
        return records
    seen: dict[str, int] = {}
    capped: list[dict] = []
    for rec in records:
        key = str(rec.get("source_name") or "_default")
        n = seen.get(key, 0)
        if n >= max_per_source:
            continue
        seen[key] = n + 1
        capped.append(rec)
    return capped


def _log_mode_to_trackers(
    accelerator: Accelerator,
    *,
    mode: str,
    records: list[dict],
    sample_rate: int,
    fps: float,
    step: int,
    cfg_mode: str = "simple",
    max_wandb_samples_per_source: Optional[int] = None,
) -> None:
    if not records:
        return
    # Cap upload count per prompt source. We keep the full manifest /
    # files on disk; this only limits what we push to trackers (so
    # joint_av at 501 prompts doesn't drown the wandb panel).
    records = _cap_records_per_source(records, max_per_source=max_wandb_samples_per_source)
    if not records:
        return
    # Tracker key tag used to keep simple- and dual-CFG samples in
    # different rows / TensorBoard sections without changing the wandb
    # group hierarchy.
    mode_tag = f"{mode}/{cfg_mode}"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for record in records:
                idx = int(record["sample_index"])
                if record.get("video_path"):
                    frames = imageio.mimread(str(record["video_path"]), memtest=False)
                    if frames:
                        video_tensor = (
                            torch.from_numpy(np.stack([np.asarray(f, dtype=np.uint8) for f in frames]))
                            .permute(0, 3, 1, 2)
                            .unsqueeze(0)
                        )
                        tracker.writer.add_video(
                            f"validation/{mode_tag}/video/sample-{idx:02d}",
                            video_tensor,
                            global_step=step,
                            fps=int(round(float(fps))),
                        )
                if record.get("audio_path"):
                    wave, sr = torchaudio.load(record["audio_path"])
                    tracker.writer.add_audio(
                        f"validation/{mode_tag}/audio/sample-{idx:02d}",
                        wave,
                        global_step=step,
                        sample_rate=int(sr),
                    )
                tracker.writer.add_text(
                    f"validation/{mode_tag}/prompt/sample-{idx:02d}",
                    str(record.get("prompt", "")),
                    global_step=step,
                )
        elif tracker.name == "wandb":
            import wandb

            payload: dict[str, Any] = {}
            video_payload: list = []
            audio_payload: list = []
            av_payload: list = []
            for record in records:
                idx = int(record["sample_index"])
                tag_bits: list[str] = [f"#{idx}"]
                if record.get("source_name"):
                    tag_bits.append(str(record["source_name"]))
                if record.get("type_label"):
                    tag_bits.append(str(record["type_label"]))
                if record.get("task_kind"):
                    tag_bits.append(str(record["task_kind"]))
                caption = f"[{' | '.join(tag_bits)}] {record.get('prompt', '')}"
                if record.get("av_path"):
                    # wandb.Video ignores fps=... when given a file path
                    # (the mp4 already encodes its own fps), and emits a
                    # noisy warning if we pass it. We've already muxed the
                    # mp4 at the right fps via imageio / ffmpeg upstream.
                    av_payload.append(
                        wandb.Video(record["av_path"], caption=caption, format="mp4")
                    )
                if record.get("video_path"):
                    video_payload.append(
                        wandb.Video(record["video_path"], caption=caption, format="mp4")
                    )
                if record.get("audio_path"):
                    audio_payload.append(
                        wandb.Audio(record["audio_path"], sample_rate=int(sample_rate), caption=caption)
                    )
            if av_payload:
                payload[f"validation/{mode_tag}/av"] = av_payload
            if video_payload:
                payload[f"validation/{mode_tag}/video"] = video_payload
            if audio_payload:
                payload[f"validation/{mode_tag}/audio"] = audio_payload
            if payload:
                tracker.log(payload, step=step)
