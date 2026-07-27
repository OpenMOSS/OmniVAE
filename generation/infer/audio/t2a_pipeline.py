"""Shared T2A inference pipeline for OmniVAE audio (Z-Image) checkpoints.

Used by ``infer/audio/streamlit_app.py`` (interactive UI) and
intended to stay drop-in compatible with the validation primitives in
``omnivae_generation.trainer.audio_validation`` so per-prompt generations are numerically
identical to a one-prompt training-time validation pass.

The single-prompt loop is the **same** body as ``omnivae_generation.trainer.audio_validation
._generate_one_set`` (encode_prompts -> flow-match euler + CFG -> VAE
decode), with the multi-rank sharding and ASR / WER bookkeeping stripped
out so streamlit can drive it directly.

Public surface
--------------
* :class:`T2APipeline` -- dataclass holding the loaded text encoder /
  tokenizer / audio VAE / transformer (+ forward wrapper) / scheduler,
  plus precomputed negative-prompt embeddings and the scalar config
  knobs needed for the inference loop.

* :func:`load_t2a_pipeline` -- build the full pipeline from a checkpoint
  directory + a training YAML. ``config_path=None`` auto-detects a
  ``resolved_config.yaml`` two levels up from the snapshot (the t2v /
  t2av convention); if that file is missing the caller must pass an
  explicit YAML path because t2a checkpoints do not snapshot the
  resolved config by default.

* :func:`generate_one_audio` -- run one prompt through the denoising
  loop, decode to a waveform and save a ``.wav`` to disk. Returns a
  record dict suitable for a ``result.json`` sidecar.

Layout assumption for ``checkpoint_dir``
----------------------------------------
The t2a trainer writes::

    checkpoint-XXXXXXXX/
      transformer/        diffusers ZImageTransformer2DModel
      scheduler/          optional; falls back to the YAML's scheduler block
      metadata.json       optional; informational only

The text encoder + tokenizer and the audio VAE are *not* in the
snapshot (frozen at training time); they are reloaded from the YAML
and can be overridden via ``vae_*_override``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
@dataclass
class T2APipeline:
    """Container for an instantiated T2A inference pipeline.

    Behaves like a ``SimpleNamespace`` -- accessed by attribute from the
    streamlit app to keep call sites readable.
    """

    tokenizer: Any
    text_encoder: Any                      # on device, eval()
    audio_vae: Any                         # on device, eval()
    transformer: Any                       # on device, eval()
    forward_transformer: Any               # zimage forward wrapper
    scheduler: Any                         # FlowMatchEulerDiscreteScheduler
    negative_prompt_embeds: Any            # encoded empty prompt, on device
    config: dict                           # deep-copied training YAML
    config_path: Optional[Path]
    checkpoint_dir: Path
    checkpoint_step: int
    device: torch.device
    in_channels: int
    max_seq_len: int
    cache_enabled: bool
    sample_rate: int
    hop_length: int
    duration_precision: int
    append_duration_suffix: bool
    default_duration_seconds: float
    default_num_inference_steps: int
    default_guidance_scale: float
    default_cfg_normalization: bool


def _apply_runtime_patches() -> None:
    """Mirror the patches that audio training / run_tts / run_eval apply
    before instantiating diffusers' Z-Image transformer. Idempotent.
    """
    from omnivae_generation.trainer.runtime_patches import (
        patch_diffusers_zimage_forward_block_stacks,
        patch_diffusers_zimage_real_rope,
        patch_transformers_qwen3_5_disable_fast_path,
    )

    patch_diffusers_zimage_real_rope()
    patch_diffusers_zimage_forward_block_stacks()
    # Frozen Qwen3.5 text encoder needs the torch fallback path; the patch
    # is idempotent so it's safe to apply even when the YAML disagrees.
    patch_transformers_qwen3_5_disable_fast_path()


def _resolve_config(
    checkpoint_dir: Path,
    config_path: Optional[str | Path],
) -> tuple[dict, Optional[Path]]:
    """Resolve the training YAML for this checkpoint.

    Order:
      1. Explicit ``config_path`` -- ``omnivae_generation.trainer.config.load_config`` it.
      2. Auto-detect ``<ckpt>/../../resolved_config.yaml`` (t2v / t2av
         convention); use ``omnivae_generation.trainer.eval.guided_diffusion.resolve_run_dir``
         + ``load_run_config_for_eval`` to pull it in.
      3. Raise a clear error.
    """
    from omnivae_generation.trainer.config import load_config
    from omnivae_generation.trainer.eval.guided_diffusion import (
        load_run_config_for_eval,
        resolve_run_dir,
    )

    if config_path:
        p = Path(config_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"Training config yaml not found: {p}. Pass the same --config "
                "you used at training / run_tts.py time."
            )
        return load_config(p), p

    # Auto-detect path (mirrors t2v/t2av): resolve_run_dir returns
    # checkpoint_dir.parents[2], and load_run_config_for_eval reads
    # `<run_dir>/resolved_config.yaml`. The t2a trainer does not always
    # persist resolved_config.yaml, so this branch may legitimately fail.
    try:
        run_dir = resolve_run_dir(None, checkpoint_dir)
        config = load_run_config_for_eval(run_dir)
        return config, run_dir / "resolved_config.yaml"
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No training config yaml provided and could not auto-detect a "
            f"resolved_config.yaml next to {checkpoint_dir}. Either pass an "
            f"explicit --config / yaml path, or place a resolved_config.yaml "
            f"at <ckpt>/../../resolved_config.yaml. Underlying error: {exc}"
        ) from exc


def _try_extract_step(checkpoint_dir: Path) -> int:
    """Best-effort step extraction; returns 0 for non-``checkpoint-NNNN`` names
    so the UI still works on directories like ``checkpoint_latest``.
    """
    from omnivae_generation.trainer.eval.guided_diffusion import extract_checkpoint_step

    try:
        return int(extract_checkpoint_step(checkpoint_dir))
    except (ValueError, TypeError):
        return 0


def load_t2a_pipeline(
    checkpoint_dir: str | Path,
    config_path: Optional[str | Path] = None,
    *,
    device: str | torch.device = "cuda",
    vae_path_override: Optional[str | Path] = None,
    vae_type_override: Optional[str] = None,
) -> T2APipeline:
    """Build the full T2A inference pipeline from a saved checkpoint dir.

    Parameters
    ----------
    checkpoint_dir
        ``.../checkpoint-XXXXXXXX`` produced by the audio trainer; must
        contain ``transformer/`` (and optionally ``scheduler/``).
    config_path
        Training YAML (e.g. ``configs/audio/t2a.yaml``). When ``None`` we
        try to auto-detect ``<ckpt>/../../resolved_config.yaml``; if
        that's missing the call raises with an actionable message.
    device
        Torch device for the transformer + text encoder + VAE.
    vae_path_override, vae_type_override
        Override ``audio_vae.model_path`` / ``audio_vae.type`` from the
        YAML. Order of resolution: override > YAML > error.
    """
    overall_t0 = time.time()

    _apply_runtime_patches()

    from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel

    from omnivae_generation.trainer.audio_validation import _audio_vae_hop_length
    from omnivae_generation.trainer.data import maybe_format_chat_prompt
    from omnivae_generation.trainer.modeling import (
        configure_transformer_prediction_target,
        configure_transformer_timestep_usage,
        encode_prompts,
        load_audio_vae,
        load_scheduler,
        load_text_components,
        resolve_dtype,
    )
    from omnivae_generation.trainer.forward_transformer import build_forward_transformer

    cdir = Path(checkpoint_dir).expanduser().resolve()
    if not cdir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {cdir}")

    config_raw, resolved_config_path = _resolve_config(cdir, config_path)
    config = copy.deepcopy(config_raw)

    if not isinstance(config.get("transformer"), dict):
        raise ValueError(
            "T2A pipeline requires the training yaml to contain a "
            "'transformer' block (single-branch zimage transformer). "
            f"Got keys: {sorted(config.keys())}."
        )

    # Audio VAE overrides apply BEFORE we read the block so downstream
    # code (load_audio_vae, _audio_vae_hop_length) sees the user choice.
    audio_vae_cfg = dict(config.get("audio_vae") or {})
    if vae_type_override:
        audio_vae_cfg["type"] = str(vae_type_override).strip()
    if vae_path_override:
        # mirror the run_tts.py convention: write into `model_path` (the
        # canonical audio_vae field), the loader also tolerates
        # `model_name_or_path` for legacy yamls.
        audio_vae_cfg["model_path"] = str(Path(str(vae_path_override)).expanduser())
    if not (audio_vae_cfg.get("model_path") or audio_vae_cfg.get("model_name_or_path")):
        raise ValueError(
            "audio_vae has no model_path (and no override given). Either set "
            "`audio_vae.model_path` in the YAML or pass --vae-path."
        )
    config["audio_vae"] = audio_vae_cfg

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {dev} but cuda is not available.")

    text_cfg = config["text_encoder"]
    text_dtype = resolve_dtype(text_cfg.get("torch_dtype"), fallback=torch.bfloat16)
    transformer_dtype = text_dtype if dev.type == "cuda" else torch.float32

    # ---- text encoder + tokenizer ----
    tokenizer, text_encoder, _cap_feat_dim = load_text_components(text_cfg, text_dtype)
    text_encoder.eval()
    text_encoder.to(dev)

    # ---- audio VAE ----
    audio_vae = load_audio_vae(audio_vae_cfg)
    audio_vae.eval()
    audio_vae.to(dev)
    hop_length = _audio_vae_hop_length(audio_vae, audio_vae_cfg)

    # ---- transformer ----
    transformer_dir = cdir / "transformer"
    if not (transformer_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"Checkpoint at {cdir} is missing {transformer_dir / 'config.json'}; "
            "this directory does not look like a t2a snapshot."
        )
    transformer = ZImageTransformer2DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=transformer_dtype,
        low_cpu_mem_usage=True,
    )
    configure_transformer_timestep_usage(
        transformer, bool(config["transformer"].get("use_timestep", True))
    )
    configure_transformer_prediction_target(
        transformer, config["transformer"].get("predict_target", "v")
    )
    if hasattr(transformer, "set_forward_compilation"):
        transformer.set_forward_compilation(False)
    transformer.eval()
    transformer.to(dev)
    if hasattr(transformer, "materialize_rope_cache"):
        transformer.materialize_rope_cache(dev)

    forward_transformer = build_forward_transformer(
        transformer,
        transformer,
        train_patch_size=int(config["transformer"]["all_patch_size"][0]),
        train_f_patch_size=int(config["transformer"]["all_f_patch_size"][0]),
    )

    # ---- scheduler (prefer snapshot-local, fall back to yaml) ----
    scheduler_dir = cdir / "scheduler"
    if scheduler_dir.is_dir():
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_dir))
    else:
        scheduler = load_scheduler(config["scheduler"])

    # ---- negative prompt embeddings (cached for whole session) ----
    max_seq_len = int(text_cfg.get("max_sequence_length", 512))
    cache_enabled = bool(text_cfg.get("cache_enabled", False))
    empty_chat = maybe_format_chat_prompt("", tokenizer)
    negative_prompt_embeds = encode_prompts(
        [empty_chat],
        tokenizer,
        text_encoder,
        dev,
        max_seq_len,
        cache_enabled=cache_enabled,
    )

    # ---- scalar defaults from yaml ----
    dataset_cfg = config.get("dataset", {}) or {}
    val_cfg = config.get("validation", {}) or {}
    sample_rate = int(dataset_cfg.get("sample_rate", 48000))
    duration_precision = int(dataset_cfg.get("duration_precision", 1))
    append_duration_suffix = bool(dataset_cfg.get("append_duration_suffix", True))
    default_duration_seconds = float(val_cfg.get("duration_seconds", 30.0))
    default_num_inference_steps = int(val_cfg.get("num_inference_steps", 50))
    default_guidance_scale = float(val_cfg.get("guidance_scale", 3.0))
    default_cfg_normalization = bool(val_cfg.get("cfg_normalization", False))

    elapsed = time.time() - overall_t0
    logger.info(
        "[t2a_pipeline] loaded ckpt=%s config=%s device=%s in %.1fs",
        cdir,
        resolved_config_path,
        dev,
        elapsed,
    )

    return T2APipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        audio_vae=audio_vae,
        transformer=transformer,
        forward_transformer=forward_transformer,
        scheduler=scheduler,
        negative_prompt_embeds=negative_prompt_embeds,
        config=config,
        config_path=resolved_config_path,
        checkpoint_dir=cdir,
        checkpoint_step=_try_extract_step(cdir),
        device=dev,
        in_channels=int(config["transformer"]["in_channels"]),
        max_seq_len=max_seq_len,
        cache_enabled=cache_enabled,
        sample_rate=sample_rate,
        hop_length=hop_length,
        duration_precision=duration_precision,
        append_duration_suffix=append_duration_suffix,
        default_duration_seconds=default_duration_seconds,
        default_num_inference_steps=default_num_inference_steps,
        default_guidance_scale=default_guidance_scale,
        default_cfg_normalization=default_cfg_normalization,
    )


# ----------------------------------------------------------------------
# Per-prompt generation
# ----------------------------------------------------------------------
_VALID_TASK_KINDS = ("tts", "tta")
_VALID_DURATION_STRATEGIES = ("auto", "fixed", "f5")


@torch.no_grad()
def generate_one_audio(
    pipe: T2APipeline,
    *,
    prompt: str,
    task_kind: str = "tts",
    task_prefix_enabled: bool = True,
    duration_seconds: float | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    cfg_normalization: bool | None = None,
    prompt_duration_strategy: str = "f5",
    auto_duration_words_per_second: float = 3.0,
    auto_duration_margin_seconds: float = 0.5,
    auto_duration_min_seconds: float = 3.0,
    auto_duration_max_seconds: float | None = None,
    f5_bytes_per_second: float = 17.0,
    f5_local_speed: float = 1.0,
    f5_short_text_threshold_bytes: int = 10,
    f5_short_text_local_speed: float = 0.3,
    f5_margin_seconds: float = 0.0,
    seed: int = 20260508,
    output_dir: str | Path,
    file_stem: str = "sample",
) -> dict[str, Any]:
    """Run one denoising pass and write a ``.wav`` under ``output_dir``.

    Returns a record dict (audio_path, prompt_with_suffix, target duration,
    sample_rate, seed, elapsed) so the caller can persist a JSON sidecar.

    Reuses the same scheduler / CFG / encoding helpers as
    ``omnivae_generation.trainer.audio_validation._generate_one_set``; the only difference is
    that we drive a single prompt with no accelerator / WER / sharding,
    so the loop is inlined here.
    """
    from omnivae_generation.trainer.audio_duration import make_duration_estimator
    from omnivae_generation.trainer.audio_task_prefix import apply_task_prefix
    from omnivae_generation.trainer.audio_validation import (
        _build_inference_scheduler,
        _build_validation_prompt_text,
    )
    from omnivae_generation.trainer.data import maybe_format_chat_prompt
    from omnivae_generation.trainer.modeling import encode_prompts
    from omnivae_generation.trainer.video_validation import apply_zimage_cfg

    task = str(task_kind or "tts").strip().lower()
    if task not in _VALID_TASK_KINDS:
        raise ValueError(
            f"task_kind must be one of {_VALID_TASK_KINDS}; got {task_kind!r}."
        )
    strategy = str(prompt_duration_strategy or "fixed").strip().lower()
    if strategy not in _VALID_DURATION_STRATEGIES:
        raise ValueError(
            f"prompt_duration_strategy must be one of {_VALID_DURATION_STRATEGIES}; "
            f"got {prompt_duration_strategy!r}."
        )

    # Resolve sampling knobs (CLI overrides win, fall back to pipe defaults).
    latent_duration = (
        float(duration_seconds)
        if duration_seconds is not None
        else float(pipe.default_duration_seconds)
    )
    num_steps = int(
        num_inference_steps
        if num_inference_steps is not None
        else pipe.default_num_inference_steps
    )
    cfg_value = float(
        guidance_scale if guidance_scale is not None else pipe.default_guidance_scale
    )
    cfg_norm = bool(
        cfg_normalization
        if cfg_normalization is not None
        else pipe.default_cfg_normalization
    )

    auto_max = (
        float(auto_duration_max_seconds)
        if auto_duration_max_seconds is not None
        else float(latent_duration)
    )
    duration_estimator = make_duration_estimator(
        strategy=strategy,
        fixed_duration=latent_duration,
        words_per_second=float(auto_duration_words_per_second),
        margin_seconds=float(auto_duration_margin_seconds),
        min_seconds=float(auto_duration_min_seconds),
        max_seconds=auto_max,
        bytes_per_second=float(f5_bytes_per_second),
        f5_local_speed=float(f5_local_speed),
        f5_short_text_threshold_bytes=int(f5_short_text_threshold_bytes),
        f5_short_text_local_speed=float(f5_short_text_local_speed),
        f5_margin_seconds=float(f5_margin_seconds),
    )

    # Reproducible task-prefix template pick (offset matches _generate_one_set).
    template_rng = random.Random(int(seed) + 31337)
    raw_text = str(prompt or "")
    if task_prefix_enabled:
        wrapped_text = apply_task_prefix(task, raw_text, rng=template_rng)
    else:
        wrapped_text = raw_text

    target_duration = float(duration_estimator(raw_text))
    prompt_with_suffix = _build_validation_prompt_text(
        wrapped_text,
        duration_seconds=target_duration,
        duration_precision=pipe.duration_precision,
        append_duration_suffix=pipe.append_duration_suffix,
    )

    # Latent length follows the user-facing latent budget (latent_duration);
    # the per-prompt estimator only controls the "duration: X.Xs" suffix the
    # text encoder sees, matching the t2a training contract.
    t_latent = max(
        1, int(round(latent_duration * pipe.sample_rate / pipe.hop_length))
    )

    device = pipe.device
    formatted_prompt = maybe_format_chat_prompt(prompt_with_suffix, pipe.tokenizer)
    prompt_embeds = encode_prompts(
        [formatted_prompt],
        pipe.tokenizer,
        pipe.text_encoder,
        device,
        pipe.max_seq_len,
        cache_enabled=pipe.cache_enabled,
    )

    inference_scheduler = _build_inference_scheduler(
        pipe.config,
        pipe.scheduler,
        pipe.transformer,
        device,
        num_steps,
    )

    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn(
        (1, pipe.in_channels, t_latent),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    t0 = time.time()
    transformer_dtype = getattr(pipe.transformer, "dtype", latents.dtype)
    for timestep_value in inference_scheduler.timesteps:
        timestep = timestep_value.expand(latents.shape[0])
        model_timesteps = (
            float(inference_scheduler.config.num_train_timesteps) - timestep
        ) / float(inference_scheduler.config.num_train_timesteps)
        model_timesteps = model_timesteps.to(device=device, dtype=torch.float32)

        latent_model_input = latents.repeat(2, 1, 1)
        prompt_embeds_model_input = prompt_embeds + pipe.negative_prompt_embeds
        timestep_model_input = model_timesteps.repeat(2)

        model_pred, _ = pipe.forward_transformer(
            latent_model_input.to(dtype=transformer_dtype),
            timestep_model_input,
            prompt_embeds_model_input,
        )
        pos_pred = model_pred[:1].float()
        neg_pred = model_pred[1:].float()
        cfg_pred = apply_zimage_cfg(pos_pred, neg_pred, cfg_value, cfg_norm)
        cfg_pred = -cfg_pred
        latents = inference_scheduler.step(
            cfg_pred.to(torch.float32),
            timestep_value,
            latents,
            return_dict=False,
        )[0].to(torch.float32)

    audio = pipe.audio_vae.decode(
        latents.to(dtype=getattr(pipe.audio_vae, "dtype", latents.dtype))
    )
    wave = audio[0, 0].float().clamp(-1.0, 1.0).cpu().unsqueeze(0)
    elapsed = time.time() - t0

    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    audio_path = out_root / f"{file_stem}.wav"
    torchaudio.save(str(audio_path), wave, sample_rate=int(pipe.sample_rate))

    record: dict[str, Any] = {
        "audio_path": str(audio_path),
        "prompt": raw_text,
        "prompt_wrapped": wrapped_text,
        "prompt_with_suffix": prompt_with_suffix,
        "task_kind": task,
        "task_prefix_enabled": bool(task_prefix_enabled),
        "prompt_duration_strategy": strategy,
        "target_duration_seconds": float(target_duration),
        "latent_duration_seconds": float(latent_duration),
        "t_latent": int(t_latent),
        "decoded_num_samples": int(wave.shape[-1]),
        "decoded_seconds": float(wave.shape[-1]) / float(pipe.sample_rate),
        "sample_rate": int(pipe.sample_rate),
        "hop_length": int(pipe.hop_length),
        "num_inference_steps": int(num_steps),
        "guidance_scale": float(cfg_value),
        "cfg_normalization": bool(cfg_norm),
        "seed": int(seed),
        "elapsed_s": float(elapsed),
    }

    del prompt_embeds, latents, model_pred, audio, cfg_pred, pos_pred, neg_pred
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    return record


def write_record_sidecar(record: dict[str, Any], sidecar_path: str | Path) -> None:
    """Write a ``.json`` sidecar with the full request + result record."""
    Path(sidecar_path).write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
