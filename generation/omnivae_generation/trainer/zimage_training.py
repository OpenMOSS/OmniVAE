from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from omnivae_generation.trainer.runtime_env import ensure_hf_home

ensure_hf_home()

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from omnivae_generation.trainer.accelerate_utils import configure_wandb_env, get_tracker_init, get_weight_dtype
from omnivae_generation.trainer.audio_data import AudioJsonlT2ADataset
from omnivae_generation.trainer.config import apply_model_size_preset_force, flatten_config, load_config
from omnivae_generation.trainer.data import ImageNetTextToImageDataset
from omnivae_generation.trainer.modeling import (
    adapt_model_prediction,
    build_transformer_init_expected_config,
    build_transformer,
    count_parameters,
    encode_audio_to_latents,
    encode_images_to_latents,
    encode_prompts,
    encode_tokenized_prompts,
    load_audio_vae,
    load_pretrained_transformer_weights,
    load_qwen3_vl_tokenizer_and_hidden_size,
    load_scheduler,
    load_text_components,
    load_vae,
    normalize_vae_type,
    resolve_dtype,
    save_checkpoint_artifacts,
)
from omnivae_generation.trainer.optim import HybridMuonAdamw
from omnivae_generation.trainer.qwen3_vl_dit import (
    Qwen3VLDiffusionTransformer,
    is_qwen3_vl_dit_arch,
    prompt_token_tensors_to_payloads,
    tokenize_prompt_payloads,
)
from omnivae_generation.trainer.relaion_data import RelaionDataset
from omnivae_generation.trainer.stateful_dataloader import (
    DATALOADER_RESUME_STRATEGY,
    build_train_dataloader,
    restore_training_state,
    save_dataloader_state,
)
from omnivae_generation.trainer.utils import ensure_dir, mark_checkpoint_complete, rotate_checkpoints, save_json
from omnivae_generation.trainer.audio_vae_validation import run_audio_vae_validation
from omnivae_generation.trainer.vae_validation import run_vae_validation as _run_visual_vae_validation
from omnivae_generation.trainer.validation import run_validation


def run_vae_validation(
    *,
    accelerator,
    config: dict,
    step: int,
    batch: dict,
    vae,
    vae_dtype,
):
    """Dispatch VAE encode->decode reconstruction validation to the right
    backend. Audio datasets get a 1D-waveform recon path that mirrors the
    visual side's interface (L1 / SNR / mel-L1 + wandb.Audio previews + local
    .wav backups); everything else routes to the original visual recon.
    """
    if is_audio_dataset(config):
        return run_audio_vae_validation(
            accelerator=accelerator,
            config=config,
            step=step,
            batch=batch,
            vae=vae,
            vae_dtype=vae_dtype,
        )
    return _run_visual_vae_validation(
        accelerator=accelerator,
        config=config,
        step=step,
        batch=batch,
        vae=vae,
        vae_dtype=vae_dtype,
    )
from omnivae_generation.trainer.video_data import (
    SUPPORTED_DECODE_BACKENDS,
    VideoJsonlDataset,
    prebuild_video_jsonl_indexes,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingPaths:
    output_dir: Path
    checkpoints_dir: Path
    snapshot_root: Path
    persistent_root: Path
    logging_dir: Path


@dataclass
class TrainingRuntime:
    accelerator: Accelerator
    paths: TrainingPaths
    manual_gc_every_steps: int | None


@dataclass
class TextSetup:
    tokenizer: Any
    text_encoder: Any
    text_hidden_size: int
    weight_dtype: torch.dtype
    text_encoder_dtype: torch.dtype


@dataclass
class FinalizedModels:
    text_encoder: Any
    vae_dtype: torch.dtype
    transformer_model: Any
    predict_target: str
    train_patch_size: int
    train_f_patch_size: int
    num_update_steps_per_epoch: int
    num_train_epochs: int


@dataclass
class DiffusionBatch:
    batch_size: int
    sigmas: torch.Tensor
    noisy_latents: torch.Tensor
    model_timesteps: torch.Tensor
    target: torch.Tensor
    weighting: torch.Tensor


def uses_unified_qwen3_vl_transformer(config: dict) -> bool:
    return is_qwen3_vl_dit_arch(config)


def parse_train_args(description: str = "Train a random-init Z-Image transformer with Accelerate."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Override experiment.name from the YAML. The trainer always keeps "
            "experiment.name, basename(experiment.output_dir), and wandb.run_name "
            "aligned (single source of truth), so this flag also rewrites the "
            "output_dir leaf and the wandb run_name."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or `latest`/`latest_persistent`. Overrides the YAML setting.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help=(
            "Disable all torch.compile sites (transformer forward, text encoder, "
            "vae encode, optimizer step) for fast first-step debug / validation runs. "
            "Forces transformer.compile_model / text_encoder.compile_model / "
            "audio_vae.compile_encode / vae.compile_encode to False and skips the "
            "@torch.compile wrapper around optimizer_step_transformer."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help=(
            "Override train.validation_steps from the YAML. Use a small value "
            "(e.g. 1) to trigger validation immediately for debug, or 0 to disable."
        ),
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help=(
            "Override transformer.model_size with a preset (e.g. 1b, 2.5b, 5b). "
            "Forces transformer.dim/n_layers/n_heads/n_kv_heads/n_refiner_layers "
            "to the preset values regardless of what the YAML says. "
            "See omnivae_generation.trainer.config.MODEL_SIZE_PRESETS for the registered presets."
        ),
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Override train.per_device_batch_size from the YAML.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override train.learning_rate from the YAML.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override train.gradient_accumulation_steps from the YAML.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=(
            "Override vae.model_name_or_path from the YAML (visual / video VAE "
            "weight location). Useful for swapping an OmniVAE state_dict.pt or a "
            "Wan2.2-VAE directory without editing the YAML."
        ),
    )
    parser.add_argument(
        "--audio_vae_path",
        type=str,
        default=None,
        help=(
            "Override audio_vae.model_path from the YAML (audio VAE weight "
            "location). Useful for swapping an OmniVAE state_dict.pt or a DAC "
            "checkpoint without editing the YAML."
        ),
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default=None,
        help=(
            "Override vae.type from the YAML (e.g. 'omnivae', "
            "'wan2_2_native_vae'). Pair with --vae_path when switching VAE families."
        ),
    )
    parser.add_argument(
        "--audio_vae_type",
        type=str,
        default=None,
        help=(
            "Override audio_vae.type from the YAML (e.g. 'dac', 'omnivae'). "
            "Pair with --audio_vae_path when switching VAE families."
        ),
    )
    parser.add_argument(
        "--vae_branch",
        type=str,
        default=None,
        choices=["video", "audio", "both"],
        help=(
            "Only relevant when (audio_)vae.type='omnivae'. Controls which "
            "branch of the OmniVAE ckpt is built. Mirrored into both "
            "config.vae.branch and config.audio_vae.branch (the trainer reads "
            "only the one matching dataset.type)."
        ),
    )
    parser.add_argument(
        "--vae_use_ema",
        type=str,
        default=None,
        choices=["true", "false"],
        help=(
            "Only relevant when (audio_)vae.type='omnivae'. Toggles whether "
            "ema_state_dict.shadow weights are merged on top of "
            "model_state_dict. Mirrored into both vae.use_ema and "
            "audio_vae.use_ema."
        ),
    )
    parser.add_argument(
        "--init_from_transformer",
        type=str,
        default=None,
        help=(
            "Override transformer.init_from_transformer from the YAML. "
            "Pass a checkpoint directory (e.g. .../checkpoint-XXXXXX/transformer) "
            "to warm-start the transformer weights, or pass an empty string "
            "to force random initialization regardless of the YAML."
        ),
    )
    return parser.parse_args()


def _sync_run_identity(config: dict, override_name: str | None = None) -> None:
    """Keep ``experiment.name``, ``experiment.output_dir`` leaf, and
    ``wandb.run_name`` aligned to a single source of truth.

    When ``override_name`` is provided it replaces ``experiment.name``. The
    final ``experiment.output_dir`` is then composed so its leaf is the
    resolved name; ``wandb.run_name`` is forced to match. ``wandb`` is left
    absent if the YAML didn't declare it.

    Two YAML conventions are supported (auto-detected; you can pick either):

    - **Base-only style (preferred, no duplication)**: ``output_dir`` is just
      the parent directory and the leaf comes from ``experiment.name``::

          experiment:
            name: my-run
            output_dir: /scratch/.../runs        # leaf auto-appended

      Result: ``/scratch/.../runs/my-run``.

    - **Legacy full-path style**: ``output_dir`` already ends with the YAML's
      declared ``experiment.name``. We treat that leaf as a placeholder and
      rewrite it with the resolved name::

          experiment:
            name: my-run
            output_dir: /scratch/.../runs/my-run

      Result: ``/scratch/.../runs/my-run`` (idempotent), or
      ``/scratch/.../runs/<override>`` when ``--name <override>`` is passed.

    This runs every time the config is loaded so even runs without ``--name``
    have the three values in sync (the YAML can drift and we want one knob,
    not three).
    """
    exp_cfg = config.setdefault("experiment", {})
    yaml_name = str(exp_cfg.get("name") or "").strip()
    if override_name:
        cleaned = str(override_name).strip()
        if cleaned:
            exp_cfg["name"] = cleaned

    exp_name = str(exp_cfg.get("name") or "").strip()
    if not exp_name:
        return

    yaml_output_dir = exp_cfg.get("output_dir")
    if yaml_output_dir:
        out_path = Path(str(yaml_output_dir)).expanduser()
        # If the yaml output_dir leaf already matches the yaml-declared name,
        # it's the legacy full-path style: replace the leaf with the (possibly
        # overridden) resolved name. Otherwise it's the base-only style and
        # we just append the resolved name as a new leaf.
        if yaml_name and out_path.name == yaml_name:
            final = out_path.parent / exp_name
        else:
            final = out_path / exp_name
        exp_cfg["output_dir"] = str(final)

    wandb_cfg = config.get("wandb")
    if isinstance(wandb_cfg, dict):
        wandb_cfg["run_name"] = exp_name


def load_train_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    # Single source of truth for the run identity. Done before any other
    # config-touching path so downstream code (output_dir reads, wandb init,
    # validation sample dirs, etc.) all see the same name.
    _sync_run_identity(config, override_name=getattr(args, "name", None))
    if args.resume_from_checkpoint is not None:
        config["train"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    if getattr(args, "no_compile", False):
        # Force every torch.compile site off so the first step doesn't pay the
        # Inductor / cudagraph capture cost. The "_no_compile" sentinel is read
        # by train_zimage.py to also bypass the optimizer_step_transformer
        # decorator that lives outside this config block.
        transformer_cfg = config.setdefault("transformer", {})
        transformer_cfg["compile_model"] = False
        text_encoder_cfg = config.setdefault("text_encoder", {})
        text_encoder_cfg["compile_model"] = False
        if "audio_vae" in config and isinstance(config["audio_vae"], dict):
            config["audio_vae"]["compile_encode"] = False
        if "vae" in config and isinstance(config["vae"], dict):
            config["vae"]["compile_encode"] = False
        config["_no_compile"] = True
    if getattr(args, "validation_steps", None) is not None:
        validation_steps_override = int(args.validation_steps)
        if validation_steps_override < 0:
            raise ValueError(
                f"--validation_steps must be >= 0 (got {validation_steps_override})."
            )
        config.setdefault("train", {})["validation_steps"] = validation_steps_override

    # CLI --size is a hard override: it stomps any yaml-side dim/n_layers/etc
    # because the user explicitly asked for this preset on the command line.
    if getattr(args, "size", None):
        apply_model_size_preset_force(config, args.size)

    if getattr(args, "per_device_batch_size", None) is not None:
        bs_override = int(args.per_device_batch_size)
        if bs_override <= 0:
            raise ValueError(
                f"--per_device_batch_size must be > 0 (got {bs_override})."
            )
        config.setdefault("train", {})["per_device_batch_size"] = bs_override

    if getattr(args, "learning_rate", None) is not None:
        lr_override = float(args.learning_rate)
        if lr_override <= 0:
            raise ValueError(
                f"--learning_rate must be > 0 (got {lr_override})."
            )
        config.setdefault("train", {})["learning_rate"] = lr_override

    if getattr(args, "gradient_accumulation_steps", None) is not None:
        ga_override = int(args.gradient_accumulation_steps)
        if ga_override <= 0:
            raise ValueError(
                f"--gradient_accumulation_steps must be > 0 (got {ga_override})."
            )
        config.setdefault("train", {})["gradient_accumulation_steps"] = ga_override

    # ----- VAE weight / type / branch / EMA CLI overrides ----- #
    # Direct path / type overrides hit a single yaml block each. The
    # OmniVAE-only `branch` and `use_ema` knobs are mirrored into BOTH blocks
    # because the trainer reads only the one matching dataset.type, and
    # we don't want users to have to think about which one to flip.
    vae_path_override = getattr(args, "vae_path", None)
    if vae_path_override:
        config.setdefault("vae", {})["model_name_or_path"] = str(vae_path_override)

    audio_vae_path_override = getattr(args, "audio_vae_path", None)
    if audio_vae_path_override:
        config.setdefault("audio_vae", {})["model_path"] = str(audio_vae_path_override)

    vae_type_override = getattr(args, "vae_type", None)
    if vae_type_override:
        config.setdefault("vae", {})["type"] = str(vae_type_override).strip().lower()

    audio_vae_type_override = getattr(args, "audio_vae_type", None)
    if audio_vae_type_override:
        config.setdefault("audio_vae", {})["type"] = str(audio_vae_type_override).strip().lower()

    vae_branch_override = getattr(args, "vae_branch", None)
    if vae_branch_override:
        branch_value = str(vae_branch_override).strip().lower()
        for block in ("vae", "audio_vae"):
            if block in config and isinstance(config[block], dict):
                config[block]["branch"] = branch_value

    vae_use_ema_override = getattr(args, "vae_use_ema", None)
    if vae_use_ema_override is not None:
        ema_value = str(vae_use_ema_override).strip().lower() == "true"
        for block in ("vae", "audio_vae"):
            if block in config and isinstance(config[block], dict):
                config[block]["use_ema"] = ema_value

    init_from_transformer_override = getattr(args, "init_from_transformer", None)
    if init_from_transformer_override is not None:
        init_path_value = str(init_from_transformer_override).strip()
        config.setdefault("transformer", {})["init_from_transformer"] = (
            init_path_value or None
        )

    validate_training_config(config)
    return config


def get_dataset_type(config: dict) -> str:
    return str(config.get("dataset", {}).get("type", "imagenet")).strip().lower()


def is_video_dataset(config: dict) -> bool:
    return get_dataset_type(config) == "video_jsonl"


def is_audio_dataset(config: dict) -> bool:
    return get_dataset_type(config) == "audio_jsonl"


def get_active_vae_config(config: dict) -> dict:
    return config.get("audio_vae", {}) if is_audio_dataset(config) else config["vae"]


def get_validation_interval(config: dict) -> int:
    return max(0, int(config.get("train", {}).get("validation_steps") or 0))


def should_run_validation(config: dict, global_step: int) -> bool:
    validation_interval = get_validation_interval(config)
    return validation_interval > 0 and global_step % validation_interval == 0


def get_vae_validation_interval(config: dict) -> int:
    return max(0, int(config.get("train", {}).get("vae_validation_steps") or 0))


def should_run_vae_validation(config: dict, global_step: int) -> bool:
    # Audio datasets used to short-circuit here; they now get a real
    # encode->decode sanity check via run_audio_vae_validation (gated by the
    # same train.vae_validation_steps knob).
    validation_interval = get_vae_validation_interval(config)
    return validation_interval > 0 and global_step % validation_interval == 0


def prepare_pixel_values_for_vae(pixel_values: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    source_dtype = pixel_values.dtype
    prepared = pixel_values.to(device, dtype=dtype, non_blocking=True)
    if source_dtype == torch.uint8:
        prepared.mul_(1.0 / 127.5).sub_(1.0)
    return prepared


def validate_training_config(config: dict) -> None:
    dataset_type = get_dataset_type(config)
    if dataset_type == "audio_jsonl":
        dataset_cfg = config.get("dataset", {})
        sources = dataset_cfg.get("sources")
        metadata_paths = dataset_cfg.get("metadata_paths") or dataset_cfg.get("dataset_metadata_path")
        if sources:
            if not isinstance(sources, (list, tuple)) or len(sources) == 0:
                raise ValueError("dataset.sources must be a non-empty list.")
            valid_kinds = {"tts", "tta", "legacy"}
            total_weight = 0.0
            for src_index, raw_src in enumerate(sources):
                if not isinstance(raw_src, dict):
                    raise ValueError(
                        f"dataset.sources[{src_index}] must be a dict, got {raw_src!r}."
                    )
                src_path = raw_src.get("path")
                if not src_path:
                    raise ValueError(f"dataset.sources[{src_index}].path is required.")
                if not Path(str(src_path)).expanduser().exists():
                    raise ValueError(f"dataset.sources[{src_index}].path does not exist: {src_path!r}.")
                kind = str(raw_src.get("kind") or "legacy").strip().lower()
                if kind not in valid_kinds:
                    raise ValueError(
                        f"dataset.sources[{src_index}].kind must be one of {sorted(valid_kinds)}, "
                        f"got {kind!r}."
                    )
                weight = float(raw_src.get("weight", 1.0))
                if weight < 0:
                    raise ValueError(
                        f"dataset.sources[{src_index}].weight must be >= 0, got {weight}."
                    )
                total_weight += weight
                if kind == "tts" and not str(raw_src.get("text_field", "text")).strip():
                    raise ValueError(
                        f"dataset.sources[{src_index}].text_field must be a non-empty string."
                    )
                if kind == "tta" and not str(raw_src.get("prompt_field", "prompt_en")).strip():
                    raise ValueError(
                        f"dataset.sources[{src_index}].prompt_field must be a non-empty string."
                    )
            if total_weight <= 0:
                raise ValueError("Sum of dataset.sources[*].weight must be positive.")
        else:
            if not metadata_paths:
                raise ValueError(
                    "dataset.type='audio_jsonl' requires dataset.sources or dataset.metadata_paths."
                )
            if isinstance(metadata_paths, (str, Path)):
                metadata_paths = [metadata_paths]
            for metadata_path in metadata_paths:
                if not Path(str(metadata_path)).expanduser().exists():
                    raise ValueError(f"Audio metadata path does not exist: {metadata_path!r}.")
        audio_vae_cfg = config.get("audio_vae", {})
        supported_audio_vae_types = {"dac", "omnivae", "univae"}
        if normalize_vae_type(audio_vae_cfg.get("type"), default="dac") not in {"dac", "univae"}:
            raise ValueError(
                "dataset.type='audio_jsonl' currently requires audio_vae.type in "
                f"{sorted(supported_audio_vae_types)}."
            )
        model_path = audio_vae_cfg.get("model_path") or audio_vae_cfg.get("model_name_or_path")
        if not model_path:
            raise ValueError("dataset.type='audio_jsonl' requires audio_vae.model_path.")
        if not Path(str(model_path)).expanduser().exists():
            raise ValueError(f"audio_vae.model_path does not exist: {model_path!r}.")
        if int(dataset_cfg.get("sample_rate", 0)) <= 0:
            raise ValueError("dataset.sample_rate must be a positive integer.")
        if int(dataset_cfg.get("num_audio_samples", 0)) <= 0:
            raise ValueError("dataset.num_audio_samples must be a positive integer.")
        if bool(config.get("scheduler", {}).get("use_dynamic_shifting", False)):
            raise NotImplementedError(
                "dataset.type='audio_jsonl' requires scheduler.use_dynamic_shifting=false in v1."
            )
        return

    if dataset_type != "video_jsonl":
        return

    if bool(config.get("scheduler", {}).get("use_dynamic_shifting", False)):
        raise NotImplementedError(
            "dataset.type=video_jsonl requires scheduler.use_dynamic_shifting=false in v1."
        )
    vae_cfg = config.get("vae", {})
    vae_type = normalize_vae_type(vae_cfg.get("type"), default="")
    supported_video_vae_types = {"wan2_2_vae", "kei_vivit2_vae", "omnivae", "univae"}
    if vae_type not in supported_video_vae_types:
        raise ValueError(
            "dataset.type=video_jsonl requires vae.type to be one of: "
            f"{', '.join(sorted(supported_video_vae_types))}."
        )
    if vae_type == "wan2_2_vae" and str(vae_cfg.get("wan_chunk_mode", "cache")).strip().lower() == "parallel":
        temporal_scale = int(vae_cfg.get("scale_factor_temporal", 4) or 4)
        num_frames = int(config.get("dataset", {}).get("num_frames", 0))
        if temporal_scale > 1 and (num_frames - 1) % temporal_scale != 0:
            raise ValueError(
                "vae.wan_chunk_mode='parallel' requires dataset.num_frames=scale_factor_temporal*k+1 "
                f"(scale_factor_temporal={temporal_scale}, got dataset.num_frames={num_frames})."
            )

    init_from_transformer = config.get("transformer", {}).get("init_from_transformer")
    if init_from_transformer is not None and str(init_from_transformer).strip():
        if str(config.get("transformer", {}).get("arch", "zimage")).strip().lower() != "zimage":
            raise ValueError("transformer.init_from_transformer is only supported for transformer.arch='zimage'.")
        if not Path(str(init_from_transformer)).expanduser().exists():
            raise ValueError(f"transformer.init_from_transformer does not exist: {init_from_transformer!r}.")

    dataset_cfg = config.get("dataset", {})
    if not dataset_cfg.get("meta_path"):
        raise ValueError("dataset.type=video_jsonl requires dataset.meta_path.")
    if not Path(str(dataset_cfg.get("meta_path"))).expanduser().exists():
        raise ValueError(f"dataset.meta_path does not exist: {dataset_cfg.get('meta_path')!r}.")
    jsonl_index_path = dataset_cfg.get("jsonl_index_path")
    if jsonl_index_path is not None and not str(jsonl_index_path).strip():
        raise ValueError("dataset.jsonl_index_path must be null or a non-empty path.")
    if jsonl_index_path is not None and not Path(str(jsonl_index_path)).expanduser().exists():
        raise ValueError(f"dataset.jsonl_index_path does not exist: {jsonl_index_path!r}.")
    jsonl_prompt_field = str(dataset_cfg.get("jsonl_prompt_field", "prompt_v2")).strip()
    if not jsonl_prompt_field:
        raise ValueError("dataset.jsonl_prompt_field must be a non-empty string.")
    jsonl_prompt_index_path = dataset_cfg.get("jsonl_prompt_index_path")
    if jsonl_prompt_index_path is not None and not str(jsonl_prompt_index_path).strip():
        raise ValueError("dataset.jsonl_prompt_index_path must be null or a non-empty path.")
    if jsonl_prompt_index_path is not None and not Path(str(jsonl_prompt_index_path)).expanduser().exists():
        raise ValueError(f"dataset.jsonl_prompt_index_path does not exist: {jsonl_prompt_index_path!r}.")
    jsonl_path_field = str(dataset_cfg.get("jsonl_path_field", "video_path")).strip()
    if not jsonl_path_field:
        raise ValueError("dataset.jsonl_path_field must be a non-empty string.")

    frame_size = dataset_cfg.get("frame_size")
    if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
        raise ValueError("dataset.type=video_jsonl requires dataset.frame_size=[height, width].")
    if any(int(item) <= 0 for item in frame_size):
        raise ValueError(f"dataset.frame_size must contain positive integers, got {frame_size!r}.")
    if int(dataset_cfg.get("num_frames", 0)) <= 0:
        raise ValueError(f"dataset.num_frames must be a positive integer, got {dataset_cfg.get('num_frames')!r}.")
    if float(dataset_cfg.get("target_fps", 0.0)) <= 0.0:
        raise ValueError(f"dataset.target_fps must be positive, got {dataset_cfg.get('target_fps')!r}.")

    decode_backend = str(dataset_cfg.get("decode_backend", "auto")).strip().lower()
    if decode_backend not in SUPPORTED_DECODE_BACKENDS:
        raise ValueError(
            f"Unsupported dataset.decode_backend={decode_backend!r}. Expected one of: {sorted(SUPPORTED_DECODE_BACKENDS)}."
        )


def get_sigmas_for_timestep_indices(
    scheduler,
    timestep_indices: torch.Tensor,
    *,
    n_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)[timestep_indices]
    while sigmas.ndim < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas


def apply_prompt_dropout(
    prompts: List[str],
    dropout_prob: float,
    empty_prompts: List[str] | None = None,
    dropout_mask: torch.Tensor | None = None,
) -> List[str]:
    if dropout_prob <= 0:
        return prompts
    if empty_prompts is None:
        empty_prompts = [""] * len(prompts)
    if len(empty_prompts) != len(prompts):
        raise ValueError("Prompt dropout expects `empty_prompts` to match `prompts` length.")
    if dropout_mask is None:
        dropout_mask = torch.rand(len(prompts)) < dropout_prob
    if int(dropout_mask.numel()) != len(prompts):
        raise ValueError("Prompt dropout mask must match `prompts` length.")
    return [
        empty_prompt if bool(dropout_mask[index].item()) else prompt
        for index, (prompt, empty_prompt) in enumerate(zip(prompts, empty_prompts))
    ]


def get_scheduler_set_timesteps_kwargs(config: dict, scheduler, vae) -> dict:
    if not scheduler.config.use_dynamic_shifting:
        return {}
    if is_audio_dataset(config):
        raise NotImplementedError("Dynamic shifting is not supported for audio_jsonl training.")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_size = int(config["dataset"]["image_size"])
    latent_height = 2 * (image_size // (vae_scale_factor * 2))
    latent_width = 2 * (image_size // (vae_scale_factor * 2))
    patch_size = int(config["transformer"]["all_patch_size"][0])
    image_seq_len = (latent_height // patch_size) * (latent_width // patch_size)

    return {
        "mu": calculate_shift(
            image_seq_len=image_seq_len,
            base_seq_len=scheduler.config.base_image_seq_len,
            max_seq_len=scheduler.config.max_image_seq_len,
            base_shift=scheduler.config.base_shift,
            max_shift=scheduler.config.max_shift,
        )
    }


def configure_torch_dynamo(config: dict) -> int | None:
    recompile_limit = config["accelerate"].get("dynamo_recompile_limit")
    if recompile_limit is None or not hasattr(torch, "_dynamo"):
        return None

    recompile_limit = int(recompile_limit)
    if recompile_limit <= 0:
        raise ValueError("accelerate.dynamo_recompile_limit must be a positive integer.")

    torch._dynamo.config.recompile_limit = recompile_limit
    return int(torch._dynamo.config.recompile_limit)


def configure_manual_gc(config: dict) -> int | None:
    if not config["train"].get("manual_gc", False):
        return None

    manual_gc_every_steps = int(config["train"].get("manual_gc_every_steps", 0))
    if manual_gc_every_steps <= 0:
        raise ValueError("train.manual_gc_every_steps must be a positive integer when train.manual_gc is enabled.")

    gc.disable()
    return manual_gc_every_steps


def build_training_paths(config: dict) -> TrainingPaths:
    output_dir = Path(config["experiment"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    return TrainingPaths(
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        snapshot_root=checkpoints_dir / "snapshots",
        persistent_root=checkpoints_dir / "persistent",
        logging_dir=output_dir / "logs",
    )


def create_training_runtime(config: dict, *, extra_find_unused_parameters: bool = False) -> TrainingRuntime:
    paths = build_training_paths(config)

    configure_wandb_env(config)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(config["train"]["train_text_encoder"] or extra_find_unused_parameters)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        mixed_precision=config["accelerate"]["mixed_precision"],
        log_with=None if config["accelerate"]["log_with"] in {None, "none"} else config["accelerate"]["log_with"],
        project_config=ProjectConfiguration(project_dir=str(paths.output_dir), logging_dir=str(paths.logging_dir)),
        kwargs_handlers=[ddp_kwargs],
    )

    if config["train"]["seed"] is not None:
        set_seed(config["train"]["seed"])

    if config["train"]["allow_tf32"] and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if bool(config["train"].get("detect_anomaly", False)):
        torch.autograd.set_detect_anomaly(True)

    dynamo_recompile_limit = configure_torch_dynamo(config)
    if dynamo_recompile_limit is not None:
        logger.info("Using torch._dynamo.config.recompile_limit=%s", dynamo_recompile_limit)
    manual_gc_every_steps = configure_manual_gc(config)
    if manual_gc_every_steps is not None:
        logger.info("Disabled automatic Python GC; running gc.collect() every %s optimizer steps", manual_gc_every_steps)

    if accelerator.is_main_process:
        ensure_dir(paths.output_dir)
        ensure_dir(paths.checkpoints_dir)
        ensure_dir(paths.snapshot_root)
        ensure_dir(paths.persistent_root)
        shutil.copy2(config["_config_path"], paths.output_dir / "resolved_config.yaml")
        save_json(paths.output_dir / "resolved_config.json", config)

    if config["accelerate"]["log_with"] not in {None, "none"}:
        tracker_project_name, tracker_init_kwargs = get_tracker_init(
            config["experiment"]["name"],
            config,
            paths.logging_dir,
        )
        accelerator.init_trackers(
            tracker_project_name,
            config=flatten_config(config),
            init_kwargs=tracker_init_kwargs,
        )

    return TrainingRuntime(
        accelerator=accelerator,
        paths=paths,
        manual_gc_every_steps=manual_gc_every_steps,
    )


def load_text_setup(config: dict, accelerator: Accelerator) -> TextSetup:
    logger.info("Loading tokenizer and text encoder.")
    logger.info("Using HF_HOME=%s", os.environ.get("HF_HOME", ""))
    logger.info("Using HF_HUB_OFFLINE=%s", os.environ.get("HF_HUB_OFFLINE", ""))
    logger.info("Using TORCHINDUCTOR_CACHE_DIR=%s", os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""))
    if config["accelerate"]["log_with"] == "wandb":
        logger.info("Using WANDB_BASE_URL=%s", os.environ.get("WANDB_BASE_URL", ""))
    weight_dtype = get_weight_dtype(accelerator)
    if uses_unified_qwen3_vl_transformer(config):
        tokenizer, text_hidden_size = load_qwen3_vl_tokenizer_and_hidden_size(config["transformer"])
        return TextSetup(
            tokenizer=tokenizer,
            text_encoder=None,
            text_hidden_size=text_hidden_size,
            weight_dtype=weight_dtype,
            text_encoder_dtype=weight_dtype,
        )
    text_encoder_dtype = (
        torch.float32
        if config["train"]["train_text_encoder"]
        else resolve_dtype(config["text_encoder"].get("torch_dtype"), fallback=weight_dtype)
    )
    tokenizer, text_encoder, text_hidden_size = load_text_components(config["text_encoder"], text_encoder_dtype)
    return TextSetup(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_hidden_size=text_hidden_size,
        weight_dtype=weight_dtype,
        text_encoder_dtype=text_encoder_dtype,
    )


def load_base_models(config: dict, text_hidden_size: int):
    logger.info("Loading VAE and scheduler.")
    vae = load_audio_vae(config["audio_vae"]) if is_audio_dataset(config) else load_vae(config["vae"])
    noise_scheduler = load_scheduler(config["scheduler"])

    if uses_unified_qwen3_vl_transformer(config):
        logger.info("Building Qwen3-VL unified diffusion transformer.")
    elif config["transformer"].get("init_from_transformer"):
        logger.info("Building Z-Image transformer for warm-start initialization.")
    else:
        logger.info("Building random-init Z-Image transformer.")
    transformer = build_transformer(config["transformer"], cap_feat_dim=text_hidden_size)
    if not uses_unified_qwen3_vl_transformer(config):
        transformer._laion_caption_target_length = int(config["text_encoder"]["max_sequence_length"])
    init_from_transformer = config["transformer"].get("init_from_transformer")
    if init_from_transformer:
        if uses_unified_qwen3_vl_transformer(config):
            raise ValueError("transformer.init_from_transformer is only supported for arch='zimage'.")
        expected_transformer_config = build_transformer_init_expected_config(
            config["transformer"],
            cap_feat_dim=text_hidden_size,
        )
        logger.info("Loading transformer warm-start weights from %s.", init_from_transformer)
        loaded_transformer_dir = load_pretrained_transformer_weights(
            transformer,
            init_from_transformer,
            expected_config=expected_transformer_config,
        )
        logger.info("Loaded transformer warm-start weights from %s.", loaded_transformer_dir)
    transformer_param_count = count_parameters(transformer)
    if config["train"]["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
    attention_backend = config["transformer"].get("attention_backend")
    if attention_backend and hasattr(transformer, "set_attention_backend"):
        transformer.set_attention_backend(attention_backend)

    return vae, noise_scheduler, transformer, transformer_param_count


def configure_model_modes(config: dict, text_encoder, vae, *, train_vae: bool) -> None:
    vae.requires_grad_(train_vae)
    vae.train(train_vae)
    if train_vae and bool(config["train"].get("gradient_checkpointing", False)):
        gradient_checkpointing_target = None
        if hasattr(vae, "enable_gradient_checkpointing"):
            gradient_checkpointing_target = vae
        elif hasattr(getattr(vae, "inner", None), "enable_gradient_checkpointing"):
            gradient_checkpointing_target = vae.inner
        if gradient_checkpointing_target is not None and bool(
            getattr(gradient_checkpointing_target, "_supports_gradient_checkpointing", False)
        ):
            gradient_checkpointing_target.enable_gradient_checkpointing()
    if text_encoder is None:
        return
    if config["train"]["train_text_encoder"]:
        text_encoder.train()
    else:
        text_encoder.requires_grad_(False)
        text_encoder.eval()


def get_prompt_max_sequence_length(config: dict) -> int:
    if uses_unified_qwen3_vl_transformer(config):
        return int(
            config["transformer"].get(
                "max_sequence_length",
                config.get("text_encoder", {}).get("max_sequence_length", 512),
            )
        )
    return int(config["text_encoder"]["max_sequence_length"])


def build_dataset_and_dataloader(config: dict, tokenizer, accelerator: Accelerator, *, include_raw_pixel_values: bool):
    dataset_type = get_dataset_type(config)
    prompt_max_sequence_length = get_prompt_max_sequence_length(config) if tokenizer is not None else None
    if dataset_type == "audio_jsonl":
        dataset_cfg = config["dataset"]
        sources = dataset_cfg.get("sources")
        metadata_paths = None
        if not sources:
            metadata_paths = dataset_cfg.get("metadata_paths") or dataset_cfg.get("dataset_metadata_path")
            if isinstance(metadata_paths, (str, Path)):
                metadata_paths = [metadata_paths]
        dataset = AudioJsonlT2ADataset(
            sources=sources,
            metadata_paths=metadata_paths,
            dataset_base_path=dataset_cfg.get("dataset_base_path", "/"),
            sample_rate=dataset_cfg.get("sample_rate", 48000),
            num_audio_samples=dataset_cfg.get("num_audio_samples", 1440000),
            max_num_audio_samples=dataset_cfg.get("max_num_audio_samples", 1440000),
            mono=dataset_cfg.get("mono", True),
            append_duration_suffix=dataset_cfg.get("append_duration_suffix", True),
            duration_precision=dataset_cfg.get("duration_precision", 1),
            max_samples=dataset_cfg.get("max_samples"),
            tokenizer=tokenizer,
            task_prefix_enabled=dataset_cfg.get("task_prefix_enabled", True),
        )
    elif dataset_type == "relaion":
        dataset = RelaionDataset(
            root=config["dataset"]["root"],
            slave_path=config["dataset"]["slave_path"],
            base_image_path=config["dataset"]["base_image_path"],
            split=config["dataset"]["split"],
            image_size=config["dataset"]["image_size"],
            center_crop=config["dataset"]["center_crop"],
            random_flip=config["dataset"]["random_flip"],
            max_samples=config["dataset"]["max_samples"],
            cache_dir=config["dataset"]["cache_dir"],
            repeat=config["dataset"].get("repeat", 1),
            recaption_prob=config["dataset"].get("recaption_prob", 0.0),
            tokenizer=tokenizer,
            include_raw_pixel_values=include_raw_pixel_values,
            prompt_max_sequence_length=prompt_max_sequence_length,
        )
    elif dataset_type == "video_jsonl":
        # Without coordination every rank would call _load_or_build_offsets
        # (and build_string_sidecar) concurrently on a cold cache, scanning
        # the same multi-GB jsonl 8x and racing each other on the same
        # output files. Have the main process build everything (with a tqdm
        # progress bar) while the rest wait at the barrier; downstream
        # ``VideoJsonlDataset(...)`` calls then cache-hit on every rank.
        if accelerator.is_main_process:
            prebuild_video_jsonl_indexes(config["dataset"])
        accelerator.wait_for_everyone()

        dataset = VideoJsonlDataset(
            meta_path=config["dataset"]["meta_path"],
            frame_size=config["dataset"]["frame_size"],
            num_frames=config["dataset"]["num_frames"],
            target_fps=config["dataset"]["target_fps"],
            center_crop=config["dataset"]["center_crop"],
            random_flip=config["dataset"]["random_flip"],
            max_samples=config["dataset"]["max_samples"],
            cache_dir=config["dataset"]["cache_dir"],
            decode_backend=config["dataset"].get("decode_backend", "auto"),
            jsonl_index_path=config["dataset"].get("jsonl_index_path"),
            jsonl_path_field=config["dataset"].get("jsonl_path_field", "video_path"),
            jsonl_prompt_field=config["dataset"].get("jsonl_prompt_field", "prompt_v2"),
            jsonl_prompt_index_path=config["dataset"].get("jsonl_prompt_index_path"),
            tokenizer=tokenizer,
            include_raw_pixel_values=include_raw_pixel_values,
            return_uint8=config["dataset"].get("return_uint8", True),
            prompt_max_sequence_length=prompt_max_sequence_length,
        )
    else:
        dataset = ImageNetTextToImageDataset(
            root=config["dataset"]["root"],
            split=config["dataset"]["split"],
            image_size=config["dataset"]["image_size"],
            center_crop=config["dataset"]["center_crop"],
            random_flip=config["dataset"]["random_flip"],
            max_samples=config["dataset"]["max_samples"],
            captions_file=config["dataset"]["captions_file"],
            prompt_templates=config["dataset"]["prompt_templates"],
            use_full_label=config["dataset"]["use_full_label"],
            tokenizer=tokenizer,
            include_raw_pixel_values=include_raw_pixel_values,
            prompt_max_sequence_length=prompt_max_sequence_length,
        )
    train_dataloader = build_train_dataloader(accelerator, dataset, config)
    return dataset, train_dataloader


def append_trainable_named_parameters(named_params: list[tuple[str, torch.nn.Parameter]], module, prefix: str) -> None:
    named_params.extend(
        (f"{prefix}.{name}", parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )


def build_main_optimizer(
    config: dict,
    accelerator: Accelerator,
    named_params_to_optimize: list[tuple[str, torch.nn.Parameter]],
):
    params_to_optimize = [parameter for _, parameter in named_params_to_optimize]

    learning_rate = config["train"]["learning_rate"]
    if config["train"]["scale_lr"]:
        learning_rate = learning_rate * config["train"]["gradient_accumulation_steps"]
        learning_rate = learning_rate * config["train"]["per_device_batch_size"] * accelerator.num_processes

    optimizer_name = str(config["train"].get("optimizer", "adamw")).strip().lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=torch.tensor(learning_rate),
            betas=(config["train"]["adam_beta1"], config["train"]["adam_beta2"]),
            weight_decay=config["train"]["adam_weight_decay"],
            eps=config["train"]["adam_epsilon"],
        )
        logger.info("Using optimizer: AdamW")
        return optimizer
    if optimizer_name == "hybrid":
        optimizer = HybridMuonAdamw(named_params_to_optimize, config["train"], torch.tensor(learning_rate))
        logger.info("Using optimizer: %s", optimizer.describe())
        return optimizer
    raise ValueError(f"Unsupported train.optimizer='{optimizer_name}'. Expected one of: adamw, Hybrid.")


def build_lr_scheduler(config: dict, accelerator: Accelerator, optimizer):
    max_train_steps = int(config["train"]["max_train_steps"])
    return get_scheduler(
        config["train"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["train"]["lr_warmup_steps"] * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )


def finalize_models_and_schedules(
    config: dict,
    accelerator: Accelerator,
    text_encoder,
    text_encoder_dtype: torch.dtype,
    vae,
    noise_scheduler,
    transformer,
    train_dataloader,
    *,
    train_vae: bool,
) -> FinalizedModels:
    transformer_model = accelerator.unwrap_model(transformer)
    predict_target = getattr(transformer_model, "_laion_predict_target", "v")
    if hasattr(transformer_model, "materialize_rope_cache"):
        transformer_model.materialize_rope_cache(accelerator.device)
    train_patch_size = int(config["transformer"]["all_patch_size"][0])
    train_f_patch_size = int(config["transformer"]["all_f_patch_size"][0])
    if hasattr(transformer_model, "set_forward_compilation"):
        transformer_model.set_forward_compilation(bool(config["transformer"].get("compile_model", True)))

    if text_encoder is not None:
        from omnivae_generation.trainer.runtime_patches import patch_transformers_qwen3_5_compile_friendly_linear_attn_mask
        from transformers import Qwen3_5Model
        from transformers.utils.output_capturing import maybe_install_capturing_hooks

        patch_transformers_qwen3_5_compile_friendly_linear_attn_mask()
        text_encoder: Qwen3_5Model
        if bool(config["text_encoder"].get("compile_model", True)):
            maybe_install_capturing_hooks(text_encoder)
            text_encoder = torch.compile(text_encoder, fullgraph=True, mode="reduce-overhead")

    active_vae_config = get_active_vae_config(config)
    vae_dtype = resolve_dtype(active_vae_config.get("torch_dtype"), fallback=torch.float32)
    vae.to(accelerator.device, dtype=vae_dtype)
    if train_vae:
        vae.train()
    else:
        vae.eval()
        if bool(active_vae_config.get("compile_encode", True)):
            compile_encode_fullgraph = active_vae_config.get("compile_encode_fullgraph")
            if compile_encode_fullgraph is None:
                compile_encode_fullgraph = getattr(vae, "wan_chunk_mode", "cache") != "parallel"
            vae.encode = torch.compile(
                vae.encode,
                fullgraph=bool(compile_encode_fullgraph),
                mode=str(active_vae_config.get("compile_encode_mode", "reduce-overhead")),
            )

    noise_scheduler.config.use_dynamic_shifting = bool(config["scheduler"].get("use_dynamic_shifting", False))
    noise_scheduler.set_timesteps(
        noise_scheduler.config.num_train_timesteps,
        device=accelerator.device,
        **get_scheduler_set_timesteps_kwargs(config, noise_scheduler, vae),
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["train"]["gradient_accumulation_steps"])
    num_train_epochs = math.ceil(int(config["train"]["max_train_steps"]) / num_update_steps_per_epoch)

    return FinalizedModels(
        text_encoder=text_encoder,
        vae_dtype=vae_dtype,
        transformer_model=transformer_model,
        predict_target=predict_target,
        train_patch_size=train_patch_size,
        train_f_patch_size=train_f_patch_size,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        num_train_epochs=num_train_epochs,
    )


def save_run_metadata(output_dir: Path, config: dict, transformer_param_count: int, text_hidden_size: int) -> None:
    save_json(
        output_dir / "run_metadata.json",
        {
            "transformer_parameters": transformer_param_count,
            "transformer_parameters_billion": round(transformer_param_count / 1e9, 4),
            "text_hidden_size": text_hidden_size,
            "config_path": config["_config_path"],
        },
    )


def build_forward_transformer(transformer, transformer_model, train_patch_size: int, train_f_patch_size: int):
    def forward_transformer(
        noisy_latents: torch.Tensor,
        model_timesteps: torch.Tensor,
        prompt_embeds,
    ):
        squeeze_frame_dim = False
        squeeze_audio_spatial_dims = False
        if noisy_latents.ndim == 3:
            model_input = noisy_latents.unsqueeze(-1).unsqueeze(-1)
            squeeze_audio_spatial_dims = True
        elif noisy_latents.ndim == 4:
            model_input = noisy_latents.unsqueeze(2)
            squeeze_frame_dim = True
        elif noisy_latents.ndim == 5:
            model_input = noisy_latents
        else:
            raise ValueError(
                "Expected latents with 3 dims [B, C, T], 4 dims [B, C, H, W], "
                f"or 5 dims [B, C, T, H, W], got ndim={noisy_latents.ndim}."
            )

        if isinstance(transformer_model, Qwen3VLDiffusionTransformer):
            outputs = transformer(
                list(model_input.unbind(dim=0)),
                model_timesteps,
                prompt_embeds,
                return_dict=False,
                patch_size=train_patch_size,
                f_patch_size=train_f_patch_size,
            )
            model_output = outputs[0]
            model_pred = torch.stack([item.float() for item in model_output], dim=0)
            if squeeze_frame_dim:
                model_pred = model_pred.squeeze(2)
            if squeeze_audio_spatial_dims:
                model_pred = model_pred.squeeze(-1).squeeze(-1)
            return model_pred, None

        packed_inputs = transformer_model.prepare_dense_inputs(
            list(model_input.unbind(dim=0)),
            prompt_embeds,
            train_patch_size,
            train_f_patch_size,
        )
        outputs = transformer(
            packed_inputs["x"],
            model_timesteps,
            packed_inputs["cap_feats"],
            return_dict=False,
            x_size=packed_inputs["x_size"],
            x_freqs=packed_inputs["x_freqs"],
            cap_freqs=packed_inputs["cap_freqs"],
            x_mask=packed_inputs["x_mask"],
            cap_mask=packed_inputs["cap_mask"],
            siglip_feats=packed_inputs["siglip_feats"],
            siglip_freqs=packed_inputs["siglip_freqs"],
            siglip_mask=packed_inputs["siglip_mask"],
            x_noise_tensor=packed_inputs["x_noise_tensor"],
            cap_noise_tensor=packed_inputs["cap_noise_tensor"],
            siglip_noise_tensor=packed_inputs["siglip_noise_tensor"],
            omni_mode=packed_inputs["omni_mode"],
            patch_size=train_patch_size,
            f_patch_size=train_f_patch_size,
        )
        model_output = outputs[0]
        model_pred = torch.stack([item.float() for item in model_output], dim=0)
        if squeeze_frame_dim:
            model_pred = model_pred.squeeze(2)
        if squeeze_audio_spatial_dims:
            model_pred = model_pred.squeeze(-1).squeeze(-1)
        return model_pred, None

    return forward_transformer


def _build_prompt_dropout_mask(num_prompts: int, dropout_prob: float) -> torch.Tensor | None:
    if dropout_prob <= 0:
        return None
    return torch.rand(int(num_prompts)) < float(dropout_prob)


def _select_tokenized_prompt_batch(batch: dict, dropout_mask: torch.Tensor | None):
    required_fields = (
        "prompt_input_ids",
        "prompt_attention_mask",
        "empty_prompt_input_ids",
        "empty_prompt_attention_mask",
    )
    if not all(field_name in batch for field_name in required_fields):
        return None

    input_ids = batch["prompt_input_ids"]
    attention_mask = batch["prompt_attention_mask"]
    if dropout_mask is None:
        return input_ids, attention_mask

    dropout_mask = dropout_mask.to(device=input_ids.device, dtype=torch.bool).view(-1, 1)
    return (
        torch.where(dropout_mask, batch["empty_prompt_input_ids"], input_ids),
        torch.where(dropout_mask, batch["empty_prompt_attention_mask"], attention_mask),
    )


def prepare_prompt_embeddings(
    config: dict,
    batch: dict,
    tokenizer,
    text_encoder,
    accelerator: Accelerator,
    *,
    train_text_encoder: bool,
) -> tuple[Any, Any]:
    dropout_prob = float(config["train"]["caption_dropout_prob"])
    dropout_mask = _build_prompt_dropout_mask(len(batch["prompts"]), dropout_prob)
    prompts = apply_prompt_dropout(
        batch["prompts"],
        dropout_prob,
        batch.get("empty_prompts"),
        dropout_mask=dropout_mask,
    )
    tokenized_prompt_batch = _select_tokenized_prompt_batch(batch, dropout_mask)
    if text_encoder is None:
        max_sequence_length = get_prompt_max_sequence_length(config)
        if tokenized_prompt_batch is not None:
            prompt_payloads = prompt_token_tensors_to_payloads(
                tokenized_prompt_batch[0],
                tokenized_prompt_batch[1],
                device=accelerator.device,
            )
            return prompt_payloads, prompt_payloads
        prompt_payloads = tokenize_prompt_payloads(
            prompts,
            tokenizer,
            device=accelerator.device,
            max_sequence_length=max_sequence_length,
        )
        return prompt_payloads, prompt_payloads

    text_context = nullcontext() if train_text_encoder else torch.no_grad()
    with text_context:
        if tokenized_prompt_batch is None:
            prompt_embeds = encode_prompts(
                prompts=prompts,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                device=accelerator.device,
                max_sequence_length=config["text_encoder"]["max_sequence_length"],
                cache_enabled=bool(config["text_encoder"].get("cache_enabled", True)),
            )
        else:
            prompt_embeds = encode_tokenized_prompts(
                prompts=prompts,
                input_ids=tokenized_prompt_batch[0],
                attention_mask=tokenized_prompt_batch[1],
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                device=accelerator.device,
                max_sequence_length=config["text_encoder"]["max_sequence_length"],
                cache_enabled=bool(config["text_encoder"].get("cache_enabled", True)),
            )
    detached_prompt_embeds = [embedding.detach() for embedding in prompt_embeds] if train_text_encoder else prompt_embeds
    return prompt_embeds, detached_prompt_embeds


def prepare_diffusion_batch(
    config: dict,
    noise_scheduler,
    latents: torch.Tensor,
    *,
    detach_target_latents: bool,
) -> DiffusionBatch:
    batch_size = latents.shape[0]
    noise = torch.randn_like(latents)
    density = compute_density_for_timestep_sampling(
        weighting_scheme=config["train"]["weighting_scheme"],
        batch_size=batch_size,
        logit_mean=config["train"]["logit_mean"],
        logit_std=config["train"]["logit_std"],
        mode_scale=config["train"]["mode_scale"],
        device=latents.device,
    )
    timestep_indices = (density * noise_scheduler.config.num_train_timesteps).long()
    timestep_indices = timestep_indices.clamp(0, noise_scheduler.config.num_train_timesteps - 1)
    schedule_timesteps = noise_scheduler.timesteps.to(device=latents.device)
    timesteps = schedule_timesteps[timestep_indices]
    sigmas = get_sigmas_for_timestep_indices(
        noise_scheduler,
        timestep_indices,
        n_dim=latents.ndim,
        dtype=latents.dtype,
        device=latents.device,
    )
    noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
    model_timesteps = (noise_scheduler.config.num_train_timesteps - timesteps) / float(
        noise_scheduler.config.num_train_timesteps
    )
    model_timesteps = model_timesteps.to(device=latents.device, dtype=torch.float32)

    target_latents = latents.detach() if detach_target_latents else latents
    target = target_latents.float() - noise.float()
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=config["train"]["weighting_scheme"],
        sigmas=sigmas,
    )
    return DiffusionBatch(
        batch_size=batch_size,
        sigmas=sigmas,
        noisy_latents=noisy_latents,
        model_timesteps=model_timesteps,
        target=target,
        weighting=weighting,
    )


def compute_per_sample_denoising_loss(
    weighting: torch.Tensor,
    model_pred_for_loss: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample weighted MSE between ``model_pred_for_loss`` and ``target``.

    Returns a 1D tensor of shape ``[batch_size]``. Used by the loss-spike
    debugger to attribute high loss to specific samples; the standard
    training loss path further reduces this with ``.mean()`` (see
    :func:`compute_denoising_loss`), so behaviour stays identical.
    """
    return torch.mean(
        (weighting.float() * (model_pred_for_loss.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        dim=1,
    )


def compute_denoising_loss(weighting: torch.Tensor, model_pred_for_loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return compute_per_sample_denoising_loss(weighting, model_pred_for_loss, target).mean()


def get_vae_model_reference(config: dict) -> str:
    if is_audio_dataset(config):
        audio_vae_config = config.get("audio_vae", {})
        return str(audio_vae_config.get("model_path") or audio_vae_config.get("model_name_or_path") or "")
    return str(config["vae"]["model_name_or_path"])


def build_checkpoint_metadata(
    config: dict,
    *,
    global_step: int,
    transformer_param_count: int,
    checkpoint_kind: str | None,
    save_vae: bool,
) -> dict:
    metadata = {
        "global_step": global_step,
        "transformer_parameters": transformer_param_count,
        "vae_model_name_or_path": get_vae_model_reference(config),
        "text_encoder_model_name_or_path": None
        if uses_unified_qwen3_vl_transformer(config)
        else config["text_encoder"]["model_name_or_path"],
        "transformer_backbone_name_or_path": config["transformer"].get("backbone_name_or_path"),
        "dataloader_resume_strategy": DATALOADER_RESUME_STRATEGY,
        "local_vae_subdir": "vae" if save_vae else None,
    }
    if checkpoint_kind is not None:
        metadata["checkpoint_kind"] = checkpoint_kind
    return metadata


def export_checkpoint_artifacts(
    *,
    accelerator: Accelerator,
    output_dir: Path,
    transformer,
    tokenizer,
    scheduler,
    config: dict,
    global_step: int,
    transformer_param_count: int,
    train_text_encoder: bool,
    text_encoder,
    vae=None,
    checkpoint_kind: str | None,
) -> None:
    save_vae = vae is not None
    save_checkpoint_artifacts(
        output_dir=output_dir,
        transformer=accelerator.unwrap_model(transformer),
        tokenizer=tokenizer,
        scheduler=scheduler,
        text_encoder=accelerator.unwrap_model(text_encoder) if train_text_encoder else None,
        vae=accelerator.unwrap_model(vae) if save_vae else None,
        metadata=build_checkpoint_metadata(
            config,
            global_step=global_step,
            transformer_param_count=transformer_param_count,
            checkpoint_kind=checkpoint_kind,
            save_vae=save_vae,
        ),
    )


def save_managed_checkpoint(
    *,
    accelerator: Accelerator,
    checkpoint_root: Path,
    checkpoint_kind: str,
    checkpoints_limit,
    train_dataloader,
    process_index: int,
    config: dict,
    global_step: int,
    transformer_param_count: int,
    transformer,
    tokenizer,
    scheduler,
    train_text_encoder: bool,
    text_encoder,
    vae=None,
) -> None:
    checkpoint_dir = checkpoint_root / f"checkpoint-{global_step:08d}"
    accelerator.save_state(str(checkpoint_dir))
    save_dataloader_state(train_dataloader, checkpoint_dir, process_index)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        export_checkpoint_artifacts(
            accelerator=accelerator,
            output_dir=checkpoint_dir,
            transformer=transformer,
            tokenizer=tokenizer,
            scheduler=scheduler,
            config=config,
            global_step=global_step,
            transformer_param_count=transformer_param_count,
            train_text_encoder=train_text_encoder,
            text_encoder=text_encoder,
            vae=vae,
            checkpoint_kind=checkpoint_kind,
        )
        mark_checkpoint_complete(checkpoint_dir)
        rotate_checkpoints(checkpoint_root, checkpoints_limit)
    accelerator.wait_for_everyone()


def finish_training(manual_gc_every_steps: int | None, accelerator: Accelerator) -> None:
    if manual_gc_every_steps is not None:
        gc.enable()
    accelerator.end_training()
