from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "imagenet-zimage-qwen35-1b",
        "output_dir": "./outputs/imagenet-zimage-qwen35-1b",
    },
    "accelerate": {
        "mixed_precision": "bf16",
        "log_with": "tensorboard",
        "dynamo_recompile_limit": 32,
        "use_stateful_dataloader": True,
        "stateful_snapshot_every_n_steps": 1,
    },
    "wandb": {
        "base_url": None,
        "entity": None,
        "project": "omnivae-generation",
        "run_name": None,
        "run_id": None,
        "resume": "allow",
    },
    "profiler": {
        "enabled": False,
        "trace_dir": None,
        "activities": ["cpu", "cuda"],
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "with_modules": False,
        "wait": 1,
        "warmup": 1,
        "active": 3,
        "repeat": 1,
        "skip_first": 0,
        "all_ranks": False,
    },
    "dataset": {
        "type": "imagenet",
        "root": "data/imagenet",
        "meta_path": None,
        "jsonl_index_path": None,
        "jsonl_path_field": "video_path",
        "jsonl_prompt_field": "prompt_v2",
        "jsonl_prompt_index_path": None,
        "slave_path": "",
        "base_image_path": "",
        "split": "train",
        "image_size": 256,
        "frame_size": [256, 256],
        "num_frames": 1,
        "target_fps": 1.0,
        "decode_backend": "torchcodec",
        "return_uint8": True,
        "center_crop": False,
        "random_flip": True,
        "num_workers": 8,
        "prefetch_factor": 2,
        "pin_memory": True,
        "drop_last": True,
        "max_samples": None,
        "cache_dir": ".cache/datasets",
        "repeat": 1,
        "recaption_prob": 0.0,
        "captions_file": None,
        "prompt_templates": ["a high-quality photo of {label}"],
        "use_full_label": False,
        "metadata_paths": [],
        "dataset_base_path": "/",
        "sample_rate": 48000,
        "num_audio_samples": 1440000,
        "max_num_audio_samples": 1440000,
        "mono": True,
        "append_duration_suffix": True,
        "duration_precision": 1,
    },
    "audio_vae": {
        "type": "dac",
        "model_path": "checkpoints/audio_vae/vae_128d_48k.pth",
        "hop_length": 960,
        "latent_channels": 128,
        "torch_dtype": "float32",
        "compile_encode": False,
        "compile_encode_fullgraph": False,
        "compile_encode_mode": "reduce-overhead",
    },
    "text_encoder": {
        "model_name_or_path": "Qwen/Qwen3.5-0.8B-Base",
        "trust_remote_code": False,
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "disable_qwen3_5_fast_path": False,
        "cache_enabled": False,
        "compile_model": True,
        "max_sequence_length": 512,
    },
    "vae": {
        "type": "autoencoder_kl",
        "model_name_or_path": "Tongyi-MAI/Z-Image-Turbo",
        "subfolder": "vae",
        "torch_dtype": "float32",
        "local_files_only": False,
        "wan_chunk_mode": "cache",
        "compile_encode": True,
        "compile_encode_mode": "reduce-overhead",
        "compile_encode_fullgraph": None,
    },
    "vae_loss": {
        "discriminator_start": 0,
        "discriminator_factor": 1.0,
        "discriminator_weight": 0.1,
        "perceptual_loss": "lpips",
        "perceptual_weight": 1.0,
        "reconstruction_loss": "l1",
        "reconstruction_weight": 1.0,
        "lecam_regularization_weight": 0.0,
        "lecam_ema_decay": 0.999,
        "kl_weight": 1e-6,
        "logvar_init": 0.0,
    },
    "scheduler": {
        "model_name_or_path": "Tongyi-MAI/Z-Image-Turbo",
        "subfolder": "scheduler",
        "local_files_only": False,
        "use_dynamic_shifting": False,
    },
    "transformer": {
        "arch": "zimage",
        "backbone_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
        "backbone_torch_dtype": "bfloat16",
        "init_from_pretrained_backbone": True,
        "text_config": None,
        "trust_remote_code": False,
        "local_files_only": False,
        "init_from_transformer": None,
        "attn_implementation": "flex_attention",
        "max_sequence_length": 512,
        "all_patch_size": [2],
        "all_f_patch_size": [1],
        "in_channels": 16,
        "dim": 2048,
        "n_layers": 18,
        "n_refiner_layers": 2,
        "n_heads": 16,
        "n_kv_heads": 16,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "use_timestep": True,
        "predict_target": "v",
        "axes_dims": [16, 56, 56],
        "axes_lens": [1024, 512, 512],
        "compile_model": True,
        "attention_backend": None,
    },
    "train": {
        "seed": 42,
        "train_text_encoder": False,
        "gradient_checkpointing": True,
        "allow_tf32": True,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "max_train_steps": 200000,
        "learning_rate": 1e-4,
        "vae_learning_rate": 1e-4,
        "disc_learning_rate": 1e-4,
        "scale_lr": False,
        "optimizer": "adamw",
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 2000,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        # Optional Muon optimizer overrides. Keep None to use torch defaults.
        "muon_momentum": None,
        "muon_nesterov": None,
        "muon_weight_decay": None,
        "muon_eps": None,
        "muon_ns_steps": None,
        "muon_ns_coefficients": None,
        "muon_adjust_lr_fn": None,
        # 0 => shard across all ranks, 1 => disable Muon sharding,
        # N => shard within groups of min(N, world_size) ranks.
        "muon_shard_across_ranks": 8,
        # -1 => unbounded, 0 => serial, 1 => 1 buffer to await tensor
        "muon_max_inflight_buckets": 1,
        # zxu TOFIX: under torch compile, tensor gather will be sunk to tail 
        "max_grad_norm": 1.0,
        "caption_dropout_prob": 0.1,
        "weighting_scheme": "none",
        "logit_mean": 0.0,
        "logit_std": 1.0,
        "mode_scale": 1.29,
        # Backward-compatible legacy keys (kept to avoid breaking existing YAMLs).
        "checkpointing_steps": 1000,
        "checkpoints_total_limit": 5,

        # New dual-checkpoint scheme.
        # - snapshots: frequent checkpoints, rotated to a limit (good for resume)
        # - persistent: infrequent checkpoints, intended to be kept long-term
        "snapshot_checkpointing_steps": None,  # defaults to `checkpointing_steps` when unset
        "snapshots_total_limit": None,  # defaults to `checkpoints_total_limit` when unset
        "persistent_checkpointing_steps": None,  # disabled when unset/<=0
        "persistent_total_limit": None,  # optional; unset/<=0 means keep all
        "validation_steps": 1000,
        "validation_prompts": [
            "a high-quality photo of a goldfish",
            "a high-quality photo of a tiger shark",
            "a high-quality photo of an ostrich",
            "a high-quality photo of a tabby cat",
        ],
        "validation_num_inference_steps": 20,
        "validation_guidance_scale": 4.0,
        "validation_cfg_normalization": False,
        "validation_cfg_truncation": 1.0,
        "validation_num_frames": None,
        "validation_frame_size": None,
        "validation_fps": None,
        "validation_negative_prompts": None,
        "vae_validation_steps": 0,
        "vae_validation_data": None,
        "vae_validation_path_list": None,
        "vae_validation_dataset_limit": None,
        "vae_validation_batch_size": 1,
        "vae_validation_num_samples": 4,
        "vae_validation_num_frames": None,
        "vae_validation_frame_size": None,
        "vae_validation_max_video_frames": 8,
        "vae_validation_fps": None,
        "vae_validation_num_ffmpeg_threads": 1,
        "vae_validation_pad_mode": "gray",
        "vae_validation_sample_mode": "argmax",
        "vae_validation_metrics": ["psnr"],
        "vae_validation_psnr_video_aggregation": "frame_mean",
        "resume_from_checkpoint": None,
        "log_every_steps": 10,
        "manual_gc": False,
        "manual_gc_every_steps": 1000,
        "detect_anomaly": False,
    },
}


# --------------------------------------------------------------------------- #
# Model size presets                                                           #
#                                                                              #
# A single ``transformer.model_size`` knob picks the (dim, n_layers, n_heads)  #
# bundle so users do not need to remember the exact numbers for each scale.    #
# Estimates assume the LLaMA SwiGLU FFN ratio (8/3 * dim) used by Z-Image,     #
# i.e. ~13.7 * n_layers * dim^2 trainable parameters.                          #
#                                                                              #
# Precedence (highest first):                                                  #
#   1. CLI ``--size``      forces every preset field (explicit user choice)    #
#   2. yaml explicit field e.g. ``transformer.dim`` overrides the preset       #
#   3. yaml ``transformer.model_size`` fills missing preset fields             #
#   4. DEFAULT_CONFIG values                                                   #
#                                                                              #
# Adding a new size: register here and (optionally) document it in train.sh.   #
# --------------------------------------------------------------------------- #
MODEL_SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    "1b": {
        "dim": 1920,
        "n_layers": 18,
        "n_refiner_layers": 2,
        "n_heads": 15,
        "n_kv_heads": 15,
    },
    "2.5b": {
        "dim": 2560,
        "n_layers": 28,
        "n_refiner_layers": 2,
        "n_heads": 20,
        "n_kv_heads": 20,
    },
    "5b": {
        "dim": 3072,
        "n_layers": 40,
        "n_refiner_layers": 2,
        "n_heads": 24,
        "n_kv_heads": 24,
    },
}


def _normalize_model_size(size: Any) -> str:
    """Canonicalize the ``model_size`` token (case/whitespace insensitive).

    Accepts e.g. ``"5B"``, ``"5b"``, ``" 2.5B "`` -> ``"5b"`` / ``"2.5b"``.
    """
    if size is None:
        return ""
    norm = str(size).strip().lower()
    return norm


def _apply_model_size_preset_to_base(
    base_config: Dict[str, Any], user_config: Dict[str, Any]
) -> None:
    """If the user yaml selects a ``transformer.model_size`` preset, splice the
    preset's fields into ``base_config`` *only* for keys the user did not
    explicitly set. This way any field the user wrote in yaml still wins after
    the subsequent ``_deep_merge(base_config, user_config)``.

    Mutates ``base_config`` in place. Safe to call with no preset selected.
    """
    user_transformer = user_config.get("transformer") if isinstance(user_config, dict) else None
    if not isinstance(user_transformer, dict):
        return

    size = _normalize_model_size(user_transformer.get("model_size"))
    if not size:
        return

    if size not in MODEL_SIZE_PRESETS:
        raise ValueError(
            f"Unsupported transformer.model_size={user_transformer.get('model_size')!r}. "
            f"Choose from: {sorted(MODEL_SIZE_PRESETS.keys())}"
        )

    preset = MODEL_SIZE_PRESETS[size]
    base_transformer = base_config.setdefault("transformer", {})
    for key, value in preset.items():
        if key not in user_transformer:
            base_transformer[key] = value


def apply_model_size_preset_force(config: Dict[str, Any], size: str) -> None:
    """CLI-side helper: apply a preset *forcefully* over an already-loaded
    config (yaml values are overwritten). Use this for ``--size`` overrides.

    Raises ValueError on unknown size.
    """
    norm = _normalize_model_size(size)
    if not norm:
        return
    if norm not in MODEL_SIZE_PRESETS:
        raise ValueError(
            f"Unsupported --size={size!r}. Choose from: {sorted(MODEL_SIZE_PRESETS.keys())}"
        )
    preset = MODEL_SIZE_PRESETS[norm]
    transformer_cfg = config.setdefault("transformer", {})
    transformer_cfg["model_size"] = norm
    for key, value in preset.items():
        transformer_cfg[key] = value


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    # Splice in any model_size preset before merging, so user yaml values still
    # take precedence (preset only fills keys the user did NOT explicitly set).
    base_config = deepcopy(DEFAULT_CONFIG)
    _apply_model_size_preset_to_base(base_config, user_config)
    config = _deep_merge(base_config, user_config)
    config["_config_path"] = str(config_path.resolve())
    config["experiment"]["output_dir"] = str(Path(config["experiment"]["output_dir"]).expanduser().resolve())

    # Backward compatibility for pre-dual-checkpoint configs.
    train_cfg = config.get("train", {})
    if train_cfg.get("snapshot_checkpointing_steps") is None:
        train_cfg["snapshot_checkpointing_steps"] = train_cfg.get("checkpointing_steps")
    if train_cfg.get("snapshots_total_limit") is None:
        train_cfg["snapshots_total_limit"] = train_cfg.get("checkpoints_total_limit")
    return config


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        current_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=current_key))
        else:
            flat[current_key] = _sanitize_tracker_value(value)
    return flat


def _sanitize_tracker_value(value: Any):
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)
