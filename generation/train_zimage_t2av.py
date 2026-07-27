"""Entry point for joint Text-to-Audio-Video (T2AV) training.

Composes two pretrained Z-Image transformer branches (t2v + t2a) via
the bridge cross-attention module from ``omnivae_generation.trainer.joint_av``. Reuses
runtime patches, accelerator setup, optimizer routing, dataloader, and
checkpointing helpers from :mod:`omnivae_generation.trainer.zimage_training` so this entry
point stays a thin wrapper that swaps in:

  * paired AV dataset (:class:`AVPairedJsonlDataset`)
  * dual VAEs (video + audio)
  * joint model (:class:`BridgedZImageJointModel`) with two
    transformer configs (``transformer_video`` + ``transformer_audio``)
  * dual sigma-shift batch prep (:func:`prepare_dual_diffusion_batch`)
  * heterogeneous-LR optimizer (:class:`HybridMuonAdamwTagged`)
  * three-mode validation (:func:`run_joint_av_validation`)
  * split-branch checkpoint save (:func:`save_split_branches`)

Usage mirrors ``train_zimage.py``::

    accelerate launch train_zimage_t2av.py --config configs/av/t2av.yaml

Run ``--help`` for the full CLI surface.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from omnivae_generation.trainer.runtime_env import ensure_hf_home

ensure_hf_home()

import torch
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

import omnivae_generation.trainer.zimage_training as zt
from omnivae_generation.trainer.config import flatten_config, load_config
from omnivae_generation.trainer.joint_av import (
    AVPairedJsonlDataset,
    BridgedZImageJointModel,
    HybridMuonAdamwTagged,
    collate_av_paired_samples,
    load_bridges_from_dir,
    load_pretrained_branches,
    prepare_dual_diffusion_batch,
    run_joint_av_validation,
    save_split_branches,
)
from omnivae_generation.trainer.modeling import (
    adapt_model_prediction,
    build_transformer,
    count_parameters,
    encode_audio_to_latents,
    encode_images_to_latents,
    load_audio_vae,
    load_scheduler,
    load_text_components,
    load_vae,
    resolve_dtype,
)
from omnivae_generation.trainer.profiler import build_profiler
from omnivae_generation.trainer.utils import ensure_dir, mark_checkpoint_complete, rotate_checkpoints, save_json


logger = get_logger(__name__)


# --------------------------------------------------------------------- args
def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the joint T2AV bridge model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    parser.add_argument("--name", type=str, default=None, help="Override experiment.name.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or `latest`/`latest_persistent`.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable all torch.compile sites for fast first-step debug.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Override train.validation_steps from the YAML.",
    )
    parser.add_argument(
        "--validate_at",
        type=str,
        default=None,
        help=(
            "One-shot validation step trigger(s), comma-separated. "
            "Validation runs once when global_step matches any value "
            "here, even if `global_step %% validation_steps != 0`. "
            "Special: `0` triggers a pre-train-loop validation on the "
            "warm-start weights *before* any optimizer step (handy for "
            "smoke-testing the pipeline / pretrained checkpoints). "
            "Examples: `--validate_at 0`, `--validate_at 0,1,500,1000`. "
            "Merged with train.validation_force_steps from YAML."
        ),
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Override train.per_device_batch_size from the YAML.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override train.gradient_accumulation_steps from the YAML.",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=None,
        help="Override train.backbone_lr from the YAML.",
    )
    parser.add_argument(
        "--bridge_lr",
        type=float,
        default=None,
        help="Override train.bridge_lr from the YAML.",
    )
    parser.add_argument(
        "--bridge_interval",
        type=int,
        default=None,
        help="Override transformer.bridge_interval from the YAML.",
    )
    parser.add_argument(
        "--pretrained_t2v",
        type=str,
        default=None,
        help="Override transformer_video.init_from_transformer.",
    )
    parser.add_argument(
        "--pretrained_t2a",
        type=str,
        default=None,
        help="Override transformer_audio.init_from_transformer.",
    )
    parser.add_argument(
        "--video_vae_path",
        type=str,
        default=None,
        help=(
            "Override `vae.model_name_or_path` (video VAE). Accepts a HF "
            "repo id, a local directory (`{vae_dir}/`), or a single-file "
            "checkpoint (`*.pth`/`*.pt`/`*.safetensors`)."
        ),
    )
    parser.add_argument(
        "--audio_vae_path",
        type=str,
        default=None,
        help=(
            "Override `audio_vae.model_path`. Accepts an OmniVAE "
            "state_dict.pt or a standalone DAC checkpoint."
        ),
    )
    parser.add_argument(
        "--video_vae_type",
        type=str,
        default=None,
        help=(
            "Override `vae.type` (e.g. wan2_2_vae / wan2_2_native_vae / "
            "omnivae). Combine with --video_vae_path when switching VAE "
            "families. See `trainer/modeling.py::load_vae` for the "
            "supported set."
        ),
    )
    parser.add_argument(
        "--audio_vae_type",
        type=str,
        default=None,
        help=(
            "Override `audio_vae.type` (e.g. omnivae / dac). The public "
            "T2AV recipe uses `omnivae`."
        ),
    )
    parser.add_argument(
        "--vae_branch",
        type=str,
        default=None,
        choices=["video", "both"],
        help=(
            "Override `vae.branch` (only used when --video_vae_type=omnivae). "
            "`video`=load just the video sub-VAE; `both`=also attach the "
            "OmniVAE audio companion to the video VAE wrapper."
        ),
    )
    parser.add_argument(
        "--vae_use_ema",
        type=str,
        default=None,
        choices=["true", "false"],
        help=(
            "Override `vae.use_ema` (only used when --video_vae_type=omnivae). "
            "Loads the EMA weights from the OmniVAE training checkpoint when "
            "true."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Override train.max_train_steps from the YAML.",
    )
    parser.add_argument(
        "--muon_shard_across_ranks",
        type=int,
        default=None,
        help=(
            "Override train.muon_shard_across_ranks. "
            "1=disable sharding, 0=shard across all ranks, N>1=shard across N ranks "
            "(N must divide world_size). Defaults to YAML value (8)."
        ),
    )
    parser.add_argument(
        "--shift_v",
        type=float,
        default=None,
        help="Override train.shift_v (per-modality video sigma shift; default 5.0).",
    )
    parser.add_argument(
        "--shift_a",
        type=float,
        default=None,
        help="Override train.shift_a (per-modality audio sigma shift; default 1.0).",
    )
    parser.add_argument(
        "--bridge_dropout_prob",
        type=float,
        default=None,
        help=(
            "Override train.bridge_dropout_prob (per-sample probability of "
            "dropping the bridge cross-modal injection during training; "
            "default 0.1). Independent from caption_dropout_prob."
        ),
    )
    return parser.parse_args()


def load_train_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    zt._sync_run_identity(config, override_name=getattr(args, "name", None))
    if args.resume_from_checkpoint is not None:
        config["train"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    if getattr(args, "no_compile", False):
        for block_name in ("transformer", "transformer_video", "transformer_audio", "text_encoder"):
            config.setdefault(block_name, {})["compile_model"] = False
        for block_name in ("vae", "audio_vae"):
            if block_name in config and isinstance(config[block_name], dict):
                config[block_name]["compile_encode"] = False
        config["_no_compile"] = True
    if getattr(args, "validation_steps", None) is not None:
        config.setdefault("train", {})["validation_steps"] = int(args.validation_steps)
    if getattr(args, "validate_at", None):
        # CLI accepts "1,500,1000"; merge with anything already declared in
        # YAML under train.validation_force_steps so YAML + CLI compose
        # rather than override.
        cli_steps: list[int] = []
        for tok in str(args.validate_at).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                cli_steps.append(int(tok))
            except ValueError:
                raise ValueError(
                    f"--validate_at expects comma-separated ints, got token {tok!r}"
                )
        existing = config.get("train", {}).get("validation_force_steps") or []
        if isinstance(existing, int):
            existing = [existing]
        # Allow 0 (pre-train-loop sanity check at warm-start state) and any
        # non-negative step. Negative values are still dropped.
        merged = sorted({int(s) for s in list(existing) + cli_steps if int(s) >= 0})
        config.setdefault("train", {})["validation_force_steps"] = merged
    if getattr(args, "per_device_batch_size", None) is not None:
        config.setdefault("train", {})["per_device_batch_size"] = int(args.per_device_batch_size)
    if getattr(args, "gradient_accumulation_steps", None) is not None:
        config.setdefault("train", {})["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if getattr(args, "backbone_lr", None) is not None:
        config.setdefault("train", {})["backbone_lr"] = float(args.backbone_lr)
    if getattr(args, "bridge_lr", None) is not None:
        config.setdefault("train", {})["bridge_lr"] = float(args.bridge_lr)
    if getattr(args, "max_train_steps", None) is not None:
        config.setdefault("train", {})["max_train_steps"] = int(args.max_train_steps)
    if getattr(args, "bridge_interval", None) is not None:
        config.setdefault("transformer", {})["bridge_interval"] = int(args.bridge_interval)
    if getattr(args, "pretrained_t2v", None) is not None:
        config.setdefault("transformer_video", {})["init_from_transformer"] = str(args.pretrained_t2v).strip() or None
    if getattr(args, "pretrained_t2a", None) is not None:
        config.setdefault("transformer_audio", {})["init_from_transformer"] = str(args.pretrained_t2a).strip() or None
    if getattr(args, "video_vae_path", None) is not None:
        # vae block uses HF-style `model_name_or_path` (repo id, local dir,
        # or single-file checkpoint).
        config.setdefault("vae", {})["model_name_or_path"] = str(args.video_vae_path).strip() or None
    if getattr(args, "audio_vae_path", None) is not None:
        # audio_vae (DAC) uses `model_path` (local .pth); make sure both legacy
        # readers see the override.
        audio_vae_path = str(args.audio_vae_path).strip() or None
        config.setdefault("audio_vae", {})["model_path"] = audio_vae_path
    if getattr(args, "video_vae_type", None) is not None:
        config.setdefault("vae", {})["type"] = str(args.video_vae_type).strip()
    if getattr(args, "audio_vae_type", None) is not None:
        config.setdefault("audio_vae", {})["type"] = str(args.audio_vae_type).strip()
    if getattr(args, "vae_branch", None) is not None:
        # Only meaningful for vae.type=omnivae but harmless to set otherwise.
        config.setdefault("vae", {})["branch"] = str(args.vae_branch).strip()
    if getattr(args, "vae_use_ema", None) is not None:
        config.setdefault("vae", {})["use_ema"] = (
            str(args.vae_use_ema).strip().lower() == "true"
        )
    if getattr(args, "muon_shard_across_ranks", None) is not None:
        config.setdefault("train", {})["muon_shard_across_ranks"] = int(args.muon_shard_across_ranks)
    if getattr(args, "shift_v", None) is not None:
        config.setdefault("train", {})["shift_v"] = float(args.shift_v)
    if getattr(args, "shift_a", None) is not None:
        config.setdefault("train", {})["shift_a"] = float(args.shift_a)
    if getattr(args, "bridge_dropout_prob", None) is not None:
        config.setdefault("train", {})["bridge_dropout_prob"] = float(args.bridge_dropout_prob)
    validate_t2av_config(config)
    return config


def validate_t2av_config(config: dict) -> None:
    dataset_cfg = config.get("dataset", {}) or {}
    if dataset_cfg.get("type") not in {"av_paired_jsonl", None}:
        raise ValueError("T2AV trainer requires dataset.type='av_paired_jsonl'.")
    has_sources = bool(dataset_cfg.get("sources"))
    has_jsonl = bool(dataset_cfg.get("jsonl_path"))
    if has_sources and has_jsonl:
        raise ValueError(
            "dataset.sources and dataset.jsonl_path are mutually exclusive."
        )
    if not has_sources and not has_jsonl:
        raise ValueError(
            "T2AV trainer requires either dataset.sources (multi-source) or "
            "dataset.jsonl_path (legacy single-source)."
        )
    for block in ("transformer_video", "transformer_audio", "vae", "audio_vae", "text_encoder", "scheduler"):
        if block not in config or not isinstance(config[block], dict):
            raise ValueError(f"T2AV config missing '{block}' block.")
    train_cfg = config.get("train", {})
    if float(train_cfg.get("backbone_lr", 0.0)) <= 0:
        raise ValueError("train.backbone_lr must be a positive float (default 5e-6).")
    if float(train_cfg.get("bridge_lr", 0.0)) <= 0:
        raise ValueError("train.bridge_lr must be a positive float (default 2e-5).")
    if float(train_cfg.get("shift_v", 0.0)) <= 0:
        raise ValueError("train.shift_v must be a positive float (default 5.0).")
    if float(train_cfg.get("shift_a", 0.0)) <= 0:
        raise ValueError("train.shift_a must be a positive float (default 1.0).")
    bridge_dropout_prob = float(train_cfg.get("bridge_dropout_prob", 0.0))
    if bridge_dropout_prob < 0.0 or bridge_dropout_prob >= 1.0:
        raise ValueError(
            "train.bridge_dropout_prob must be in [0.0, 1.0); got "
            f"{bridge_dropout_prob}."
        )


# --------------------------------------------------------------- model bld
def _build_branch_transformer(branch_config: dict, *, cap_feat_dim: int):
    branch_cfg = dict(branch_config)
    branch_cfg.setdefault("arch", "zimage")
    return build_transformer(branch_cfg, cap_feat_dim=cap_feat_dim)


def _expected_transformer_init_config(branch_config: dict, cap_feat_dim: int) -> dict:
    from omnivae_generation.trainer.modeling import build_transformer_init_expected_config

    return build_transformer_init_expected_config(branch_config, cap_feat_dim=cap_feat_dim)


def build_joint_transformer(
    config: dict,
    cap_feat_dim: int,
    *,
    info_fn=None,
) -> tuple[BridgedZImageJointModel, dict[str, str]]:
    """Build both branches, compose them under :class:`BridgedZImageJointModel`,
    and warm-start from the configured pretrained checkpoints. Returns the
    joint module plus the resolved warm-start paths (for logging).

    ``info_fn`` is an optional ``str -> None`` callable used to surface
    progress to stdout (the slowest step here is loading two ~6 GB
    safetensors blobs back to back; without progress prints it looks
    like the process has hung).
    """
    if info_fn is None:
        info_fn = logger.info
    video_cfg = config["transformer_video"]
    audio_cfg = config["transformer_audio"]

    info_fn("  building video branch ZImageTransformer2DModel ...")
    video_transformer = _build_branch_transformer(video_cfg, cap_feat_dim=cap_feat_dim)
    info_fn("  building audio branch ZImageTransformer2DModel ...")
    audio_transformer = _build_branch_transformer(audio_cfg, cap_feat_dim=cap_feat_dim)

    bridge_interval = int(config.get("transformer", {}).get("bridge_interval", 2))
    use_asymmetric_ati = bool(config.get("transformer", {}).get("use_asymmetric_ati", False))
    a2v_window_size = int(config.get("transformer", {}).get("a2v_window_size", 1))
    bridge_enabled = bool(config.get("transformer", {}).get("bridge_enabled", True))
    qk_norm = bool(video_cfg.get("qk_norm", True) and audio_cfg.get("qk_norm", True))
    norm_eps = float(min(video_cfg.get("norm_eps", 1e-5), audio_cfg.get("norm_eps", 1e-5)))

    info_fn(
        f"  composing bridges (interval={bridge_interval}, "
        f"asymmetric={use_asymmetric_ati}, enabled={bridge_enabled}) ..."
    )
    joint = BridgedZImageJointModel(
        video_transformer=video_transformer,
        audio_transformer=audio_transformer,
        bridge_interval=bridge_interval,
        bridge_enabled=bridge_enabled,
        use_asymmetric_ati=use_asymmetric_ati,
        a2v_window_size=a2v_window_size,
        qk_norm=qk_norm,
        norm_eps=norm_eps,
    )

    expected_video = _expected_transformer_init_config(video_cfg, cap_feat_dim)
    expected_audio = _expected_transformer_init_config(audio_cfg, cap_feat_dim)
    info_fn(
        f"  warm-start: t2v={video_cfg.get('init_from_transformer') or '<random>'}"
    )
    info_fn(
        f"  warm-start: t2a={audio_cfg.get('init_from_transformer') or '<random>'}"
    )
    loaded = load_pretrained_branches(
        joint,
        pretrained_t2v=video_cfg.get("init_from_transformer"),
        pretrained_t2a=audio_cfg.get("init_from_transformer"),
        expected_video_config=expected_video,
        expected_audio_config=expected_audio,
    )
    info_fn("  warm-start complete; verifying bridge zero-init ... OK")

    if config["train"].get("gradient_checkpointing", True):
        joint.enable_gradient_checkpointing()
    attention_backend = config.get("transformer", {}).get("attention_backend")
    if attention_backend:
        joint.set_attention_backend(attention_backend)
    return joint, loaded


# ------------------------------------------------------------------ data
def build_paired_dataloader(config: dict, tokenizer, accelerator):
    """Build the paired AV dataset + dataloader.

    Supports two yaml schemas under ``dataset:``:

    * ``sources: [{path, path_field, prompt_field, weight, name}, ...]``
      -- multi-source weighted mixing. Engages
      :class:`WeightedShuffledCycleStatefulSampler` (per-rank
      deterministic, resume-exact) and requires
      ``accelerate.use_stateful_dataloader=true``.
    * Legacy single-source: top-level ``jsonl_path`` + optional
      ``path_field`` / ``prompt_field``. Behaves byte-for-byte like the
      pre-multi-source code path; no sampler required.
    """
    dataset_cfg = config["dataset"]
    sources_cfg = dataset_cfg.get("sources")
    legacy_jsonl_path = dataset_cfg.get("jsonl_path")
    if sources_cfg and legacy_jsonl_path:
        raise ValueError(
            "dataset.sources and dataset.jsonl_path are mutually exclusive; "
            "use one or the other (sources is the multi-source schema, "
            "jsonl_path is the legacy single-source schema)."
        )

    dataset = AVPairedJsonlDataset(
        sources=sources_cfg,
        jsonl_path=legacy_jsonl_path,
        path_field=dataset_cfg.get("path_field", "video_path"),
        prompt_field=dataset_cfg.get("prompt_field", "av_caption"),
        frame_size=tuple(dataset_cfg.get("frame_size", [256, 256])),
        num_frames=int(dataset_cfg.get("num_frames", 49)),
        target_fps=float(dataset_cfg.get("target_fps", 24.0)),
        center_crop=bool(dataset_cfg.get("center_crop", True)),
        random_flip=bool(dataset_cfg.get("random_flip", False)),
        return_uint8=bool(dataset_cfg.get("return_uint8", True)),
        sample_rate=int(dataset_cfg.get("sample_rate", 48000)),
        num_audio_samples=int(dataset_cfg.get("num_audio_samples", 1440000)),
        mono=bool(dataset_cfg.get("mono", True)),
        append_duration_suffix=bool(dataset_cfg.get("append_duration_suffix", True)),
        duration_precision=int(dataset_cfg.get("duration_precision", 1)),
        task_prefix_enabled=bool(dataset_cfg.get("task_prefix_enabled", True)),
        tokenizer=tokenizer,
        prompt_max_sequence_length=int(config["text_encoder"].get("max_sequence_length", 512)),
        max_samples=dataset_cfg.get("max_samples"),
        min_clip_duration_ratio=float(
            dataset_cfg.get("min_clip_duration_ratio", 0.99)
        ),
    )

    # Multi-source / non-uniform weights -> route through the shared
    # build_train_dataloader so we get the WeightedShuffledCycleStateful
    # sampler + StatefulDataLoader (exact resume) for free. Single-
    # source legacy yamls (one source, weight=1.0) keep the simpler
    # plain-DataLoader path so existing checkpoints / behaviour stay
    # bit-for-bit compatible.
    from omnivae_generation.trainer.stateful_dataloader import (
        _is_weighted_av_paired_dataset,
        build_train_dataloader,
    )

    if _is_weighted_av_paired_dataset(dataset):
        if not bool(config["accelerate"].get("use_stateful_dataloader", False)):
            raise ValueError(
                "Multi-source / weighted AVPairedJsonlDataset requires "
                "`accelerate.use_stateful_dataloader: true` (the weighted "
                "sampler is stateful and only the StatefulDataLoader can "
                "snapshot it across resume)."
            )
        dataloader = build_train_dataloader(accelerator, dataset, config)
        return dataset, dataloader

    # Single-source legacy path: keep the original plain DataLoader
    # behaviour so existing runs continue without any wiring changes.
    train_cfg = config["train"]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(train_cfg["per_device_batch_size"]),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=bool(dataset_cfg.get("pin_memory", True)),
        prefetch_factor=int(dataset_cfg.get("prefetch_factor", 2))
            if int(dataset_cfg.get("num_workers", 4)) > 0 else None,
        drop_last=bool(dataset_cfg.get("drop_last", True)),
        persistent_workers=bool(int(dataset_cfg.get("num_workers", 4)) > 0),
        collate_fn=collate_av_paired_samples,
    )
    # No DistributedSampler wrapping here - accelerator.prepare will handle it.
    return dataset, dataloader


# ----------------------------------------------------------- compile sites
def _compile_module_forward(
    module: torch.nn.Module,
    *,
    mode: str,
    fullgraph: bool,
    dynamic: bool | None,
) -> None:
    """Replace ``module.forward`` (a bound method) with its compiled
    counterpart.

    We deliberately compile the **bound** method rather than wrap the
    whole module. Wrapping the module with ``torch.compile(module)``
    breaks downstream code that touches attributes on the wrapper
    (``module.layers[i]``, ``module.video.layers``, etc.) and also makes
    DDP / accelerate's parameter discovery flake out. Replacing the
    bound forward keeps the original module identity intact while still
    routing every ``module(...)`` call through Inductor.
    """
    original_forward = module.forward
    compiled = torch.compile(
        original_forward,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    module.forward = compiled  # type: ignore[method-assign]


def _maybe_compile_joint_model(
    joint_model: BridgedZImageJointModel,
    *,
    config: dict,
    info_fn,
) -> None:
    """Compile each block.forward in both branches and every bridge.

    We do **not** compile the whole joint forward because
    ``_run_main_blocks_with_bridges`` contains data-dependent control
    flow (modality skip when one branch is None, scatter-add for the
    bridge delta) that would force a graph break / recompile every
    step. Compiling per-block keeps every Inductor graph fully static
    and amortises the compile cost across the 18 main blocks * 2
    branches + bridge stack.
    """
    if config.get("_no_compile"):
        info_fn("torch.compile: skipped (--no_compile)")
        return

    transformer_cfg = config.get("transformer", {}) or {}
    enable_video = bool(config.get("transformer_video", {}).get("compile_model", True))
    enable_audio = bool(config.get("transformer_audio", {}).get("compile_model", True))
    enable_bridge = bool(transformer_cfg.get("compile_bridges", True))
    # NOTE: default is "default" (Inductor codegen, no CUDA graphs), NOT
    # "reduce-overhead". The latter records CUDA graphs per-block, which
    # is incompatible with our setup: gradient checkpointing replays the
    # forward inside backward, and the second invocation of any block's
    # graph overwrites tensors held by other blocks' graphs (since they
    # all share Inductor's memory pool). The crash looks like:
    #   "Error: accessing tensor output of CUDAGraphs that has been
    #    overwritten by a subsequent run."
    # We trade ~5-10% step-time for stability. Set
    # ``transformer.compile_mode: reduce-overhead`` only if you also set
    # ``train.gradient_checkpointing: false`` (and have the VRAM for it).
    compile_mode = str(transformer_cfg.get("compile_mode", "default"))
    # fullgraph=False is the safe default for the joint trunk: the patched
    # ZImage block forward contains a few size-dependent branches (e.g.
    # mask shortcuts when caption padding fully covers the row) that may
    # graph-break under fullgraph=True. Override via
    # ``transformer.compile_fullgraph: true`` once you're confident.
    compile_fullgraph = bool(transformer_cfg.get("compile_fullgraph", False))
    compile_dynamic = transformer_cfg.get("compile_dynamic", None)
    if compile_dynamic is not None:
        compile_dynamic = bool(compile_dynamic)

    # Defensive guard: reduce-overhead + gradient checkpointing has a
    # known incompatibility (CUDA graph buffers are reused across compile
    # sites and the recomputed backward forward overwrites them). Refuse
    # the combination up-front rather than crashing 30 minutes into the
    # first step.
    grad_ckpt = bool(config.get("train", {}).get("gradient_checkpointing", False))
    if compile_mode == "reduce-overhead" and grad_ckpt:
        raise ValueError(
            "transformer.compile_mode='reduce-overhead' is incompatible with "
            "train.gradient_checkpointing=true: the per-block CUDA graphs "
            "share Inductor's memory pool, and the checkpointed backward "
            "replays the forward, which overwrites tensors held by other "
            "blocks' graphs (RuntimeError: 'accessing tensor output of "
            "CUDAGraphs that has been overwritten'). Either:\n"
            "  - set transformer.compile_mode: default (the safe choice), or\n"
            "  - set train.gradient_checkpointing: false (needs more VRAM)."
        )

    def _compile_branch(branch, label: str) -> None:
        n_main = sum(1 for _ in branch.layers)
        n_ctx = sum(1 for _ in branch.context_refiner)
        n_noise = sum(1 for _ in branch.noise_refiner)
        for layer in branch.layers:
            _compile_module_forward(
                layer, mode=compile_mode, fullgraph=compile_fullgraph, dynamic=compile_dynamic,
            )
        for layer in branch.context_refiner:
            _compile_module_forward(
                layer, mode=compile_mode, fullgraph=compile_fullgraph, dynamic=compile_dynamic,
            )
        for layer in branch.noise_refiner:
            _compile_module_forward(
                layer, mode=compile_mode, fullgraph=compile_fullgraph, dynamic=compile_dynamic,
            )
        info_fn(
            f"torch.compile: {label} branch ({n_main} main + {n_ctx} ctx + "
            f"{n_noise} noise refiners) -> {compile_mode} "
            f"fullgraph={compile_fullgraph} dynamic={compile_dynamic}"
        )

    if enable_video:
        _compile_branch(joint_model.video, "video")
    if enable_audio:
        _compile_branch(joint_model.audio, "audio")
    if enable_bridge:
        for bridge in joint_model.bridges:
            _compile_module_forward(
                bridge, mode=compile_mode, fullgraph=compile_fullgraph, dynamic=compile_dynamic,
            )
        info_fn(
            f"torch.compile: {len(joint_model.bridges)} bridges -> {compile_mode} "
            f"fullgraph={compile_fullgraph} dynamic={compile_dynamic}"
        )


def _maybe_compile_text_encoder(text_encoder, *, config: dict, info_fn):
    """Compile the Qwen3.5 text encoder forward.

    Mirrors the single-modality entry script: applies the
    Qwen3.5-friendly attn-mask patch + capturing-hooks shim before
    handing off to ``torch.compile`` so dynamo can trace the whole
    encode path without graph breaks.
    """
    if config.get("_no_compile"):
        return text_encoder
    if text_encoder is None or not bool(config["text_encoder"].get("compile_model", True)):
        return text_encoder
    try:
        from omnivae_generation.trainer.runtime_patches import (
            patch_transformers_qwen3_5_compile_friendly_linear_attn_mask,
        )
        patch_transformers_qwen3_5_compile_friendly_linear_attn_mask()
    except Exception as exc:
        info_fn(f"torch.compile: text encoder attn-mask patch unavailable ({exc!r})")
    try:
        from transformers.utils.output_capturing import maybe_install_capturing_hooks
        maybe_install_capturing_hooks(text_encoder)
    except Exception as exc:
        info_fn(f"torch.compile: text encoder capturing hooks unavailable ({exc!r})")
    info_fn("torch.compile: text encoder -> reduce-overhead, fullgraph=True")
    return torch.compile(text_encoder, fullgraph=True, mode="reduce-overhead")


def _maybe_compile_vae_encode(vae, vae_cfg: dict, *, config: dict, info_fn, label: str):
    """Compile ``vae.encode`` in-place. No-op when disabled."""
    if config.get("_no_compile"):
        return
    if vae is None or not bool(vae_cfg.get("compile_encode", True)):
        return
    fullgraph = vae_cfg.get("compile_encode_fullgraph")
    if fullgraph is None:
        # Wan2.2 in "parallel" chunk mode wants fullgraph=False; "cache"
        # mode is fully traceable.
        fullgraph = getattr(vae, "wan_chunk_mode", "cache") != "parallel"
    mode = str(vae_cfg.get("compile_encode_mode", "reduce-overhead"))
    vae.encode = torch.compile(vae.encode, fullgraph=bool(fullgraph), mode=mode)
    info_fn(
        f"torch.compile: {label} VAE encode -> {mode} "
        f"fullgraph={bool(fullgraph)}"
    )


# ----------------------------------------------------------- optimizer bld
def build_t2av_optimizer(config: dict, joint_model: BridgedZImageJointModel):
    train_cfg = config["train"]
    backbone_lr = float(train_cfg["backbone_lr"])
    bridge_lr = float(train_cfg["bridge_lr"])
    tagged = joint_model.named_branch_parameters()
    optimizer = HybridMuonAdamwTagged(
        tagged_named_params=tagged,
        train_cfg=train_cfg,
        learning_rate=torch.tensor(backbone_lr),
        per_tag_lr={"backbone": torch.tensor(backbone_lr), "bridge": torch.tensor(bridge_lr)},
    )
    logger.info("Built T2AV optimizer: %s", optimizer.describe())
    return optimizer


# -------------------------------------------------------------- ckpt save
def save_managed_t2av_checkpoint(
    *,
    accelerator,
    checkpoint_root: Path,
    checkpoint_kind: str,
    checkpoints_limit,
    train_dataloader,
    config: dict,
    global_step: int,
    transformer_param_count: int,
    joint_model,
    tokenizer,
    scheduler,
) -> None:
    checkpoint_dir = checkpoint_root / f"checkpoint-{global_step:08d}"
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(joint_model, keep_torch_compile=False)
        save_split_branches(
            output_dir=checkpoint_dir,
            joint_model=unwrapped,
            tokenizer=tokenizer,
            scheduler=scheduler,
            metadata={
                "global_step": global_step,
                "transformer_parameters": transformer_param_count,
                "checkpoint_kind": checkpoint_kind,
                "joint_av": True,
                "bridge_interval": int(unwrapped.bridge_interval),
                "use_asymmetric_ati": bool(unwrapped.use_asymmetric_ati),
                "a2v_window_size": int(unwrapped.a2v_window_size),
            },
        )
        mark_checkpoint_complete(checkpoint_dir)
        rotate_checkpoints(checkpoint_root, checkpoints_limit)
    accelerator.wait_for_everyone()


# --------------------------------------------------------------- training
def main(args=None, config=None) -> None:
    if args is None:
        args = parse_train_args()
    if config is None:
        config = load_train_config(args)

    runtime = zt.create_training_runtime(config)
    accelerator = runtime.accelerator
    paths = runtime.paths

    def _info(msg: str) -> None:
        # accelerate.logging.get_logger defaults to WARNING level so
        # `logger.info` is silent. We mirror every important step to
        # stdout from the main process so the user can see progress
        # during the (slow) checkpoint-load phase.
        logger.info(msg)
        if accelerator.is_main_process:
            print(f"[t2av] {msg}", flush=True)

    _info("Loading tokenizer + text encoder.")
    text_encoder_dtype = resolve_dtype(config["text_encoder"].get("torch_dtype"), fallback=torch.bfloat16)
    tokenizer, text_encoder, text_hidden_size = load_text_components(config["text_encoder"], text_encoder_dtype)

    _info("Loading video VAE.")
    video_vae = load_vae(config["vae"])
    _info("Loading audio VAE.")
    audio_vae = load_audio_vae(config["audio_vae"])
    _info("Loading scheduler.")
    noise_scheduler = load_scheduler(config["scheduler"])

    _info("Building joint transformer (video + audio + bridges) ...")
    joint_model, warmstart_paths = build_joint_transformer(
        config, cap_feat_dim=text_hidden_size, info_fn=_info,
    )
    joint_model._laion_caption_target_length = int(config["text_encoder"]["max_sequence_length"])
    transformer_param_count = count_parameters(joint_model)
    _info(
        f"Joint transformer ready: {transformer_param_count/1e9:.3f}B params "
        f"(bridges={len(joint_model.bridges)})."
    )

    # Freeze VAEs + text encoder.
    video_vae.requires_grad_(False); video_vae.eval()
    audio_vae.requires_grad_(False); audio_vae.eval()
    text_encoder.requires_grad_(False); text_encoder.eval()

    _info("Building paired AV dataloader.")
    dataset, train_dataloader = build_paired_dataloader(config, tokenizer, accelerator)
    _info(f"Dataset size = {len(dataset)} samples.")

    _info("Building optimizer + lr scheduler.")
    optimizer = build_t2av_optimizer(config, joint_model)
    max_train_steps = int(config["train"]["max_train_steps"])
    lr_scheduler = get_scheduler(
        config["train"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=int(config["train"]["lr_warmup_steps"]) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    _info("accelerator.prepare(...) ...")
    # The multi-source / weighted dataloader (StatefulDataLoader +
    # WeightedShuffledCycleStatefulSampler) already partitions indices
    # per-rank via (seed, rank). Letting accelerate wrap it again with
    # BatchSamplerShard would double-shard (each rank gets ~1/world_size**2
    # of the data) and silently break the stateful resume snapshot. So we
    # only feed the dataloader to ``prepare`` for the legacy single-source
    # path, which uses a plain DataLoader (shuffle=False, no
    # DistributedSampler) and relies on accelerate to insert one.
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader as _SDL
        _is_stateful_loader = isinstance(train_dataloader, _SDL)
    except ImportError:
        _is_stateful_loader = False
    if _is_stateful_loader:
        joint_model, optimizer, lr_scheduler = accelerator.prepare(
            joint_model, optimizer, lr_scheduler,
        )
    else:
        joint_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            joint_model, optimizer, lr_scheduler, train_dataloader,
        )
    _info("accelerator.prepare(...) done.")

    video_vae_dtype = resolve_dtype(config["vae"].get("torch_dtype"), fallback=torch.bfloat16)
    audio_vae_dtype = resolve_dtype(config["audio_vae"].get("torch_dtype"), fallback=torch.float32)
    video_vae.to(accelerator.device, dtype=video_vae_dtype)
    audio_vae.to(accelerator.device, dtype=audio_vae_dtype)
    text_encoder.to(accelerator.device, dtype=text_encoder_dtype)

    transformer_model = accelerator.unwrap_model(joint_model)
    transformer_model.materialize_rope_cache(accelerator.device)
    train_patch_size = int(config["transformer_video"]["all_patch_size"][0])
    train_f_patch_size = int(config["transformer_video"]["all_f_patch_size"][0])
    audio_patch_size = int(config["transformer_audio"]["all_patch_size"][0])
    audio_f_patch_size = int(config["transformer_audio"]["all_f_patch_size"][0])

    # ---- torch.compile sites (no-op when --no_compile is set) ----
    # Compile per-block forwards on both branches and every bridge so the
    # joint forward's data-dependent control flow stays in eager while the
    # heavy transformer blocks themselves run as fused inductor graphs.
    _maybe_compile_joint_model(transformer_model, config=config, info_fn=_info)
    text_encoder = _maybe_compile_text_encoder(text_encoder, config=config, info_fn=_info)
    _maybe_compile_vae_encode(
        video_vae, vae_cfg=config["vae"], config=config, info_fn=_info, label="video",
    )
    _maybe_compile_vae_encode(
        audio_vae, vae_cfg=config["audio_vae"], config=config, info_fn=_info, label="audio",
    )

    noise_scheduler.config.use_dynamic_shifting = False
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device=accelerator.device)

    if accelerator.is_main_process:
        save_json(
            paths.output_dir / "run_metadata.json",
            {
                "transformer_parameters": int(transformer_param_count),
                "transformer_parameters_billion": round(transformer_param_count / 1e9, 4),
                "warmstart": warmstart_paths,
                "config_path": config["_config_path"],
                "bridge_interval": int(transformer_model.bridge_interval),
                "n_bridges": int(len(transformer_model.bridges)),
                "shift_v": float(config["train"]["shift_v"]),
                "shift_a": float(config["train"]["shift_a"]),
            },
        )

    # Resume from checkpoint (if requested). We use a plain DataLoader
    # rather than the stateful wrapper, so resume is "best-effort": we
    # restore the accelerator state (transformer + bridge weights +
    # optimizer + lr scheduler + RNG) and parse ``global_step`` /
    # ``first_epoch`` from the checkpoint directory name. The exact
    # within-epoch sample order is not preserved, which mirrors what the
    # other trainers do when ``use_stateful_dataloader=false``.
    global_step = 0
    first_epoch = 0
    resume_target = config["train"].get("resume_from_checkpoint")
    if resume_target:
        from omnivae_generation.trainer.stateful_dataloader import resolve_latest_resume_checkpoint

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / int(config["train"]["gradient_accumulation_steps"])
        )
        resume_root = paths.snapshot_root
        if resume_target == "latest_persistent":
            resume_root = paths.persistent_root
            resume_target = "latest"
        if resume_target == "latest":
            resume_checkpoint = resolve_latest_resume_checkpoint(resume_root)
        else:
            resume_checkpoint = Path(resume_target)
        if resume_checkpoint is not None and resume_checkpoint.is_dir():
            accelerator.load_state(str(resume_checkpoint))
            try:
                global_step = int(resume_checkpoint.name.split("-")[-1])
            except ValueError:
                global_step = 0
            first_epoch = global_step // max(1, num_update_steps_per_epoch)
            if accelerator.is_main_process:
                print(
                    f"[t2av] Resumed from {resume_checkpoint} at global_step={global_step}",
                    flush=True,
                )
        elif accelerator.is_main_process:
            print(
                f"[t2av] resume_from_checkpoint={resume_target!r} requested but no usable "
                f"checkpoint found under {resume_root}; starting from global_step=0.",
                flush=True,
            )

    profiler = build_profiler(accelerator, config, paths.output_dir)
    if profiler.enabled:
        logger.info("Torch profiler chrome traces -> %s", profiler.trace_dir)
    profiler.start()

    progress_bar = tqdm(
        initial=global_step,
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="train_t2av",
    )

    accumulated = {
        "loss": 0.0, "loss_v": 0.0, "loss_a": 0.0,
        "sigma_v": 0.0, "sigma_a": 0.0, "model_t_v": 0.0, "model_t_a": 0.0,
        "bridge_keep_ratio": 0.0,
    }
    accumulated_steps = 0
    log_every = int(config["train"].get("log_every_steps", 10))
    grad_accum = int(config["train"]["gradient_accumulation_steps"])
    loss_weight_v = float(config["train"].get("loss_weight_v", 1.0))
    loss_weight_a = float(config["train"].get("loss_weight_a", 1.0))
    shift_v = float(config["train"]["shift_v"])
    shift_a = float(config["train"]["shift_a"])
    max_grad_norm = float(config["train"].get("max_grad_norm", 1.0))
    bridge_dropout_prob = float(config["train"].get("bridge_dropout_prob", 0.0))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # If transformer compile_mode is "reduce-overhead" we have to manually
    # mark the start of every training iteration so cudagraph_trees treats
    # all 53 per-block compile sites within the same step as belonging to
    # the same "iteration". Without this signal, every site sees the
    # previous site's backward as "pending" and falls back to the slow
    # re-record path (~10x step-time blowup; see
    # transformer.compile_mode comments in t2av.yaml).
    transformer_compile_mode = str(
        config.get("transformer", {}).get("compile_mode", "default")
    )
    needs_cudagraph_mark_step = (
        not bool(config.get("_no_compile"))
        and transformer_compile_mode == "reduce-overhead"
    )
    if needs_cudagraph_mark_step:
        try:
            from torch.compiler import cudagraph_mark_step_begin as _cudagraph_mark_step_begin
        except ImportError:
            try:
                from torch._inductor.cudagraph_trees import (
                    mark_step_begin as _cudagraph_mark_step_begin,  # type: ignore[no-redef]
                )
            except ImportError:
                _cudagraph_mark_step_begin = None  # type: ignore[assignment]
        if _cudagraph_mark_step_begin is None:
            _info(
                "WARNING: transformer.compile_mode='reduce-overhead' but "
                "torch.compiler.cudagraph_mark_step_begin is unavailable. "
                "You'll see 'Unable to hit fast path of CUDAGraphs' "
                "warnings and ~10x step-time blowup. Switch to "
                "compile_mode=default or upgrade torch."
            )
        else:
            _info(
                "compile_mode=reduce-overhead -> calling "
                "torch.compiler.cudagraph_mark_step_begin() at the start "
                "of every training step to keep the cudagraph fast path."
            )
    else:
        _cudagraph_mark_step_begin = None  # type: ignore[assignment]

    # ----- Pre-train-loop validation hook ---------------------------- #
    # If `global_step` (== 0 for fresh runs, or the resume step otherwise)
    # is in `validation_force_steps`, run a one-shot validation BEFORE the
    # training loop ever calls backward. The most common use case is
    # `--validate_at 0`: a smoke test on the warm-start weights / pipeline
    # before any optimizer step has perturbed them.
    pretrain_force_steps_cfg = config["train"].get("validation_force_steps") or []
    if isinstance(pretrain_force_steps_cfg, int):
        pretrain_force_steps_cfg = [pretrain_force_steps_cfg]
    pretrain_force_steps_set = {
        int(s) for s in pretrain_force_steps_cfg if int(s) >= 0
    }
    if global_step in pretrain_force_steps_set:
        if accelerator.is_local_main_process:
            logger.info(
                "[val] pre-train-loop forced validation at global_step=%d "
                "(validation_force_steps=%s).",
                global_step, sorted(pretrain_force_steps_set),
            )
        accelerator.wait_for_everyone()
        run_joint_av_validation(
            accelerator=accelerator,
            config=config,
            step=global_step,
            joint_model=joint_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            scheduler=noise_scheduler,
        )
        accelerator.wait_for_everyone()

    try:
        for epoch in range(first_epoch, num_train_epochs):
            joint_model.train()
            if hasattr(getattr(train_dataloader, "sampler", None), "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            for batch in train_dataloader:
                # Per-step cudagraph step marker (only active when
                # compile_mode=reduce-overhead). Must happen *outside*
                # accelerator.accumulate so it always runs once per
                # micro-batch, regardless of gradient_accumulation_steps.
                if _cudagraph_mark_step_begin is not None:
                    _cudagraph_mark_step_begin()
                with accelerator.accumulate(joint_model):
                    pixel_values = zt.prepare_pixel_values_for_vae(
                        batch["pixel_values"], accelerator.device, video_vae_dtype,
                    )
                    audio_input = batch["audio"].to(
                        accelerator.device, dtype=audio_vae_dtype, non_blocking=True,
                    )
                    with torch.no_grad():
                        video_latents = encode_images_to_latents(pixel_values, video_vae).to(torch.float32)
                        audio_latents = encode_audio_to_latents(audio_input, audio_vae).to(torch.float32)

                    dual = prepare_dual_diffusion_batch(
                        config=config,
                        video_latents=video_latents,
                        audio_latents=audio_latents,
                        shift_v=shift_v,
                        shift_a=shift_a,
                        detach_target_latents=False,
                    )

                    prompts = list(batch["prompts"])
                    empty_prompts = list(batch.get("empty_prompts", [""] * len(prompts)))
                    dropout_prob = float(config["train"].get("caption_dropout_prob", 0.0))
                    if dropout_prob > 0.0:
                        mask = (torch.rand(len(prompts)) < dropout_prob).tolist()
                        prompts = [empty_prompts[i] if mask[i] else prompts[i] for i in range(len(prompts))]
                    from omnivae_generation.trainer.modeling import encode_prompts

                    with torch.no_grad():
                        prompt_embeds = encode_prompts(
                            prompts=prompts,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            device=accelerator.device,
                            max_sequence_length=int(config["text_encoder"]["max_sequence_length"]),
                            cache_enabled=bool(config["text_encoder"].get("cache_enabled", False)),
                        )

                    # Per-sample bridge dropout: independently from caption
                    # dropout, with probability ``bridge_dropout_prob`` we
                    # zero out the bridge cross-modal contribution for that
                    # sample. Aligns the training distribution with the
                    # NFE=3 dual CFG inference path (which evaluates a
                    # ``bridge=off`` slot).
                    bsz_train = int(dual.video.noisy_latents.shape[0])
                    if bridge_dropout_prob > 0.0:
                        bridge_mask = (
                            torch.rand(bsz_train, device=accelerator.device)
                            >= bridge_dropout_prob
                        )
                    else:
                        bridge_mask = None

                    video_pred_list, audio_pred_list = joint_model(
                        video_x=dual.video.noisy_latents.to(
                            dtype=getattr(transformer_model.video, "dtype", dual.video.noisy_latents.dtype)
                        ),
                        video_t=dual.video.model_timesteps,
                        audio_x=dual.audio.noisy_latents.to(
                            dtype=getattr(transformer_model.audio, "dtype", dual.audio.noisy_latents.dtype)
                        ),
                        audio_t=dual.audio.model_timesteps,
                        prompt_embeds_video=prompt_embeds,
                        prompt_embeds_audio=prompt_embeds,
                        video_patch_size=train_patch_size,
                        video_f_patch_size=train_f_patch_size,
                        audio_patch_size=audio_patch_size,
                        audio_f_patch_size=audio_f_patch_size,
                        bridge_mask=bridge_mask,
                    )
                    video_pred = transformer_model.stack_branch_predictions(video_pred_list)
                    audio_pred_5d = transformer_model.stack_branch_predictions(audio_pred_list)
                    if audio_pred_5d.dim() == 5:
                        audio_pred = audio_pred_5d.squeeze(-1).squeeze(-1)
                    else:
                        audio_pred = audio_pred_5d

                    predict_target = getattr(transformer_model.video, "_laion_predict_target", "v")
                    video_pred_for_loss = adapt_model_prediction(
                        video_pred, dual.video.noisy_latents, dual.video.sigmas, predict_target,
                    )
                    audio_pred_for_loss = adapt_model_prediction(
                        audio_pred, dual.audio.noisy_latents, dual.audio.sigmas, predict_target,
                    )

                    per_sample_v = zt.compute_per_sample_denoising_loss(
                        dual.video.weighting, video_pred_for_loss, dual.video.target,
                    )
                    per_sample_a = zt.compute_per_sample_denoising_loss(
                        dual.audio.weighting, audio_pred_for_loss, dual.audio.target,
                    )
                    loss_v = per_sample_v.mean()
                    loss_a = per_sample_a.mean()
                    loss = loss_weight_v * loss_v + loss_weight_a * loss_a

                    averaged_loss = float(accelerator.gather(loss.detach()).mean().item())
                    averaged_loss_v = float(accelerator.gather(loss_v.detach()).mean().item())
                    averaged_loss_a = float(accelerator.gather(loss_a.detach()).mean().item())
                    averaged_sigma_v = float(dual.video.sigmas.detach().mean().item())
                    averaged_sigma_a = float(dual.audio.sigmas.detach().mean().item())
                    averaged_t_v = float(dual.video.model_timesteps.detach().mean().item())
                    averaged_t_a = float(dual.audio.model_timesteps.detach().mean().item())
                    if bridge_mask is not None:
                        averaged_bridge_keep = float(
                            accelerator.gather(bridge_mask.float().mean()).mean().item()
                        )
                    else:
                        averaged_bridge_keep = 1.0

                    accumulated["loss"] += averaged_loss / grad_accum
                    accumulated["loss_v"] += averaged_loss_v / grad_accum
                    accumulated["loss_a"] += averaged_loss_a / grad_accum
                    accumulated["sigma_v"] += averaged_sigma_v / grad_accum
                    accumulated["sigma_a"] += averaged_sigma_a / grad_accum
                    accumulated["model_t_v"] += averaged_t_v / grad_accum
                    accumulated["model_t_a"] += averaged_t_a / grad_accum
                    accumulated["bridge_keep_ratio"] += averaged_bridge_keep / grad_accum

                    accelerator.backward(loss)

                    grad_norm = None
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(joint_model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accumulated_steps += 1

                    if global_step % log_every == 0 and accumulated_steps > 0:
                        # accelerator.prepare wraps the optimizer in
                        # AcceleratedOptimizer, which on some accelerate
                        # versions doesn't forward custom methods via
                        # __getattr__. Unwrap one level so per_tag_lrs() and
                        # the param_groups (with the "tag" key) are reachable.
                        underlying_opt = getattr(optimizer, "optimizer", optimizer)
                        if hasattr(underlying_opt, "per_tag_lrs"):
                            per_tag_lr = underlying_opt.per_tag_lrs()
                        else:
                            # Last-resort fallback: bucket param_groups by
                            # their "tag" key so we still get per-tag curves
                            # even without per_tag_lrs().
                            per_tag_lr = {}
                            for group in underlying_opt.param_groups:
                                tag = str(group.get("tag", "untagged"))
                                lr_val = group.get("lr")
                                if isinstance(lr_val, torch.Tensor):
                                    lr_val = float(lr_val.detach().cpu().item())
                                else:
                                    lr_val = float(lr_val)
                                per_tag_lr.setdefault(tag, []).append(lr_val)
                            per_tag_lr = {
                                tag: sum(vs) / max(1, len(vs)) for tag, vs in per_tag_lr.items()
                            }
                        log_payload: dict[str, Any] = {
                            "train/loss": accumulated["loss"] / accumulated_steps,
                            "train/loss_video": accumulated["loss_v"] / accumulated_steps,
                            "train/loss_audio": accumulated["loss_a"] / accumulated_steps,
                            "train/sigma_video": accumulated["sigma_v"] / accumulated_steps,
                            "train/sigma_audio": accumulated["sigma_a"] / accumulated_steps,
                            "train/model_t_video": accumulated["model_t_v"] / accumulated_steps,
                            "train/model_t_audio": accumulated["model_t_a"] / accumulated_steps,
                            "train/bridge_keep_ratio": accumulated["bridge_keep_ratio"] / accumulated_steps,
                            "train/lr": float(lr_scheduler.get_last_lr()[0]),
                        }
                        for tag, lr in per_tag_lr.items():
                            log_payload[f"train/lr_{tag}"] = float(lr)
                        if per_tag_lr.get("backbone") and per_tag_lr.get("bridge"):
                            log_payload["train/lr_ratio_bridge_over_backbone"] = (
                                float(per_tag_lr["bridge"]) / max(1e-12, float(per_tag_lr["backbone"]))
                            )
                        if global_step <= log_every and accelerator.is_main_process:
                            # One-shot startup print so the user can verify
                            # that the per-tag lrs are flowing into wandb.
                            print(
                                f"[t2av] first lr log step={global_step} "
                                f"lr_keys={[k for k in log_payload if k.startswith('train/lr')]}",
                                flush=True,
                            )
                        if grad_norm is not None:
                            log_payload["train/grad_norm"] = float(grad_norm)
                        accelerator.log(log_payload, step=global_step)
                        accumulated = {k: 0.0 for k in accumulated}
                        accumulated_steps = 0

                    snapshot_every = int(config["train"].get("snapshot_checkpointing_steps") or 0)
                    snapshots_limit = config["train"].get("snapshots_total_limit")
                    if snapshot_every > 0 and global_step % snapshot_every == 0:
                        save_managed_t2av_checkpoint(
                            accelerator=accelerator,
                            checkpoint_root=paths.snapshot_root,
                            checkpoint_kind="snapshot",
                            checkpoints_limit=snapshots_limit,
                            train_dataloader=train_dataloader,
                            config=config,
                            global_step=global_step,
                            transformer_param_count=transformer_param_count,
                            joint_model=joint_model,
                            tokenizer=tokenizer,
                            scheduler=noise_scheduler,
                        )

                    persistent_every = int(config["train"].get("persistent_checkpointing_steps") or 0)
                    persistent_limit = config["train"].get("persistent_total_limit")
                    if persistent_every > 0 and global_step % persistent_every == 0:
                        save_managed_t2av_checkpoint(
                            accelerator=accelerator,
                            checkpoint_root=paths.persistent_root,
                            checkpoint_kind="persistent",
                            checkpoints_limit=persistent_limit,
                            train_dataloader=train_dataloader,
                            config=config,
                            global_step=global_step,
                            transformer_param_count=transformer_param_count,
                            joint_model=joint_model,
                            tokenizer=tokenizer,
                            scheduler=noise_scheduler,
                        )

                    val_every = int(config["train"].get("validation_steps") or 0)
                    force_steps_cfg = config["train"].get("validation_force_steps") or []
                    if isinstance(force_steps_cfg, int):
                        force_steps_cfg = [force_steps_cfg]
                    force_steps_set = {int(s) for s in force_steps_cfg if int(s) > 0}
                    is_periodic = val_every > 0 and global_step % val_every == 0
                    # `global_step >= 1` here because we're inside the
                    # sync_gradients branch; step=0 is handled separately
                    # via the pre-train-loop hook below.
                    is_forced = global_step in force_steps_set
                    if is_periodic or is_forced:
                        if is_forced and not is_periodic and accelerator.is_local_main_process:
                            logger.info(
                                "[val] global_step=%d hit a forced validation step "
                                "(validation_force_steps=%s); running validation now.",
                                global_step, sorted(force_steps_set),
                            )
                        accelerator.wait_for_everyone()
                        run_joint_av_validation(
                            accelerator=accelerator,
                            config=config,
                            step=global_step,
                            joint_model=joint_model,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            video_vae=video_vae,
                            audio_vae=audio_vae,
                            scheduler=noise_scheduler,
                        )
                        accelerator.wait_for_everyone()

                    if runtime.manual_gc_every_steps and global_step % runtime.manual_gc_every_steps == 0:
                        gc.collect()

                    progress_bar.set_postfix(
                        loss=f"{loss.detach().float().item():.4f}",
                        v=f"{loss_v.detach().float().item():.4f}",
                        a=f"{loss_a.detach().float().item():.4f}",
                    )

                profiler.step()

                if global_step >= max_train_steps:
                    break

            if global_step >= max_train_steps:
                break
    finally:
        profiler.stop()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(joint_model, keep_torch_compile=False)
        save_split_branches(
            output_dir=paths.output_dir / "final",
            joint_model=unwrapped,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            metadata={
                "global_step": global_step,
                "transformer_parameters": transformer_param_count,
                "checkpoint_kind": None,
                "joint_av": True,
                "bridge_interval": int(unwrapped.bridge_interval),
            },
        )

    zt.finish_training(runtime.manual_gc_every_steps, accelerator)


if __name__ == "__main__":
    main()
