"""Run trainer-style T2AV validation on historical checkpoints.

This is an inference-only entrypoint. It loads one saved joint-AV
checkpoint at a time, optionally overrides the validation jsonl / type
filter, then calls ``omnivae_generation.trainer.joint_av.validation.run_joint_av_validation``
so the on-disk layout is identical to validation during training:

    <output-dir>/samples/step-XXXXXXXX/<mode>/cfg_<simple|dual>/

When ``--cfg`` sweeps numeric values, the value is added to the CFG directory
name to avoid overwriting samples, e.g. ``cfg_dual_g4``.

All ranks validate the same checkpoint concurrently and split prompts by
``sample_index % world_size`` inside ``run_joint_av_validation``. That keeps
the normal manifest gathering / save path intact.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from t2av_pipeline import load_joint_av_pipeline  # noqa: E402
from omnivae_generation.trainer.joint_av.validation import run_joint_av_validation, validation_outputs_complete  # noqa: E402


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_VALID_MODES = ("joint_av", "video_only", "audio_only")
_VALID_CFG_MODES = ("simple", "dual")


def _split_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            out.extend(_split_values(item))
        return out
    text = str(raw).strip()
    return [text] if text else []


def _nonneg_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _format_cfg_for_dir(value: float) -> str:
    parsed = float(value)
    if parsed.is_integer():
        return str(int(parsed))
    return ("%g" % parsed).replace("-", "m").replace(".", "p")


def _parse_cfg_values(raw: Any) -> list[float]:
    values: list[float] = []
    for token in _split_values(raw):
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"--cfg got non-numeric value {token!r}") from exc
    return values


def _parse_step_selection(raw: Any) -> tuple[Optional[set[int]], bool]:
    steps: set[int] = set()
    include_latest = False
    for token in _split_values(raw):
        if token.lower() == "latest":
            include_latest = True
            continue
        try:
            step = int(token)
        except ValueError as exc:
            raise ValueError(
                f"--steps got {token!r}; expected non-negative integer or 'latest'"
            ) from exc
        if step < 0:
            raise ValueError(f"--steps got negative step {token!r}")
        steps.add(step)
    return (steps or None), include_latest


def _split_vae_spec(spec: str) -> tuple[str, Optional[str]]:
    """Parse ``TYPE:PATH`` while leaving plain paths untouched."""
    text = str(spec).strip()
    match = re.match(r"^([A-Za-z0-9_.+-]+):(.*)$", text)
    if match and match.group(2):
        return match.group(2), match.group(1)
    return text, None


def _parse_vae_overrides(raw_items: Any, *, flag_name: str) -> dict[str, dict[str, Optional[str]]]:
    overrides: dict[str, dict[str, Optional[str]]] = {}
    for raw in raw_items or []:
        text = str(raw).strip()
        if "=" not in text:
            raise ValueError(f"{flag_name} expects EXP=PATH or EXP=TYPE:PATH, got {text!r}")
        exp, spec = text.split("=", 1)
        exp = exp.strip()
        path, vae_type = _split_vae_spec(spec)
        path = path.strip()
        if not exp or not path:
            raise ValueError(f"{flag_name} expects non-empty EXP and PATH, got {text!r}")
        overrides[exp] = {"path": path, "type": vae_type}
    return overrides


def _infer_override_vae_type(path: Optional[str], explicit_type: Optional[str]) -> Optional[str]:
    if explicit_type:
        return str(explicit_type)
    if not path:
        return None
    lowered = str(path).lower()
    if lowered.endswith("state_dict.pt") or "omnivae" in lowered or "univae" in lowered:
        return "omnivae"
    return None


def _resolve_vae_override(
    args: argparse.Namespace,
    target: dict[str, Any],
    *,
    kind: str,
) -> tuple[Optional[str], Optional[str]]:
    if kind == "video":
        base_type = args.video_vae_type
        base_path = args.video_vae_path
        overrides = args.video_vae_overrides
    elif kind == "audio":
        base_type = args.audio_vae_type
        base_path = args.audio_vae_path
        overrides = args.audio_vae_overrides
    else:  # pragma: no cover - defensive programming for future callers.
        raise ValueError(f"unknown VAE override kind {kind!r}")

    override = overrides.get(str(target["experiment"]))
    if not override:
        return base_type, base_path
    path = override.get("path") or base_path
    vae_type = _infer_override_vae_type(path, override.get("type")) or base_type
    return vae_type, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt-root", "--ckpt_root", "--ckpt", dest="ckpt_root", required=True,
        help=(
            "A run dir, a sweep root containing <experiment>/checkpoints/snapshots, "
            "or one direct checkpoint-XXXXXXXX dir."
        ),
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="When --ckpt-root is a sweep root, keep only these experiment directory names.",
    )
    parser.add_argument("--min-step", "--min_step", type=_nonneg_int, default=None)
    parser.add_argument("--max-step", "--max_step", type=_nonneg_int, default=None)
    parser.add_argument(
        "--step-multiple", "--step_multiple", "--step-stride", "--step_stride",
        type=_nonneg_int, default=0,
        help="Keep only checkpoints whose step is a positive multiple of this value. 0 disables.",
    )
    parser.add_argument(
        "--steps", action="append", default=None,
        help=(
            "Explicit step whitelist, comma-separated or repeated. May include 'latest'. "
            "Overrides --step-multiple unless only --latest/--only-latest is used."
        ),
    )
    parser.add_argument(
        "--latest", "--only-latest", dest="only_latest", action="store_true",
        help="Keep only the latest checkpoint per experiment after min/max and experiment filters.",
    )
    parser.add_argument("--max-ckpts", "--max_ckpts", type=_nonneg_int, default=0, help="0 = no cap.")
    parser.add_argument("--order", choices=("asc", "desc"), default="asc")
    parser.add_argument(
        "--target-shard-index", "--target_shard_index",
        type=_nonneg_int, default=None,
        help="Keep only one checkpoint-target shard after discovery. 0-based.",
    )
    parser.add_argument(
        "--target-shard-count", "--target_shard_count",
        type=_nonneg_int, default=0,
        help="Number of checkpoint-target shards. 0 disables sharding.",
    )
    parser.add_argument(
        "--target-shard-weights", "--target_shard_weights",
        default="",
        help=(
            "Comma-separated positive weights for all target shards. "
            "When set, length must match --target-shard-count."
        ),
    )

    parser.add_argument(
        "--output-root", "--output_root", default=None,
        help=(
            "Optional root for validation outputs. Writes to "
            "<output-root>/<experiment>/samples/step-XXXXXXXX. "
            "Default: the checkpoint's original run dir."
        ),
    )
    parser.add_argument(
        "--resume-inference",
        dest="resume_inference",
        action="store_true",
        default=True,
        help=(
            "Skip checkpoint/cfg/sample outputs already present under --output-root. "
            "This is enabled by default for validation sweeps."
        ),
    )
    parser.add_argument(
        "--no-resume-inference",
        dest="resume_inference",
        action="store_false",
        help="Regenerate inference outputs even when existing samples are present.",
    )

    parser.add_argument(
        "--validation-jsonl", "--valid-jsonl", "--validation_jsonl", "--valid_jsonl", default=None,
        help="Override validation.joint_av_prompts.jsonl_path for this run.",
    )
    parser.add_argument("--validation-name", "--validation_name", default=None, help="Source name stored in manifest / file stems.")
    parser.add_argument("--text-field", "--prompt-field", "--text_field", "--prompt_field", default="av_caption")
    parser.add_argument("--type-field", "--type_field", default="type")
    parser.add_argument("--index-field", "--index_field", default="index")
    parser.add_argument(
        "--types", "--type-filter", action="append", default=None,
        help="Exact type whitelist, comma-separated or repeated. Example: --types set3-large,set4-large.",
    )
    parser.add_argument(
        "--type-filter-field", "--type_filter_field", default=None,
        help="Field checked by --types. Defaults to --type-field.",
    )
    parser.add_argument("--max-examples", "--max_examples", type=_nonneg_int, default=None, help="Applied after type filtering. 0 = all.")
    parser.add_argument(
        "--no-task-prefix", "--no_task_prefix", action="store_true",
        help="Do not apply the t2av task prefix to manually supplied joint_av prompts.",
    )
    parser.add_argument(
        "--append-duration-suffix", "--append_duration_suffix", action="store_true",
        help="Append ' duration: X.Xs' to manually supplied prompts.",
    )
    parser.add_argument("--duration-precision", "--duration_precision", type=_nonneg_int, default=1)

    parser.add_argument("--modes", nargs="+", choices=list(_VALID_MODES), default=["joint_av"])
    parser.add_argument("--cfg-modes", "--cfg_modes", nargs="+", choices=list(_VALID_CFG_MODES), default=None)
    parser.add_argument(
        "--cfg", action="append", default=None,
        help=(
            "Comma-separated or repeated numeric CFG values to sweep. For T2AV this sets "
            "video/audio guidance plus text/cross-modal guidance to the same value. "
            "When --cfg-modes is omitted, --cfg defaults validation.cfg_modes to dual."
        ),
    )
    parser.add_argument("--num-inference-steps", "--num_inference_steps", type=_nonneg_int, default=None)
    parser.add_argument("--video-guidance-scale", "--video_guidance_scale", type=float, default=None)
    parser.add_argument("--audio-guidance-scale", "--audio_guidance_scale", type=float, default=None)
    parser.add_argument("--video-text-guidance", "--video_text_guidance", type=float, default=None)
    parser.add_argument("--video-modality-guidance", "--video_modality_guidance", type=float, default=None)
    parser.add_argument("--audio-text-guidance", "--audio_text_guidance", type=float, default=None)
    parser.add_argument("--audio-modality-guidance", "--audio_modality_guidance", type=float, default=None)
    parser.add_argument(
        "--max-wandb-samples-per-source", "--max_wandb_samples_per_source", type=_nonneg_int, default=None,
        help="Forwarded to validation config. 0 disables the cap.",
    )

    parser.add_argument("--video-vae-type", "--video_vae_type", "--vae-type", "--vae_type", dest="video_vae_type", default=None)
    parser.add_argument("--video-vae-path", "--video_vae_path", "--vae-path", "--vae_path", dest="video_vae_path", default=None)
    parser.add_argument("--audio-vae-type", "--audio_vae_type", default=None)
    parser.add_argument("--audio-vae-path", "--audio_vae_path", default=None)
    parser.add_argument(
        "--video-vae-override", "--video_vae_override", action="append", default=None,
        help="Per-experiment video VAE override, repeatable: EXP=PATH or EXP=TYPE:PATH.",
    )
    parser.add_argument(
        "--audio-vae-override", "--audio_vae_override", action="append", default=None,
        help="Per-experiment audio VAE override, repeatable: EXP=PATH or EXP=TYPE:PATH.",
    )

    parser.add_argument("--mixed-precision", "--mixed_precision", default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Print selected checkpoints and exit.")
    parser.add_argument("--fail-fast", "--fail_fast", action="store_true", help="Stop on the first failed checkpoint.")
    args = parser.parse_args()
    try:
        args.cfg_values = _parse_cfg_values(args.cfg)
        args.steps_filter, args.steps_include_latest = _parse_step_selection(args.steps)
        args.video_vae_overrides = _parse_vae_overrides(
            args.video_vae_override, flag_name="--video-vae-override",
        )
        args.audio_vae_overrides = _parse_vae_overrides(
            args.audio_vae_override, flag_name="--audio-vae-override",
        )
        if args.target_shard_index is not None or int(args.target_shard_count or 0) > 0:
            if args.target_shard_index is None or int(args.target_shard_count or 0) <= 0:
                raise ValueError("--target-shard-index and --target-shard-count must be set together")
            if int(args.target_shard_index) >= int(args.target_shard_count):
                raise ValueError(
                    "--target-shard-index must be < --target-shard-count, "
                    f"got {args.target_shard_index} >= {args.target_shard_count}"
                )
            args.target_shard_weight_values = _parse_target_shard_weights(
                str(args.target_shard_weights or ""),
                int(args.target_shard_count),
            )
        else:
            args.target_shard_weight_values = []
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _parse_target_shard_weights(raw: str, shard_count: int) -> list[int]:
    if not raw:
        return [1] * shard_count
    try:
        weights = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--target-shard-weights must contain positive integers: {raw!r}") from exc
    if len(weights) != shard_count:
        raise ValueError(
            f"--target-shard-weights length ({len(weights)}) must equal "
            f"--target-shard-count ({shard_count})"
        )
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"--target-shard-weights must be positive: {raw!r}")
    return weights


def _weighted_chunk_sizes(total: int, weights: list[int]) -> list[int]:
    if not weights:
        return []
    weight_sum = sum(weights)
    raw_sizes = [total * weight / weight_sum for weight in weights]
    sizes = [int(math.floor(size)) for size in raw_sizes]
    remainder = total - sum(sizes)
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw_sizes[index] - sizes[index], weights[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        sizes[index] += 1
    return sizes


def _apply_target_shard(args: argparse.Namespace, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shard_count = int(args.target_shard_count or 0)
    if args.target_shard_index is None and shard_count <= 0:
        return targets
    weights = list(args.target_shard_weight_values)
    sizes = _weighted_chunk_sizes(len(targets), weights)
    start = sum(sizes[: int(args.target_shard_index)])
    end = start + sizes[int(args.target_shard_index)]
    return targets[start:end]


def _step_from_checkpoint_dir(path: Path) -> Optional[int]:
    match = _CKPT_RE.match(path.name)
    if match:
        return int(match.group(1))
    metadata_path = path / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("global_step", "checkpoint_step", "release_checkpoint_step", "source_step"):
                if metadata.get(key) is not None:
                    return int(metadata[key])
            if metadata.get("inference_only") or metadata.get("export_format") == "omnivae_generation_inference_package_v1":
                return 0
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
    return None


def _looks_like_t2av_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "transformer_video" / "config.json").is_file()
        and (path / "transformer_audio" / "config.json").is_file()
        and (path / "bridges" / "bridges.safetensors").is_file()
        and (path / "metadata.json").is_file()
    )


def _target_for_direct_checkpoint(path: Path) -> Optional[dict[str, Any]]:
    step = _step_from_checkpoint_dir(path)
    if step is None or not _looks_like_t2av_checkpoint(path):
        return None
    if path.parent.name == "snapshots" and path.parent.parent.name == "checkpoints":
        experiment = path.parent.parent.parent.name
    else:
        experiment = path.name or path.parent.name or "direct_ckpt"
    return {"experiment": experiment, "step": step, "ckpt_dir": path}


def _iter_checkpoint_dirs(root: Path) -> list[dict[str, Any]]:
    direct = _target_for_direct_checkpoint(root)
    if direct is not None:
        return [direct]

    # One experiment run dir: <run>/checkpoints/snapshots/checkpoint-*
    snap = root / "checkpoints" / "snapshots"
    if snap.is_dir():
        out = []
        for ckpt in snap.iterdir():
            target = _target_for_direct_checkpoint(ckpt)
            if target is not None:
                target["experiment"] = root.name
                out.append(target)
        final_target = _target_for_direct_checkpoint(root / "final")
        if final_target is not None:
            final_target["experiment"] = root.name
            out.append(final_target)
        return out

    # Sweep root: <root>/<experiment>/checkpoints/snapshots/checkpoint-*
    out = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        direct_target = _target_for_direct_checkpoint(exp_dir)
        if direct_target is not None:
            direct_target["experiment"] = exp_dir.name
            out.append(direct_target)
            continue
        snap = exp_dir / "checkpoints" / "snapshots"
        if not snap.is_dir():
            continue
        for ckpt in snap.iterdir():
            target = _target_for_direct_checkpoint(ckpt)
            if target is not None:
                target["experiment"] = exp_dir.name
                out.append(target)
        final_target = _target_for_direct_checkpoint(exp_dir / "final")
        if final_target is not None:
            final_target["experiment"] = exp_dir.name
            out.append(final_target)
    return out


def discover_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.ckpt_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--ckpt-root not found: {root}")

    experiments = set(args.experiments or []) or None
    steps_filter = args.steps_filter
    include_latest = bool(args.steps_include_latest)

    candidates = []
    for target in _iter_checkpoint_dirs(root):
        step = int(target["step"])
        if experiments is not None and str(target["experiment"]) not in experiments:
            continue
        if args.min_step is not None and step < int(args.min_step):
            continue
        if args.max_step is not None and step > int(args.max_step):
            continue
        candidates.append(target)

    latest_by_experiment: dict[str, int] = {}
    if include_latest or args.only_latest:
        for target in candidates:
            exp = str(target["experiment"])
            step = int(target["step"])
            latest_by_experiment[exp] = max(step, latest_by_experiment.get(exp, -1))

    targets = []
    for target in candidates:
        step = int(target["step"])
        exp = str(target["experiment"])
        if args.only_latest:
            if step != latest_by_experiment.get(exp):
                continue
        elif steps_filter is not None or include_latest:
            explicit_match = steps_filter is not None and step in steps_filter
            latest_match = include_latest and step == latest_by_experiment.get(exp)
            if not (explicit_match or latest_match):
                continue
        elif int(args.step_multiple or 0) > 0 and (step <= 0 or step % int(args.step_multiple) != 0):
            continue
        targets.append(target)

    reverse = args.order == "desc"
    targets.sort(key=lambda t: (int(t["step"]), str(t["experiment"])), reverse=reverse)
    if int(args.max_ckpts or 0) > 0:
        targets = targets[: int(args.max_ckpts)]
    args.target_count_before_shard = len(targets)
    targets = _apply_target_shard(args, targets)
    return targets


def _inject_type_filter_into_existing_validation(config: dict, args: argparse.Namespace) -> None:
    types = _split_values(args.types)
    if not types:
        return
    val_cfg = config.setdefault("validation", {})
    filter_field = args.type_filter_field or args.type_field
    for key in ("joint_av_prompts", "video_only_prompts", "audio_only_prompts"):
        spec = val_cfg.get(key)
        if isinstance(spec, dict):
            spec["type_filter"] = types
            spec["type_filter_field"] = filter_field
    sets = val_cfg.get("audio_only_prompt_sets")
    if isinstance(sets, list):
        for spec in sets:
            if isinstance(spec, dict):
                spec["type_filter"] = types
                spec["type_filter_field"] = filter_field


def build_validation_config_from_run_config(
    run_config: dict,
    args: argparse.Namespace,
    target: dict[str, Any],
    cfg_value: Optional[float] = None,
    *,
    default_output_dir: Optional[Path] = None,
) -> dict:
    config = copy.deepcopy(run_config)
    exp_cfg = config.setdefault("experiment", {})
    if args.output_root:
        output_dir = Path(args.output_root).expanduser().resolve() / str(target["experiment"])
    else:
        output_dir = Path(default_output_dir).expanduser().resolve() if default_output_dir else Path(exp_cfg.get("output_dir", ".")).expanduser().resolve()
    exp_cfg["output_dir"] = str(output_dir)
    exp_cfg.setdefault("name", str(target["experiment"]))

    val_cfg = config.setdefault("validation", {})
    val_cfg["modes"] = list(args.modes)

    if cfg_value is not None:
        cfg = float(cfg_value)
        val_cfg["cfg_value"] = cfg
        val_cfg["cfg_output_suffix"] = f"g{_format_cfg_for_dir(cfg)}"
        if args.cfg_modes is None:
            val_cfg["cfg_modes"] = ["dual"]
        for cfg_name in (
            "video_guidance_scale",
            "audio_guidance_scale",
            "video_text_guidance",
            "video_modality_guidance",
            "audio_text_guidance",
            "audio_modality_guidance",
        ):
            val_cfg[cfg_name] = cfg

    if args.cfg_modes is not None:
        val_cfg["cfg_modes"] = list(args.cfg_modes)
    if args.num_inference_steps is not None and int(args.num_inference_steps) > 0:
        val_cfg["num_inference_steps"] = int(args.num_inference_steps)
    for cli_name, cfg_name in (
        ("video_guidance_scale", "video_guidance_scale"),
        ("audio_guidance_scale", "audio_guidance_scale"),
        ("video_text_guidance", "video_text_guidance"),
        ("video_modality_guidance", "video_modality_guidance"),
        ("audio_text_guidance", "audio_text_guidance"),
        ("audio_modality_guidance", "audio_modality_guidance"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            val_cfg[cfg_name] = float(value)
    if args.max_wandb_samples_per_source is not None:
        val_cfg["max_wandb_samples_per_source"] = (
            None if int(args.max_wandb_samples_per_source) <= 0
            else int(args.max_wandb_samples_per_source)
        )

    if args.validation_jsonl:
        jsonl_path = Path(args.validation_jsonl).expanduser().resolve()
        spec: dict[str, Any] = {
            "name": args.validation_name or jsonl_path.stem,
            "jsonl_path": str(jsonl_path),
            "text_field": str(args.text_field),
            "type_field": str(args.type_field),
            "index_field": str(args.index_field),
            "task_kind": None if args.no_task_prefix else "t2av",
            "append_duration_suffix": bool(args.append_duration_suffix),
            "duration_precision": int(args.duration_precision),
        }
        types = _split_values(args.types)
        if types:
            spec["type_filter"] = types
            spec["type_filter_field"] = str(args.type_filter_field or args.type_field)
        if args.max_examples is not None and int(args.max_examples) > 0:
            spec["max_examples"] = int(args.max_examples)
        val_cfg["joint_av_prompts"] = spec
    else:
        _inject_type_filter_into_existing_validation(config, args)
        if args.max_examples is not None:
            spec = val_cfg.get("joint_av_prompts")
            if isinstance(spec, dict) and int(args.max_examples) > 0:
                spec["max_examples"] = int(args.max_examples)

    return config


def build_validation_config(
    pipe,
    args: argparse.Namespace,
    target: dict[str, Any],
    cfg_value: Optional[float] = None,
) -> dict:
    return build_validation_config_from_run_config(
        pipe.run_config,
        args,
        target,
        cfg_value=cfg_value,
        default_output_dir=Path(pipe.run_dir).expanduser().resolve(),
    )


def load_run_config_for_probe(ckpt_dir: Path) -> tuple[dict[str, Any], Path]:
    """Load the resolved run config without constructing any model weights."""
    from omnivae_generation.trainer.eval.guided_diffusion import load_run_config_for_eval, resolve_run_dir

    run_dir = resolve_run_dir(None, ckpt_dir)
    json_cfg_path = run_dir / "resolved_config.json"
    yaml_cfg_path = run_dir / "resolved_config.yaml"
    if json_cfg_path.is_file():
        return json.loads(json_cfg_path.read_text(encoding="utf-8")), run_dir
    if yaml_cfg_path.is_file():
        return load_run_config_for_eval(run_dir), run_dir
    raise FileNotFoundError(
        f"Neither resolved_config.json nor resolved_config.yaml found under {run_dir}"
    )


def log(accelerator: Accelerator, message: str) -> None:
    if accelerator.is_local_main_process:
        print(message, flush=True)


def _append_timing_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    targets = discover_targets(args)
    cfg_values: list[Optional[float]] = list(args.cfg_values) if args.cfg_values else [None]
    cfg_labels = [
        "default" if value is None else f"cfg={float(value):g}"
        for value in cfg_values
    ]
    if args.dry_run:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[validate_t2av] selected {len(targets)} checkpoint(s)", flush=True)
            if int(args.target_shard_count or 0) > 0:
                print(
                    f"[validate_t2av] target shard: {args.target_shard_index}/"
                    f"{args.target_shard_count} from {args.target_count_before_shard} target(s)",
                    flush=True,
                )
            print(f"[validate_t2av] cfg sweep: {', '.join(cfg_labels)}", flush=True)
            if args.video_vae_overrides:
                print(f"[validate_t2av] video VAE overrides: {args.video_vae_overrides}", flush=True)
            if args.audio_vae_overrides:
                print(f"[validate_t2av] audio VAE overrides: {args.audio_vae_overrides}", flush=True)
            for target in targets:
                video_vae_type, video_vae_path = _resolve_vae_override(args, target, kind="video")
                audio_vae_type, audio_vae_path = _resolve_vae_override(args, target, kind="audio")
                print(
                    f"[validate_t2av]   step={int(target['step']):08d} "
                    f"experiment={target['experiment']} ckpt={target['ckpt_dir']} "
                    f"video_vae={video_vae_type or '<config>'}:{video_vae_path or '<config>'} "
                    f"audio_vae={audio_vae_type or '<config>'}:{audio_vae_path or '<config>'}",
                    flush=True,
                )
        return

    if not targets and int(args.target_shard_count or 0) > 0:
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[validate_t2av] target shard {args.target_shard_index}/"
                f"{args.target_shard_count} is empty; exit.",
                flush=True,
            )
        return

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    if accelerator.is_main_process:
        print(f"[validate_t2av] selected {len(targets)} checkpoint(s)", flush=True)
        if int(args.target_shard_count or 0) > 0:
            print(
                f"[validate_t2av] target shard: {args.target_shard_index}/"
                f"{args.target_shard_count} from {args.target_count_before_shard} target(s)",
                flush=True,
            )
        print(f"[validate_t2av] cfg sweep: {', '.join(cfg_labels)}", flush=True)
        for target in targets:
            print(
                f"[validate_t2av]   step={int(target['step']):08d} "
                f"experiment={target['experiment']} ckpt={target['ckpt_dir']}",
                flush=True,
            )
    accelerator.wait_for_everyone()
    if not targets:
        raise SystemExit("No checkpoints matched the requested filters.")

    failures: list[tuple[dict[str, Any], str]] = []
    cfg_pairs = list(zip(cfg_values, cfg_labels))
    for ordinal, target in enumerate(targets, start=1):
        step = int(target["step"])
        ckpt_dir = Path(target["ckpt_dir"]).expanduser().resolve()
        pending_cfg_pairs = list(cfg_pairs)
        if args.resume_inference:
            try:
                probe_run_config, probe_run_dir = load_run_config_for_probe(ckpt_dir)
                pending_cfg_pairs = []
                for cfg_value, cfg_label in cfg_pairs:
                    probe_config = build_validation_config_from_run_config(
                        probe_run_config,
                        args,
                        target,
                        cfg_value=cfg_value,
                        default_output_dir=probe_run_dir,
                    )
                    if validation_outputs_complete(probe_config, step):
                        log(
                            accelerator,
                            f"[validate_t2av] skip complete step={step:08d} {cfg_label} "
                            f"experiment={target['experiment']}",
                        )
                    else:
                        pending_cfg_pairs.append((cfg_value, cfg_label))
                if not pending_cfg_pairs:
                    log(
                        accelerator,
                        f"[validate_t2av] ({ordinal}/{len(targets)}) all requested cfgs complete; "
                        f"skip loading step={step:08d} {ckpt_dir}",
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                log(
                    accelerator,
                    f"[validate_t2av] resume probe failed for step={step:08d}; "
                    f"will load and run normal resume checks: {type(exc).__name__}: {exc}",
                )
        log(
            accelerator,
            f"[validate_t2av] ({ordinal}/{len(targets)}) loading step={step:08d} {ckpt_dir} "
            f"pending_cfgs={','.join(label for _, label in pending_cfg_pairs)}",
        )
        pipe = None
        failed = False
        load_started_at = time.time()
        load_completed_at: Optional[float] = None
        try:
            video_vae_type, video_vae_path = _resolve_vae_override(args, target, kind="video")
            audio_vae_type, audio_vae_path = _resolve_vae_override(args, target, kind="audio")
            log(
                accelerator,
                "[validate_t2av] VAE overrides "
                f"video={video_vae_type or '<config>'}:{video_vae_path or '<config>'} "
                f"audio={audio_vae_type or '<config>'}:{audio_vae_path or '<config>'}",
            )
            pipe = load_joint_av_pipeline(
                ckpt_dir,
                device=accelerator.device,
                vae_type_override=video_vae_type,
                vae_path_override=video_vae_path,
                audio_vae_type_override=audio_vae_type,
                audio_vae_path_override=audio_vae_path,
            )
            load_completed_at = time.time()
            for cfg_value, cfg_label in pending_cfg_pairs:
                config = build_validation_config(pipe, args, target, cfg_value=cfg_value)
                out_dir = Path(config["experiment"]["output_dir"])
                cfg_suffix = (
                    ""
                    if cfg_value is None
                    else f" dirs=cfg_<mode>_g{_format_cfg_for_dir(float(cfg_value))}"
                )
                log(
                    accelerator,
                    f"[validate_t2av] validating step={step:08d} {cfg_label}; "
                    f"output={out_dir / 'samples' / f'step-{step:08d}'}{cfg_suffix}",
                )
                accelerator.wait_for_everyone()
                started_at = time.time()
                run_joint_av_validation(
                    accelerator=accelerator,
                    config=config,
                    step=step,
                    joint_model=pipe.joint_model,
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    video_vae=pipe.video_vae,
                    audio_vae=pipe.audio_vae,
                    scheduler=pipe.scheduler,
                    skip_completed=bool(args.resume_inference),
                )
                accelerator.wait_for_everyone()
                completed_at = time.time()
                if accelerator.is_main_process:
                    _append_timing_jsonl(
                        out_dir / "samples" / f"step-{step:08d}" / "validation_timing.jsonl",
                        {
                            "experiment": str(target["experiment"]),
                            "step": step,
                            "checkpoint_dir": str(ckpt_dir),
                            "cfg_label": cfg_label,
                            "cfg_value": None if cfg_value is None else float(cfg_value),
                            "modes": list(args.modes),
                            "cfg_modes": list(config.get("validation", {}).get("cfg_modes", [])),
                            "num_inference_steps": config.get("validation", {}).get("num_inference_steps"),
                            "pipeline_load_started_at_unix": load_started_at,
                            "pipeline_load_completed_at_unix": load_completed_at,
                            "pipeline_load_elapsed_sec": (
                                load_completed_at - load_started_at
                                if load_completed_at is not None else None
                            ),
                            "started_at_unix": started_at,
                            "completed_at_unix": completed_at,
                            "generation_started_at_unix": started_at,
                            "generation_completed_at_unix": completed_at,
                            "generation_elapsed_sec": completed_at - started_at,
                            "elapsed_sec": completed_at - started_at,
                        },
                    )
                log(accelerator, f"[validate_t2av] finished step={step:08d} {cfg_label}")
        except Exception as exc:
            failed = True
            message = f"{type(exc).__name__}: {exc}"
            failures.append((target, message))
            if accelerator.is_main_process:
                print(f"[validate_t2av] FAILED step={step:08d}: {message}", flush=True)
            # In distributed validation, continuing after a rank-local
            # generation failure is not safe: other ranks may be inside a
            # gather/barrier from run_joint_av_validation. Let torchrun tear
            # down the whole job instead of risking a silent hang.
            if args.fail_fast or int(accelerator.num_processes) > 1:
                raise
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not failed or int(accelerator.num_processes) == 1:
                accelerator.wait_for_everyone()

    if failures:
        if accelerator.is_main_process:
            print(f"[validate_t2av] {len(failures)} checkpoint(s) failed:", flush=True)
            for target, message in failures:
                print(
                    f"[validate_t2av]   step={int(target['step']):08d} "
                    f"experiment={target['experiment']}: {message}",
                    flush=True,
                )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
