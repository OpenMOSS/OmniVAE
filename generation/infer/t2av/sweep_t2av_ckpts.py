"""Multi-node / multi-GPU T2AV checkpoint sweep inference driver.

Launched by ``sweep_t2av_ckpts.sh`` via torchrun. Each rank binds to one
local GPU and pulls a static (rank-strided) slice of targets from the
global checkpoint pool sorted ``(-step, experiment)`` so the newest
checkpoint across all experiments is processed first.

For every (experiment, checkpoint) target this driver:

1. Loads the joint-AV pipeline with :func:`t2av_pipeline.load_joint_av_pipeline`.
2. Iterates the prompt manifest (default key ``av_caption``) and runs
   :func:`t2av_pipeline.generate_one_av` per prompt, writing one
   ``sample-<dataset>-<index>-<type>.{mp4,wav,av.mp4,json}`` triplet per
   sample plus a ``samples.jsonl`` append-only log.
3. Writes ``done.json`` so subsequent rescan rounds (and the eval
   passes downstream) can tell the target is complete.

Filename layout matches the evaluator's
``build_sample_manifest.FILENAME_RE`` regex
(``sample-<dataset>-<index>-<category>.<ext>``) so the existing
``generation/infer/t2av/my_eval`` pipeline can ingest the produced
directory tree without any modifications.

``--ckpt-root`` may point either at a sweep root containing one or more
``<experiment>/checkpoints/snapshots/checkpoint-*`` directories, or directly
at one ``checkpoint-XXXXXXXX`` directory. The latter is convenient for a
single-checkpoint smoke test.

Rescan loop
-----------

After each round all ranks barrier; rank 0 re-runs target discovery and
broadcasts the new pending count. When it reaches zero the loop exits
(or sleeps ``--watch-interval`` seconds before re-checking, useful
while a training job is still producing fresh checkpoints).

Distributed protocol
--------------------

* Backend: ``gloo`` (we only ship CPU-side scalars across ranks).
* Collectives: ``barrier`` and ``broadcast`` of a single int.
* ``world_size == 1`` short-circuits init_process_group, so the same
  script runs standalone on a single GPU without a launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import traceback
from datetime import datetime, timedelta
from itertools import zip_longest
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist


# --- sys.path bootstrap ---------------------------------------------------
INFER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = INFER_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(INFER_ROOT))

from t2av_pipeline import (  # noqa: E402
    check_ffmpeg_available,
    generate_one_av,
    load_joint_av_pipeline,
)
from infer_t2av import (  # noqa: E402
    _coerce_text,
    _filename_safe,
    _resolve_sample_index,
    load_manifest,
)


_VALID_MODES = ("joint_av", "video_only", "audio_only")
_VALID_CFG_MODES = ("simple", "dual")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def log(rank: int, message: str) -> None:
    ts = datetime.now().strftime("%F %T")
    print(f"[{ts}][rank{rank}] {message}", flush=True)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive float, got {value!r}")
    return parsed


def nonneg_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt-root", "--ckpt", dest="ckpt_root", required=True,
        help=(
            "Root directory containing <experiment>/checkpoints/snapshots/checkpoint-XXXXXXXX, "
            "or a single checkpoint-XXXXXXXX directory."
        ),
    )
    parser.add_argument("--output-root", required=True,
                        help="Inference output root; per-target dir = <root>/<exp>/samples/step-<NNNNNNNN>/joint_av/cfg_<cfg>.")
    parser.add_argument("--prompt-manifest", required=True,
                        help="JSONL with one prompt per row (av_caption / index / type fields).")

    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Restrict to these experiment subdirectory names. Default = auto-discover all.")
    parser.add_argument("--dataset-tag", default="vabench",
                        help="Middle segment of the file stem 'sample-<tag>-<index>-<type>'. Default: vabench.")
    parser.add_argument("--modes", nargs="+", default=["joint_av"],
                        choices=list(_VALID_MODES),
                        help="Generation modes (default: joint_av).")
    parser.add_argument("--cfg-mode", default="dual", choices=list(_VALID_CFG_MODES),
                        help="CFG variant. dual is only honoured for joint_av; collapses to simple otherwise.")

    parser.add_argument("--num-inference-steps", type=positive_int, default=50)
    parser.add_argument("--video-guidance-scale", type=float, default=4.0)
    parser.add_argument("--audio-guidance-scale", type=float, default=4.0)
    parser.add_argument("--cfg-normalization", action="store_true")
    parser.add_argument("--video-text-guidance", type=float, default=None)
    parser.add_argument("--video-modality-guidance", type=float, default=None)
    parser.add_argument("--audio-text-guidance", type=float, default=None)
    parser.add_argument("--audio-modality-guidance", type=float, default=None)

    parser.add_argument("--num-frames", type=positive_int, default=None)
    parser.add_argument("--fps", type=positive_float, default=None)
    parser.add_argument("--height", type=positive_int, default=None)
    parser.add_argument("--width", type=positive_int, default=None)
    parser.add_argument("--audio-duration-seconds", type=positive_float, default=None)
    parser.add_argument("--seed", type=int, default=20260508,
                        help="Base seed. Per-prompt seed = seed + row_index * 100003.")
    parser.add_argument("--video-quality", type=int, default=8)
    parser.add_argument("--negative-prompt", default="")

    parser.add_argument("--vae-type", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--audio-vae-type", default=None)
    parser.add_argument("--audio-vae-path", default=None)

    parser.add_argument("--no-task-prefix", action="store_true",
                        help="Disable the t2av template wrapping (NOT recommended for distilled checkpoints).")
    parser.add_argument("--no-duration-suffix", action="store_true",
                        help="Disable the ' duration: X.Xs' suffix (mirrors trainer validation).")

    parser.add_argument("--prompt-key", default="av_caption")
    parser.add_argument("--prompt-fallback-keys", default="prompt,video_caption,caption,text")
    parser.add_argument("--negative-prompt-key", default="negative_prompt")

    parser.add_argument("--max-total-ckpts", type=nonneg_int, default=0,
                        help="Globally keep only the top-N most recent ckpts (across all experiments). 0 = unlimited.")
    parser.add_argument("--max-ckpts-per-experiment", type=nonneg_int, default=0,
                        help="Per experiment keep at most the N most recent ckpts. 0 = unlimited.")
    parser.add_argument("--step-multiple", type=nonneg_int, default=0,
                        help="Only consider checkpoints whose training step is a positive multiple of this "
                             "value (e.g. --step-multiple 10000 keeps step-10000, step-20000, ... and skips "
                             "step-5000 / off-grid snapshots). 0 = unlimited (all checkpoints kept). "
                             "Ignored when --steps is set.")
    parser.add_argument("--steps", nargs="+", type=nonneg_int, default=None,
                        metavar="STEP",
                        help="Explicit whitelist of training steps to process (e.g. "
                             "--steps 5000 10000 15000). Only checkpoints whose step matches one of these "
                             "values are kept. When set, --step-multiple is ignored (this flag is the "
                             "more specific override). Composes with --experiments, --max-total-ckpts "
                             "and --max-ckpts-per-experiment.")
    parser.add_argument("--limit-prompts", type=nonneg_int, default=0,
                        help="Truncate the prompt list to the first N rows. 0 = use all.")

    parser.add_argument("--watch-interval", type=nonneg_int, default=0,
                        help="When rescan returns 0 pending: 0 = exit, >0 = sleep N seconds then rescan again.")
    parser.add_argument("--watch-max-empty-rescans", type=nonneg_int, default=1,
                        help="Only meaningful when --watch-interval > 0. After this many *consecutive* rescans "
                             "with 0 pending targets, give up watching and exit (so the calling shell can "
                             "proceed to eval). 0 = unlimited (legacy behaviour, never exit on empty rescans). "
                             "Default: 1 (exit on the first empty rescan -- 'do what I mean' for the common "
                             "case where you're not actively waiting on a training job to produce new ckpts).")
    parser.add_argument("--max-rescan-rounds", type=positive_int, default=10000,
                        help="Hard cap on the number of rescan iterations (safety net).")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Ignore failed.json markers from a previous run and retry those targets.")
    parser.add_argument("--max-empty-rounds", type=positive_int, default=2,
                        help="If this many consecutive rounds produce zero new successes, exit (hot-spin guard).")

    parser.add_argument(
        "--prompt-parallel", action="store_true",
        help=(
            "Coalesce all ranks onto every target and split prompts across ranks "
            "(prompt-level data parallelism). Use when N_ckpts < world_size "
            "(e.g. --test with 1 ckpt) so every GPU stays busy; otherwise stick "
            "with the default ckpt-level parallelism for best aggregate throughput."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------------------------------
# Distributed boilerplate (mirrors run_eval_group.py)
# --------------------------------------------------------------------------
def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        # Default gloo barrier timeout is 30min. A single ckpt fans out across
        # hundreds of vabench prompts and joint_av generation can be 30-60s
        # per sample on a single GPU, so the rank that gets the slowest ckpt
        # easily takes >30min while peers idle at the end-of-round barrier.
        # 1.5h leaves plenty of headroom; overridable via DIST_TIMEOUT_HOURS.
        timeout_hours = float(os.environ.get("DIST_TIMEOUT_HOURS", "1.5"))
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(hours=timeout_hours),
        )
    return rank, local_rank, world_size


def barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def broadcast_int(value: int, world_size: int, src: int = 0) -> int:
    if world_size <= 1 or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.long)
    dist.broadcast(tensor, src=src)
    return int(tensor.item())


def gather_int_list(local_value: int, world_size: int) -> list[int]:
    if world_size <= 1 or not dist.is_initialized():
        return [local_value]
    gathered = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([local_value], dtype=torch.long))
    return [int(t.item()) for t in gathered]


# --------------------------------------------------------------------------
# Target discovery
# --------------------------------------------------------------------------
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def _looks_like_t2av_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "transformer_video" / "config.json").is_file()
        and (path / "transformer_audio" / "config.json").is_file()
        and (path / "bridges" / "bridges.safetensors").is_file()
        and (path / "metadata.json").is_file()
    )


def _checkpoint_step(path: Path) -> Optional[int]:
    match = _CKPT_RE.match(path.name)
    if match:
        return int(match.group(1))
    metadata_path = path / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("global_step", "checkpoint_step"):
                if metadata.get(key) is not None:
                    return int(metadata[key])
            if metadata.get("inference_only") or metadata.get("export_format") == "omnivae_generation_inference_package_v1":
                return 0
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def _direct_checkpoint_target(ckpt_dir: Path) -> Optional[dict[str, Any]]:
    """Return a single target when ``ckpt_dir`` is itself checkpoint-XXXXXXXX."""
    step = _checkpoint_step(ckpt_dir)
    if step is None or not _looks_like_t2av_checkpoint(ckpt_dir):
        return None

    # Standard layout: <experiment>/checkpoints/snapshots/checkpoint-XXXXXXXX.
    # For ad-hoc checkpoint locations, fall back to the immediate parent so the
    # output path still has a stable experiment-like directory segment.
    if (
        ckpt_dir.parent.name == "snapshots"
        and ckpt_dir.parent.parent.name == "checkpoints"
        and ckpt_dir.parent.parent.parent != ckpt_dir.parent.parent
    ):
        experiment = ckpt_dir.parent.parent.parent.name
    else:
        experiment = ckpt_dir.name or ckpt_dir.parent.name or "direct_ckpt"

    return {"experiment": experiment, "step": int(step), "ckpt_dir": str(ckpt_dir)}


def discover_targets(
    ckpt_root: Path,
    experiments_filter: Optional[set[str]] = None,
    max_per_experiment: int = 0,
    max_total: int = 0,
    step_multiple: int = 0,
    steps_filter: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Emit checkpoint targets from a sweep root or one direct checkpoint dir.

    In sweep-root mode, walks
    ``ckpt_root/<exp>/checkpoints/snapshots/checkpoint-*`` and emits one target
    dict per discovered checkpoint.

    Ordering: per-experiment lists are sorted by descending step, then
    interleaved round-robin across experiments (latest of exp1, latest of
    exp2, ..., 2nd-latest of exp1, 2nd-latest of exp2, ...). This way the
    sweep makes one "fresh" eval available per experiment before going
    back for older snapshots, which is what you want when several
    experiments are training in parallel and you mostly care about the
    leaderboard at the most recent step.

    ``steps_filter`` (when not None): explicit whitelist of training
    steps; only checkpoints whose step is in this set are kept. This is
    the most specific filter and takes precedence over ``step_multiple``
    (which is silently ignored when ``steps_filter`` is given).

    ``step_multiple`` (when > 0 and ``steps_filter`` is None): only keep
    checkpoints whose step is a *positive* multiple of this value.
    Step-0 checkpoints (rarely useful) are also discarded so the filter
    behaves predictably even when the trainer writes a step-0 snapshot
    for warm-up. Use this to coarse-grain eval to e.g. every 10k step.

    ``max_per_experiment`` keeps only the top-N steps per experiment
    (applied *after* the step filters);
    ``max_total`` truncates the final interleaved list to N entries.
    """
    if not ckpt_root.is_dir():
        return []
    direct = _direct_checkpoint_target(ckpt_root)
    if direct is not None:
        step = int(direct["step"])
        if experiments_filter is not None and direct["experiment"] not in experiments_filter:
            return []
        if steps_filter is not None:
            if step not in steps_filter:
                return []
        elif step_multiple > 0 and (step <= 0 or step % step_multiple != 0):
            return []
        return [direct]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for exp_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
        if experiments_filter is not None and exp_dir.name not in experiments_filter:
            continue
        direct = _direct_checkpoint_target(exp_dir)
        if direct is not None:
            step = int(direct["step"])
            if steps_filter is not None:
                if step not in steps_filter:
                    continue
            elif step_multiple > 0 and (step <= 0 or step % step_multiple != 0):
                continue
            direct["experiment"] = exp_dir.name
            grouped.setdefault(exp_dir.name, []).append(direct)
            continue
        snap = exp_dir / "checkpoints" / "snapshots"
        if not snap.is_dir():
            continue
        for ckpt in snap.iterdir():
            if not ckpt.is_dir():
                continue
            m = _CKPT_RE.match(ckpt.name)
            if not m:
                continue
            step = int(m.group(1))
            if steps_filter is not None:
                if step not in steps_filter:
                    continue
            elif step_multiple > 0 and (step <= 0 or step % step_multiple != 0):
                continue
            grouped.setdefault(exp_dir.name, []).append(
                {"experiment": exp_dir.name, "step": step, "ckpt_dir": str(ckpt)}
            )

    per_exp_lists: list[list[dict[str, Any]]] = []
    for exp in sorted(grouped.keys()):
        items = grouped[exp]
        items.sort(key=lambda t: -t["step"])
        if max_per_experiment > 0:
            items = items[:max_per_experiment]
        if items:
            per_exp_lists.append(items)

    targets: list[dict[str, Any]] = []
    for tier in zip_longest(*per_exp_lists, fillvalue=None):
        for entry in tier:
            if entry is not None:
                targets.append(entry)

    if max_total > 0:
        targets = targets[:max_total]
    return targets


def target_label(target: dict[str, Any]) -> str:
    return f"{target['experiment']}/step-{target['step']:08d}"


def target_out_dir(output_root: Path, target: dict[str, Any], cfg_mode: str) -> Path:
    return (
        output_root
        / target["experiment"]
        / "samples"
        / f"step-{target['step']:08d}"
        / "joint_av"
        / f"cfg_{cfg_mode}"
    )


def is_target_done(out_dir: Path) -> bool:
    done = out_dir / "done.json"
    return done.is_file() and done.stat().st_size > 0


def is_target_failed(out_dir: Path) -> bool:
    failed = out_dir / "failed.json"
    return failed.is_file() and failed.stat().st_size > 0


def effective_cfg_for_mode(mode: str, cfg_mode: str) -> str:
    """Mirror omnivae_generation.trainer.joint_av.validation: dual collapses to simple in
    single-modality modes."""
    if mode != "joint_av" and cfg_mode == "dual":
        return "simple"
    return cfg_mode


def write_failed_marker(
    output_root: Path,
    target: dict[str, Any],
    modes: list[str],
    cfg_mode: str,
    reason: str,
    *,
    rank: int,
) -> None:
    """Write a ``failed.json`` marker per (mode, eff_cfg) output dir so the
    rescan loop won't keep picking this target. Best-effort: a write error
    here is logged but does not crash the sweep (the hot-spin guard will
    still terminate the loop)."""
    payload = {
        "experiment": target["experiment"],
        "step": int(target["step"]),
        "reason": reason,
        "failed_at": time.strftime("%Y%m%d_%H%M%S"),
        "host": socket.gethostname(),
        "rank": rank,
        "remediation": "Fix the issue, then delete failed.json (or pass --retry-failed) and re-run.",
    }
    for mode in modes:
        eff_cfg = effective_cfg_for_mode(mode, cfg_mode)
        out_dir = target_out_dir(output_root, target, eff_cfg)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "failed.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            log(rank, f"WARN: failed to write failed.json in {out_dir}: {exc!r}")


# --------------------------------------------------------------------------
# Prompt manifest helpers
# --------------------------------------------------------------------------
def make_stem(record: dict[str, Any], *, dataset_tag: str) -> str:
    """Three-segment stem ``sample-<tag>-<index>-<type>`` matching
    ``build_sample_manifest.FILENAME_RE``.

    Index resolution defers to :func:`_resolve_sample_index` (shared with
    ``infer_t2av.make_file_stem``) so the inference and eval sides agree
    on the disjoint fallback namespace. Without that offset, versebench
    set2 rows whose ``index`` is non-numeric (e.g. ``"clip_05f5760d"``)
    fall back to ``_row_index`` and collide with sibling rows whose
    numeric ``index`` happens to equal that line number -- on versebench
    that wiped 64/600 samples off disk.
    """
    row_index = int(record.get("_row_index", 0))
    index = _resolve_sample_index(record.get("index"), row_index)
    type_label = _filename_safe(record.get("type", "unknown"), fallback="unknown")
    return f"sample-{dataset_tag}-{index:04d}-{type_label}"


def load_prompt_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    fallback_keys = [
        token.strip()
        for token in (args.prompt_fallback_keys or "").split(",")
        if token.strip()
    ]
    records = load_manifest(
        Path(args.prompt_manifest).expanduser().resolve(),
        prompt_key=args.prompt_key,
        fallback_keys=fallback_keys,
    )
    if args.limit_prompts > 0:
        records = records[: args.limit_prompts]
    if not records:
        raise SystemExit(
            f"[sweep] no usable prompts found in {args.prompt_manifest!r}"
        )
    return records


# --------------------------------------------------------------------------
# Per-target inference
# --------------------------------------------------------------------------
def write_run_json(
    out_dir: Path,
    *,
    pipe: Any,
    target: dict[str, Any],
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    cfg_mode: str,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    audio_duration_seconds: float,
    device: torch.device,
) -> None:
    run_manifest = {
        "experiment": target["experiment"],
        "step": int(target["step"]),
        "checkpoint_dir": str(pipe.checkpoint_dir),
        "checkpoint_step": int(pipe.checkpoint_step),
        "run_dir": str(pipe.run_dir),
        "modes": list(args.modes),
        "cfg_mode": cfg_mode,
        "num_inference_steps": int(args.num_inference_steps),
        "video_guidance_scale": float(args.video_guidance_scale),
        "audio_guidance_scale": float(args.audio_guidance_scale),
        "cfg_normalization": bool(args.cfg_normalization),
        "video_text_guidance": args.video_text_guidance,
        "video_modality_guidance": args.video_modality_guidance,
        "audio_text_guidance": args.audio_text_guidance,
        "audio_modality_guidance": args.audio_modality_guidance,
        "shift_v": pipe.shift_v,
        "shift_a": pipe.shift_a,
        "num_frames": num_frames,
        "fps": fps,
        "height": height,
        "width": width,
        "audio_duration_seconds": audio_duration_seconds,
        "base_seed": int(args.seed),
        "prompt_manifest": str(args.prompt_manifest),
        "prompt_count": len(records),
        "dataset_tag": args.dataset_tag,
        "device": str(device),
        "vae_type_override": args.vae_type,
        "vae_path_override": args.vae_path,
        "audio_vae_type_override": args.audio_vae_type,
        "audio_vae_path_override": args.audio_vae_path,
        "wrap_task_prefix": not bool(args.no_task_prefix),
        "append_duration_suffix": not bool(args.no_duration_suffix),
        "started_at": time.strftime("%Y%m%d_%H%M%S"),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def process_target(
    rank: int,
    local_rank: int,
    world_size: int,
    target: dict[str, Any],
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_root: Path,
    *,
    prompt_parallel: bool = False,
) -> str:
    """Run all (mode, prompt) pairs for one ckpt target. Returns one of:

    * ``"success"`` -- at least one new sample was produced
    * ``"skipped"`` -- no new work to do (all sidecars already on disk)
    * ``"failed"``  -- pipeline could not load; ``failed.json`` written to all
      configured (mode, eff_cfg) dirs so the rescan loop skips this target.

    Parallelism modes
    -----------------
    * ``prompt_parallel=False`` (default): the caller has already partitioned
      ckpts across ranks (``targets[rank::ws]``); this rank owns the entire
      record set for its assigned targets and writes both ``samples.jsonl``
      and ``done.json``.
    * ``prompt_parallel=True``: every rank is called with the *same* target.
      This rank takes the slice ``records[rank::ws]``, writes to
      ``samples.rank<NN>.jsonl`` (private per rank, no append-race), and
      does **NOT** write ``done.json``. The caller is responsible for the
      barrier + per-rank file merge + consolidated done.json. Resume is
      handled via per-prompt sidecar existence which is race-free.
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    label = target_label(target)
    log(rank, f"loading pipeline for {label} on {device}")
    try:
        pipe = load_joint_av_pipeline(
            target["ckpt_dir"],
            device=device,
            vae_type_override=args.vae_type,
            vae_path_override=args.vae_path,
            audio_vae_type_override=args.audio_vae_type,
            audio_vae_path_override=args.audio_vae_path,
        )
    except Exception as exc:  # noqa: BLE001
        log(rank, f"FAILED to load pipeline for {label}: {exc!r}")
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_failed_marker(
            output_root, target, list(args.modes), args.cfg_mode,
            reason=f"load_joint_av_pipeline raised {type(exc).__name__}: {exc}",
            rank=rank,
        )
        return "failed"

    val_cfg = pipe.run_config.get("validation", {}) or {}
    dataset_cfg = pipe.run_config.get("dataset", {}) or {}
    frame_size = val_cfg.get("video_frame_size") or dataset_cfg.get("frame_size") or [256, 256]
    num_frames = int(args.num_frames or val_cfg.get("video_num_frames") or dataset_cfg.get("num_frames") or 121)
    fps = float(args.fps or val_cfg.get("video_fps") or dataset_cfg.get("target_fps") or 24.0)
    height = int(args.height or frame_size[0])
    width = int(args.width or frame_size[1])
    audio_duration_seconds = float(
        args.audio_duration_seconds
        or val_cfg.get("audio_duration_seconds", 5.04)
    )

    records_for_run = list(records)
    if prompt_parallel and world_size > 1:
        records_for_run = records_for_run[rank::world_size]
        log(
            rank,
            f"prompt-parallel: rank {rank}/{world_size} owns "
            f"{len(records_for_run)} / {len(records)} prompts",
        )

    success_count = 0
    failure_count = 0
    skipped_count = 0
    for mode in args.modes:
        effective_cfg_mode = args.cfg_mode
        if mode != "joint_av" and effective_cfg_mode == "dual":
            effective_cfg_mode = "simple"
        out_dir = target_out_dir(output_root, target, effective_cfg_mode)
        out_dir.mkdir(parents=True, exist_ok=True)
        if prompt_parallel and world_size > 1:
            samples_path = out_dir / f"samples.rank{rank:02d}.jsonl"
        else:
            samples_path = out_dir / "samples.jsonl"

        # Only rank 0 writes run.json in prompt-parallel to avoid a write race.
        if (not prompt_parallel) or rank == 0:
            write_run_json(
                out_dir, pipe=pipe, target=target, args=args, records=records,
                cfg_mode=effective_cfg_mode, num_frames=num_frames, fps=fps,
                height=height, width=width, audio_duration_seconds=audio_duration_seconds,
                device=device,
            )

        # append-only; existing rows preserved across resumes
        samples_handle = samples_path.open("a", encoding="utf-8")
        try:
            for record in records_for_run:
                row_index = int(record.get("_row_index", 0))
                prompt = str(record.get("_resolved_prompt", "")).strip()
                if not prompt:
                    skipped_count += 1
                    continue
                file_stem = make_stem(record, dataset_tag=args.dataset_tag)
                # Resume guard: a non-empty sidecar means the sample
                # was fully produced and recorded in a prior run.
                # Sidecar-existence is race-free (each file_stem is
                # owned by exactly one rank in prompt-parallel mode and
                # one ckpt-rank in the default mode).
                sidecar = out_dir / f"{file_stem}.json"
                if sidecar.is_file() and sidecar.stat().st_size > 0:
                    skipped_count += 1
                    continue

                negative_prompt = (
                    _coerce_text(record.get(args.negative_prompt_key))
                    or args.negative_prompt
                )
                sample_seed = int(args.seed) + row_index * 100003

                t0 = time.time()
                try:
                    result = generate_one_av(
                        pipe,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        mode=mode,
                        cfg_mode=effective_cfg_mode,
                        num_inference_steps=int(args.num_inference_steps),
                        video_guidance_scale=float(args.video_guidance_scale),
                        audio_guidance_scale=float(args.audio_guidance_scale),
                        cfg_normalization=bool(args.cfg_normalization),
                        video_text_guidance=args.video_text_guidance,
                        video_modality_guidance=args.video_modality_guidance,
                        audio_text_guidance=args.audio_text_guidance,
                        audio_modality_guidance=args.audio_modality_guidance,
                        num_frames=num_frames,
                        fps=fps,
                        height=height,
                        width=width,
                        audio_duration_seconds=audio_duration_seconds,
                        seed=sample_seed,
                        output_dir=out_dir,
                        file_stem=file_stem,
                        video_quality=int(args.video_quality),
                        wrap_task_prefix=not bool(args.no_task_prefix),
                        append_duration_suffix=not bool(args.no_duration_suffix),
                    )
                except Exception as exc:  # noqa: BLE001
                    failure_count += 1
                    log(
                        rank,
                        f"FAILED {label} stem={file_stem} mode={mode}: {exc!r}",
                    )
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                elapsed = time.time() - t0
                result.update(
                    experiment=target["experiment"],
                    step=int(target["step"]),
                    mode=mode,
                    cfg_mode=effective_cfg_mode,
                    file_stem=file_stem,
                    row_index=row_index,
                    sample_seed=sample_seed,
                    source_record=record,
                    checkpoint_step=int(pipe.checkpoint_step),
                    wall_elapsed_sec=float(elapsed),
                    rank=rank,
                )
                sidecar.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                samples_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                samples_handle.flush()
                success_count += 1
                if success_count % 10 == 0 or success_count == 1:
                    log(
                        rank,
                        f"{label} mode={mode}: produced {success_count} / "
                        f"{len(records_for_run)} (last={file_stem}, {elapsed:.1f}s)",
                    )
        finally:
            samples_handle.close()

        # In prompt-parallel mode, done.json is written by the *caller*
        # after all ranks barrier + rank-0 merges per-rank samples files.
        # Each rank also leaves its private samples.rank<NN>.jsonl behind
        # for the caller's merge step.
        if not (prompt_parallel and world_size > 1):
            done_payload = {
                "experiment": target["experiment"],
                "step": int(target["step"]),
                "mode": mode,
                "cfg_mode": effective_cfg_mode,
                "prompt_count": len(records),
                "produced": success_count,
                "failed": failure_count,
                "skipped": skipped_count,
                "completed_at": time.strftime("%Y%m%d_%H%M%S"),
                "host": socket.gethostname(),
                "rank": rank,
            }
            (out_dir / "done.json").write_text(
                json.dumps(done_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        log(
            rank,
            f"slice done {label} mode={mode} cfg={effective_cfg_mode}: "
            f"produced={success_count} skipped={skipped_count} failed={failure_count}",
        )

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if success_count > 0:
        return "success"
    return "skipped"


def consolidate_prompt_parallel_target(
    target: dict[str, Any],
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_root: Path,
    world_size: int,
) -> None:
    """Rank-0 post-pass for a single target after all ranks finished their
    prompt slice in prompt-parallel mode. For each (mode, eff_cfg) dir:

    1. Merge every ``samples.rank<NN>.jsonl`` into ``samples.jsonl``
       (dedup on raw line; per-rank files are then unlinked).
    2. Tally success / skip / failure counts across rank shards and
       write the consolidated ``done.json`` so the rescan loop treats
       this target as complete.

    Callers must already have done a barrier so every rank's writes
    are durable on the shared filesystem.
    """
    for mode in args.modes:
        eff_cfg = effective_cfg_for_mode(mode, args.cfg_mode)
        out_dir = target_out_dir(output_root, target, eff_cfg)
        if not out_dir.is_dir():
            continue

        merged_path = out_dir / "samples.jsonl"
        seen: set[str] = set()
        if merged_path.is_file():
            for line in merged_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped:
                    seen.add(stripped)

        rank_files = sorted(out_dir.glob("samples.rank*.jsonl"))
        produced = 0
        with merged_path.open("a", encoding="utf-8") as out_f:
            for rf in rank_files:
                try:
                    for line in rf.read_text(encoding="utf-8").splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped in seen:
                            continue
                        out_f.write(stripped + "\n")
                        seen.add(stripped)
                        produced += 1
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[consolidate] WARN: failed to read {rf}: {exc!r}",
                        flush=True,
                    )
                # Best-effort cleanup of the per-rank shard. If unlink
                # fails we leave it in place; the file is non-fatal.
                try:
                    rf.unlink()
                except Exception:  # noqa: BLE001
                    pass

        # Cross-check sidecar count against expected prompt count so
        # done.json reflects ground truth.
        sidecar_count = sum(
            1 for p in out_dir.glob("sample-*.json")
            if p.name not in {"done.json", "run.json", "samples.jsonl"}
            and not p.name.startswith("samples.rank")
        )
        done_payload = {
            "experiment": target["experiment"],
            "step": int(target["step"]),
            "mode": mode,
            "cfg_mode": eff_cfg,
            "prompt_count": len(records),
            "sidecar_count": sidecar_count,
            "merged_from_rank_shards": produced,
            "consolidated_prompt_parallel": True,
            "world_size": world_size,
            "completed_at": time.strftime("%Y%m%d_%H%M%S"),
            "host": socket.gethostname(),
        }
        (out_dir / "done.json").write_text(
            json.dumps(done_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def all_pending(
    output_root: Path,
    ckpt_root: Path,
    experiments_filter: Optional[set[str]],
    cfg_mode: str,
    modes: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    targets = discover_targets(
        ckpt_root,
        experiments_filter=experiments_filter,
        max_per_experiment=args.max_ckpts_per_experiment,
        max_total=args.max_total_ckpts,
        step_multiple=args.step_multiple,
        steps_filter=set(args.steps) if args.steps else None,
    )

    def target_blocked(t: dict[str, Any]) -> bool:
        """A target is "no more work" when every configured (mode, eff_cfg)
        has either a done.json or a failed.json marker. With --retry-failed,
        failed.json is ignored."""
        for mode in modes:
            eff_cfg = effective_cfg_for_mode(mode, cfg_mode)
            out_dir = target_out_dir(output_root, t, eff_cfg)
            if is_target_done(out_dir):
                continue
            if (not args.retry_failed) and is_target_failed(out_dir):
                continue
            return False
        return True

    return [t for t in targets if not target_blocked(t)]


def main() -> int:
    args = parse_args()

    ckpt_root = Path(args.ckpt_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not ckpt_root.is_dir():
        raise SystemExit(f"--ckpt-root/--ckpt not found or not a directory: {ckpt_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    experiments_filter = set(args.experiments) if args.experiments else None

    if not check_ffmpeg_available():
        print(
            "[sweep] WARNING: ffmpeg not on PATH; joint_av will produce .mp4 + .wav "
            "without the muxed .av.mp4.",
            file=sys.stderr,
        )

    rank, local_rank, world_size = setup_distributed()
    if rank == 0:
        log(
            rank,
            f"host={socket.gethostname()} world_size={world_size} local_rank={local_rank} "
            f"cuda={torch.cuda.is_available()}",
        )
        log(rank, f"ckpt_root={ckpt_root}")
        log(rank, f"output_root={output_root}")
        log(rank, f"modes={args.modes} cfg_mode={args.cfg_mode} dataset_tag={args.dataset_tag}")
        log(
            rank,
            f"steps={args.num_inference_steps} vg={args.video_guidance_scale} "
            f"ag={args.audio_guidance_scale} seed={args.seed}",
        )
        log(rank, f"watch_interval={args.watch_interval}s max_rescan_rounds={args.max_rescan_rounds}")
        if args.steps:
            log(
                rank,
                f"steps_filter={sorted(set(args.steps))} "
                f"(only these explicit steps will be processed; --step-multiple ignored)",
            )
        elif args.step_multiple > 0:
            log(rank, f"step_multiple={args.step_multiple} (only checkpoints with step%N==0 will be processed)")
        if args.max_total_ckpts > 0 or args.max_ckpts_per_experiment > 0:
            log(
                rank,
                f"max_total_ckpts={args.max_total_ckpts} "
                f"max_ckpts_per_experiment={args.max_ckpts_per_experiment}",
            )

    # Prompt manifest is small + identical on every rank; load it once locally.
    records = load_prompt_records(args)
    if rank == 0:
        log(rank, f"loaded {len(records)} prompts from {args.prompt_manifest}")

    barrier(world_size)

    round_idx = 0
    total_processed = 0
    total_success = 0
    total_failed = 0
    consecutive_empty_rounds = 0
    consecutive_empty_rescans = 0
    while round_idx < args.max_rescan_rounds:
        pending: list[dict[str, Any]] = []
        if rank == 0:
            pending = all_pending(
                output_root, ckpt_root, experiments_filter,
                args.cfg_mode, args.modes, args,
            )
            log(rank, f"round {round_idx}: {len(pending)} pending targets")
            for t in pending[:20]:
                log(rank, f"  -> {target_label(t)}")
            if len(pending) > 20:
                log(rank, f"  ... and {len(pending) - 20} more")
        n_pending = broadcast_int(len(pending), world_size, src=0)

        if n_pending == 0:
            if args.watch_interval <= 0:
                if rank == 0:
                    log(rank, "no pending targets; exiting (watch_interval=0)")
                break
            consecutive_empty_rescans += 1
            # ``watch_max_empty_rescans == 0`` means "watch forever" (legacy
            # behaviour). Any positive value means "give up watching after N
            # consecutive empty rescans" so the calling shell can proceed to
            # eval. Default is 1: as soon as a rescan finds nothing pending,
            # exit. Increase it (or set to 0) when training is still actively
            # producing new ckpts you want to wait for.
            if (
                args.watch_max_empty_rescans > 0
                and consecutive_empty_rescans >= args.watch_max_empty_rescans
            ):
                if rank == 0:
                    log(
                        rank,
                        f"no pending; reached watch_max_empty_rescans="
                        f"{args.watch_max_empty_rescans} consecutive empty rescan(s); "
                        f"exiting watch loop.",
                    )
                break
            if rank == 0:
                log(
                    rank,
                    f"no pending; sleeping {args.watch_interval}s then rescanning "
                    f"(empty_rescan={consecutive_empty_rescans}"
                    f"{'/'+str(args.watch_max_empty_rescans) if args.watch_max_empty_rescans > 0 else ''})",
                )
            time.sleep(float(args.watch_interval))
            round_idx += 1
            continue

        # Found pending work this round -> reset the empty-rescan counter so
        # subsequent empty rounds get the full grace budget again.
        consecutive_empty_rescans = 0

        # All ranks re-discover independently. Shared FS + deterministic
        # sort key gives every rank the same list.
        pending = all_pending(
            output_root, ckpt_root, experiments_filter,
            args.cfg_mode, args.modes, args,
        )

        prompt_parallel = bool(args.prompt_parallel) and world_size > 1
        if prompt_parallel:
            # Every rank processes every target; records get sliced inside
            # process_target. Best for "few ckpts, many GPUs" workloads
            # (e.g. --test with 1 ckpt and 8 ranks).
            my_targets = pending
            if rank == 0:
                log(
                    rank,
                    f"round {round_idx}: prompt-parallel mode "
                    f"(world_size={world_size}, every rank processes all "
                    f"{len(pending)} target(s))",
                )
        else:
            # Default: ckpt-level data parallelism. Each rank fully owns
            # its slice of targets and writes its own done.json.
            my_targets = pending[rank::max(1, world_size)]
            if rank == 0:
                log(
                    rank,
                    f"round {round_idx}: world_size={world_size}, "
                    f"rank0 will process {len(my_targets)} of {len(pending)}",
                )

        round_success_local = 0
        round_failed_local = 0
        for i, target in enumerate(my_targets, 1):
            log(rank, f"round {round_idx} [{i}/{len(my_targets)}]: {target_label(target)}")
            status = process_target(
                rank, local_rank, world_size, target, args, records, output_root,
                prompt_parallel=prompt_parallel,
            )
            total_processed += 1
            if status == "success":
                round_success_local += 1
                total_success += 1
            elif status == "failed":
                round_failed_local += 1
                total_failed += 1

            # In prompt-parallel mode, every rank just finished its slice
            # of this target. Barrier, then rank 0 merges per-rank
            # samples.jsonl shards and writes the consolidated done.json
            # (unless *any* rank reported a hard failure -- in that case
            # failed.json was already written by each rank and we leave
            # done.json absent so the next rescan treats the target as
            # failed, not done).
            if prompt_parallel:
                barrier(world_size)
                status_code = {"success": 0, "failed": 1, "skipped": 2}.get(status, 2)
                all_codes = gather_int_list(status_code, world_size)
                any_hard_failure = 1 in all_codes
                if rank == 0 and not any_hard_failure:
                    consolidate_prompt_parallel_target(
                        target, args, records, output_root, world_size,
                    )
                barrier(world_size)
        barrier(world_size)

        # Aggregate across ranks so the hot-spin guard is global, not per-rank.
        all_success = sum(gather_int_list(round_success_local, world_size))
        all_failed = sum(gather_int_list(round_failed_local, world_size))
        if rank == 0:
            log(
                rank,
                f"round {round_idx} summary: succeeded={all_success} failed_hard={all_failed}",
            )

        # Hot-spin guard: only count rounds where *something* hard-failed
        # AND nothing succeeded. "Successful resume" rounds (success=0 and
        # failed=0, e.g. every sample was already in samples.jsonl) are
        # benign and must not trip the guard.
        if all_failed > 0 and all_success == 0:
            consecutive_empty_rounds += 1
            if consecutive_empty_rounds >= args.max_empty_rounds:
                if rank == 0:
                    log(
                        rank,
                        f"hot-spin guard: {consecutive_empty_rounds} consecutive "
                        f"rounds where every target hit a hard failure; exiting. "
                        f"Inspect failed.json markers under {output_root} and "
                        f"re-run with --retry-failed after fixing the issue.",
                    )
                round_idx += 1
                break
        elif all_success > 0:
            consecutive_empty_rounds = 0

        round_idx += 1

    barrier(world_size)
    if rank == 0:
        log(
            rank,
            f"sweep complete. rounds={round_idx} this-rank-processed={total_processed} "
            f"success={total_success} failed_hard={total_failed}",
        )

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
