#!/usr/bin/env python3
"""Distributed T2AV evaluation dispatcher.

Launched by ``run_my_eval.sh`` via ``torchrun``. Each rank binds to one local
GPU (``CUDA_VISIBLE_DEVICES=local_rank`` is set by the launcher); the dispatcher

1. discovers (experiment, step, cfg) targets under ``--sample-root``,
2. (rank 0 only) materialises ``metadata/manifest.json`` for each target by
   shelling out to ``generation/infer/t2av/build_sample_manifest.py``,
3. by default loops over metric kinds first, reusing that kind's model cache
   across all pending targets/checkpoints; within each target it slices records
   by ``records[rank::world_size]`` so every rank participates,
4. (rank 0 only) consolidates ``per_sample/<kind>/*.json`` into
   ``summary/<kind>.json`` and finally writes ``all_metrics_summary.json``.

No overall score is computed; every metric writes its own JSON.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Set

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
for _warning_module in ("torch", "torchaudio", "torchvision", "transformers", "timm"):
    warnings.filterwarnings("ignore", category=UserWarning, module=fr"{_warning_module}.*")

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]  # GitHub repo root
INFER_T2AV = REPO_ROOT / "generation" / "infer" / "t2av"
DEFAULT_BUILD_SCRIPT = REPO_ROOT / "generation" / "infer" / "t2av" / "eval_tools" / "build_sample_manifest.py"
DEFAULT_VALID_JSONL = Path(os.environ.get("OMNIVAE_RELEASE_ROOT", os.environ.get("OPEN_SOURCE_ROOT", str(REPO_ROOT / "open_source")))) / "eval" / "data" / "t2av" / "versebench_minimal" / "versebench_t2av_infer_minimal.jsonl"

# Make ``import my_eval...`` work no matter where torchrun was started.
sys.path.insert(0, str(INFER_T2AV))

from my_eval.tasks import (  # noqa: E402
    ALL_KIND_METRIC_KEYS,
    DEFAULT_KIND_ORDER,
    KIND_ELIGIBLE_CATEGORIES,
    KIND_METRIC_KEYS,
    KIND_ORDER,
    DATASET_LEVEL_SUMMARY_KINDS,
    clear_model_cache,
    get_preload_task,
    get_run_task,
    metric_keys_for_kind,
)
from my_eval.utils.distributed import (  # noqa: E402
    barrier,
    broadcast_object,
    log,
    slice_for_rank,
    setup_distributed,
)
from my_eval.utils.io_utils import (  # noqa: E402
    already_done,
    consolidate_summary,
    merge_all_metrics_summary,
)
from my_eval.utils.manifest import (  # noqa: E402
    Target,
    discover_targets,
    ensure_manifest,
    limit_records,
    load_manifest,
    parse_steps_arg,
    target_output_dir,
)
from my_eval.utils.preprocess import prewarm_preprocess_records  # noqa: E402
from my_eval.utils.quiet import silence_known_warnings  # noqa: E402

silence_known_warnings()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed T2AV evaluation dispatcher.")
    p.add_argument("--sample-root", required=True, type=Path,
                   help="Root containing <experiment>/samples/step-*/joint_av/cfg_*/")
    p.add_argument("--eval-output-root", required=True, type=Path)
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Restrict to these experiment directory names.")
    p.add_argument("--steps", default="",
                   help='Whitelist of training-step integers (e.g. "40000 80000").')
    p.add_argument("--cfg", choices=["dual", "simple", "both"], default="dual")
    p.add_argument("--kinds", default="",
                   help=f"Comma/space separated whitelist of metric kinds; default all of {DEFAULT_KIND_ORDER}")
    p.add_argument("--extra-kinds", default="",
                   help=(
                       "Comma/space separated metric kinds to append to the selected set. "
                       "Useful for optional kinds such as audio_amplitude."
                   ))
    p.add_argument("--skip-kinds", default="",
                   help=(
                       "Comma/space separated blacklist of metric kinds. Applied "
                       "after --kinds. Example: --skip-kinds video_aesthetic to "
                       "drop the slowest per-frame inferencer."
                   ))
    p.add_argument("--optional-metrics", default="",
                   help=(
                       "Comma/space separated optional submetrics. Supported: "
                       "AV-Align, LSE-D. These are off by default."
                   ))
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of samples per target (0 = all).")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip samples that already have a per_sample/<kind>/<stem>.json.")
    p.add_argument("--scan-workers", type=int, default=int(os.environ.get("MY_EVAL_SCAN_WORKERS", "1")),
                   help=(
                       "Rank-0 parallelism for the --skip-completed target-completeness scan. "
                       "This is thread-based metadata I/O parallelism; default 1, or "
                       "MY_EVAL_SCAN_WORKERS."
                   ))
    p.add_argument("--build-manifest-script", type=Path, default=DEFAULT_BUILD_SCRIPT)
    p.add_argument(
        "--valid-jsonl",
        "--input-jsonl",
        dest="valid_jsonl",
        type=Path,
        default=DEFAULT_VALID_JSONL,
        help=(
            "Prompt/metadata JSONL for manifest building. Rows may be original "
            "input records or generation logs with source_record."
        ),
    )
    p.add_argument("--max-ckpt-per-experiment", type=int, default=0,
                   help="Keep at most N largest step values per experiment (0 = no limit).")
    p.add_argument("--target-order", choices=["interleave", "linear"], default="interleave",
                   help=(
                       "Order in which (experiment, step, cfg) targets are evaluated. "
                       "'interleave' (default): newest step first, round-robin across "
                       "experiments so a partial run still produces at least one "
                       "checkpoint per experiment early. 'linear': experiment "
                       "alphabetical -> step ascending (the pre-2026-05 behaviour)."
                   ))
    p.add_argument(
        "--dispatch-mode",
        choices=["kind-major-reuse", "subtask", "data-parallel", "sample-major"],
        default="kind-major-reuse",
        help=(
            "How work is sliced across ranks. 'kind-major-reuse' (default): "
            "for each metric kind, all ranks evaluate all targets with that "
            "kind's model kept resident, then release it before the next kind; "
            "this avoids per-checkpoint reloads without keeping every model in "
            "VRAM. 'subtask': flatten into (target, kind) subtasks and "
            "round-robin by rank, so different ranks run different kinds at "
            "the same time. 'data-parallel': older behaviour where all ranks "
            "cooperate on the same (target, kind) by slicing samples; useful "
            "when there are very few subtasks. 'sample-major': experimental "
            "mode that preloads selected metric models on each rank, then "
            "evaluates this rank's samples/chunks through the selected "
            "non-dataset kinds."
        ),
    )
    p.add_argument("--sample-major-chunk-size", type=int,
                   default=int(os.environ.get("MY_EVAL_SAMPLE_MAJOR_CHUNK_SIZE", "1")),
                   help=(
                       "Records per microbatch in --dispatch-mode sample-major. "
                       "1 matches strict sample-by-sample scheduling; larger values "
                       "keep more per-kind batching while all models stay resident."
                   ))
    p.add_argument("--watch", action="store_true",
                   help="Keep polling --sample-root for newly discovered targets instead of exiting.")
    p.add_argument("--watch-interval", type=float, default=180.0,
                   help="Seconds between watch-mode scans (default: 180).")
    p.add_argument("--watch-min-age", type=float, default=0.0,
                   help=(
                       "Only evaluate targets whose newest direct sample file is at least this many "
                       "seconds old. Use this to avoid reading a step while generation is still writing."
                   ))
    p.add_argument("--watch-skip-existing", action="store_true",
                   help="In watch mode, ignore targets already present at startup and only evaluate future ones.")
    p.add_argument("--watch-max-passes", type=int, default=0,
                   help="Debug/testing only: stop watch mode after N scans (0 = never stop).")
    return p.parse_args()


def filter_targets_max_ckpt(targets: List[Target], max_ckpt: int) -> List[Target]:
    if max_ckpt <= 0:
        return targets
    by_exp: dict[str, list[int]] = {}
    for t in targets:
        by_exp.setdefault(t.experiment, []).append(t.step_num)
    keep: dict[str, set[int]] = {}
    for exp, nums in by_exp.items():
        keep[exp] = set(sorted(set(nums), reverse=True)[:max_ckpt])
    return [t for t in targets if t.step_num in keep.get(t.experiment, set())]


def order_targets_interleave(targets: List[Target]) -> List[Target]:
    """Round-robin across experiments, newest step first within each experiment.

    Within one experiment the per-step ordering is ``(step_num desc, cfg asc)`` so
    when ``--cfg both`` is given, the two cfg variants of the same step stay
    adjacent in that experiment's local queue (but the global iteration still
    rotates experiment by experiment).

    Example with experiments A, B, C and steps:
        A: [200, 100]
        B: [500, 300, 100]
        C: [50]
    yields:
        A/200, B/500, C/50, A/100, B/300, B/100
    so the moment evaluation crashes / runs out of time, every experiment has
    at least its latest checkpoint scored.
    """
    by_exp: dict[str, list[Target]] = {}
    for t in targets:
        by_exp.setdefault(t.experiment, []).append(t)
    queues: list[list[Target]] = []
    for exp_name in sorted(by_exp.keys()):
        bucket = sorted(by_exp[exp_name], key=lambda t: (-t.step_num, t.cfg))
        queues.append(bucket)
    ordered: List[Target] = []
    while any(queues):
        for q in queues:
            if q:
                ordered.append(q.pop(0))
    return ordered


def target_key(target: Target) -> str:
    return f"{target.experiment}/{target.step}/{target.cfg}"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _append_timing_record(target_dir: Path, rank: int, payload: dict) -> None:
    timing_dir = target_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    path = timing_dir / f"rank-{rank:05d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _merge_timing_summary(target_dir: Path) -> Path | None:
    timing_dir = target_dir / "timing"
    if not timing_dir.is_dir():
        return None
    records: list[dict] = []
    for path in sorted(timing_dir.glob("rank-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    records.append(item)

    by_kind: dict[str, list[dict]] = {}
    for item in records:
        by_kind.setdefault(str(item.get("kind") or "unknown"), []).append(item)

    summary = {
        "num_records": len(records),
        "records": records,
        "by_kind": {},
    }
    timing_fields = [
        "elapsed_sec",
        "module_import_elapsed_sec",
        "model_load_elapsed_sec",
        "task_run_elapsed_sec",
        "metric_compute_elapsed_sec",
        "summary_elapsed_sec",
        "barrier_elapsed_sec",
    ]
    timing_fields.extend(
        field
        for field in sorted({
            str(key)
            for item in records
            for key in item.keys()
            if str(key).endswith("_elapsed_sec")
        })
        if field not in timing_fields
    )
    for kind, items in sorted(by_kind.items()):
        field_stats = {}
        for field in timing_fields:
            values = [float(x.get(field, 0.0)) for x in items]
            field_stats[field] = {
                "sum": float(sum(values)),
                "max": float(max(values)) if values else 0.0,
                "min": float(min(values)) if values else 0.0,
            }
        elapsed = field_stats["elapsed_sec"]
        summary["by_kind"][kind] = {
            "count": len(items),
            "elapsed_sec_sum": elapsed["sum"],
            "elapsed_sec_max": elapsed["max"],
            "elapsed_sec_min": elapsed["min"],
            "timing_fields": field_stats,
            "statuses": {
                status: sum(1 for x in items if str(x.get("status")) == status)
                for status in sorted({str(x.get("status")) for x in items})
            },
        }

    out = target_dir / "eval_timing_summary.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return out


def _task_timing_payload(result: object) -> dict:
    if isinstance(result, dict):
        timing = result.get("timing")
        if isinstance(timing, dict):
            return dict(timing)
        return {k: v for k, v in result.items() if str(k).endswith("_elapsed_sec")}
    return {}


def select_kinds(args: argparse.Namespace, rank: int) -> List[str]:
    kinds_whitelist = [k.strip() for k in args.kinds.replace(",", " ").split() if k.strip()]
    extra_kinds = [k.strip() for k in args.extra_kinds.replace(",", " ").split() if k.strip()]
    kinds_blacklist = {k.strip() for k in args.skip_kinds.replace(",", " ").split() if k.strip()}
    unknown = (set(kinds_whitelist) | set(extra_kinds) | kinds_blacklist) - set(KIND_ORDER)
    if unknown:
        log(rank, f"WARNING: unknown kinds requested (ignored): {sorted(unknown)}; "
                  f"supported={KIND_ORDER}")
    base_order = KIND_ORDER if kinds_whitelist else DEFAULT_KIND_ORDER
    selected_kinds = [
        k for k in base_order
        if (not kinds_whitelist or k in kinds_whitelist) and k not in kinds_blacklist
    ]
    for kind in KIND_ORDER:
        if kind in extra_kinds and kind not in kinds_blacklist and kind not in selected_kinds:
            selected_kinds.append(kind)
    log(rank, f"Selected kinds (in order): {selected_kinds}"
              + (f"   skipped={sorted(kinds_blacklist & set(KIND_ORDER))}" if kinds_blacklist else ""))
    return selected_kinds


def parse_optional_metrics(raw: str) -> Set[str]:
    return {t.strip() for t in raw.replace(",", " ").split() if t.strip()}


def selected_metric_keys(args: argparse.Namespace, selected_kinds: List[str], rank: int) -> dict[str, List[str]]:
    optional_metrics = parse_optional_metrics(args.optional_metrics)
    known_optional = {"all", *KIND_ORDER}
    for keys in ALL_KIND_METRIC_KEYS.values():
        known_optional.update(keys)
    unknown = optional_metrics - known_optional
    if unknown:
        log(rank, f"WARNING: unknown optional metrics requested (ignored): {sorted(unknown)}")
        optional_metrics -= unknown
    by_kind = {
        kind: metric_keys_for_kind(kind, optional_metrics)
        for kind in selected_kinds
    }
    log(rank, f"Selected metric keys: {by_kind}")
    return by_kind


def _target_latest_direct_mtime(target: Target) -> float:
    latest = target.sample_dir.stat().st_mtime
    for p in target.sample_dir.iterdir():
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
    return latest


def _target_complete(
    args: argparse.Namespace,
    target: Target,
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> bool:
    target_dir = target_output_dir(args.eval_output_root.resolve(), target).resolve()
    manifest_file = target_dir / "metadata" / "manifest.json"
    manifest = load_manifest(manifest_file) if manifest_file.is_file() else None
    records = limit_records(manifest.get("records", []), args.limit) if manifest else []

    for kind in selected_kinds:
        if kind in DATASET_LEVEL_SUMMARY_KINDS:
            if not _summary_ready(target_dir, kind, manifest, metric_keys_by_kind.get(kind)):
                return False
            continue
        if not records:
            return False
        metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
        if not _summary_ready(target_dir, kind, manifest, metric_keys):
            return False
        for rec in records:
            if not already_done(target_dir, kind, rec["file_stem"], metric_keys):
                return False
    return True


def _expected_summary_categories(manifest: dict | None) -> set[str]:
    if not manifest:
        return set()
    categories = {
        str(rec.get("category") or "")
        for rec in manifest.get("records", [])
        if isinstance(rec, dict) and rec.get("category")
    }
    if categories:
        categories.add("all")
    return categories


def _summary_ready(
    target_dir: Path,
    kind: str,
    manifest: dict | None = None,
    metric_keys: List[str] | None = None,
) -> bool:
    summary_file = target_dir / "summary" / f"{kind}.json"
    if not summary_file.is_file() or summary_file.stat().st_size == 0:
        return False
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not all(k in data for k in ("num_success", "num_failed", "num_skipped")):
        return False
    if metric_keys is not None and set(data.get("scores", {}).keys()) != set(metric_keys):
        return False
    for key in metric_keys or []:
        if key not in data.get("scores", {}):
            return False
        if key not in data.get("num_success", {}):
            return False
        if key not in data.get("num_failed", {}):
            return False
        if key not in data.get("num_skipped", {}):
            return False
    expected = _expected_summary_categories(manifest)
    if expected:
        num_samples = data.get("num_samples")
        if not isinstance(num_samples, dict):
            return False
        if not expected.issubset(set(num_samples.keys())):
            return False
    return True


def _per_sample_complete(target_dir: Path, kind: str, manifest: dict, metric_keys: List[str]) -> bool:
    records = manifest.get("records", [])
    if not records:
        return False
    return all(already_done(target_dir, kind, rec["file_stem"], metric_keys) for rec in records)


def discover_targets_for_pass(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> List[Target]:
    if rank == 0:
        steps_filter = parse_steps_arg(args.steps)
        targets = discover_targets(
            sample_root=args.sample_root.resolve(),
            cfg_filter=args.cfg,
            experiments=args.experiments,
            steps_whitelist=steps_filter,
        )
        targets = filter_targets_max_ckpt(targets, args.max_ckpt_per_experiment)

        if args.watch and args.watch_min_age > 0:
            now = time.time()
            kept: List[Target] = []
            for t in targets:
                age = now - _target_latest_direct_mtime(t)
                if age >= args.watch_min_age:
                    kept.append(t)
                else:
                    log(rank, f"[watch] hold {t.label}: newest sample file age {age:.1f}s "
                              f"< --watch-min-age {args.watch_min_age:.1f}s")
            targets = kept

        if args.skip_completed:
            pending: List[Target] = []
            scan_workers = max(1, int(args.scan_workers))

            def _check_complete(t: Target) -> tuple[Target, bool, str | None]:
                try:
                    return t, _target_complete(args, t, selected_kinds, metric_keys_by_kind), None
                except Exception as exc:
                    return t, False, str(exc)

            if scan_workers > 1 and len(targets) > 1:
                workers = min(scan_workers, len(targets))
                log(rank, f"[discover] checking completion with {workers} scan workers "
                          f"over {len(targets)} target(s)")
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    checked = list(ex.map(_check_complete, targets))
            else:
                checked = [_check_complete(t) for t in targets]

            for t, complete, err in checked:
                if err:
                    log(rank, f"[discover] completion check failed for {t.label}: {err}; keep pending")
                if complete:
                    log(rank, f"[discover] skip completed target {t.label}")
                else:
                    pending.append(t)
            targets = pending

        if args.target_order == "interleave":
            targets = order_targets_interleave(targets)
        log(rank, f"Discovered {len(targets)} pending targets (order={args.target_order}):")
        for t in targets:
            log(rank, f"  - {t.label}  sample_dir={t.sample_dir}")
    else:
        targets = []
    return broadcast_object(targets, world_size, src=0)


def run_evaluation_pass(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    targets: List[Target],
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> int:
    # rank-0 pre-builds manifests; the other ranks wait at the barrier below.
    if rank == 0:
        for t in targets:
            try:
                ensure_manifest(
                    target=t,
                    eval_output_root=args.eval_output_root.resolve(),
                    build_script=args.build_manifest_script.resolve(),
                    valid_jsonl=args.valid_jsonl.resolve(),
                )
            except Exception as exc:
                log(rank, f"FAILED to build manifest for {t.label}: {exc}\n{traceback.format_exc()}")
    barrier(world_size)

    if args.dispatch_mode == "subtask":
        total_failures = run_subtask_mode(
            args, rank, local_rank, world_size, targets, selected_kinds, metric_keys_by_kind,
        )
    elif args.dispatch_mode == "kind-major-reuse":
        total_failures = run_kind_major_reuse_mode(
            args, rank, local_rank, world_size, targets, selected_kinds, metric_keys_by_kind,
        )
    elif args.dispatch_mode == "sample-major":
        total_failures = run_sample_major_mode(
            args, rank, local_rank, world_size, targets, selected_kinds, metric_keys_by_kind,
        )
    else:
        total_failures = run_data_parallel_mode(
            args, rank, local_rank, world_size, targets, selected_kinds, metric_keys_by_kind,
        )

    barrier(world_size)
    if rank == 0:
        for t in targets:
            tdir = target_output_dir(args.eval_output_root, t).resolve()
            if (tdir / "summary").is_dir():
                merge_all_metrics_summary(tdir)
            _merge_timing_summary(tdir)

    log(rank, f"Done. failures={total_failures}")
    barrier(world_size)
    return total_failures


def run_watch_loop(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> int:
    seen: Set[str] = set()
    total_failures = 0
    pass_idx = 0

    log(rank, f"[watch] enabled: interval={args.watch_interval}s "
              f"min_age={args.watch_min_age}s skip_existing={args.watch_skip_existing}")
    while True:
        pass_idx += 1
        log(rank, f"[watch] scan pass {pass_idx}")
        targets = discover_targets_for_pass(args, rank, world_size, selected_kinds, metric_keys_by_kind)

        if pass_idx == 1 and args.watch_skip_existing:
            seen.update(target_key(t) for t in targets)
            log(rank, f"[watch] skipped {len(targets)} startup targets; waiting for future targets")
            pending: List[Target] = []
        else:
            pending = [t for t in targets if target_key(t) not in seen]

        if pending:
            log(rank, f"[watch] evaluating {len(pending)} new target(s): "
                      f"{[target_key(t) for t in pending]}")
            total_failures += run_evaluation_pass(
                args, rank, local_rank, world_size, pending, selected_kinds, metric_keys_by_kind,
            )
            seen.update(target_key(t) for t in pending)
        else:
            log(rank, "[watch] no new pending targets")

        if args.watch_max_passes > 0 and pass_idx >= args.watch_max_passes:
            log(rank, f"[watch] reached --watch-max-passes={args.watch_max_passes}; exit")
            break

        barrier(world_size)
        sleep_s = max(0.0, float(args.watch_interval))
        log(rank, f"[watch] sleeping {sleep_s:.1f}s")
        time.sleep(sleep_s)

    return 0 if total_failures == 0 else 1


def main() -> int:
    args = parse_args()
    args.scan_workers = max(1, int(args.scan_workers))
    rank, local_rank, world_size = setup_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    log(rank, f"sample_root={args.sample_root}  eval_output_root={args.eval_output_root}")
    log(rank, f"world_size={world_size}  local_rank={local_rank}  cfg={args.cfg}  "
              f"limit={args.limit}  scan_workers={args.scan_workers}")

    selected_kinds = select_kinds(args, rank)
    metric_keys_by_kind = selected_metric_keys(args, selected_kinds, rank)
    if args.watch:
        return run_watch_loop(args, rank, local_rank, world_size, selected_kinds, metric_keys_by_kind)

    targets = discover_targets_for_pass(args, rank, world_size, selected_kinds, metric_keys_by_kind)
    if not targets:
        log(rank, "No targets discovered; exit.")
        return 0
    total_failures = run_evaluation_pass(
        args, rank, local_rank, world_size, targets, selected_kinds, metric_keys_by_kind,
    )
    return 0 if total_failures == 0 else 1


def _load_target_manifest(args: argparse.Namespace, t: Target) -> tuple[Path, dict | None]:
    tdir = target_output_dir(args.eval_output_root, t).resolve()
    manifest_file = tdir / "metadata" / "manifest.json"
    if not manifest_file.is_file():
        return tdir, None
    manifest = load_manifest(manifest_file)
    manifest["records"] = limit_records(manifest.get("records", []), args.limit)
    if not manifest["records"]:
        return tdir, None
    return tdir, manifest


def run_subtask_mode(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    targets: List[Target],
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> int:
    """Subtask-queue dispatcher.

    Build a flat list of (target, kind) subtasks ordered kind-major (all targets
    of kind0 first, then all targets of kind1, ...). After rank-strided slicing
    ``subtasks[rank::ws]`` each rank ends up with a diverse mix of kinds and
    targets -- different ranks therefore process different kinds simultaneously,
    so CPU-bound kinds no longer block GPU-bound ones.

    Each (target, kind) is fully owned by exactly one rank. We pass
    ``rank=0, world_size=1`` into ``run_task`` so the task's internal
    ``slice_for_rank`` returns the full record list and any ``all_gather_object``
    calls degenerate to a no-op. consolidate_summary for that (target, kind) is
    done by the owning rank immediately after the task finishes -- no global
    barrier needed mid-flight.
    """
    all_subtasks: List[tuple[Target, str]] = []
    for kind in selected_kinds:
        for t in targets:
            all_subtasks.append((t, kind))
    my_subtasks = all_subtasks[rank::max(1, world_size)]

    log(rank, f"[subtask] {len(my_subtasks)} subtasks on this rank "
              f"(total={len(all_subtasks)}, world_size={world_size})")
    if rank == 0:
        for i, (t, k) in enumerate(all_subtasks, 1):
            owner = (i - 1) % max(1, world_size)
            log(rank, f"  [{i:3d}/{len(all_subtasks)}] {t.label} :: {k}  -> rank{owner}")

    # Avoid re-reading the manifest JSON for every kind on the same target.
    manifest_cache: dict[Path, tuple[Path, dict | None]] = {}
    failures = 0
    for sub_idx, (t, kind) in enumerate(my_subtasks, 1):
        if t.sample_dir not in manifest_cache:
            manifest_cache[t.sample_dir] = _load_target_manifest(args, t)
        tdir, manifest = manifest_cache[t.sample_dir]
        if manifest is None:
            log(rank, f"SKIP [{sub_idx}/{len(my_subtasks)}] {t.label} :: {kind}: "
                      f"manifest missing or empty")
            continue
        metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])

        if args.skip_completed:
            if kind in DATASET_LEVEL_SUMMARY_KINDS:
                if _summary_ready(tdir, kind, manifest, metric_keys):
                    log(rank, f"SKIP [{sub_idx}/{len(my_subtasks)}] {t.label} :: {kind}: "
                              f"summary already exists")
                    now = time.time()
                    _append_timing_record(
                        tdir,
                        rank,
                        {
                            "target": t.label,
                            "kind": kind,
                            "dispatch_mode": "subtask",
                            "rank": rank,
                            "world_size": world_size,
                            "num_records": len(manifest["records"]),
                            "status": "skipped_summary_ready",
                            "started_at_unix": now,
                            "completed_at_unix": now,
                            "elapsed_sec": 0.0,
                        },
                    )
                    continue
            elif _per_sample_complete(tdir, kind, manifest, metric_keys):
                if _summary_ready(tdir, kind, manifest, metric_keys):
                    log(rank, f"SKIP [{sub_idx}/{len(my_subtasks)}] {t.label} :: {kind}: "
                              f"per-sample + summary already complete")
                    now = time.time()
                    _append_timing_record(
                        tdir,
                        rank,
                        {
                            "target": t.label,
                            "kind": kind,
                            "dispatch_mode": "subtask",
                            "rank": rank,
                            "world_size": world_size,
                            "num_records": len(manifest["records"]),
                            "status": "skipped_complete",
                            "started_at_unix": now,
                            "completed_at_unix": now,
                            "elapsed_sec": 0.0,
                        },
                    )
                    continue
                try:
                    log(rank, f"CONSOLIDATE [{sub_idx}/{len(my_subtasks)}] {t.label} :: {kind}: "
                              f"per-sample complete, summary missing")
                    started_at = time.time()
                    consolidate_summary(
                        target_dir=tdir,
                        kind=kind,
                        manifest=manifest,
                        metric_keys=metric_keys,
                        eligible_categories=KIND_ELIGIBLE_CATEGORIES.get(kind),
                    )
                    completed_at = time.time()
                    _append_timing_record(
                        tdir,
                        rank,
                        {
                            "target": t.label,
                            "kind": kind,
                            "dispatch_mode": "subtask",
                            "rank": rank,
                            "world_size": world_size,
                            "num_records": len(manifest["records"]),
                            "status": "consolidated_only",
                            "started_at_unix": started_at,
                            "completed_at_unix": completed_at,
                            "elapsed_sec": completed_at - started_at,
                        },
                    )
                except Exception as exc:
                    log(rank, f"  !! consolidate_summary failed {t.label} :: {kind}: {exc}")
                continue

        log(rank, f"==> [{sub_idx}/{len(my_subtasks)}] {t.label} :: {kind}  "
                  f"({len(manifest['records'])} records)")
        started_at = time.time()
        import_started_at = started_at
        import_completed_at = started_at
        task_started_at = started_at
        task_completed_at = started_at
        summary_started_at = None
        summary_completed_at = None
        task_result: object = None
        status = "success"
        try:
            import_started_at = time.time()
            run_task = get_run_task(kind)
            import_completed_at = time.time()
            task_started_at = time.time()
            task_result = run_task(
                rank=0,
                local_rank=local_rank,
                world_size=1,
                target_dir=tdir,
                manifest=manifest,
                skip_completed=args.skip_completed,
                metric_keys=metric_keys,
            )
            task_completed_at = time.time()
        except Exception as exc:
            task_completed_at = time.time()
            log(rank, f"  !! FAILED {t.label} :: {kind}: {exc}\n{traceback.format_exc()}")
            failures += 1
            status = "failed"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if kind not in DATASET_LEVEL_SUMMARY_KINDS:
            try:
                summary_started_at = time.time()
                consolidate_summary(
                    target_dir=tdir,
                    kind=kind,
                    manifest=manifest,
                    metric_keys=metric_keys,
                    eligible_categories=KIND_ELIGIBLE_CATEGORIES.get(kind),
                )
                summary_completed_at = time.time()
            except Exception as exc:
                summary_completed_at = time.time()
                log(rank, f"  !! consolidate_summary failed {t.label} :: {kind}: {exc}")
                if status == "success":
                    status = "summary_failed"
        completed_at = time.time()
        task_extra_timing = _task_timing_payload(task_result)
        task_run_elapsed_sec = task_completed_at - task_started_at
        model_load_elapsed_sec = float(task_extra_timing.get("model_load_elapsed_sec", 0.0))
        _append_timing_record(
            tdir,
            rank,
            {
                "target": t.label,
                "kind": kind,
                "dispatch_mode": "subtask",
                "rank": rank,
                "world_size": world_size,
                "num_records": len(manifest["records"]),
                "status": status,
                "started_at_unix": started_at,
                "completed_at_unix": completed_at,
                "elapsed_sec": completed_at - started_at,
                "module_import_elapsed_sec": import_completed_at - import_started_at,
                "task_run_elapsed_sec": task_run_elapsed_sec,
                "metric_compute_elapsed_sec": max(0.0, task_run_elapsed_sec - model_load_elapsed_sec),
                "summary_elapsed_sec": (
                    summary_completed_at - summary_started_at
                    if summary_started_at is not None and summary_completed_at is not None
                    else 0.0
                ),
                "barrier_elapsed_sec": 0.0,
                **task_extra_timing,
            },
        )

    log(rank, f"[subtask] this rank done, local failures = {failures}")
    return failures


def _chunks(items: list, size: int) -> list[list]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _pending_records_for_kind(
    target_dir: Path,
    kind: str,
    records: list[dict],
    metric_keys: List[str],
    skip_completed: bool,
) -> list[dict]:
    if not skip_completed:
        return list(records)
    return [
        rec for rec in records
        if not already_done(target_dir, kind, rec["file_stem"], metric_keys)
    ]


def _append_sample_major_timing(
    target_dir: Path,
    rank: int,
    target: Target,
    kind: str,
    world_size: int,
    status: str,
    started_at: float,
    completed_at: float,
    num_records: int,
    result: object = None,
    *,
    task_run_elapsed_sec: float | None = None,
    summary_elapsed_sec: float = 0.0,
    dispatch_mode: str = "sample-major",
) -> None:
    extra = _task_timing_payload(result)
    elapsed = max(0.0, completed_at - started_at)
    task_elapsed = elapsed if task_run_elapsed_sec is None else float(task_run_elapsed_sec)
    model_load_elapsed_sec = float(extra.get("model_load_elapsed_sec", 0.0))
    _append_timing_record(
        target_dir,
        rank,
        {
            "target": target.label,
            "kind": kind,
            "dispatch_mode": dispatch_mode,
            "rank": rank,
            "world_size": world_size,
            "num_records": int(num_records),
            "status": status,
            "started_at_unix": started_at,
            "completed_at_unix": completed_at,
            "elapsed_sec": elapsed,
            "module_import_elapsed_sec": 0.0,
            "task_run_elapsed_sec": task_elapsed,
            "metric_compute_elapsed_sec": max(0.0, task_elapsed - model_load_elapsed_sec),
            "summary_elapsed_sec": float(summary_elapsed_sec),
            "barrier_elapsed_sec": 0.0,
            **extra,
        },
    )


def _sample_major_chunk_size_for_kind(kind: str, default: int) -> int:
    if kind == "speech_wer":
        raw = os.environ.get("MY_EVAL_WER_BATCH_SIZE")
        if raw is not None:
            return max(1, int(raw))
        return max(default, 8)
    return max(1, int(default))


def _rank_has_pending_records_for_kind(
    target_dir: Path,
    kind: str,
    manifest: dict,
    metric_keys: List[str],
    rank: int,
    world_size: int,
    skip_completed: bool,
) -> bool:
    if skip_completed and kind in DATASET_LEVEL_SUMMARY_KINDS:
        if _summary_ready(target_dir, kind, manifest, metric_keys):
            return False
    records = list(manifest.get("records", []))
    if not records:
        return False
    my_records = slice_for_rank(records, rank, world_size)
    return bool(_pending_records_for_kind(target_dir, kind, my_records, metric_keys, skip_completed))


def _clear_cached_models_for_kind(kind: str, rank: int) -> None:
    try:
        cleared = clear_model_cache(kind)
        if cleared:
            log(rank, f"[kind-major-reuse] cleared {kind} caches: {', '.join(cleared)}")
    except Exception as exc:
        log(rank, f"[kind-major-reuse] WARNING: failed to clear caches for {kind}: {exc}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_kind_major_reuse_mode(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    targets: List[Target],
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> int:
    """Evaluate all targets kind-by-kind while reusing one kind's model cache.

    This keeps the current metric model resident across checkpoints in the same
    process, then explicitly clears that kind before moving to the next metric.
    It avoids repeated checkpoint-level model loads without retaining all metric
    models at once.
    """
    loaded_targets: list[tuple[Target, Path, dict]] = []
    for t in targets:
        tdir, manifest = _load_target_manifest(args, t)
        if manifest is None:
            log(rank, f"SKIP {t.label}: manifest missing or empty")
            continue
        loaded_targets.append((t, tdir, manifest))

    if not loaded_targets:
        return 0

    total_failures = 0
    log(rank, f"[kind-major-reuse] targets={len(loaded_targets)} kinds={len(selected_kinds)}")
    for kind_idx, kind in enumerate(selected_kinds, 1):
        metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
        log(rank, f"[kind-major-reuse] kind {kind_idx}/{len(selected_kinds)}: {kind}")

        pending_here = [
            (t, tdir, manifest)
            for t, tdir, manifest in loaded_targets
            if _rank_has_pending_records_for_kind(
                tdir, kind, manifest, metric_keys, rank, world_size, args.skip_completed
            )
        ]
        preload = get_preload_task(kind)
        if preload is not None and pending_here:
            preload_target, preload_tdir, _ = pending_here[0]
            started_at = time.time()
            result: object = None
            status = "preloaded"
            try:
                result = preload(rank=rank, local_rank=local_rank, metric_keys=metric_keys)
            except Exception as exc:
                status = "preload_failed"
                total_failures += 1
                log(rank, f"[kind-major-reuse] preload FAILED {kind}: {exc}\n{traceback.format_exc()}")
            completed_at = time.time()
            _append_sample_major_timing(
                preload_tdir,
                rank,
                preload_target,
                kind,
                world_size,
                status,
                started_at,
                completed_at,
                num_records=0,
                result=result,
                task_run_elapsed_sec=0.0,
                dispatch_mode="kind-major-reuse",
            )
        elif preload is not None and rank == 0:
            log(rank, f"[kind-major-reuse] skip preload {kind}: no pending records on this rank")

        barrier(world_size)

        for target_idx, (t, tdir, manifest) in enumerate(loaded_targets, 1):
            log(rank, f"==> [kind-major-reuse {target_idx}/{len(loaded_targets)}] "
                      f"{t.label} :: {kind} ({len(manifest['records'])} records)")
            started_at = time.time()
            import_started_at = started_at
            import_completed_at = started_at
            task_started_at = started_at
            task_completed_at = started_at
            summary_started_at = None
            summary_completed_at = None
            barrier_elapsed_sec = 0.0
            task_result: object = None
            status = "success"
            try:
                import_started_at = time.time()
                run_task = get_run_task(kind)
                import_completed_at = time.time()
                task_started_at = time.time()
                task_result = run_task(
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    target_dir=tdir,
                    manifest=manifest,
                    skip_completed=args.skip_completed,
                    metric_keys=metric_keys,
                    reuse_models=True,
                )
                task_completed_at = time.time()
            except Exception as exc:
                task_completed_at = time.time()
                log(rank, f"  !! FAILED kind={kind} on {t.label}: {exc}\n{traceback.format_exc()}")
                total_failures += 1
                status = "failed"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            barrier_started_at = time.time()
            barrier(world_size)
            barrier_elapsed_sec += time.time() - barrier_started_at

            if rank == 0 and kind not in DATASET_LEVEL_SUMMARY_KINDS:
                try:
                    summary_started_at = time.time()
                    consolidate_summary(
                        target_dir=tdir,
                        kind=kind,
                        manifest=manifest,
                        metric_keys=metric_keys,
                        eligible_categories=KIND_ELIGIBLE_CATEGORIES.get(kind),
                    )
                    summary_completed_at = time.time()
                except Exception as exc:
                    summary_completed_at = time.time()
                    log(rank, f"  !! consolidate_summary failed kind={kind}: {exc}")
                    if status == "success":
                        status = "summary_failed"

            barrier_started_at = time.time()
            barrier(world_size)
            barrier_elapsed_sec += time.time() - barrier_started_at

            completed_at = time.time()
            task_extra_timing = _task_timing_payload(task_result)
            task_run_elapsed_sec = task_completed_at - task_started_at
            model_load_elapsed_sec = float(task_extra_timing.get("model_load_elapsed_sec", 0.0))
            _append_timing_record(
                tdir,
                rank,
                {
                    "target": t.label,
                    "kind": kind,
                    "dispatch_mode": "kind-major-reuse",
                    "rank": rank,
                    "world_size": world_size,
                    "num_records": len(manifest["records"]),
                    "status": status,
                    "started_at_unix": started_at,
                    "completed_at_unix": completed_at,
                    "elapsed_sec": completed_at - started_at,
                    "module_import_elapsed_sec": import_completed_at - import_started_at,
                    "task_run_elapsed_sec": task_run_elapsed_sec,
                    "metric_compute_elapsed_sec": max(0.0, task_run_elapsed_sec - model_load_elapsed_sec),
                    "summary_elapsed_sec": (
                        summary_completed_at - summary_started_at
                        if summary_started_at is not None and summary_completed_at is not None
                        else 0.0
                    ),
                    "barrier_elapsed_sec": barrier_elapsed_sec,
                    **task_extra_timing,
                },
            )

        barrier(world_size)
        _clear_cached_models_for_kind(kind, rank)
        barrier(world_size)

    return total_failures


def run_sample_major_mode(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    targets: List[Target],
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> int:
    """Experimental sample-major dispatcher with rank-local resident models.

    Each rank owns ``records[rank::world_size]`` for a target. For non-dataset
    metrics we run this rank's sample chunks through each selected kind with
    ``reuse_models=True`` so task modules keep their heavy models in module-level
    caches. Dataset-level metrics such as audio_is still run once per target at
    the end because their summary requires all ranks to participate in a single
    collective.
    """
    chunk_size = max(1, int(args.sample_major_chunk_size))
    failures = 0
    log(rank, f"[sample-major] chunk_size={chunk_size} selected_kinds={selected_kinds}")

    for target_idx, t in enumerate(targets, 1):
        tdir, manifest = _load_target_manifest(args, t)
        if manifest is None:
            log(rank, f"SKIP {t.label}: manifest missing or empty")
            continue
        records = list(manifest.get("records", []))
        my_records = slice_for_rank(records, rank, world_size)
        log(rank, f"==> [sample-major {target_idx}/{len(targets)}] {t.label} "
                  f"my_records={len(my_records)}/{len(records)}")

        sample_kinds = [k for k in selected_kinds if k not in DATASET_LEVEL_SUMMARY_KINDS]
        dataset_kinds = [k for k in selected_kinds if k in DATASET_LEVEL_SUMMARY_KINDS]
        active_sample_kinds: list[str] = []

        for kind in sample_kinds:
            metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
            pending = _pending_records_for_kind(tdir, kind, my_records, metric_keys, args.skip_completed)
            if pending:
                active_sample_kinds.append(kind)
            elif rank == 0:
                log(rank, f"[sample-major] skip preload {t.label} :: {kind}: no pending records on this rank")

        distributed_sample_kinds: list[str] = []
        for kind in sample_kinds:
            if kind != "lip_sync":
                continue
            metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
            pending_global = _pending_records_for_kind(tdir, kind, records, metric_keys, args.skip_completed)
            if pending_global:
                distributed_sample_kinds.append(kind)
                if kind not in active_sample_kinds:
                    active_sample_kinds.append(kind)

        # Preload all selected models that this rank will actually use. The
        # first normal run_task call would also populate the cache, but doing it
        # explicitly separates load time from metric compute in the timing log.
        for kind in active_sample_kinds + dataset_kinds:
            metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
            should_preload = True
            if kind in DATASET_LEVEL_SUMMARY_KINDS:
                dataset_pending = _pending_records_for_kind(
                    tdir, kind, my_records, metric_keys, args.skip_completed
                )
                should_preload = bool(dataset_pending)
            if not should_preload:
                continue
            preload = get_preload_task(kind)
            if preload is None:
                continue
            started_at = time.time()
            result: object = None
            status = "preloaded"
            try:
                result = preload(rank=rank, local_rank=local_rank, metric_keys=metric_keys)
            except Exception as exc:
                status = "preload_failed"
                failures += 1
                log(rank, f"[sample-major] preload FAILED {t.label} :: {kind}: "
                          f"{exc}\n{traceback.format_exc()}")
                if kind in active_sample_kinds:
                    active_sample_kinds.remove(kind)
            completed_at = time.time()
            _append_sample_major_timing(
                tdir,
                rank,
                t,
                kind,
                world_size,
                status,
                started_at,
                completed_at,
                num_records=0,
                result=result,
                task_run_elapsed_sec=0.0,
            )

        prewarm_started_at = time.time()
        prewarm_result = prewarm_preprocess_records(
            records=my_records,
            kinds=active_sample_kinds + dataset_kinds,
            rank=rank,
        )
        prewarm_completed_at = time.time()
        if prewarm_result.get("elapsed_sec", 0.0) > 0:
            _append_timing_record(
                tdir,
                rank,
                {
                    "target": t.label,
                    "kind": "preprocess",
                    "dispatch_mode": "sample-major",
                    "rank": rank,
                    "world_size": world_size,
                    "num_records": len(my_records),
                    "status": "success" if not prewarm_result.get("num_errors") else "completed_with_errors",
                    "started_at_unix": prewarm_started_at,
                    "completed_at_unix": prewarm_completed_at,
                    "elapsed_sec": prewarm_completed_at - prewarm_started_at,
                    "module_import_elapsed_sec": 0.0,
                    "model_load_elapsed_sec": 0.0,
                    "task_run_elapsed_sec": prewarm_completed_at - prewarm_started_at,
                    "metric_compute_elapsed_sec": 0.0,
                    "summary_elapsed_sec": 0.0,
                    "barrier_elapsed_sec": 0.0,
                    "preprocess_prewarm_elapsed_sec": float(prewarm_result.get("elapsed_sec", 0.0)),
                    "preprocess_workers": int(prewarm_result.get("workers", 0)),
                    "preprocess_num_errors": int(prewarm_result.get("num_errors", 0)),
                    "preprocess_decode_video": bool(prewarm_result.get("decode_video", False)),
                },
            )

        if active_sample_kinds:
            log(rank, f"[sample-major] active sample kinds: {active_sample_kinds}")
        batched_sample_kinds = [k for k in active_sample_kinds if k == "speech_wer"]
        distributed_sample_kind_set = set(distributed_sample_kinds)
        generic_sample_kinds = [
            k for k in active_sample_kinds
            if k not in batched_sample_kinds and k not in distributed_sample_kind_set
        ]
        for chunk_idx, chunk in enumerate(_chunks(my_records, chunk_size), 1):
            for kind in generic_sample_kinds:
                metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
                pending_chunk = _pending_records_for_kind(
                    tdir, kind, chunk, metric_keys, args.skip_completed
                )
                if not pending_chunk:
                    continue
                run_task = get_run_task(kind)
                chunk_manifest = dict(manifest)
                chunk_manifest["records"] = list(pending_chunk)
                started_at = time.time()
                task_result: object = None
                status = "success"
                try:
                    task_result = run_task(
                        rank=rank,
                        local_rank=local_rank,
                        world_size=1,
                        target_dir=tdir,
                        manifest=chunk_manifest,
                        skip_completed=args.skip_completed,
                        metric_keys=metric_keys,
                        reuse_models=True,
                    )
                except Exception as exc:
                    status = "failed"
                    failures += 1
                    log(rank, f"[sample-major] FAILED {t.label} :: {kind} "
                              f"chunk={chunk_idx}: {exc}\n{traceback.format_exc()}")
                completed_at = time.time()
                _append_sample_major_timing(
                    tdir,
                    rank,
                    t,
                    kind,
                    world_size,
                    status,
                    started_at,
                    completed_at,
                    num_records=len(pending_chunk),
                    result=task_result,
                )
            if chunk_idx % 10 == 0:
                log(rank, f"[sample-major] {t.label} chunks {chunk_idx}/"
                          f"{max(1, (len(my_records) + chunk_size - 1) // chunk_size)}")

        for kind in batched_sample_kinds:
            kind_chunk_size = _sample_major_chunk_size_for_kind(kind, chunk_size)
            metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
            log(rank, f"[sample-major] {kind} uses metric chunk_size={kind_chunk_size}")
            for chunk_idx, chunk in enumerate(_chunks(my_records, kind_chunk_size), 1):
                pending_chunk = _pending_records_for_kind(
                    tdir, kind, chunk, metric_keys, args.skip_completed
                )
                if not pending_chunk:
                    continue
                run_task = get_run_task(kind)
                chunk_manifest = dict(manifest)
                chunk_manifest["records"] = list(pending_chunk)
                started_at = time.time()
                task_result: object = None
                status = "success"
                try:
                    task_result = run_task(
                        rank=rank,
                        local_rank=local_rank,
                        world_size=1,
                        target_dir=tdir,
                        manifest=chunk_manifest,
                        skip_completed=args.skip_completed,
                        metric_keys=metric_keys,
                        reuse_models=True,
                    )
                except Exception as exc:
                    status = "failed"
                    failures += 1
                    log(rank, f"[sample-major] FAILED {t.label} :: {kind} "
                              f"chunk={chunk_idx}: {exc}\n{traceback.format_exc()}")
                completed_at = time.time()
                _append_sample_major_timing(
                    tdir,
                    rank,
                    t,
                    kind,
                    world_size,
                    status,
                    started_at,
                    completed_at,
                    num_records=len(pending_chunk),
                    result=task_result,
                )
                if chunk_idx % 10 == 0:
                    log(rank, f"[sample-major] {t.label} {kind} chunks {chunk_idx}/"
                              f"{max(1, (len(my_records) + kind_chunk_size - 1) // kind_chunk_size)}")

        for kind in distributed_sample_kinds:
            metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
            pending_global = _pending_records_for_kind(tdir, kind, records, metric_keys, args.skip_completed)
            if not pending_global:
                continue
            run_task = get_run_task(kind)
            started_at = time.time()
            task_result: object = None
            status = "success"
            try:
                task_result = run_task(
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    target_dir=tdir,
                    manifest=manifest,
                    skip_completed=args.skip_completed,
                    metric_keys=metric_keys,
                    reuse_models=True,
                )
            except Exception as exc:
                status = "failed"
                failures += 1
                log(rank, f"[sample-major] FAILED distributed kind {t.label} :: {kind}: "
                          f"{exc}\n{traceback.format_exc()}")
            completed_at = time.time()
            _append_sample_major_timing(
                tdir,
                rank,
                t,
                kind,
                world_size,
                status,
                started_at,
                completed_at,
                num_records=len(pending_global),
                result=task_result,
            )

        # Dataset-level metrics need one full-manifest collective after all
        # per-sample work for this target. audio_is uses this to write summary.
        for kind in dataset_kinds:
            metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
            if args.skip_completed and _summary_ready(tdir, kind, manifest, metric_keys):
                now = time.time()
                _append_sample_major_timing(
                    tdir, rank, t, kind, world_size, "skipped_summary_ready",
                    now, now, num_records=len(records), task_run_elapsed_sec=0.0
                )
                continue
            run_task = get_run_task(kind)
            started_at = time.time()
            task_result = None
            status = "success"
            try:
                task_result = run_task(
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    target_dir=tdir,
                    manifest=manifest,
                    skip_completed=args.skip_completed,
                    metric_keys=metric_keys,
                    reuse_models=True,
                )
            except Exception as exc:
                status = "failed"
                failures += 1
                log(rank, f"[sample-major] FAILED dataset kind {t.label} :: {kind}: "
                          f"{exc}\n{traceback.format_exc()}")
            completed_at = time.time()
            _append_sample_major_timing(
                tdir,
                rank,
                t,
                kind,
                world_size,
                status,
                started_at,
                completed_at,
                num_records=len(my_records),
                result=task_result,
            )

        barrier_started_at = time.time()
        barrier(world_size)
        barrier_elapsed_sec = time.time() - barrier_started_at

        if rank == 0:
            for kind in sample_kinds:
                metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
                summary_started_at = time.time()
                status = "summary_success"
                try:
                    consolidate_summary(
                        target_dir=tdir,
                        kind=kind,
                        manifest=manifest,
                        metric_keys=metric_keys,
                        eligible_categories=KIND_ELIGIBLE_CATEGORIES.get(kind),
                    )
                except Exception as exc:
                    status = "summary_failed"
                    failures += 1
                    log(rank, f"[sample-major] consolidate_summary failed "
                              f"{t.label} :: {kind}: {exc}")
                summary_completed_at = time.time()
                _append_sample_major_timing(
                    tdir,
                    rank,
                    t,
                    kind,
                    world_size,
                    status,
                    summary_started_at,
                    summary_completed_at,
                    num_records=len(records),
                    task_run_elapsed_sec=0.0,
                    summary_elapsed_sec=summary_completed_at - summary_started_at,
                )
        if barrier_elapsed_sec > 0:
            log(rank, f"[sample-major] {t.label} final barrier {barrier_elapsed_sec:.3f}s")
        barrier(world_size)

    log(rank, f"[sample-major] this rank done, local failures = {failures}")
    return failures


def run_data_parallel_mode(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    targets: List[Target],
    selected_kinds: List[str],
    metric_keys_by_kind: dict[str, List[str]],
) -> int:
    """All ranks cooperate on the same (target, kind) by slicing samples.

    This is the original behaviour. Each kind has a global barrier before the
    next kind starts; rank 0 consolidates summaries after each kind. Useful
    when there is only one target and you want every kind sample-sharded across
    all GPUs.
    """
    total_failures = 0
    for t in targets:
        tdir, manifest = _load_target_manifest(args, t)
        if manifest is None:
            log(rank, f"SKIP {t.label}: manifest missing or empty")
            continue
        log(rank, f"==> {t.label}  ({len(manifest['records'])} records)")

        if _env_flag("MY_EVAL_DATA_PARALLEL_PREWARM", False):
            prewarm_started_at = time.time()
            my_records = slice_for_rank(list(manifest.get("records", [])), rank, world_size)
            prewarm_result = prewarm_preprocess_records(
                records=my_records,
                kinds=selected_kinds,
                rank=rank,
            )
            prewarm_completed_at = time.time()
            barrier_started_at = time.time()
            barrier(world_size)
            barrier_elapsed_sec = time.time() - barrier_started_at
            if prewarm_result.get("elapsed_sec", 0.0) > 0 or prewarm_result.get("num_errors", 0):
                _append_timing_record(
                    tdir,
                    rank,
                    {
                        "target": t.label,
                        "kind": "preprocess",
                        "dispatch_mode": "data-parallel",
                        "rank": rank,
                        "world_size": world_size,
                        "num_records": len(my_records),
                        "status": (
                            "success"
                            if not prewarm_result.get("num_errors")
                            else "completed_with_errors"
                        ),
                        "started_at_unix": prewarm_started_at,
                        "completed_at_unix": prewarm_completed_at,
                        "elapsed_sec": prewarm_completed_at - prewarm_started_at,
                        "module_import_elapsed_sec": 0.0,
                        "model_load_elapsed_sec": 0.0,
                        "task_run_elapsed_sec": prewarm_completed_at - prewarm_started_at,
                        "metric_compute_elapsed_sec": 0.0,
                        "summary_elapsed_sec": 0.0,
                        "barrier_elapsed_sec": barrier_elapsed_sec,
                        "preprocess_prewarm_elapsed_sec": float(prewarm_result.get("elapsed_sec", 0.0)),
                        "preprocess_workers": int(prewarm_result.get("workers", 0)),
                        "preprocess_backend": str(prewarm_result.get("backend", "")),
                        "preprocess_num_errors": int(prewarm_result.get("num_errors", 0)),
                        "preprocess_decode_video": bool(prewarm_result.get("decode_video", False)),
                        "preprocess_audio_srs": list(prewarm_result.get("audio_srs", [])),
                    },
                )

        for kind in selected_kinds:
            metric_keys = metric_keys_by_kind.get(kind, KIND_METRIC_KEYS[kind])
            log(rank, f"  -- kind={kind}")
            started_at = time.time()
            import_started_at = started_at
            import_completed_at = started_at
            task_started_at = started_at
            task_completed_at = started_at
            summary_started_at = None
            summary_completed_at = None
            barrier_elapsed_sec = 0.0
            task_result: object = None
            status = "success"
            try:
                import_started_at = time.time()
                run_task = get_run_task(kind)
                import_completed_at = time.time()
                task_started_at = time.time()
                task_result = run_task(
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    target_dir=tdir,
                    manifest=manifest,
                    skip_completed=args.skip_completed,
                    metric_keys=metric_keys,
                )
                task_completed_at = time.time()
            except Exception as exc:
                task_completed_at = time.time()
                log(rank, f"  !! FAILED kind={kind} on {t.label}: {exc}\n{traceback.format_exc()}")
                total_failures += 1
                status = "failed"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            barrier_started_at = time.time()
            barrier(world_size)
            barrier_elapsed_sec += time.time() - barrier_started_at

            if rank == 0 and kind not in DATASET_LEVEL_SUMMARY_KINDS:
                try:
                    summary_started_at = time.time()
                    consolidate_summary(
                        target_dir=tdir,
                        kind=kind,
                        manifest=manifest,
                        metric_keys=metric_keys,
                        eligible_categories=KIND_ELIGIBLE_CATEGORIES.get(kind),
                    )
                    summary_completed_at = time.time()
                except Exception as exc:
                    summary_completed_at = time.time()
                    log(rank, f"  !! consolidate_summary failed kind={kind}: {exc}")
                    if status == "success":
                        status = "summary_failed"
            barrier_started_at = time.time()
            barrier(world_size)
            barrier_elapsed_sec += time.time() - barrier_started_at
            completed_at = time.time()
            task_extra_timing = _task_timing_payload(task_result)
            task_run_elapsed_sec = task_completed_at - task_started_at
            model_load_elapsed_sec = float(task_extra_timing.get("model_load_elapsed_sec", 0.0))
            _append_timing_record(
                tdir,
                rank,
                {
                    "target": t.label,
                    "kind": kind,
                    "dispatch_mode": "data-parallel",
                    "rank": rank,
                    "world_size": world_size,
                    "num_records": len(manifest["records"]),
                    "status": status,
                    "started_at_unix": started_at,
                    "completed_at_unix": completed_at,
                    "elapsed_sec": completed_at - started_at,
                    "module_import_elapsed_sec": import_completed_at - import_started_at,
                    "task_run_elapsed_sec": task_run_elapsed_sec,
                    "metric_compute_elapsed_sec": max(0.0, task_run_elapsed_sec - model_load_elapsed_sec),
                    "summary_elapsed_sec": (
                        summary_completed_at - summary_started_at
                        if summary_started_at is not None and summary_completed_at is not None
                        else 0.0
                    ),
                    "barrier_elapsed_sec": barrier_elapsed_sec,
                    **task_extra_timing,
                },
            )
    return total_failures


if __name__ == "__main__":
    sys.exit(main())
