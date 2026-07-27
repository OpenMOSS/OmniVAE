"""Lip-sync metrics (LSE-D + LSE-C).

Driven by ``generation/evaluation/models/wav2lip/evaluation/syncnet_python/eval_lip_sync.py``.
The upstream pipeline expects an mp4 with an embedded audio track, so we mux on
the fly using the rank-owned tmp directory. We only run the SyncNet pipeline for
speech categories (``set3`` and its size variants); other records get a
placeholder with NaN scores so the summary aggregator can still differentiate
them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[5]
SYNCNET_DIR = REPO_ROOT / "generation" / "evaluation" / "models" / "wav2lip" / "evaluation" / "syncnet_python"


def _ensure_pythonpath() -> None:
    if str(SYNCNET_DIR) not in sys.path:
        sys.path.insert(0, str(SYNCNET_DIR))


_ensure_pythonpath()

from my_eval.utils.audio_video import mux_av, rank_tmp_dir
from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample
from my_eval.utils.quiet import fd_redirect, open_rank_log


def _cuda_visible_device_for_local_rank(local_rank: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if 0 <= local_rank < len(devices):
            return devices[local_rank]
    return str(local_rank)


def _evaluate_one(muxed_path: str, work_root: Path) -> Dict[str, Any]:
    from eval_lip_sync import evaluate_one_video  # type: ignore

    batch_size = max(1, int(os.environ.get("MY_EVAL_LIPSYNC_BATCH_SIZE", "20")))
    per_person, mean_d, mean_c, max_c, min_d = evaluate_one_video(
        video_path=muxed_path,
        tmp_work_root=str(work_root),
        batch_size=batch_size,
        facedet_scale=0.25,
        min_face_size=100,
        conf_th=0.9,
        use_multi_scale=False,
    )
    if per_person is None:
        return {"LSE-D": float("nan"), "LSE-C": float("nan"), "n_persons": 0}
    out: Dict[str, Any] = dict(per_person)
    out["LSE-D"] = float(mean_d)
    out["LSE-C"] = float(mean_c)
    if max_c is not None:
        out["max_C"] = float(max_c)
    if min_d is not None:
        out["min_D"] = float(min_d)
    out["n_persons"] = int(len(per_person) // 2)
    return out


def run_task(
    rank: int,
    local_rank: int,
    world_size: int,
    target_dir: Path,
    manifest: Dict[str, Any],
    skip_completed: bool = True,
    metric_keys: tuple[str, ...] | list[str] | None = None,
    eligible_categories: tuple[str, ...] = ("set3", "set3-large", "set3-medium-large"),
    **_: Any,
) -> Dict[str, float]:
    metric_keys = list(metric_keys or ["LSE-C"])
    metric_key_set = set(metric_keys)
    records = list(manifest.get("records", []))
    eligible_set = set(eligible_categories)
    eligible_records = [rec for rec in records if rec.get("category", "") in eligible_set]
    skipped_records = [rec for rec in records if rec.get("category", "") not in eligible_set]
    my_scored_records = slice_for_rank(eligible_records, rank, world_size)
    my_skipped_records = slice_for_rank(skipped_records, rank, world_size)
    log(
        rank,
        f"[lip_sync] scored_records={len(my_scored_records)}/{len(eligible_records)} "
        f"skipped_records={len(my_skipped_records)}/{len(skipped_records)} "
        f"world_size={world_size} eligible={eligible_categories}",
    )
    if not my_scored_records and not my_skipped_records:
        return {"model_load_elapsed_sec": 0.0}

    work_root = rank_tmp_dir(target_dir, "lip_sync", rank)
    # SyncNet uses bare .cuda() internally, so bind the current process to this
    # rank's CUDA device before constructing/evaluating it.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    os.environ["MY_EVAL_LIPSYNC_PIPELINE_CUDA_VISIBLE_DEVICES"] = _cuda_visible_device_for_local_rank(local_rank)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except Exception:
        pass

    # Per-rank log file collects all the ffmpeg / S3FD chatter so the tty stays
    # readable. Tail this if a sample fails and you want the raw output.
    rank_log_path = open_rank_log(target_dir, "lip_sync", rank)

    n_scored = 0
    n_skipped_cat = 0
    n_failed = 0

    with rank_log_path.open("a", encoding="utf-8") as rank_log:
        rank_log.write(f"\n========== lip_sync rank{rank} new run "
                       f"(scored={len(my_scored_records)} skipped={len(my_skipped_records)}) ==========\n")
        rank_log.flush()

        for rec in my_skipped_records:
            stem = rec["file_stem"]
            if skip_completed and already_done(target_dir, "lip_sync", stem, metric_keys):
                continue
            cat = rec.get("category", "")
            payload: Dict[str, Any] = {
                "LSE-C": float("nan"),
                "category": cat,
            }
            if "LSE-D" in metric_key_set:
                payload["LSE-D"] = float("nan")
            write_per_sample(target_dir, "lip_sync", stem, payload)
            n_skipped_cat += 1

        for idx, rec in enumerate(my_scored_records):
            stem = rec["file_stem"]
            if skip_completed and already_done(target_dir, "lip_sync", stem, metric_keys):
                continue
            cat = rec.get("category", "")
            payload: Dict[str, Any] = {
                "LSE-C": float("nan"),
                "category": cat,
            }
            if "LSE-D" in metric_key_set:
                payload["LSE-D"] = float("nan")
            rank_log.write(f"\n---- {stem} ----\n")
            rank_log.flush()
            try:
                muxed = work_root / f"{stem}.av.mp4"
                mux_av(rec["video_path"], rec["audio_path"], str(muxed))
                with fd_redirect(rank_log):
                    result = _evaluate_one(str(muxed), work_root)
                if "LSE-D" not in metric_key_set:
                    result.pop("LSE-D", None)
                payload.update(result)
                n_scored += 1
            except Exception as exc:
                log(rank, f"[lip_sync] failed for {stem}: {exc}  (see {rank_log_path})")
                n_failed += 1
            write_per_sample(target_dir, "lip_sync", stem, payload)
            if (idx + 1) % 10 == 0:
                log(rank, f"  lip_sync {idx + 1}/{len(my_scored_records)} "
                          f"(scored={n_scored} skipped_cat={n_skipped_cat} failed={n_failed})")

    log(rank, f"[lip_sync] done: scored={n_scored} skipped_cat={n_skipped_cat} "
              f"failed={n_failed}, log={rank_log_path}")
    return {
        "model_load_elapsed_sec": 0.0,
        "lip_sync_num_scored_records": float(n_scored),
        "lip_sync_num_skipped_category": float(n_skipped_cat),
        "lip_sync_num_failed": float(n_failed),
    }
