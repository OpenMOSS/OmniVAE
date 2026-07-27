"""Inception Score over the PANNs Cnn14 32 kHz softmax distribution.

IS is a dataset-level metric: per-sample we only collect the softmax probability
vector; the actual IS is computed by rank 0 after every rank finishes its slice.
Rank 0 broadcasts a per-category IS to ``summary/audio_is.json`` (covering the
manifest categories plus ``all``).

The per-sample JSON stores the prob vector so we can recompute IS for arbitrary
subsets after the run without rerunning the PANNs forward.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
AUDIO_IS_CLAP_CODE_DIR = REPO_ROOT / "generation" / "evaluation" / "metrics" / "audio_is_clap"
AUDIO_IS_CLAP_ASSET_DIR = Path(
    os.environ.get("MY_EVAL_AUDIO_IS_CLAP_DIR", str(AUDIO_IS_CLAP_CODE_DIR))
).expanduser()
os.environ.setdefault("MY_EVAL_PANNS_HOME", str(AUDIO_IS_CLAP_ASSET_DIR / "pann_home"))


def _ensure_pythonpath() -> None:
    if str(AUDIO_IS_CLAP_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(AUDIO_IS_CLAP_CODE_DIR))


_ensure_pythonpath()

from my_eval.utils.audio_video import load_wav_mono
from my_eval.utils.distributed import all_gather_object, log, slice_for_rank
from my_eval.utils.io_utils import (
    already_done,
    per_sample_path,
    summary_path,
    write_per_sample,
)


_SR = 32000
_MODEL_CACHE: Dict[str, Any] = {}


def _ordered_count_buckets(*maps: Dict[str, Any]) -> List[str]:
    preferred = ["set1", "set2", "set3", "set3-large", "set3-medium-large", "all"]
    seen = set()
    for m in maps:
        seen.update(m.keys())
    return [b for b in preferred if b in seen] + sorted(b for b in seen if b not in preferred)


def _load_panns():
    import IS
    extractor = IS.Cnn14Extractor()
    return extractor


def _bind_device(local_rank: int) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def _get_panns(local_rank: int, *, reuse_models: bool) -> tuple[Any, float]:
    _bind_device(local_rank)
    key = f"local_rank={local_rank}|cuda={torch.cuda.is_available()}"
    if reuse_models and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], 0.0
    started_at = time.time()
    extractor = _load_panns()
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = extractor
    return extractor, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    _, elapsed = _get_panns(local_rank, reuse_models=True)
    log(rank, f"[audio_is] preload complete model_load={elapsed:.3f}s")
    return {"model_load_elapsed_sec": elapsed}


def run_task(
    rank: int,
    local_rank: int,
    world_size: int,
    target_dir: Path,
    manifest: Dict[str, Any],
    skip_completed: bool = True,
    metric_keys: list[str] | None = None,
    reuse_models: bool = False,
    **_: Any,
) -> Dict[str, float]:
    model_load_elapsed_sec = 0.0
    metric_keys = metric_keys or ["IS"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[audio_is] my_records={len(my_records)}/{len(records)}")

    _bind_device(local_rank)

    # Only load the model if there is something to do (skip_completed may render
    # this rank empty).
    pending = []
    for rec in my_records:
        if skip_completed and already_done(target_dir, "audio_is", rec["file_stem"], metric_keys):
            continue
        pending.append(rec)

    extractor = None
    if pending:
        extractor, model_load_elapsed_sec = _get_panns(local_rank, reuse_models=reuse_models)

    for idx, rec in enumerate(pending):
        stem = rec["file_stem"]
        payload: Dict[str, Any] = {
            "IS": None,  # filled at the dataset level
            "prob": None,
            "category": rec.get("category", ""),
            "audio_path": rec["audio_path"],
        }
        try:
            audio, _ = load_wav_mono(rec["audio_path"], _SR)
            wav = torch.tensor(audio, dtype=torch.float32)
            prob = extractor(wav).squeeze().detach().cpu().numpy()
            payload["prob"] = prob.tolist()
        except Exception as exc:
            log(rank, f"[audio_is] failed for {stem}: {exc}")
        write_per_sample(target_dir, "audio_is", stem, payload)
        if (idx + 1) % 100 == 0:
            log(rank, f"  audio_is {idx + 1}/{len(pending)}")

    if extractor is not None and not reuse_models:
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Each rank collects (stem, category) pairs for its records so rank 0 can
    # walk the disk afterwards (per_sample JSONs hold the actual probs).
    my_meta = [(rec["file_stem"], rec.get("category", "")) for rec in my_records]
    gathered = all_gather_object(my_meta, world_size)

    if rank != 0:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    # Rank 0: walk per_sample/audio_is/*.json to recover probs and run IS.
    from IS import calculate_inception_score  # type: ignore
    flat: List[tuple[str, str]] = [m for chunk in gathered for m in chunk]
    by_cat: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_cat_all: List[np.ndarray] = []
    expected_counts: Dict[str, int] = defaultdict(int)
    success_counts: Dict[str, int] = defaultdict(int)
    for stem, cat in flat:
        expected_counts[cat] += 1
        expected_counts["all"] += 1
        p = per_sample_path(target_dir, "audio_is", stem)
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        prob = data.get("prob")
        if not prob:
            continue
        arr = np.asarray(prob, dtype=np.float32)
        by_cat[cat].append(arr)
        by_cat_all.append(arr)
        success_counts[cat] += 1
        success_counts["all"] += 1

    scores_by_cat: Dict[str, float] = {}
    for cat, probs in by_cat.items():
        if probs:
            scores_by_cat[cat] = float(calculate_inception_score(np.vstack(probs)))
    if by_cat_all:
        scores_by_cat["all"] = float(calculate_inception_score(np.vstack(by_cat_all)))

    buckets = _ordered_count_buckets(expected_counts, success_counts)
    failed_counts = {
        b: int(expected_counts.get(b, 0) - success_counts.get(b, 0))
        for b in buckets
    }

    summary = {
        "metric_kind": "audio_is",
        "scores": {"IS": scores_by_cat},
        "num_samples": {b: int(expected_counts.get(b, 0)) for b in buckets},
        "num_success": {"IS": {b: int(success_counts.get(b, 0)) for b in buckets}},
        "num_failed": {"IS": failed_counts},
        "num_skipped": {"IS": {b: 0 for b in buckets}},
    }
    out = summary_path(target_dir, "audio_is")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(rank, f"[audio_is] summary written to {out}")
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
