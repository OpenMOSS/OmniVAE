"""Speech Word Error Rate (WER) against Verse-Bench speech_prompt text."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
VERSE_BENCH_DIR = REPO_ROOT / "generation" / "evaluation" / "verse_bench"
DEFAULT_VERSE_MODELS = VERSE_BENCH_DIR / "models"


def _ensure_pythonpath() -> None:
    p = str(VERSE_BENCH_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_pythonpath()

from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample
from my_eval.utils.versebench_refs import resolve_speech_text

_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_models_dir() -> str:
    explicit = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if explicit:
        return str(Path(explicit).expanduser())
    return str(DEFAULT_VERSE_MODELS)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_wer(models_dir: str, device: str):
    from wer.wer_inferencer import WERInferencer  # type: ignore
    print(f"[speech_wer] loading WER ASR model on {device}", flush=True)
    return WERInferencer(models_dir, device=device)


def _resolve_device(local_rank: int) -> str:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    os.environ["MY_EVAL_WER_DEVICE"] = device
    return device


def _get_wer(models_dir: str, device: str, *, reuse_models: bool) -> tuple[Any, float]:
    key = f"{models_dir}|{device}"
    if reuse_models and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], 0.0
    started_at = time.time()
    model = _load_wer(models_dir, device=device)
    _cuda_sync()
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = model
    return model, elapsed


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    device = _resolve_device(local_rank)
    _, elapsed = _get_wer(_resolve_models_dir(), device, reuse_models=True)
    log(rank, f"[speech_wer] preload complete model_load={elapsed:.3f}s")
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
    metric_keys = metric_keys or ["WER"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[speech_wer] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    pending = []
    for rec in my_records:
        if skip_completed and already_done(target_dir, "speech_wer", rec["file_stem"], metric_keys):
            continue
        pending.append(rec)
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    device = _resolve_device(local_rank)

    texts_by_stem: dict[str, str] = {
        rec["file_stem"]: resolve_speech_text(rec)
        for rec in pending
    }
    if any(texts_by_stem.values()):
        wer_model, model_load_elapsed_sec = _get_wer(
            _resolve_models_dir(), device, reuse_models=reuse_models
        )
    else:
        wer_model = None

    wer_asr_infer_elapsed_sec = 0.0
    batch_size = max(1, int(os.environ.get("MY_EVAL_WER_BATCH_SIZE", "8")))
    valid_items: List[tuple[Dict[str, Any], str]] = []
    for rec in pending:
        stem = rec["file_stem"]
        target_text = texts_by_stem.get(stem, "")
        if not target_text:
            write_per_sample(
                target_dir,
                "speech_wer",
                stem,
                {
                    "WER": float("nan"),
                    "audio_path": rec["audio_path"],
                    "target_text": target_text,
                    "_skipped_metrics": ["WER"],
                    "skip_reason": "missing_speech_text",
                },
            )
            continue
        valid_items.append((rec, target_text))

    log(rank, f"[speech_wer] valid_records={len(valid_items)}/{len(pending)} batch_size={batch_size}")
    num_batches = 0
    done = 0

    for batch in _chunks(valid_items, batch_size):
        num_batches += 1
        try:
            assert wer_model is not None
            infer_started_at = time.time()
            scores = list(wer_model.infer_audio_text_batch(
                [rec["audio_path"] for rec, _ in batch],
                [target_text for _, target_text in batch],
            ))
            _cuda_sync()
            wer_asr_infer_elapsed_sec += time.time() - infer_started_at
        except Exception as exc:
            log(rank, f"[speech_wer] batch failed ({len(batch)} samples): {exc}; retry one-by-one")
            scores = []
            for rec, target_text in batch:
                try:
                    assert wer_model is not None
                    infer_started_at = time.time()
                    score = float(wer_model.infer_audio_text(rec["audio_path"], target_text))
                    _cuda_sync()
                    wer_asr_infer_elapsed_sec += time.time() - infer_started_at
                except Exception as one_exc:
                    log(rank, f"[speech_wer] failed for {rec['file_stem']}: {one_exc}")
                    score = float("nan")
                scores.append(score)

        if len(scores) < len(batch):
            scores.extend([float("nan")] * (len(batch) - len(scores)))
        for (rec, target_text), score in zip(batch, scores):
            stem = rec["file_stem"]
            payload: Dict[str, Any] = {
                "WER": float(score),
                "audio_path": rec["audio_path"],
                "target_text": target_text,
            }
            write_per_sample(target_dir, "speech_wer", stem, payload)
            done += 1
        if done and done % 20 == 0:
            log(rank, f"  speech_wer {done}/{len(valid_items)} (batch_size={batch_size})")

    if wer_model is not None and not reuse_models:
        del wer_model
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {
        "model_load_elapsed_sec": model_load_elapsed_sec,
        "wer_asr_infer_elapsed_sec": wer_asr_infer_elapsed_sec,
        "wer_batch_size": float(batch_size),
        "wer_num_batches": float(num_batches),
        "wer_num_valid_records": float(len(valid_items)),
    }
