"""DNSMOS P808 speech-quality score.

Driven by MOVA's ``ComputeScore`` (ONNX). Each torchrun rank passes its
``local_rank`` as ONNX Runtime's CUDA ``device_id`` so multi-GPU runs do not all
bind to visible device 0.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
DNSMOS_CODE_DIR = REPO_ROOT / "generation" / "evaluation" / "metrics" / "dnsmos"
DNSMOS_MODEL_DIR = Path(os.environ.get("MY_EVAL_DNSMOS_DIR", str(DNSMOS_CODE_DIR))).expanduser()


def _ensure_pythonpath() -> None:
    if str(DNSMOS_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(DNSMOS_CODE_DIR))


_ensure_pythonpath()

from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample


_SAMPLING_RATE = 16000
_MODEL_CACHE: Dict[str, Any] = {}


def _build_scorer(local_rank: int):
    from eval_dnsmos import ComputeScore  # type: ignore
    p808_model = str(DNSMOS_MODEL_DIR / "DNSMOS" / "model_v8.onnx")
    primary_model = str(DNSMOS_MODEL_DIR / "DNSMOS" / "sig_bak_ovr.onnx")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        os.environ["MY_EVAL_ONNX_DEVICE_ID"] = str(local_rank)
    print(f"[audio_dnsmos] primary={primary_model}", flush=True)
    print(f"[audio_dnsmos] p808={p808_model}", flush=True)
    return ComputeScore(primary_model, p808_model, use_gpu=use_gpu, gpu_device_id=local_rank)


def _bind_device(local_rank: int) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def _get_scorer(local_rank: int, *, reuse_models: bool) -> tuple[Any, float]:
    key = f"local_rank={local_rank}|cuda={torch.cuda.is_available()}"
    if reuse_models and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], 0.0
    started_at = time.time()
    scorer = _build_scorer(local_rank)
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = scorer
    return scorer, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    _bind_device(local_rank)
    _, elapsed = _get_scorer(local_rank, reuse_models=True)
    log(rank, f"[audio_dnsmos] preload complete model_load={elapsed:.3f}s")
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
    metric_keys = metric_keys or ["P808_MOS"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[audio_dnsmos] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    _bind_device(local_rank)
    pending = [
        rec for rec in my_records
        if not (skip_completed and already_done(target_dir, "audio_dnsmos", rec["file_stem"], metric_keys))
    ]
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    scorer, model_load_elapsed_sec = _get_scorer(local_rank, reuse_models=reuse_models)

    for idx, rec in enumerate(pending):
        stem = rec["file_stem"]
        payload: Dict[str, Any] = {"P808_MOS": float("nan"), "audio_path": rec["audio_path"]}
        try:
            result = scorer(rec["audio_path"], _SAMPLING_RATE, False, only_p808=True)
            payload["P808_MOS"] = float(result["P808_MOS"])
        except Exception as exc:
            log(rank, f"[audio_dnsmos] failed for {stem}: {exc}")
        write_per_sample(target_dir, "audio_dnsmos", stem, payload)
        if (idx + 1) % 100 == 0:
            log(rank, f"  audio_dnsmos {idx + 1}/{len(pending)}")

    if not reuse_models:
        del scorer
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
