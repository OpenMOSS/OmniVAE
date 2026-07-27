"""Reference-audio FD and KL metrics from Verse-Bench."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
from my_eval.utils.versebench_refs import resolve_reference_audio

_FD_MODEL_CACHE: Dict[str, Any] = {}
_KL_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_models_dir() -> str:
    explicit = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if explicit:
        return str(Path(explicit).expanduser())
    return str(DEFAULT_VERSE_MODELS)


def _load_fd_model(models_dir: str, device: torch.device):
    from fd.clap_inferencer import ClapInferencer  # type: ignore
    print(f"[audio_fd_kl] loading CLAP FD model on {device}", flush=True)
    return ClapInferencer(models_dir, device=str(device))


def _load_kl_model(device: torch.device):
    from kl.kld_inferencer import KLDInferencer  # type: ignore
    print(f"[audio_fd_kl] loading PaSST KLD model on {device}", flush=True)
    return KLDInferencer(device=device)


def _resolve_device(local_rank: int) -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device


def _get_fd_model(models_dir: str, device: torch.device, *, reuse_models: bool) -> tuple[Any, float]:
    key = f"{models_dir}|{device}"
    if reuse_models and key in _FD_MODEL_CACHE:
        return _FD_MODEL_CACHE[key], 0.0
    started_at = time.time()
    model = _load_fd_model(models_dir, device)
    elapsed = time.time() - started_at
    if reuse_models:
        _FD_MODEL_CACHE[key] = model
    return model, elapsed


def _get_kl_model(device: torch.device, *, reuse_models: bool) -> tuple[Any, float]:
    key = str(device)
    if reuse_models and key in _KL_MODEL_CACHE:
        return _KL_MODEL_CACHE[key], 0.0
    started_at = time.time()
    model = _load_kl_model(device)
    elapsed = time.time() - started_at
    if reuse_models:
        _KL_MODEL_CACHE[key] = model
    return model, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    metric_key_set = set(metric_keys or ["FD", "KL"])
    device = _resolve_device(local_rank)
    elapsed = 0.0
    if "FD" in metric_key_set:
        _, part = _get_fd_model(_resolve_models_dir(), device, reuse_models=True)
        elapsed += part
    if "KL" in metric_key_set:
        _, part = _get_kl_model(device, reuse_models=True)
        elapsed += part
    log(rank, f"[audio_fd_kl] preload complete model_load={elapsed:.3f}s")
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
    metric_keys = metric_keys or ["FD", "KL"]
    metric_key_set = set(metric_keys)
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[audio_fd_kl] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    device = _resolve_device(local_rank)

    pending = []
    for rec in my_records:
        if skip_completed and already_done(target_dir, "audio_fd_kl", rec["file_stem"], metric_keys):
            continue
        pending.append(rec)
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    refs_by_stem: dict[str, str | None] = {
        rec["file_stem"]: resolve_reference_audio(rec)
        for rec in pending
    }
    need_models = any(refs_by_stem.values())
    model_load_elapsed_sec = 0.0
    fd_model = None
    kl_model = None
    if need_models and "FD" in metric_key_set:
        fd_model, part = _get_fd_model(_resolve_models_dir(), device, reuse_models=reuse_models)
        model_load_elapsed_sec += part
    if need_models and "KL" in metric_key_set:
        kl_model, part = _get_kl_model(device, reuse_models=reuse_models)
        model_load_elapsed_sec += part

    for idx, rec in enumerate(pending):
        stem = rec["file_stem"]
        reference_audio = refs_by_stem.get(stem)
        payload: Dict[str, Any] = {
            "FD": float("nan"),
            "KL": float("nan"),
            "audio_path": rec["audio_path"],
            "reference_audio_path": reference_audio,
        }
        if not reference_audio:
            payload["_skipped_metrics"] = list(metric_keys)
            payload["skip_reason"] = "missing_reference_audio"
            write_per_sample(target_dir, "audio_fd_kl", stem, payload)
            continue

        if fd_model is not None and "FD" in metric_key_set:
            try:
                payload["FD"] = float(fd_model.infer_fd(rec["audio_path"], reference_audio))
            except Exception as exc:
                log(rank, f"[audio_fd_kl] FD failed for {stem}: {exc}")
        if kl_model is not None and "KL" in metric_key_set:
            try:
                payload["KL"] = float(kl_model.infer(rec["audio_path"], reference_audio))
            except Exception as exc:
                log(rank, f"[audio_fd_kl] KL failed for {stem}: {exc}")
        write_per_sample(target_dir, "audio_fd_kl", stem, payload)
        if (idx + 1) % 20 == 0:
            log(rank, f"  audio_fd_kl {idx + 1}/{len(pending)}")

    if not reuse_models:
        del fd_model, kl_model
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
