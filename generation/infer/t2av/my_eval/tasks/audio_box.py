"""AudioBox-Aesthetics CE / CU / PC / PQ.

Wraps ``audiobox_aesthetics.infer.initialize_predictor`` (the same code path
Verse-Bench's ``audio_box/audio_box_inferencer.py`` uses). Reads the .wav from
``record["audio_path"]`` directly -- no mux needed.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_VERSE_MODELS = REPO_ROOT / "generation" / "evaluation" / "verse_bench" / "models"
_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_ckpt() -> str:
    explicit = os.environ.get("MY_EVAL_AUDIOBOX_CKPT")
    if explicit:
        return str(Path(explicit).expanduser())
    models_root = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if models_root:
        candidate = Path(models_root).expanduser() / "audiobox-aesthetics" / "checkpoint.pt"
        if candidate.is_file():
            return str(candidate)
    candidate = DEFAULT_VERSE_MODELS / "audiobox-aesthetics" / "checkpoint.pt"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        "audiobox-aesthetics checkpoint not found. Set MY_EVAL_AUDIOBOX_CKPT or place it under "
        "<models>/audiobox-aesthetics/checkpoint.pt"
    )


def _load_predictor(device: torch.device):
    from audiobox_aesthetics.infer import initialize_predictor
    ckpt = _resolve_ckpt()
    print(f"[audio_box] loading checkpoint {ckpt} on {device}", flush=True)

    # Avoid checkpoint tensors being restored onto a stale device from another
    # rank; move the model explicitly after construction.
    import torch as _torch
    _orig_load = _torch.load

    def _safe_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return _orig_load(*args, **kwargs)

    _torch.load = _safe_load
    try:
        predictor = initialize_predictor(ckpt=ckpt)
    finally:
        _torch.load = _orig_load

    # Force the predictor's internal model and forward tensors onto this rank's GPU.
    if torch.cuda.is_available():
        for attr in ("model", "_model", "module"):
            mdl = getattr(predictor, attr, None)
            if mdl is not None and hasattr(mdl, "to"):
                try:
                    mdl.to(device)
                except Exception:
                    pass
        if hasattr(predictor, "device"):
            try:
                predictor.device = device
            except Exception:
                pass
    return predictor


def _resolve_device(local_rank: int) -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device


def _get_predictor(device: torch.device, *, reuse_models: bool) -> tuple[Any, float]:
    key = f"{_resolve_ckpt()}|{device}"
    if reuse_models and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], 0.0
    started_at = time.time()
    predictor = _load_predictor(device)
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = predictor
    return predictor, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    device = _resolve_device(local_rank)
    _, elapsed = _get_predictor(device, reuse_models=True)
    log(rank, f"[audio_box] preload complete model_load={elapsed:.3f}s")
    return {"model_load_elapsed_sec": elapsed}


def _chunks(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _empty_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "CE": float("nan"), "CU": float("nan"),
        "PC": float("nan"), "PQ": float("nan"),
        "audio_path": rec["audio_path"],
    }


def _normalise_results(results: Any, expected: int) -> List[Dict[str, Any]]:
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list):
        raise TypeError(f"unexpected AudioBox result type: {type(results)}")
    if len(results) != expected:
        raise RuntimeError(f"AudioBox returned {len(results)} results for batch of {expected}")
    return results


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
    metric_keys = metric_keys or ["CE", "CU", "PC", "PQ"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[audio_box] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    device = _resolve_device(local_rank)
    pending = [
        rec for rec in my_records
        if not (skip_completed and already_done(target_dir, "audio_box", rec["file_stem"], metric_keys))
    ]
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    predictor, model_load_elapsed_sec = _get_predictor(device, reuse_models=reuse_models)
    batch_size = int(os.environ.get("MY_EVAL_AUDIOBOX_BATCH_SIZE", "8"))

    done = 0
    for batch in _chunks(pending, batch_size):
        try:
            results = _normalise_results(
                predictor.forward([{"path": rec["audio_path"]} for rec in batch]),
                len(batch),
            )
        except Exception as exc:
            if len(batch) == 1:
                rec = batch[0]
                log(rank, f"[audio_box] failed for {rec['file_stem']}: {exc}")
                write_per_sample(target_dir, "audio_box", rec["file_stem"], _empty_payload(rec))
                done += 1
                continue
            log(rank, f"[audio_box] batch failed ({len(batch)} samples): {exc}; retry one-by-one")
            for rec in batch:
                payload = _empty_payload(rec)
                try:
                    result = _normalise_results(
                        predictor.forward([{"path": rec["audio_path"]}]),
                        1,
                    )[0]
                    payload["CE"] = float(result.get("CE", float("nan")))
                    payload["CU"] = float(result.get("CU", float("nan")))
                    payload["PC"] = float(result.get("PC", float("nan")))
                    payload["PQ"] = float(result.get("PQ", float("nan")))
                except Exception as one_exc:
                    log(rank, f"[audio_box] failed for {rec['file_stem']}: {one_exc}")
                write_per_sample(target_dir, "audio_box", rec["file_stem"], payload)
                done += 1
            continue

        for rec, result in zip(batch, results):
            payload = _empty_payload(rec)
            payload["CE"] = float(result.get("CE", float("nan")))
            payload["CU"] = float(result.get("CU", float("nan")))
            payload["PC"] = float(result.get("PC", float("nan")))
            payload["PQ"] = float(result.get("PQ", float("nan")))
            write_per_sample(target_dir, "audio_box", rec["file_stem"], payload)
            done += 1
        if done % 50 == 0:
            log(rank, f"  audio_box {done}/{len(pending)} (batch_size={max(1, batch_size)})")

    if not reuse_models:
        del predictor
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
