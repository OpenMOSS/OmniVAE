"""DINOv3 identity/reference-image consistency (ID)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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
from my_eval.utils.audio_video import load_video_rgb_pil
from my_eval.utils.versebench_refs import resolve_reference_image

_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_models_dir() -> str:
    explicit = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if explicit:
        return str(Path(explicit).expanduser())
    return str(DEFAULT_VERSE_MODELS)


def _resolve_dino_device(local_rank: int) -> str:
    explicit = os.environ.get("MY_EVAL_DINO_DEVICE")
    if explicit:
        return explicit
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cpu"


def _load_dino(models_dir: str, device: str):
    from dino.dinov3_inferencer import DinoV3Inferencer  # type: ignore
    print(f"[identity_dino] loading DINOv3 on {device}", flush=True)
    os.environ["MY_EVAL_DINO_DEVICE"] = device
    return DinoV3Inferencer(models_dir, device=device)


def _get_dino(models_dir: str, device: str, *, reuse_models: bool) -> tuple[Any, float]:
    key = f"{models_dir}|{device}"
    if reuse_models and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], 0.0
    started_at = time.time()
    dino = _load_dino(models_dir, device)
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = dino
    return dino, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    device = _resolve_dino_device(local_rank)
    _, elapsed = _get_dino(_resolve_models_dir(), device, reuse_models=True)
    log(rank, f"[identity_dino] preload complete model_load={elapsed:.3f}s")
    return {"model_load_elapsed_sec": elapsed}


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _decode_video_frames(video_path: str) -> List["Image.Image"]:  # type: ignore[name-defined]  # noqa: F821
    frames = load_video_rgb_pil(video_path, convert_rgb=False)
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def _feature_vector(output: Any) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return arr[-2]
    if arr.ndim == 1:
        return arr
    raise RuntimeError(f"unexpected DINO feature shape: {arr.shape}")


def _batch_frame_features(dino, frames: List[Any], batch_size: int) -> np.ndarray:
    features: List[np.ndarray] = []
    for batch in _chunks(frames, batch_size):
        raw = dino.feature_extractor(batch, batch_size=max(1, int(batch_size)))
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == len(batch):
            features.extend(arr[:, -2, :])
        elif arr.ndim == 4 and arr.shape[0] == len(batch) and arr.shape[1] == 1:
            features.extend(arr[:, 0, -2, :])
        else:
            features.extend(_feature_vector(item) for item in raw)
    if not features:
        raise RuntimeError("DINO returned no frame features")
    return np.stack(features, axis=0)


def _identity_score(video_path: str, reference_image_path: str, dino, batch_size: int) -> float:
    from PIL import Image
    anchor_feature = _feature_vector(dino.get_feature(Image.open(reference_image_path).convert("RGB")))
    frame_features = _batch_frame_features(dino, _decode_video_frames(video_path), batch_size)
    denom = np.linalg.norm(frame_features, axis=1) * np.linalg.norm(anchor_feature)
    scores = np.divide(
        frame_features @ anchor_feature,
        denom,
        out=np.full(frame_features.shape[0], np.nan, dtype=np.float32),
        where=denom > 0,
    )
    return float(np.nanmean(scores)) if scores.size else float("nan")


def _identity_score_single(video_path: str, reference_image_path: str, dino) -> float:
    from PIL import Image
    anchor_feature = dino.get_feature(Image.open(reference_image_path).convert("RGB"))
    scores: List[float] = []
    for frame in _decode_video_frames(video_path):
        feature = dino.get_feature(frame)
        scores.append(float(dino.infer_feature(feature, anchor_feature)))
    return float(np.mean(scores)) if scores else float("nan")


def _identity_score_with_fallback(
    video_path: str,
    reference_image_path: str,
    dino,
    batch_size: int,
) -> float:
    try:
        return _identity_score(video_path, reference_image_path, dino, batch_size)
    except Exception:
        if batch_size <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _identity_score_single(video_path, reference_image_path, dino)


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
    metric_keys = metric_keys or ["ID"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[identity_dino] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    pending = []
    for rec in my_records:
        if skip_completed and already_done(target_dir, "identity_dino", rec["file_stem"], metric_keys):
            continue
        pending.append(rec)
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    device = _resolve_dino_device(local_rank)
    log(
        rank,
        f"[identity_dino] torch_cuda={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
        f"dino_device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
    )
    dino, model_load_elapsed_sec = _get_dino(_resolve_models_dir(), device, reuse_models=reuse_models)
    batch_size = int(os.environ.get("MY_EVAL_DINO_BATCH_SIZE", "16"))

    for idx, rec in enumerate(pending):
        stem = rec["file_stem"]
        reference_image = resolve_reference_image(rec)
        payload: Dict[str, Any] = {
            "ID": float("nan"),
            "video_path": rec["video_path"],
            "reference_image_path": reference_image,
        }
        if not reference_image:
            payload["_skipped_metrics"] = ["ID"]
            payload["skip_reason"] = "missing_reference_image"
            write_per_sample(target_dir, "identity_dino", stem, payload)
            continue
        try:
            payload["ID"] = _identity_score_with_fallback(
                rec["video_path"], reference_image, dino, max(1, batch_size)
            )
        except Exception as exc:
            log(rank, f"[identity_dino] failed for {stem}: {exc}")
        write_per_sample(target_dir, "identity_dino", stem, payload)
        if (idx + 1) % 10 == 0:
            log(rank, f"  identity_dino {idx + 1}/{len(pending)}")

    if not reuse_models:
        del dino
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
