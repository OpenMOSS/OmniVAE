"""Motion Score (MS) using Verse-Bench's RAFT inferencer."""
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

_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_models_dir() -> str:
    explicit = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if explicit:
        return str(Path(explicit).expanduser())
    return str(DEFAULT_VERSE_MODELS)


def _load_raft(models_dir: str, device: torch.device):
    from raft.raft_inferencer import RAFTInferencer  # type: ignore
    print(f"[video_motion] loading RAFT on {device}", flush=True)
    return RAFTInferencer(models_dir, device=str(device))


def _resolve_device(local_rank: int) -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device


def _get_raft(models_dir: str, device: torch.device, *, reuse_models: bool) -> tuple[Any, float]:
    key = f"{models_dir}|{device}"
    if reuse_models and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], 0.0
    started_at = time.time()
    raft = _load_raft(models_dir, device)
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = raft
    return raft, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    device = _resolve_device(local_rank)
    _, elapsed = _get_raft(_resolve_models_dir(), device, reuse_models=True)
    log(rank, f"[video_motion] preload complete model_load={elapsed:.3f}s")
    return {"model_load_elapsed_sec": elapsed}


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _decode_video_frames(video_path: str) -> List["Image.Image"]:  # type: ignore[name-defined]  # noqa: F821
    frames = load_video_rgb_pil(video_path, convert_rgb=True)
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def _motion_score_single(video_path: str, raft) -> float:
    frames = _decode_video_frames(video_path)
    scores: List[float] = []
    for i in range(len(frames) - 1):
        score = raft.infer(frames[i], frames[i + 1])
        if score is not None:
            scores.append(float(score))
    return float(np.mean(scores)) if scores else float("nan")


def _motion_score_batched(video_path: str, raft, batch_size: int) -> float:
    from raft.raft_inferencer import resize_image  # type: ignore
    from raft.core.utils.utils import InputPadder  # type: ignore

    device = getattr(raft, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    frames = _decode_video_frames(video_path)
    if len(frames) < 2:
        return float("nan")
    pair_indices = list(range(len(frames) - 1))
    scores: List[float] = []
    for batch_indices in _chunks(pair_indices, batch_size):
        imgs1 = []
        imgs2 = []
        for i in batch_indices:
            img1, _ = resize_image(np.array(frames[i]))
            img2, _ = resize_image(np.array(frames[i + 1]))
            imgs1.append(torch.from_numpy(img1).permute(2, 0, 1).float())
            imgs2.append(torch.from_numpy(img2).permute(2, 0, 1).float())
        with torch.inference_mode():
            image1 = torch.stack(imgs1, dim=0).to(device)
            image2 = torch.stack(imgs2, dim=0).to(device)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = raft.model(image1, image2, iters=20, test_mode=True)
            vals = torch.abs(flow_up).flatten(1).mean(dim=1).detach().float().cpu().tolist()
        scores.extend(float(v) for v in vals)
    return float(np.mean(scores)) if scores else float("nan")


def _motion_score(video_path: str, raft, batch_size: int) -> float:
    exact_mode = _env_flag("MY_EVAL_RAFT_EXACT", not _env_flag("MY_EVAL_RAFT_ALLOW_BATCH", False))
    if exact_mode or batch_size <= 1:
        return _motion_score_single(video_path, raft)
    try:
        return _motion_score_batched(video_path, raft, batch_size)
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _motion_score_single(video_path, raft)


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
    metric_keys = metric_keys or ["MS"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[video_motion] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    device = _resolve_device(local_rank)
    pending = [
        rec for rec in my_records
        if not (skip_completed and already_done(target_dir, "video_motion", rec["file_stem"], metric_keys))
    ]
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    raft, model_load_elapsed_sec = _get_raft(_resolve_models_dir(), device, reuse_models=reuse_models)
    batch_size = int(os.environ.get("MY_EVAL_RAFT_BATCH_SIZE", "4"))
    exact_mode = _env_flag("MY_EVAL_RAFT_EXACT", not _env_flag("MY_EVAL_RAFT_ALLOW_BATCH", False))
    if exact_mode:
        log(rank, "[video_motion] mode=exact original RAFTInferencer.infer per frame-pair")
    else:
        log(rank, f"[video_motion] mode=fast_pair_batch batch_size={max(1, batch_size)} "
                  "(may differ from Verse-Bench exact MS)")

    for idx, rec in enumerate(pending):
        stem = rec["file_stem"]
        payload: Dict[str, Any] = {
            "MS": float("nan"),
            "video_path": rec["video_path"],
        }
        try:
            payload["MS"] = _motion_score(rec["video_path"], raft, max(1, batch_size))
        except Exception as exc:
            log(rank, f"[video_motion] failed for {stem}: {exc}")
        write_per_sample(target_dir, "video_motion", stem, payload)
        if (idx + 1) % 10 == 0:
            log(rank, f"  video_motion {idx + 1}/{len(pending)}")

    if not reuse_models:
        del raft
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
