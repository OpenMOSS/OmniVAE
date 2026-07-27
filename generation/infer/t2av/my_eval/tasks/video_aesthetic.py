"""Video aesthetic metrics (AS = mean(Aesthetic v2.5, MusiQ, ManiQA)).

Ports the three per-frame inferencers used by
``generation/evaluation/verse_bench/calculate_metrics.py``:

* AestheticInferencer (aesthetic_predictor_v2_5 + SigLIP)  -- score / 10
* MusiqInferencer (pyiqa)                                  -- score / 100
* ManiqaInferencer (MANIQA Koniq-10k)                      -- raw score

Per-video score is the mean over all decoded frames. ``AS`` is the average of
the three per-video means; we store each component too so callers can inspect
them individually.
"""
from __future__ import annotations

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torchvision import transforms

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
from my_eval.utils.quiet import fd_redirect, open_rank_log

_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_models_dir() -> str:
    explicit = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if explicit:
        return str(Path(explicit).expanduser())
    if DEFAULT_VERSE_MODELS.is_dir():
        return str(DEFAULT_VERSE_MODELS)
    raise FileNotFoundError(
        f"Verse-Bench models directory not found. Set MY_EVAL_VERSE_MODELS or MODELS_PATH; "
        f"tried {DEFAULT_VERSE_MODELS}."
    )


def _load_inferencers(models_dir: str):
    # The aesthetic modules call .cuda() internally; run_task sets the current
    # CUDA device to local_rank before this loader is called.
    from aesthetic.aesthetic_inferencer import AestheticInferencer  # type: ignore
    from aesthetic.musiq_inferencer import MusiqInferencer  # type: ignore
    from aesthetic.maniqa_inferencer import ManiqaInferencer  # type: ignore
    print("[video_aesthetic] loading Aesthetic + MusiQ + ManiQA", flush=True)
    return (
        AestheticInferencer(models_dir),
        MusiqInferencer(),
        ManiqaInferencer(models_dir),
    )


def _bind_device(local_rank: int) -> str:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cpu"


def _get_inferencers(models_dir: str, local_rank: int, *, reuse_models: bool) -> tuple[Any, Any, Any, float]:
    device = _bind_device(local_rank)
    key = f"{models_dir}|{device}"
    if reuse_models and key in _MODEL_CACHE:
        aesthetic_infer, musiq_infer, maniqa_infer = _MODEL_CACHE[key]
        return aesthetic_infer, musiq_infer, maniqa_infer, 0.0
    started_at = time.time()
    aesthetic_infer, musiq_infer, maniqa_infer = _load_inferencers(models_dir)
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = (aesthetic_infer, musiq_infer, maniqa_infer)
    return aesthetic_infer, musiq_infer, maniqa_infer, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: List[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    _, _, _, elapsed = _get_inferencers(_resolve_models_dir(), local_rank, reuse_models=True)
    log(rank, f"[video_aesthetic] preload complete model_load={elapsed:.3f}s")
    return {"model_load_elapsed_sec": elapsed}


def _decode_video_frames(video_path: str) -> List["Image.Image"]:  # type: ignore  # noqa: F821
    """Decode all frames using moviepy (same as Verse-Bench)."""
    frames = load_video_rgb_pil(video_path, convert_rgb=False)
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _flatten_scores(values: Any) -> List[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return [float(v) for v in arr.tolist()]


def _infer_aesthetic_frames(inferencer: Any, frames: List[Any], batch_size: int) -> List[float]:
    scores: List[float] = []
    if hasattr(inferencer, "infer_batch"):
        for batch in _chunks(frames, batch_size):
            scores.extend(_flatten_scores(inferencer.infer_batch(batch)))
        return scores
    for frame in frames:
        scores.append(float(inferencer.infer(frame)))
    return scores


def _infer_musiq_frames(inferencer: Any, frames: List[Any], batch_size: int) -> List[float]:
    to_tensor = getattr(inferencer, "transform", transforms.ToTensor())
    device = getattr(inferencer, "device", "cuda")
    model = inferencer.model
    scores: List[float] = []
    for batch in _chunks(frames, batch_size):
        tensor = torch.stack([to_tensor(frame.convert("RGB")) for frame in batch], dim=0).to(device)
        with torch.no_grad():
            out = model(tensor)
        scores.extend(_flatten_scores(out.detach().float().cpu().numpy()))
    return scores


def _deterministic_rng(seed_key: str) -> Any:
    if os.environ.get("MY_EVAL_DETERMINISTIC", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return np.random
    digest = hashlib.sha1(seed_key.encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:8], 16))


def _randint(rng: Any, low: int, high: int) -> int:
    if high <= low:
        return low
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def _infer_maniqa_frames(
    inferencer: Any,
    frames: List[Any],
    patch_batch_size: int,
    seed_key: str = "",
) -> List[float]:
    net = inferencer.net
    net.eval()
    num_crops = int(inferencer.config.num_crops)
    crop_h = 224
    crop_w = 224
    sums = np.zeros(len(frames), dtype=np.float64)
    counts = np.zeros(len(frames), dtype=np.int32)
    patch_batch: List[np.ndarray] = []
    owner_batch: List[int] = []
    rng = _deterministic_rng(seed_key)

    def flush() -> None:
        nonlocal patch_batch, owner_batch
        if not patch_batch:
            return
        patches = torch.from_numpy(np.stack(patch_batch, axis=0)).float().cuda()
        with torch.no_grad():
            out = net(patches).detach().float().cpu().numpy().reshape(-1)
        for frame_idx, score in zip(owner_batch, out):
            sums[frame_idx] += float(score)
            counts[frame_idx] += 1
        patch_batch = []
        owner_batch = []

    for frame_idx, frame in enumerate(frames):
        img = np.asarray(frame.convert("RGB")).astype("float32") / 255.0
        img = np.transpose(img, (2, 0, 1))
        c, h, w = img.shape
        if h < crop_h or w < crop_w:
            raise RuntimeError(f"frame too small for MANIQA crop: {(h, w)} < {(crop_h, crop_w)}")
        for _ in range(num_crops):
            top = _randint(rng, 0, h - crop_h) if h > crop_h else 0
            left = _randint(rng, 0, w - crop_w) if w > crop_w else 0
            patch = img[:, top:top + crop_h, left:left + crop_w]
            patch = (patch - 0.5) / 0.5
            patch_batch.append(patch.astype("float32", copy=False))
            owner_batch.append(frame_idx)
            if len(patch_batch) >= patch_batch_size:
                flush()
    flush()
    return [float(sums[i] / counts[i]) if counts[i] else float("nan") for i in range(len(frames))]


def run_task(
    rank: int,
    local_rank: int,
    world_size: int,
    target_dir: Path,
    manifest: Dict[str, Any],
    skip_completed: bool = True,
    metric_keys: List[str] | None = None,
    reuse_models: bool = False,
    **_: Any,
) -> Dict[str, float]:
    model_load_elapsed_sec = 0.0
    metric_keys = metric_keys or ["Aesthetic", "MusiQ", "ManiQA", "AS"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[video_aesthetic] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    _bind_device(local_rank)
    pending = [
        rec for rec in my_records
        if not (skip_completed and already_done(target_dir, "video_aesthetic", rec["file_stem"], metric_keys))
    ]
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    models_dir = _resolve_models_dir()
    aesthetic_infer, musiq_infer, maniqa_infer, model_load_elapsed_sec = _get_inferencers(
        models_dir, local_rank, reuse_models=reuse_models
    )
    batch_size = int(os.environ.get("MY_EVAL_AESTHETIC_BATCH_SIZE", "16"))
    maniqa_patch_batch_size = int(os.environ.get("MY_EVAL_MANIQA_PATCH_BATCH_SIZE", "64"))
    metric_key_set = set(metric_keys)

    rank_log_path = open_rank_log(target_dir, "video_aesthetic", rank)

    with rank_log_path.open("a", encoding="utf-8") as rank_log:
        rank_log.write(f"\n===== video_aesthetic rank{rank} new run "
                       f"({len(pending)} records) =====\n")
        rank_log.flush()

        for idx, rec in enumerate(pending):
            stem = rec["file_stem"]
            payload: Dict[str, Any] = {
                "Aesthetic": float("nan"),
                "MusiQ": float("nan"),
                "ManiQA": float("nan"),
                "AS": float("nan"),
                "video_path": rec["video_path"],
            }
            rank_log.write(f"\n---- {stem} ----\n")
            rank_log.flush()
            try:
                frames = _decode_video_frames(rec["video_path"])
                need_aesthetic = "Aesthetic" in metric_key_set or "AS" in metric_key_set
                need_musiq = "MusiQ" in metric_key_set or "AS" in metric_key_set
                need_maniqa = "ManiQA" in metric_key_set or "AS" in metric_key_set
                aest_scores: List[float] = []
                musiq_scores: List[float] = []
                maniqa_scores: List[float] = []
                with fd_redirect(rank_log):
                    if need_aesthetic:
                        try:
                            aest_scores = _infer_aesthetic_frames(aesthetic_infer, frames, batch_size)
                        except Exception as fexc:
                            print(f"[video_aesthetic] aesthetic failed for {stem}: {fexc}")
                    if need_musiq:
                        try:
                            musiq_scores = _infer_musiq_frames(musiq_infer, frames, batch_size)
                        except Exception as fexc:
                            print(f"[video_aesthetic] musiq failed for {stem}: {fexc}")
                    if need_maniqa:
                        try:
                            maniqa_scores = _infer_maniqa_frames(
                                maniqa_infer,
                                frames,
                                maniqa_patch_batch_size,
                                seed_key=str(rec["video_path"]),
                            )
                        except Exception as fexc:
                            print(f"[video_aesthetic] maniqa failed for {stem}: {fexc}")
                avg_aesthetic = float(np.mean(aest_scores)) / 10.0 if aest_scores else float("nan")
                avg_musiq = float(np.mean(musiq_scores)) / 100.0 if musiq_scores else float("nan")
                avg_maniqa = float(np.mean(maniqa_scores)) if maniqa_scores else float("nan")
                payload["Aesthetic"] = avg_aesthetic
                payload["MusiQ"] = avg_musiq
                payload["ManiQA"] = avg_maniqa
                sub_scores = [s for s in (avg_aesthetic, avg_musiq, avg_maniqa)
                              if not (s != s)]  # filter NaNs
                payload["AS"] = float(np.mean(sub_scores)) if sub_scores else float("nan")
                payload["n_frames"] = len(frames)
            except Exception as exc:
                log(rank, f"[video_aesthetic] failed for {stem}: {exc}  (see {rank_log_path})")
            write_per_sample(target_dir, "video_aesthetic", stem, payload)
            if (idx + 1) % 5 == 0:
                log(rank, f"  video_aesthetic {idx + 1}/{len(pending)}")

    log(rank, f"[video_aesthetic] done, log={rank_log_path}")

    if not reuse_models:
        del aesthetic_infer, musiq_infer, maniqa_infer
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
