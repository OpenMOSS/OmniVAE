"""LAION-Aesthetics V2 score for generated videos (frame-averaged).

Pipeline:

    1. Read samples.jsonl produced by infer_t2v.py.
    2. Modulo-shard rows across torchrun ranks.
    3. For each video:
         a. Decode with imageio and uniformly sample N frames
            (default ``--frames-per-video 16``).
         b. Run each frame through an OpenCLIP ViT-L/14 image encoder
            (``open_clip.create_model_and_transforms("ViT-L-14",
            pretrained="openai")``) to obtain an L2-normalized 768-d image
            embedding.
         c. Feed the embedding into the MLP head from
            https://github.com/christophschuhmann/improved-aesthetic-predictor
            (``sac+logos+ava1-l14-linearMSE.pth``) which outputs a single
            aesthetic score.
         d. Aesthetic score for the video = mean over the sampled frames.
    4. Per-rank shards are merged on rank 0 into ``aesthetic_score.jsonl`` plus
       a ``aesthetic_score_summary.json`` with mean / std / count overall and
       per-dataset.

The MLP head architecture mirrors the official predictor::

    Linear(768, 1024) -> Dropout(0.2)
    Linear(1024, 128) -> Dropout(0.2)
    Linear(128, 64)   -> Dropout(0.1)
    Linear(64, 16)
    Linear(16, 1)

The .pth file from the upstream repo can be downloaded once and pointed at via
``--predictor-ckpt``. Default search path: ``$AESTHETIC_PREDICTOR_PATH``,
``./.checkpoints/sac+logos+ava1-l14-linearMSE.pth``, then
``~/.cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth``.

CLI:

    torchrun --nproc_per_node=N eval/video/t2v/compute_aesthetic_score.py \\
        --samples-jsonl <run-dir>/samples.jsonl \\
        --output-dir    <run-dir>/metrics \\
        --predictor-ckpt /path/to/sac+logos+ava1-l14-linearMSE.pth
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-key", type=str, default="video_path")
    parser.add_argument("--unified-id-key", type=str, default="unified_id")
    parser.add_argument(
        "--predictor-ckpt",
        type=Path,
        default=None,
        help="Path to sac+logos+ava1-l14-linearMSE.pth from improved-aesthetic-predictor.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-L-14",
        help="open_clip model name (default ViT-L-14, matches the predictor's training distribution).",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="openai",
        help="open_clip pretrained tag (default openai).",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=16,
        help="How many frames to uniformly sample per video.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=16,
        help="How many frames to encode per CLIP forward (memory tradeoff).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--limit",
        "--max-examples",
        "--max_examples",
        dest="limit",
        type=int,
        default=None,
        help="Process only first N rows. Aliases: --max-examples / --max_examples.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--dist-timeout-minutes", type=int, default=60)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[int, int, int, bool]:
    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    if world_size <= 1:
        return rank, world_size, local_rank, False
    backend = args.dist_backend if torch.cuda.is_available() else "gloo"
    if not torch.distributed.is_initialized():
        from datetime import timedelta

        torch.distributed.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=int(args.dist_timeout_minutes)),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def resolve_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"input jsonl not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                out.append(payload)
    return out


def existing_unified_ids(path: Path, key: str) -> set[str]:
    if not path.is_file():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            value = payload.get(key)
            if isinstance(value, str) and value:
                seen.add(value)
    return seen


def merge_shards(*, output_dir: Path, final_path: Path, rank_pattern: str, key: str, keep_existing: bool) -> dict[str, int]:
    by_id: dict[str, dict[str, Any]] = {}
    n_kept_existing = 0
    if keep_existing and final_path.is_file():
        with final_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                value = payload.get(key)
                if isinstance(value, str) and value:
                    by_id[value] = payload
                    n_kept_existing += 1
    n_added = 0
    for shard_path in sorted(output_dir.glob(rank_pattern)):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                value = payload.get(key)
                if not isinstance(value, str) or not value:
                    continue
                if value in by_id and keep_existing:
                    continue
                by_id[value] = payload
                n_added += 1
    sorted_records = [by_id[k] for k in sorted(by_id.keys())]
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as handle:
        for record in sorted_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for shard_path in output_dir.glob(rank_pattern):
        try:
            shard_path.unlink()
        except OSError:
            pass
    return {"n_kept_existing": n_kept_existing, "n_added": n_added, "n_total": len(sorted_records)}


def write_summary(records: list[dict[str, Any]], path: Path) -> None:
    by_dataset: dict[str, list[float]] = {}
    all_scores: list[float] = []
    n_failures = 0
    for record in records:
        score = record.get("aesthetic_score")
        if not isinstance(score, (int, float)) or score != score:
            n_failures += 1
            continue
        all_scores.append(float(score))
        dataset = str(record.get("dataset") or "unknown")
        by_dataset.setdefault(dataset, []).append(float(score))

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "n": len(values),
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
        }

    payload = {
        "metric": "aesthetic_score",
        "definition": "LAION-Aesthetics V2 (sac+logos+ava1-l14-linearMSE) frame-averaged over uniformly sampled frames",
        "overall": stats(all_scores),
        "per_dataset": {dataset: stats(scores) for dataset, scores in sorted(by_dataset.items())},
        "n_failures": n_failures,
        "n_records": len(records),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class AestheticPredictor(nn.Module):
    """Re-implementation of the LAION-Aesthetics V2 predictor head.

    Architecture matches sac+logos+ava1-l14-linearMSE.pth from
    https://github.com/christophschuhmann/improved-aesthetic-predictor.
    """

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _candidate_predictor_paths(explicit: Path | None) -> list[Path]:
    paths: list[Path] = []
    if explicit is not None:
        paths.append(Path(explicit).expanduser())
    env_path = os.environ.get("AESTHETIC_PREDICTOR_PATH")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(Path.cwd() / ".checkpoints" / "sac+logos+ava1-l14-linearMSE.pth")
    paths.append(Path.home() / ".cache" / "aesthetic" / "sac+logos+ava1-l14-linearMSE.pth")
    paths.append(Path.cwd() / "sac+logos+ava1-l14-linearMSE.pth")
    return paths


def load_predictor(args: argparse.Namespace, device: torch.device) -> AestheticPredictor:
    candidates = _candidate_predictor_paths(args.predictor_ckpt)
    for candidate in candidates:
        if candidate.is_file():
            ckpt_path = candidate
            break
    else:
        raise SystemExit(
            "Aesthetic predictor checkpoint not found. Tried:\n  "
            + "\n  ".join(str(p) for p in candidates)
            + "\nPass --predictor-ckpt or set AESTHETIC_PREDICTOR_PATH. The .pth lives at "
            "https://github.com/christophschuhmann/improved-aesthetic-predictor "
            "(sac+logos+ava1-l14-linearMSE.pth)."
        )

    predictor = AestheticPredictor(embed_dim=768)
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and not any(k.startswith("layers") for k in state_dict):
        state_dict = state_dict["state_dict"]
    # Original repo's keys are like "layers.0.weight", which already align.
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[aesthetic] WARNING missing keys when loading predictor: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[aesthetic] WARNING unexpected keys when loading predictor: {unexpected}", file=sys.stderr)
    predictor.eval()
    predictor.to(device=device, dtype=torch.float32)
    print(f"[aesthetic] loaded predictor head from {ckpt_path}", file=sys.stderr)
    return predictor


def load_open_clip(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "open_clip_torch is not installed. Run: pip install open_clip_torch"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, preprocess


def sample_uniform_indices(num_frames: int, n: int) -> list[int]:
    if num_frames <= 0:
        return []
    if n >= num_frames:
        return list(range(num_frames))
    # Inclusive endpoints, evenly spaced.
    return [int(round(i)) for i in np.linspace(0, num_frames - 1, num=n)]


def decode_frames(video_path: Path, n_frames: int) -> list[Image.Image]:
    import imageio.v3 as iio

    # Cheap probe: ImageIO can give us metadata via get_reader for fps/nframes.
    try:
        meta = iio.immeta(str(video_path))
    except Exception:
        meta = {}
    nframes = meta.get("n_frames")

    frames: list[Image.Image] = []
    if isinstance(nframes, int) and nframes > 0:
        target_indices = sample_uniform_indices(nframes, n_frames)
        target_set = set(target_indices)
        for idx, frame in enumerate(iio.imiter(str(video_path))):
            if idx in target_set:
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                frames.append(Image.fromarray(np.asarray(frame, dtype=np.uint8)))
            if idx >= max(target_indices):
                break
    else:
        # Fallback: read everything, then sample.
        all_frames = list(iio.imiter(str(video_path)))
        target_indices = sample_uniform_indices(len(all_frames), n_frames)
        for idx in target_indices:
            frame = all_frames[idx]
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(Image.fromarray(np.asarray(frame, dtype=np.uint8)))

    return frames


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_distributed = setup_distributed(args)
    is_main = rank == 0
    device = resolve_device(local_rank)
    dtype = resolve_dtype(args.dtype, device)

    samples_path = Path(args.samples_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        torch.distributed.barrier()

    final_path = output_dir / "aesthetic_score.jsonl"
    summary_path = output_dir / "aesthetic_score_summary.json"
    rank_shard_path = output_dir / f"aesthetic_score.rank{rank:03d}.jsonl"
    rank_pattern = "aesthetic_score.rank*.jsonl"

    rows = read_jsonl(samples_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        if is_main:
            print(f"[aesthetic] No rows in {samples_path}", file=sys.stderr)
        if is_distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    skip_ids: set[str] = set()
    if args.resume and is_main:
        skip_ids = existing_unified_ids(final_path, args.unified_id_key)
    if is_distributed:
        payload: list[Any] = [sorted(skip_ids)] if is_main else [None]
        torch.distributed.broadcast_object_list(payload, src=0)
        skip_ids = set(str(s) for s in (payload[0] or []))

    shard: list[dict[str, Any]] = []
    for global_index, record in enumerate(rows):
        if global_index % max(1, world_size) != rank:
            continue
        unified_id = str(record.get(args.unified_id_key, ""))
        if not unified_id:
            unified_id = f"row_{global_index:06d}"
        if unified_id in skip_ids:
            continue
        shard.append(record)

    if is_main:
        print(
            f"[aesthetic] world_size={world_size} total={len(rows)} "
            f"rank0_shard={len(shard)} skip_existing={len(skip_ids)}",
            flush=True,
        )

    if shard:
        clip_model, preprocess = load_open_clip(args, device, dtype)
        predictor = load_predictor(args, device)

    n_done = 0
    started_at = time.time()
    if shard:
        with rank_shard_path.open("w", encoding="utf-8") as handle:
            iterator = tqdm(
                shard,
                desc=f"rank{rank:02d} aesthetic",
                disable=not is_main,
                position=local_rank,
                leave=False,
            )
            for record in iterator:
                video_path_value = record.get(args.video_key)
                unified_id = str(record.get(args.unified_id_key, "") or f"row_{n_done:06d}")

                row_out: dict[str, Any] = {
                    args.unified_id_key: unified_id,
                    "dataset": record.get("dataset"),
                    "category": record.get("category"),
                    "same_source_id": record.get("same_source_id"),
                    "video_path": str(video_path_value) if video_path_value is not None else None,
                    "n_frames_used": 0,
                }

                if not isinstance(video_path_value, str) or not Path(video_path_value).is_file():
                    row_out["error"] = f"missing video file: {video_path_value!r}"
                    row_out["aesthetic_score"] = None
                    handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                    handle.flush()
                    n_done += 1
                    continue

                try:
                    frames = decode_frames(Path(video_path_value), int(args.frames_per_video))
                    if not frames:
                        raise RuntimeError("no frames decoded")
                    row_out["n_frames_used"] = len(frames)

                    frame_tensors = torch.stack([preprocess(frame) for frame in frames]).to(device=device, dtype=dtype)
                    frame_scores: list[float] = []
                    with torch.no_grad():
                        for start in range(0, frame_tensors.shape[0], int(args.frame_batch_size)):
                            chunk = frame_tensors[start : start + int(args.frame_batch_size)]
                            features = clip_model.encode_image(chunk)
                            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                            features = features.float()
                            scores = predictor(features).squeeze(-1).cpu().tolist()
                            frame_scores.extend(float(s) for s in scores)
                    row_out["aesthetic_score"] = float(sum(frame_scores) / len(frame_scores))
                    row_out["frame_scores_min"] = float(min(frame_scores))
                    row_out["frame_scores_max"] = float(max(frame_scores))
                except Exception as exc:  # noqa: BLE001
                    row_out["error"] = f"{type(exc).__name__}: {exc}"
                    row_out["aesthetic_score"] = None

                handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                handle.flush()
                n_done += 1

    elapsed = time.time() - started_at
    print(
        f"[aesthetic] rank={rank} world={world_size} done={n_done} elapsed={elapsed:.1f}s "
        f"shard={rank_shard_path}",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_distributed:
        torch.distributed.barrier()

    if is_main:
        merge_stats = merge_shards(
            output_dir=output_dir,
            final_path=final_path,
            rank_pattern=rank_pattern,
            key=args.unified_id_key,
            keep_existing=bool(args.resume),
        )
        merged_records = read_jsonl(final_path)
        write_summary(merged_records, summary_path)
        print(
            json.dumps(
                {
                    "metric": "aesthetic_score",
                    "samples_jsonl": str(samples_path),
                    "aesthetic_score_jsonl": str(final_path),
                    "summary_json": str(summary_path),
                    "merged": merge_stats,
                },
                indent=2,
                sort_keys=True,
            )
        )

    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
