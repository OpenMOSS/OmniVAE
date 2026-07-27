"""ImageBind-Score: text<->video alignment via the ImageBind embedding space.

This script reads a ``samples.jsonl`` produced by ``infer_t2v.py`` and writes a
``imagebind_score.jsonl`` (one JSON object per video, plus an aggregate
summary file). It is multi-rank aware: rows are sharded across torchrun
processes via the same ``i % world_size == rank`` rule used during inference.
On rank 0 the per-rank shard files are concatenated into a single deduped
output sorted by ``unified_id``.

Install (one-time)::

    pip install git+https://github.com/facebookresearch/ImageBind.git
    # imagebind_huge.pth is downloaded automatically on first model load
    # to ``$IMAGEBIND_CACHE_DIR`` (defaults to ./.checkpoints/).

Score definition
----------------
For a video V and prompt P:

    score(V, P) = cosine_similarity(
        normalize(imagebind.encode_video(V)),
        normalize(imagebind.encode_text(P))
    )

ImageBind's official video transform (``data.load_and_transform_video_data``)
samples ``clips_per_video`` non-overlapping clips of length ``clip_duration``
seconds; each clip is encoded independently and ImageBind itself averages
their embeddings inside ``forward``. We therefore obtain *one* video-side
embedding per video, regardless of clip count, and a single cosine score per
``(video, prompt)`` pair.

CLI:

    torchrun --nproc_per_node=N eval/video/t2v/compute_imagebind_score.py \\
        --samples-jsonl <run-dir>/samples.jsonl \\
        --output-dir    <run-dir>/metrics

Outputs (under ``--output-dir``):

    imagebind_score.rank{R:03d}.jsonl    # intermediate per-rank shards
    imagebind_score.jsonl                # merged & sorted; final artifact
    imagebind_score_summary.json         # mean / std / count per dataset
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

import torch
import torch.nn.functional as F
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
    parser.add_argument("--samples-jsonl", type=Path, required=True, help="Path to samples.jsonl produced by infer_t2v.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write per-rank shards + final imagebind_score.jsonl.")
    parser.add_argument("--video-key", type=str, default="video_path", help="Field in samples.jsonl pointing at the generated mp4.")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Field in samples.jsonl carrying the positive prompt.")
    parser.add_argument("--unified-id-key", type=str, default="unified_id")
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=2.0,
        help="ImageBind clip_duration in seconds (default 2.0; matches the official 2s clips).",
    )
    parser.add_argument(
        "--clips-per-video",
        type=int,
        default=5,
        help="Number of non-overlapping clips sampled per video (default 5; matches the official video_data transform).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit imagebind_huge.pth path. By default ImageBind downloads to .checkpoints/.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
        help="Forward dtype on GPU; falls back to fp32 on CPU.",
    )
    parser.add_argument(
        "--limit",
        "--max-examples",
        "--max_examples",
        dest="limit",
        type=int,
        default=None,
        help="Process only first N rows (smoke test). Aliases: --max-examples / --max_examples.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows whose unified_id already appears in imagebind_score.jsonl.")
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
        score = record.get("imagebind_score")
        if not isinstance(score, (int, float)) or score != score:  # NaN guard
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
        "metric": "imagebind_score",
        "definition": "cosine similarity between ImageBind text and video embeddings",
        "overall": stats(all_scores),
        "per_dataset": {dataset: stats(scores) for dataset, scores in sorted(by_dataset.items())},
        "n_failures": n_failures,
        "n_records": len(records),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def configure_imagebind_cache(checkpoint_path: Path | None) -> None:
    """Match ImageBind's expected layout: it looks under .checkpoints/ in cwd
    unless we point it elsewhere via env / explicit path."""

    cache_root = os.environ.get("IMAGEBIND_CACHE_DIR")
    if cache_root:
        Path(cache_root).expanduser().mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(Path(cache_root).expanduser() / "hf"))


def load_imagebind_model(checkpoint_path: Path | None, device: torch.device, dtype: torch.dtype):
    try:
        from imagebind.models import imagebind_model
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ImageBind is not installed. Run: "
            "pip install git+https://github.com/facebookresearch/ImageBind.git"
        ) from exc

    configure_imagebind_cache(checkpoint_path)
    if checkpoint_path is not None:
        # The ``pretrained=True`` call path inside imagebind_huge() looks for
        # ./.checkpoints/imagebind_huge.pth -- if the user supplied a path we
        # symlink it there (or copy as a fallback) before loading.
        ckpt_dir = Path.cwd() / ".checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / "imagebind_huge.pth"
        if not target.exists():
            try:
                target.symlink_to(checkpoint_path.resolve())
            except OSError:
                import shutil

                shutil.copy2(checkpoint_path, target)

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device=device, dtype=dtype)
    return model


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

    final_path = output_dir / "imagebind_score.jsonl"
    summary_path = output_dir / "imagebind_score_summary.json"
    rank_shard_path = output_dir / f"imagebind_score.rank{rank:03d}.jsonl"
    rank_pattern = "imagebind_score.rank*.jsonl"

    rows = read_jsonl(samples_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        if is_main:
            print(f"[imagebind] No rows read from {samples_path}; nothing to do.", file=sys.stderr)
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
            f"[imagebind] world_size={world_size} total_rows={len(rows)} "
            f"shard_size_rank0={len(shard)} skip_existing={len(skip_ids)}",
            flush=True,
        )

    # Lazy imports so empty ranks don't pay the cost.
    if shard:
        from imagebind.data import load_and_transform_text, load_and_transform_video_data
        from imagebind.models.imagebind_model import ModalityType

        model = load_imagebind_model(args.checkpoint_path, device=device, dtype=dtype)

    n_done = 0
    started_at = time.time()
    if shard:
        with rank_shard_path.open("w", encoding="utf-8") as handle:
            iterator = tqdm(
                shard,
                desc=f"rank{rank:02d} ImageBind",
                disable=not is_main,
                position=local_rank,
                leave=False,
            )
            for record in iterator:
                video_path_value = record.get(args.video_key)
                prompt_value = record.get(args.prompt_key)
                unified_id = str(record.get(args.unified_id_key, "") or f"row_{n_done:06d}")

                row_out: dict[str, Any] = {
                    args.unified_id_key: unified_id,
                    "dataset": record.get("dataset"),
                    "category": record.get("category"),
                    "same_source_id": record.get("same_source_id"),
                    "video_path": str(video_path_value) if video_path_value is not None else None,
                    "prompt": str(prompt_value) if prompt_value is not None else None,
                }

                if not isinstance(video_path_value, str) or not Path(video_path_value).is_file():
                    row_out["error"] = f"missing video file: {video_path_value!r}"
                    row_out["imagebind_score"] = None
                    handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                    handle.flush()
                    n_done += 1
                    continue
                if not isinstance(prompt_value, str) or not prompt_value.strip():
                    row_out["error"] = "empty prompt"
                    row_out["imagebind_score"] = None
                    handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                    handle.flush()
                    n_done += 1
                    continue

                try:
                    with torch.no_grad():
                        # ImageBind's video transform handles uniform clip
                        # sampling internally; we just pass the path.
                        video_inputs = load_and_transform_video_data(
                            [str(video_path_value)],
                            device,
                            clip_duration=float(args.clip_duration),
                            clips_per_video=int(args.clips_per_video),
                        ).to(device=device, dtype=dtype)
                        text_inputs = load_and_transform_text([str(prompt_value)], device)

                        embeddings = model(
                            {
                                ModalityType.VISION: video_inputs,
                                ModalityType.TEXT: text_inputs,
                            }
                        )
                        video_emb = F.normalize(embeddings[ModalityType.VISION].float(), dim=-1)
                        text_emb = F.normalize(embeddings[ModalityType.TEXT].float(), dim=-1)
                        score = float((video_emb * text_emb).sum(dim=-1).item())

                    row_out["imagebind_score"] = score
                except Exception as exc:  # noqa: BLE001
                    row_out["error"] = f"{type(exc).__name__}: {exc}"
                    row_out["imagebind_score"] = None

                handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                handle.flush()
                n_done += 1

    elapsed = time.time() - started_at
    print(
        f"[imagebind] rank={rank} world={world_size} done={n_done} elapsed={elapsed:.1f}s "
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
                    "metric": "imagebind_score",
                    "samples_jsonl": str(samples_path),
                    "imagebind_score_jsonl": str(final_path),
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
