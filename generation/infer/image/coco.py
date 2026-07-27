"""COCO 2014 caption schedule + reference image preprocessing.

The captions JSON we consume is the standard `captions_val2014.json`:

    {
      "info": {...},
      "images": [{"id": int, "file_name": str, ...}, ...],
      "annotations": [{"id": int, "image_id": int, "caption": str}, ...]
    }

For T2I FID evaluation we want **one prompt per image**. We follow the early
DALL-E / Imagen "first caption" convention: walk ``annotations`` sorted by
``annotation_id`` and remember the *first* caption seen for each image_id.
Images are then sorted by ``image_id`` and the first ``num_samples`` are kept.

Reference images are center-cropped to a square (short-side anchored) and
resized to ``image_size`` so they share preprocessing with the generated
samples (same H/W, same uint8 dtype) -- a strict precondition for a fair
``FID(real, fake)``.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


@dataclass(frozen=True)
class CocoEntry:
    """One (image, caption) pair scheduled for FID evaluation."""

    sample_index: int
    image_id: int
    file_name: str
    caption: str
    sample_seed: int


def load_coco_schedule(
    annotations_json: str | Path,
    *,
    num_samples: int,
    seed: int,
) -> list[CocoEntry]:
    """Build a deterministic ``num_samples``-long schedule from COCO captions.

    Selection rule:
      1. Sort ``annotations`` by their ``id`` and remember each image_id's
         *first* caption (caption with the smallest annotation id).
      2. Sort ``images`` by ``id`` and take the first ``num_samples`` whose
         image_id has a caption recorded in step 1.

    Each entry's ``sample_seed`` is ``seed + sample_index`` to match the
    convention used by ``omnivae_generation.trainer.eval.guided_diffusion`` so that the same
    prompt receives the same noise across reruns.
    """
    annotations_path = Path(annotations_json).expanduser().resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(
            f"COCO captions JSON not found: {annotations_path}"
        )
    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    if not images or not annotations:
        raise ValueError(
            f"Captions JSON {annotations_path} is missing 'images' or 'annotations'."
        )

    first_caption: dict[int, str] = {}
    for ann in sorted(annotations, key=lambda r: int(r.get("id", 0))):
        image_id = int(ann.get("image_id", -1))
        if image_id < 0:
            continue
        if image_id in first_caption:
            continue
        caption = str(ann.get("caption", "")).strip()
        if not caption:
            continue
        first_caption[image_id] = caption

    sorted_images = sorted(images, key=lambda r: int(r.get("id", 0)))

    schedule: list[CocoEntry] = []
    for image_record in sorted_images:
        if len(schedule) >= int(num_samples):
            break
        image_id = int(image_record.get("id", -1))
        if image_id < 0:
            continue
        caption = first_caption.get(image_id)
        if caption is None:
            continue
        file_name = str(image_record.get("file_name") or "").strip()
        if not file_name:
            continue
        sample_index = len(schedule)
        schedule.append(
            CocoEntry(
                sample_index=sample_index,
                image_id=image_id,
                file_name=file_name,
                caption=caption,
                sample_seed=int(seed) + sample_index,
            )
        )

    if len(schedule) < int(num_samples):
        # Soft cap: COCO val2014 has ~40504 unique (image, caption) pairs, so
        # users asking for ``--num-samples 100000`` should be rounded down
        # rather than crashed; emit a clear warning to stderr so it shows up
        # in the launcher logs and the run_manifest will reflect the actual
        # count.
        warnings.warn(
            f"Requested num_samples={int(num_samples)} but only "
            f"{len(schedule)} image+caption pairs were available in "
            f"{annotations_path}; capping to {len(schedule)}.",
            stacklevel=2,
        )
    return schedule


def write_coco_schedule_jsonl(entries: Sequence[CocoEntry], output_path: str | Path) -> Path:
    """Persist the schedule as ``prompts.jsonl`` for downstream tooling."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return output_file


# ---------------------------------------------------------------------------
# Reference image preprocessing
# ---------------------------------------------------------------------------


def center_crop_and_resize(image: Image.Image, image_size: int) -> np.ndarray:
    """Center-crop ``image`` to a square (short-side anchored) and resize to
    ``image_size``. Returns a contiguous ``[H, W, 3]`` uint8 ``numpy`` array.

    Matches the canonical FID reference preprocessing (Inception v3 expects
    square inputs at the eval resolution; cropping first avoids the squashing
    artifacts a naive ``resize((S, S))`` would introduce on COCO's mostly
    landscape photos).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    short_side = min(width, height)
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    cropped = image.crop((left, top, left + short_side, top + short_side))
    resized = cropped.resize((int(image_size), int(image_size)), Image.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def preprocess_reference_shard(
    *,
    schedule: Sequence[CocoEntry],
    images_dir: str | Path,
    sample_indices: np.ndarray,
    image_size: int,
    progress_desc: str | None = None,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``center_crop_and_resize`` on the schedule rows referenced by
    ``sample_indices``. Returns ``(images_uint8 [N, S, S, 3], indices_int64)``
    where rows are aligned with ``sample_indices`` (caller should hand both
    arrays plus a zeros class array to ``save_local_sample_shard``).

    ``num_workers``: when > 1, parallelize the JPEG decode + bicubic resize
    inside this rank with a ``ThreadPoolExecutor``. Both libjpeg-turbo and
    PIL's resize release the GIL, so threads give a near-linear speedup on
    HDD-backed images directories without the pickling cost that
    ``ProcessPoolExecutor`` would impose on the schedule. ``0`` / ``1``
    falls back to the original single-thread loop (kept as the default to
    avoid surprising callers).
    """
    images_root = Path(images_dir).expanduser().resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"COCO images directory not found: {images_root}")

    out_indices = np.asarray(sample_indices, dtype=np.int64)
    if out_indices.size == 0:
        return (
            np.zeros((0, int(image_size), int(image_size), 3), dtype=np.uint8),
            out_indices,
        )

    images_buffer = np.empty(
        (int(out_indices.size), int(image_size), int(image_size), 3),
        dtype=np.uint8,
    )

    def _process_one(slot: int, sample_index: int) -> None:
        entry = schedule[int(sample_index)]
        image_path = images_root / entry.file_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing COCO val image for image_id={entry.image_id} "
                f"(sample_index={entry.sample_index}): {image_path}"
            )
        with Image.open(image_path) as raw_image:
            arr = center_crop_and_resize(raw_image, int(image_size))
        images_buffer[int(slot)] = arr

    items = list(enumerate(out_indices.tolist()))

    if int(num_workers) > 1:
        # Threaded: PIL JPEG decode + resize release the GIL, so 4-8 workers
        # per rank typically give a 2-4x speedup on a busy HDD/NFS images
        # directory. Each ``slot`` is unique, so concurrent writes into
        # ``images_buffer`` can't race.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=int(num_workers)) as ex:
            futures = [
                ex.submit(_process_one, slot, sample_index)
                for slot, sample_index in items
            ]
            iterator = as_completed(futures)
            if progress_desc is not None:
                iterator = tqdm(
                    iterator,
                    total=len(items),
                    desc=f"{progress_desc} x{int(num_workers)}",
                    leave=True,
                    dynamic_ncols=True,
                    mininterval=2.0,
                    file=sys.stderr,
                )
            for fut in iterator:
                fut.result()
        return images_buffer, out_indices

    iterator = items
    if progress_desc is not None:
        iterator = tqdm(
            iterator,
            total=len(items),
            desc=progress_desc,
            leave=True,
            dynamic_ncols=True,
            mininterval=2.0,
            file=sys.stderr,
        )
    for slot, sample_index in iterator:
        _process_one(slot, sample_index)

    return images_buffer, out_indices
