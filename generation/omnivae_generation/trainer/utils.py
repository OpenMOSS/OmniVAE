from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

CHECKPOINT_COMPLETE_MARKER_NAME = ".checkpoint_complete"


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def flatten_gathered_record_chunks(gathered: list[Any]) -> list[dict]:
    """Normalize ``gather_object(local_records)`` into a flat list of dict records.

    Under distributed training, Accelerate flattens each rank's list into one list.
    In single-process mode, ``gather_object`` returns the argument unchanged (already flat).
    """
    if not gathered:
        return []
    merged: list[dict] = []
    for chunk in gathered:
        if isinstance(chunk, dict):
            merged.append(chunk)
        elif isinstance(chunk, list):
            merged.extend(chunk)
        else:
            raise TypeError(f"Unexpected gather chunk type {type(chunk)!r}")
    return merged


def get_checkpoint_complete_marker_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / CHECKPOINT_COMPLETE_MARKER_NAME


def mark_checkpoint_complete(checkpoint_dir: str | Path) -> Path:
    marker_path = get_checkpoint_complete_marker_path(checkpoint_dir)
    marker_path.write_text("complete\n", encoding="utf-8")
    return marker_path


def is_checkpoint_complete(checkpoint_dir: str | Path) -> bool:
    return get_checkpoint_complete_marker_path(checkpoint_dir).exists()


def find_latest_complete_checkpoint(checkpoint_root: str | Path) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_root)
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        [path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda path: path.name,
        reverse=True,
    )
    for checkpoint in checkpoints:
        if is_checkpoint_complete(checkpoint):
            return checkpoint
    return None


def rotate_checkpoints(checkpoint_root: str | Path, limit: Optional[int]) -> None:
    if limit is None or limit <= 0:
        return

    checkpoint_dir = Path(checkpoint_root)
    checkpoints = sorted(
        [path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda path: path.name,
    )
    while len(checkpoints) > limit:
        stale = checkpoints.pop(0)
        shutil.rmtree(stale, ignore_errors=True)
