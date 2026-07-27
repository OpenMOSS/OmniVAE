"""Synchformer-only AV sync metrics.

This wrapper keeps Synchformer/DeSync separate from ImageBind metrics while
reusing the implementation helpers in ``av_sync_imagebind``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from my_eval.tasks import av_sync_imagebind as _impl


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    return _impl.preload_task(
        rank=rank,
        local_rank=local_rank,
        metric_keys=metric_keys or ["DeSync"],
        **kwargs,
    )


def clear_model_cache() -> List[str]:
    return _impl.clear_model_cache()


def run_task(
    rank: int,
    local_rank: int,
    world_size: int,
    target_dir: Path,
    manifest: Dict[str, Any],
    skip_completed: bool = True,
    metric_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    return _impl.run_task(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        target_dir=target_dir,
        manifest=manifest,
        skip_completed=skip_completed,
        metric_keys=metric_keys or ["DeSync"],
        output_kind="av_sync_synchformer",
        **kwargs,
    )
