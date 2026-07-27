"""Metric task registry.

Each ``KIND`` corresponds to one (target x metric-group) chunk of work. Tasks
of the same kind share a model load on each rank; the dispatcher loops over
kinds in ``KIND_ORDER`` and within each kind splits ``manifest.records`` by
``records[rank::world_size]`` so every rank stays fully utilised even when only
one target exists.

A task module exposes:

    run_task(rank, local_rank, world_size, target_dir, manifest,
             skip_completed=True, **kwargs) -> None

It must write per-sample JSONs to ``target_dir/per_sample/<kind>/<stem>.json``
and may optionally write a global aggregation hook into
``target_dir/summary/<kind>.json`` (only when the metric is dataset-level, like
IS).
"""
from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Iterable, List, Set

# Default order matters: heaviest GPU consumers first so we never carry two big
# models at the same time across kinds. Lighter / CPU-only metrics run later.
DEFAULT_KIND_ORDER: List[str] = [
    "av_sync_synchformer",
    "av_sync_imagebind",
    "lip_sync",
    "pe_av",
    "audio_clap",
    "video_motion",
    "video_aesthetic",
    "identity_dino",
    "audio_fd_kl",
    "audio_box",
    "speech_wer",
    "audio_dnsmos",
    "audio_is",
]

# Supported but not run unless explicitly requested with --kinds.
OPTIONAL_KIND_ORDER: List[str] = [
    "audio_amplitude",
]

KIND_ORDER: List[str] = DEFAULT_KIND_ORDER + OPTIONAL_KIND_ORDER


# Default metric keys produced by each kind (used for summary aggregation).
KIND_METRIC_KEYS: Dict[str, List[str]] = {
    "av_sync_synchformer": ["DeSync"],
    "av_sync_imagebind": ["IB-AV", "IB-TV", "IB-TA"],
    "lip_sync":          ["LSE-C"],
    "pe_av":             [
        "PE-TV",
        "PE-TA",
        "PE-TAV",
        "PE-TV-cosine",
        "PE-TA-cosine",
        "PE-TAV-cosine",
    ],
    "audio_clap":        ["CLAP"],
    "video_motion":      ["MS"],
    "video_aesthetic":   ["Aesthetic", "MusiQ", "ManiQA", "AS"],
    "identity_dino":     ["ID"],
    "audio_fd_kl":       ["FD", "KL"],
    "audio_box":         ["CE", "CU", "PC", "PQ"],
    "speech_wer":        ["WER"],
    "audio_dnsmos":      ["P808_MOS"],
    "audio_is":          ["IS"],
    "audio_amplitude":   ["amplitude_rms", "loudness_lufs"],
}

OPTIONAL_KIND_METRIC_KEYS: Dict[str, List[str]] = {
    "av_sync_synchformer": ["AV-Align"],
    "lip_sync": ["LSE-D"],
}

ALL_KIND_METRIC_KEYS: Dict[str, List[str]] = {
    kind: list(keys) + list(OPTIONAL_KIND_METRIC_KEYS.get(kind, []))
    for kind, keys in KIND_METRIC_KEYS.items()
}

OPTIONAL_METRIC_NAMES: Set[str] = {
    key for keys in OPTIONAL_KIND_METRIC_KEYS.values() for key in keys
}


def metric_keys_for_kind(kind: str, optional_metrics: Iterable[str] | None = None) -> List[str]:
    keys = list(KIND_METRIC_KEYS[kind])
    requested = set(optional_metrics or [])
    if "all" in requested or kind in requested:
        keys.extend(OPTIONAL_KIND_METRIC_KEYS.get(kind, []))
    else:
        keys.extend(k for k in OPTIONAL_KIND_METRIC_KEYS.get(kind, []) if k in requested)
    return keys


# Kinds whose summary is computed dataset-wise (not as a mean of per-sample
# values). They write their own ``summary/<kind>.json`` and the consolidator
# leaves it alone.
DATASET_LEVEL_SUMMARY_KINDS = {"audio_is"}


# Kinds that are only meaningful on a subset of VerseBench categories. Records
# outside these categories are counted as skipped, not failed, in summaries.
KIND_ELIGIBLE_CATEGORIES: Dict[str, Set[str]] = {
    "lip_sync": {"set3", "set3-large", "set3-medium-large"},
}


def get_run_task(kind: str) -> Callable:
    if kind not in KIND_ORDER:
        raise KeyError(f"unknown metric kind: {kind!r}; supported: {KIND_ORDER}")
    module = import_module(f"my_eval.tasks.{kind}")
    return module.run_task


def get_preload_task(kind: str) -> Callable | None:
    if kind not in KIND_ORDER:
        raise KeyError(f"unknown metric kind: {kind!r}; supported: {KIND_ORDER}")
    module = import_module(f"my_eval.tasks.{kind}")
    return getattr(module, "preload_task", None)


def clear_model_cache(kind: str) -> List[str]:
    """Drop resident model caches for one metric kind, if the task exposes them."""
    if kind not in KIND_ORDER:
        raise KeyError(f"unknown metric kind: {kind!r}; supported: {KIND_ORDER}")
    module = import_module(f"my_eval.tasks.{kind}")
    clear_fn = getattr(module, "clear_model_cache", None)
    if callable(clear_fn):
        cleared = clear_fn()
        return list(cleared or [])

    cleared_names: List[str] = []
    for name, value in vars(module).items():
        if name.endswith("_CACHE") and hasattr(value, "clear"):
            value.clear()
            cleared_names.append(name)
    return cleared_names
