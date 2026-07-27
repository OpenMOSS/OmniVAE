"""Shared CPU preprocessing warmup for my_eval.

The metric tasks still own their model-specific transforms. This module only
warms reusable raw assets: mono audio at common sample rates and full RGB video
frames for the metrics that consume every frame.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List

from my_eval.utils.audio_video import load_video_rgb_array, load_wav_mono
from my_eval.utils.distributed import log


AUDIO_SR_BY_KIND: Dict[str, tuple[int, ...]] = {
    "av_sync_synchformer": (16000,),
    "av_sync_imagebind": (16000,),
    "pe_av": (48000,),
    "audio_clap": (48000,),
    "audio_fd_kl": (48000, 32000),
    "audio_dnsmos": (16000,),
    "audio_is": (32000,),
    "audio_amplitude": (48000,),
}

VIDEO_RGB_KINDS = {"pe_av", "video_motion", "video_aesthetic", "identity_dino"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_choice(name: str, default: str) -> str:
    value = os.environ.get(name)
    return (value if value is not None else default).strip().lower()


def _audio_srs_for_kinds(kinds: Iterable[str]) -> list[int]:
    out: set[int] = set()
    for kind in kinds:
        out.update(AUDIO_SR_BY_KIND.get(kind, ()))
    return sorted(out)


def _reference_audio_paths(record: dict[str, Any], kinds: Iterable[str]) -> list[str]:
    if "audio_fd_kl" not in set(kinds):
        return []
    try:
        from my_eval.utils.versebench_refs import resolve_reference_audio
        ref = resolve_reference_audio(record)
        return [ref] if ref else []
    except Exception:
        return []


class _PreprocessDataset:
    def __init__(self, records: list[dict[str, Any]], kinds: list[str], decode_video: bool) -> None:
        self.records = records
        self.kinds = kinds
        self.audio_srs = _audio_srs_for_kinds(kinds)
        self.decode_video = decode_video

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        errors: list[str] = []
        audio_paths = [str(rec.get("audio_path") or "")]
        audio_paths.extend(_reference_audio_paths(rec, self.kinds))
        for audio_path in audio_paths:
            if not audio_path:
                continue
            for sr in self.audio_srs:
                try:
                    load_wav_mono(audio_path, sr)
                except Exception as exc:
                    errors.append(f"audio:{Path(audio_path).name}:sr{sr}:{exc}")
        if self.decode_video and rec.get("video_path"):
            try:
                load_video_rgb_array(str(rec["video_path"]))
            except Exception as exc:
                errors.append(f"video:{Path(str(rec['video_path'])).name}:{exc}")
        return {"file_stem": rec.get("file_stem", ""), "errors": errors}


def _collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def _decode_video_for_prewarm(needs_video: bool, workers: int) -> bool:
    mode = _env_choice("MY_EVAL_PREPROCESS_VIDEO", "auto")
    if mode in {"1", "true", "yes", "on", "video", "all"}:
        return needs_video
    if mode in {"0", "false", "no", "off", "audio", "audio-only", "none"}:
        return False

    # Process workers cannot populate the parent process memory cache. Video
    # arrays are also large, so only pre-decode them by default when the result
    # is usable by the current process or a disk cache has explicitly been
    # enabled. Thread mode can be forced with MY_EVAL_PREPROCESS_VIDEO=1, but
    # audio-only is the safer no-disk default.
    video_disk_cache = _env_flag("MY_EVAL_VIDEO_DISK_CACHE", False)
    return needs_video and (workers == 0 or video_disk_cache)


def _run_thread_warmup(dataset: _PreprocessDataset, workers: int, rank: int) -> list[str]:
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for idx, item in enumerate(pool.map(dataset.__getitem__, range(len(dataset))), 1):
            errors.extend(str(err) for err in item.get("errors", []))
            if idx % 50 == 0:
                log(rank, f"[preprocess] warmed {idx}/{len(dataset)}")
    return errors


def prewarm_preprocess_records(
    *,
    records: list[dict[str, Any]],
    kinds: list[str],
    rank: int,
) -> dict[str, Any]:
    workers = max(0, _env_int("MY_EVAL_PREPROCESS_WORKERS", 2))
    backend = _env_choice("MY_EVAL_PREPROCESS_BACKEND", "dataloader")
    if backend in {"process", "processes", "multiprocess", "multiprocessing"}:
        backend = "dataloader"
    if not records or not kinds:
        return {
            "elapsed_sec": 0.0,
            "num_records": 0,
            "num_errors": 0,
            "workers": workers,
            "backend": backend,
        }

    needs_video = bool(set(kinds) & VIDEO_RGB_KINDS)
    decode_video = _decode_video_for_prewarm(needs_video, workers)
    if not _audio_srs_for_kinds(kinds) and not decode_video:
        return {
            "elapsed_sec": 0.0,
            "num_records": len(records),
            "num_errors": 0,
            "workers": workers,
            "backend": backend,
            "decode_video": decode_video,
            "audio_srs": [],
        }

    started_at = time.time()
    errors: list[str] = []
    dataset = _PreprocessDataset(records, kinds, decode_video=decode_video)
    batch_size = max(1, _env_int("MY_EVAL_PREPROCESS_BATCH_SIZE", 1))

    try:
        if backend in {"thread", "threads", "threadpool"} and workers > 0:
            errors.extend(_run_thread_warmup(dataset, workers, rank))
        else:
            backend = "dataloader"
            from torch.utils.data import DataLoader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                collate_fn=_collate,
                persistent_workers=workers > 0,
                prefetch_factor=max(1, _env_int("MY_EVAL_PREPROCESS_PREFETCH_FACTOR", 2)) if workers > 0 else None,
            )
            for batch_idx, batch in enumerate(loader, 1):
                for item in batch:
                    errors.extend(str(err) for err in item.get("errors", []))
                if batch_idx % 50 == 0:
                    log(rank, f"[preprocess] warmed {min(batch_idx * batch_size, len(records))}/{len(records)}")
    except Exception as exc:
        errors.append(f"{backend}:{exc}")

    elapsed = time.time() - started_at
    if errors:
        log(rank, f"[preprocess] warmup completed with {len(errors)} error(s); first={errors[0]}")
    else:
        log(rank, f"[preprocess] warmup completed records={len(records)} backend={backend} workers={workers} "
                  f"decode_video={decode_video} elapsed={elapsed:.3f}s")
    return {
        "elapsed_sec": elapsed,
        "num_records": len(records),
        "num_errors": len(errors),
        "workers": workers,
        "backend": backend,
        "decode_video": decode_video,
        "audio_srs": _audio_srs_for_kinds(kinds),
    }
