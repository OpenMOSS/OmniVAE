"""Paired audio + video dataset with multi-source weighted sampling.

Each line of every input jsonl describes one mp4 clip and (at minimum)
must provide:

  * a path to the mp4 (read from ``source["path_field"]``; common values
    are ``video_path`` for the av_caption shards and ``audio_path`` /
    ``mp4_path`` for tta-from-video legacy shards);
  * a caption text or list (read from ``source["prompt_field"]``; lists
    have one element drawn at random per ``__getitem__``).

Both modalities are decoded from the *same* mp4 so they are always
temporally byte-aligned (the dataset deliberately ignores any
``start_time`` / ``end_time`` fields that may exist in legacy shards;
see ``_load_audio`` for the strict-alignment rationale).

Multi-source weighted sampling
------------------------------
The dataset accepts ``sources: List[dict]``, each entry being::

    {
        name: <source label, used for logging>,
        path: <jsonl path>,
        path_field: <jsonl key holding the mp4 path>,
        prompt_field: <jsonl key holding the caption / caption list>,
        weight: <float, relative draw probability across sources>,
    }

Indexing uses the same encoded-index trick as ``AudioJsonlT2ADataset``::

    encoded = source_idx * _ENCODING_STRIDE + offset_idx

so the standard ``Sampler`` interface (which yields ints) keeps working
unchanged. The companion ``WeightedShuffledCycleStatefulSampler`` (in
``omnivae_generation.trainer.stateful_dataloader``) is what produces these encoded indices
when there are multiple sources or a non-1.0 weight; for a single-
source spec we fall back to the trainer's regular distributed shuffle
sampler and the encoding still decodes correctly (``source=0``).

Backwards compatibility: passing ``jsonl_path=...`` (with optional
``path_field`` / ``prompt_field``) instead of ``sources`` builds a
single-source spec internally so the old single-jsonl yamls keep
working byte-for-byte.

Sample fields produced (matches what the trainer's collator expects):

  * ``video``: ``[C=3, T, H, W]`` float32 in ``[-1, 1]`` (or uint8
    ``[0, 255]`` if ``return_uint8 = True``)
  * ``audio``: ``[1, num_audio_samples]`` float32, mono, padded
  * ``valid_num_samples``: int -- audio's pre-pad length
  * ``prompt`` / ``empty_prompt``: chat-formatted strings
  * ``audio_path`` / ``video_path``: mp4 location (same value)
  * ``duration``: float (seconds, post-clipping)
  * ``source_index`` / ``source_name``: which entry of ``sources`` the
    sample was drawn from; handy for per-source loss attribution / wandb.
  * ``prompt_input_ids`` / ... (optional, when ``tokenizer is not None``)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from tqdm.auto import tqdm

from omnivae_generation.trainer.audio_task_prefix import KIND_T2AV, apply_task_prefix
from omnivae_generation.trainer.data import (
    add_tokenized_prompt_fields,
    collate_tokenized_prompt_fields,
    maybe_format_chat_prompt,
    maybe_tokenize_prompt_to_tensors,
)


logger = logging.getLogger(__name__)

_MAX_BAD_SAMPLE_RETRIES = 16

# How often a worker process should run a manual ``gc.collect()`` to keep
# native (ffmpeg / torchcodec) buffer fragmentation from compounding under
# ``persistent_workers=True``. Empirically each torchcodec
# AudioDecoder / VideoDecoder open+close leaves a tiny amount of native
# state in the glibc arena that the GC can't reach; periodic collect calls
# don't drop the native memory directly but they free Python-side objects
# that pin C buffers, giving the next ``malloc_trim`` a chance to release
# pages back to the OS. Tuned for ~1 collection per ~150 sample ⇒ ~0.5%
# overhead on a 250ms-per-sample workload.
_GC_EVERY_N_SAMPLES = 128

# Cap on the number of bad-sample indices we remember per worker. Strictly
# bounded so a long-lived persistent worker doesn't accumulate a huge set
# (each entry is tiny but with millions of samples it adds up). FIFO drop
# is fine because a "bad" sample doesn't suddenly become good.
_BAD_INDEX_CAP = 1 << 16

# 4 MB chunk works well for the offset scan over multi-GB jsonl shards.
_OFFSET_SCAN_CHUNK_SIZE = 1 << 22

# Encoding stride for ``__getitem__``: encoded = source_idx * STRIDE + offset.
# 2**40 supports up to ~1 trillion lines per source, well beyond any
# realistic jsonl shard. Must match the value the companion sampler reads
# from ``dataset.stride``.
_ENCODING_STRIDE = 1 << 40

# Defaults applied to per-source dicts when keys are omitted.
_DEFAULT_PATH_FIELD = "video_path"
_DEFAULT_PROMPT_FIELD = "av_caption"
# Fallback fields tried when the source's path_field comes back empty
# (covers legacy tta-from-video shards which use ``audio_path``).
_FALLBACK_PATH_FIELDS = ("video_path", "audio_path", "mp4_path")


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scan_newline_offsets(path: Path, *, show_progress: bool = True) -> np.ndarray:
    """Return the byte offset of each newline-delimited line as ``np.int64``.

    Shows a ``tqdm`` progress bar (disabled on non-TTY) advancing by
    bytes read so multi-GB scans don't look like a hang.
    Hot-loop is vectorised via numpy ``np.frombuffer + np.where`` -- on
    a 5 GB jsonl this is ~10x faster than the per-byte python loop and
    keeps the GIL free so tqdm can refresh.
    """
    file_size = int(path.stat().st_size)
    if file_size == 0:
        return np.zeros(0, dtype=np.int64)
    pieces: list[np.ndarray] = [np.array([0], dtype=np.int64)]
    with path.open("rb") as handle:
        position = 0
        progress = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"scan {path.name}",
            disable=not show_progress,
            mininterval=0.5,
        )
        try:
            while True:
                chunk = handle.read(_OFFSET_SCAN_CHUNK_SIZE)
                if not chunk:
                    break
                arr = np.frombuffer(chunk, dtype=np.uint8)
                newline_positions = np.flatnonzero(arr == 0x0A)
                if newline_positions.size:
                    pieces.append(newline_positions.astype(np.int64) + (position + 1))
                position += len(chunk)
                progress.update(len(chunk))
        finally:
            progress.close()
    offsets = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    # Drop the trailing offset if the file ended with a newline (the
    # last "line" would be empty).
    if offsets.size > 0 and int(offsets[-1]) >= file_size:
        offsets = offsets[:-1]
    return offsets


def _normalize_sources(
    sources: Sequence[dict] | None,
    *,
    legacy_jsonl_path: str | Path | None,
    legacy_path_field: str,
    legacy_prompt_field: str,
) -> list[dict]:
    """Validate + materialize the user-facing ``sources`` spec.

    The returned list is uniform: every entry has ``name / path /
    path_field / prompt_field / weight`` with sensible defaults filled
    in. Unknown keys are passed through untouched so callers can stash
    debug metadata.

    When ``sources`` is None / empty we fall back to a single-source
    spec built from ``legacy_jsonl_path`` (for backwards compatibility
    with the older single-``jsonl_path`` yaml schema).
    """
    if sources:
        normalized: list[dict] = []
        for raw in sources:
            if not isinstance(raw, dict):
                raise ValueError(
                    f"AVPairedJsonlDataset.sources entries must be dicts, got: {raw!r}"
                )
            path = raw.get("path") or raw.get("jsonl_path")
            if not path:
                raise ValueError(
                    "AVPairedJsonlDataset.sources[*].path is required."
                )
            weight = float(raw.get("weight", 1.0))
            if weight < 0:
                raise ValueError(
                    f"AVPairedJsonlDataset.sources[*].weight must be >= 0, got {weight}"
                )
            entry = dict(raw)  # passthrough debug keys
            entry["path"] = str(path)
            entry["path_field"] = str(raw.get("path_field", legacy_path_field))
            entry["prompt_field"] = str(raw.get("prompt_field", legacy_prompt_field))
            entry["weight"] = weight
            entry["name"] = str(
                raw.get("name") or Path(str(path)).stem or f"source_{len(normalized)}"
            )
            normalized.append(entry)
        if not normalized:
            raise ValueError("AVPairedJsonlDataset.sources must be a non-empty list.")
        return normalized

    if not legacy_jsonl_path:
        raise ValueError(
            "AVPairedJsonlDataset requires either dataset.sources or "
            "dataset.jsonl_path (legacy single-source schema)."
        )
    return [
        {
            "name": Path(str(legacy_jsonl_path)).stem or "source_0",
            "path": str(legacy_jsonl_path),
            "path_field": str(legacy_path_field),
            "prompt_field": str(legacy_prompt_field),
            "weight": 1.0,
        }
    ]


def _is_global_main_process() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _resize_and_crop_video_frames(
    video_thwc: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    center_crop: bool,
    random_flip: bool,
) -> torch.Tensor:
    """Take ``[T, C, H, W]`` and return ``[C, T, target_height,
    target_width]`` via short-side resize + center / random crop."""
    if video_thwc.dim() != 4:
        raise ValueError(f"Expected [T, C, H, W], got shape {tuple(video_thwc.shape)}.")
    src_h = int(video_thwc.shape[-2])
    src_w = int(video_thwc.shape[-1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid frame size: H={src_h}, W={src_w}.")

    scale = max(float(target_height) / src_h, float(target_width) / src_w)
    new_h = max(target_height, int(math.ceil(src_h * scale)))
    new_w = max(target_width, int(math.ceil(src_w * scale)))

    # F.interpolate expects [N, C, H, W]; treat T as N for batched resize.
    resized = F.interpolate(
        video_thwc.float(),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if center_crop:
        crop_top = (new_h - target_height) // 2
        crop_left = (new_w - target_width) // 2
    else:
        crop_top = random.randint(0, max(0, new_h - target_height))
        crop_left = random.randint(0, max(0, new_w - target_width))
    cropped = resized[..., crop_top : crop_top + target_height, crop_left : crop_left + target_width]
    if random_flip and random.random() < 0.5:
        cropped = torch.flip(cropped, dims=[-1])
    return cropped.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]


class AVPairedJsonlDataset(Dataset):
    """Yields ``(video, audio, prompt)`` triplets sourced from the same MP4.

    Supports multi-source weighted sampling: pass ``sources=[{path,
    path_field, prompt_field, weight, name}, ...]`` to mix several jsonl
    shards under a fixed weight ratio. Single-source ``jsonl_path=...``
    is preserved for backwards compatibility and gets transparently
    converted to a one-element ``sources`` list.

    The video / audio decoder pair is constructed *per worker process*
    (lazily, at first use after a fork) so multi-worker data loading
    doesn't accidentally share a file descriptor and corrupt the read
    cursor. The same applies to the per-source jsonl read handles.

    Indexing:
        Each ``__getitem__(encoded_idx)`` decodes the integer as
        ``(source_idx, offset_idx) = divmod(encoded_idx, stride)``. The
        companion :class:`WeightedShuffledCycleStatefulSampler` produces
        these encoded indices for the multi-source case; for single-
        source we fall back to the regular distributed shuffle sampler
        and ``source_idx`` is always 0.
    """

    def __init__(
        self,
        *,
        # ----- Source spec (multi-source weighted) ----- #
        sources: Sequence[dict] | None = None,
        # ----- Legacy single-source spec (backwards compat) ----- #
        jsonl_path: str | Path | None = None,
        path_field: str = _DEFAULT_PATH_FIELD,
        prompt_field: str = _DEFAULT_PROMPT_FIELD,
        # ----- Video knobs (mirror ``VideoJsonlDataset``) ----- #
        frame_size: tuple[int, int] = (256, 256),
        num_frames: int = 121,
        target_fps: float = 24.0,
        center_crop: bool = True,
        random_flip: bool = False,
        return_uint8: bool = True,
        # ----- Audio knobs (mirror ``AudioJsonlT2ADataset``) ----- #
        sample_rate: int = 48000,
        num_audio_samples: int = 1440000,
        mono: bool = True,
        append_duration_suffix: bool = True,
        duration_precision: int = 1,
        task_prefix_enabled: bool = True,
        # ----- Prompt tokenization (matches the audio dataset's flow) ----- #
        tokenizer=None,
        prompt_max_sequence_length: int | None = None,
        max_samples: int | None = None,
        cache_dir: str | None = None,        # ignored, kept for parity
        # Strict-alignment guards (T2AV needs frame-accurate AV pairing).
        # ``min_clip_duration_ratio`` is a fraction of the requested
        # window: a clip whose mp4 header advertises a duration shorter
        # than ``ratio * (num_frames / target_fps)`` is treated as a bad
        # sample and skipped. 0 disables the check (matches the legacy
        # tta-from-video pipeline that pads with silence/last-frame).
        min_clip_duration_ratio: float = 0.99,
    ):
        super().__init__()
        self.sources = _normalize_sources(
            sources,
            legacy_jsonl_path=jsonl_path,
            legacy_path_field=path_field,
            legacy_prompt_field=prompt_field,
        )
        # Sanity-check every source path up-front so a typo aborts at
        # construction rather than mid-training when a worker tries to
        # seek into a non-existent file.
        for src in self.sources:
            src_path = Path(src["path"]).expanduser().resolve()
            if not src_path.is_file():
                raise FileNotFoundError(
                    f"AVPairedJsonlDataset: source {src['name']!r} jsonl not found at {src_path}"
                )
            src["path"] = str(src_path)

        self.frame_size = tuple(int(x) for x in frame_size)
        self.num_frames = int(num_frames)
        self.target_fps = float(target_fps)
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        self.return_uint8 = bool(return_uint8)
        self.sample_rate = int(sample_rate)
        self.num_audio_samples = int(num_audio_samples)
        self.mono = bool(mono)
        self.append_duration_suffix = bool(append_duration_suffix)
        self.duration_precision = int(duration_precision)
        self.task_prefix_enabled = bool(task_prefix_enabled)
        self.tokenizer = tokenizer
        self.prompt_max_sequence_length = (
            int(prompt_max_sequence_length) if prompt_max_sequence_length else None
        )
        # The requested clip window the AV pair must cover, in seconds.
        # Both modalities are read from [0, target_window_seconds] so the
        # bridge cross-attention sees frame-accurate AV alignment (no
        # entry["start_time"] / "end_time" branch).
        self.target_window_seconds = float(self.num_frames) / float(self.target_fps)
        self.min_clip_duration_ratio = max(0.0, float(min_clip_duration_ratio))
        self.min_clip_duration_seconds = (
            self.target_window_seconds * self.min_clip_duration_ratio
        )

        self.empty_prompt = maybe_format_chat_prompt("", tokenizer)
        self.empty_prompt_tokens = maybe_tokenize_prompt_to_tensors(
            self.empty_prompt, tokenizer, self.prompt_max_sequence_length,
        )

        # Per-source line offsets / sizes / weights, all parallel arrays.
        self.stride: int = _ENCODING_STRIDE
        self.source_offsets: list[np.ndarray] = []
        self.source_sizes: list[int] = []
        self.source_weights: list[float] = []
        self._scan_all_sources()
        self._total_len = sum(self.source_sizes)
        if max_samples is not None and int(max_samples) > 0:
            self._total_len = min(self._total_len, int(max_samples))
        self._summarize_sources()

        # Per-process file handles for each source. Reset on fork (each
        # DataLoader worker is a forked subprocess and must NOT reuse
        # the parent's fds because they share the file position).
        self._handles: dict[int, Any] = {}
        self._handles_pid: int | None = None
        # Cross-source bad-sample dedup: keys are encoded indices. Cap
        # to ``_BAD_INDEX_CAP`` so a long-running persistent worker
        # doesn't grow this monotonically across millions of samples.
        self._bad_indices: set[int] = set()
        # Per-worker sample counter that drives the periodic
        # ``gc.collect()``. Reset whenever the worker pid changes (i.e.
        # at fork into a new dataloader worker process) so each worker
        # has its own cadence.
        self._sample_counter: int = 0
        self._sample_counter_pid: int | None = None

    def _scan_all_sources(self) -> None:
        show_progress = _is_global_main_process()
        for src_idx, source in enumerate(self.sources):
            path = Path(source["path"])
            try:
                file_size = path.stat().st_size
            except OSError:
                file_size = 0
            if show_progress:
                print(
                    f"[AVPairedJsonlDataset] scanning newline offsets for "
                    f"{path} ({file_size / (1024 ** 3):.2f} GiB, source={source['name']!r}) ...",
                    flush=True,
                )
            offsets = _scan_newline_offsets(path, show_progress=show_progress)
            if offsets.size == 0:
                raise ValueError(
                    f"AVPairedJsonlDataset source {source['name']!r} ({path}) is empty (zero lines)."
                )
            if int(offsets.size) > self.stride:
                raise ValueError(
                    f"AVPairedJsonlDataset source {source['name']!r} has {int(offsets.size)} "
                    f"lines which exceeds the per-source stride of {self.stride}; "
                    f"rebuild with a larger _ENCODING_STRIDE."
                )
            self.source_offsets.append(offsets)
            self.source_sizes.append(int(offsets.size))
            self.source_weights.append(float(source["weight"]))
            if show_progress:
                print(
                    f"[AVPairedJsonlDataset] source={source['name']!r} "
                    f"line_count={int(offsets.size):,} weight={source['weight']:.4f}",
                    flush=True,
                )

    def _summarize_sources(self) -> None:
        for source, line_count in zip(self.sources, self.source_sizes):
            logger.info(
                "AVPairedJsonlDataset source name=%s weight=%.4f line_count=%d "
                "path_field=%s prompt_field=%s path=%s",
                source["name"],
                source["weight"],
                line_count,
                source["path_field"],
                source["prompt_field"],
                source["path"],
            )

    def __len__(self) -> int:
        return self._total_len

    # ----------------------------------------------------------- handle mgmt
    def _get_handle(self, source_idx: int):
        pid = os.getpid()
        if self._handles_pid != pid:
            # Forked into a worker; parent's fds are unsafe (shared
            # file position). Drop the cached references and re-open
            # in this process on demand.
            self._handles = {}
            self._handles_pid = pid
        handle = self._handles.get(source_idx)
        if handle is None:
            handle = open(self.sources[source_idx]["path"], "rb")
            self._handles[source_idx] = handle
        return handle

    def _maybe_periodic_gc(self) -> None:
        """Run ``gc.collect()`` every ``_GC_EVERY_N_SAMPLES`` calls.

        torchcodec's AudioDecoder / VideoDecoder hold native ffmpeg
        buffers via cyclic Python references; without periodic GC those
        cycles leak under ``persistent_workers=True`` and accumulate
        into a slow RSS climb that eventually trips the OOM-killer
        after several hours of training. This is a cheap insurance
        policy -- a full collection on a freshly-allocated worker
        takes well under 10 ms.
        """
        pid = os.getpid()
        if self._sample_counter_pid != pid:
            self._sample_counter = 0
            self._sample_counter_pid = pid
        self._sample_counter += 1
        if self._sample_counter % _GC_EVERY_N_SAMPLES == 0:
            gc.collect()

    def _record_bad_index(self, encoded: int) -> None:
        """Add ``encoded`` to the bad-sample set, evicting if oversized.

        Uses a simple "drop oldest" policy via ``set.pop`` (Python sets
        have no insertion order guarantee, but for a bad-sample dedup
        cache that's fine -- we just want a bounded membership test,
        not a true LRU).
        """
        self._bad_indices.add(encoded)
        if len(self._bad_indices) > _BAD_INDEX_CAP:
            try:
                self._bad_indices.pop()
            except KeyError:  # pragma: no cover -- only if set is empty
                pass

    def _read_entry(self, source_idx: int, offset_idx: int) -> dict | None:
        if source_idx < 0 or source_idx >= len(self.sources):
            raise IndexError(
                f"AVPairedJsonlDataset source index out of range: "
                f"{source_idx} (n_sources={len(self.sources)})."
            )
        offsets = self.source_offsets[source_idx]
        if offset_idx < 0 or offset_idx >= int(offsets.size):
            raise IndexError(
                f"AVPairedJsonlDataset offset index out of range: "
                f"{offset_idx} for source {source_idx} (size={int(offsets.size)})."
            )
        handle = self._get_handle(source_idx)
        handle.seek(int(offsets[offset_idx]))
        line = handle.readline()
        if not line or not line.strip():
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    # --------------------------------------------------------------- helpers
    def _resolve_caption(self, entry: dict, prompt_field: str) -> str | None:
        raw = entry.get(prompt_field)
        if raw is None:
            return None
        if isinstance(raw, list):
            candidates = [str(item).strip() for item in raw if str(item).strip()]
            if not candidates:
                return None
            return random.choice(candidates) if len(candidates) > 1 else candidates[0]
        text = str(raw).strip()
        return text or None

    def _resolve_mp4_path(self, entry: dict, primary_path_field: str) -> str | None:
        """Pull the mp4 path out of a jsonl entry.

        Tries ``primary_path_field`` first (per-source override), then
        falls back to the standard tta-from-video aliases so legacy
        shards keep working without touching the source spec.
        """
        candidate = entry.get(primary_path_field)
        if candidate:
            return str(candidate)
        for alt in _FALLBACK_PATH_FIELDS:
            if alt == primary_path_field:
                continue
            value = entry.get(alt)
            if value:
                return str(value)
        return None

    def _load_video(self, mp4_path: Path, decoder: VideoDecoder | None = None) -> torch.Tensor:
        owned = decoder is None
        if owned:
            decoder = VideoDecoder(str(mp4_path), dimension_order="NCHW")
        try:
            metadata = decoder.metadata
            fps = (
                float(getattr(metadata, "average_fps", 0.0) or 0.0)
                or float(getattr(metadata, "average_fps_from_header", 0.0) or 0.0)
                or self.target_fps
            )
            num_frames_total = int(
                getattr(metadata, "num_frames", None) or len(decoder)
            )
            stride = max(1.0, fps / self.target_fps)
            frame_indices = [
                min(int(round(i * stride)), max(0, num_frames_total - 1))
                for i in range(self.num_frames)
            ]
            decoded = decoder[frame_indices[0] : frame_indices[-1] + 1]
            gather_indices = torch.tensor(
                [idx - frame_indices[0] for idx in frame_indices], dtype=torch.long
            )
            sampled = decoded.index_select(0, gather_indices)  # [T, C, H, W]
            video = _resize_and_crop_video_frames(
                sampled,
                target_height=self.frame_size[0],
                target_width=self.frame_size[1],
                center_crop=self.center_crop,
                random_flip=self.random_flip,
            )
            if self.return_uint8:
                # torchcodec returns uint8 frames; resize cast them to float, so
                # round and clamp before returning.
                video = video.clamp(0, 255).round().to(torch.uint8)
            else:
                video = video.float() / 127.5 - 1.0
            return video
        finally:
            if owned and hasattr(decoder, "close"):
                try:
                    decoder.close()
                except Exception:  # pragma: no cover -- best-effort close
                    pass

    def _load_audio(self, mp4_path: Path, entry: dict) -> tuple[torch.Tensor, float, int]:
        """Decode audio strictly aligned with the video window ``[0, T]``.

        T2AV training requires per-sample frame-accurate AV alignment so the
        bridge cross-attention learns a meaningful temporal correspondence.
        ``_load_video`` always samples from frame 0, so we mirror that here:

        * Always read ``[0.0, num_audio_samples / sample_rate]`` from the
          source mp4. Any ``entry["start_time"]`` / ``entry["end_time"]``
          fields that may exist in legacy tta_from_video jsonls are
          *deliberately ignored* -- they would offset the audio without
          touching the video and silently break alignment.
        * If the source is shorter than the target window we still pad to
          ``num_audio_samples`` (mirroring how ``_load_video`` repeats the
          last frame), and surface ``valid_num_samples`` so downstream code
          can decide whether to drop the sample.
        """
        target_seconds = float(self.num_audio_samples) / float(self.sample_rate)
        decoder = AudioDecoder(str(mp4_path))
        try:
            duration_seconds = float(decoder.metadata.duration_seconds_from_header or 0.0)
            window_end = (
                min(target_seconds, duration_seconds) if duration_seconds > 0
                else target_seconds
            )
            if window_end > 0:
                samples = decoder.get_samples_played_in_range(0.0, window_end)
            else:
                # Header missing duration; fall back to decoding the whole
                # stream and trim afterwards. Rare but cheap on short clips.
                samples = decoder.get_all_samples()

            waveform = samples.data.float()
            source_sample_rate = int(samples.sample_rate)
            if source_sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, source_sample_rate, self.sample_rate
                )
            if self.mono and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.shape[0] == 0:
                raise ValueError(f"AudioDecoder returned empty waveform for {mp4_path}.")

            valid_num_samples = int(min(waveform.shape[-1], self.num_audio_samples))
            if waveform.shape[-1] > self.num_audio_samples:
                waveform = waveform[..., : self.num_audio_samples]
            elif waveform.shape[-1] < self.num_audio_samples:
                pad = self.num_audio_samples - waveform.shape[-1]
                waveform = F.pad(waveform, (0, pad))
            nonpad_duration = float(valid_num_samples) / float(self.sample_rate)
            return waveform.contiguous(), nonpad_duration, valid_num_samples
        finally:
            if hasattr(decoder, "close"):
                try:
                    decoder.close()
                except Exception:  # pragma: no cover
                    pass

    def _build_prompt(self, caption: str, duration_s: float) -> str:
        prompt = caption
        if self.task_prefix_enabled:
            prompt = apply_task_prefix(KIND_T2AV, prompt)
        if self.append_duration_suffix:
            fmt = f"{{:.{max(0, self.duration_precision)}f}}"
            prompt = f"{prompt} duration: {fmt.format(float(duration_s))}s"
        return maybe_format_chat_prompt(prompt, self.tokenizer)

    # --------------------------------------------------------------- __getitem__
    def __getitem__(self, encoded_idx: int) -> dict:
        if self._total_len <= 0:
            raise RuntimeError("AVPairedJsonlDataset is empty.")

        # Slow-leak insurance: periodic gc.collect inside the worker
        # process keeps torchcodec / ffmpeg cyclic references from
        # compounding into a multi-GB RSS climb across hours of
        # ``persistent_workers=True`` execution.
        self._maybe_periodic_gc()

        # Decode (source_idx, offset_idx) from the encoded sampler index.
        #
        # * Multi-source: the WeightedShuffledCycleStatefulSampler
        #   always emits encoded indices ``s * stride + i`` with
        #   ``stride = 2**40`` so divmod recovers (source, offset).
        # * Single-source legacy: the default distributed shuffle
        #   sampler emits raw indices in ``[0, len)``. Since
        #   ``len = source_sizes[0]`` and ``source_sizes[0] < stride``,
        #   ``divmod(raw_idx, stride)`` returns ``(0, raw_idx)`` which
        #   is exactly the desired decoding.
        encoded = int(encoded_idx)
        source_idx = encoded // self.stride
        offset_idx = encoded % self.stride
        if source_idx < 0 or source_idx >= len(self.sources):
            raise IndexError(
                f"AVPairedJsonlDataset received encoded index {encoded} which "
                f"resolves to source {source_idx} outside [0, {len(self.sources)}). "
                f"This usually means a flat-index sampler was attached to a "
                f"multi-source dataset; install WeightedShuffledCycleStatefulSampler "
                f"(via build_train_dataloader) instead."
            )
        source = self.sources[source_idx]
        source_size = self.source_sizes[source_idx]
        if len(self.sources) > 1 and offset_idx >= source_size:
            raise IndexError(
                f"AVPairedJsonlDataset got offset {offset_idx} >= source_size "
                f"{source_size} for source {source['name']!r}. The flat-index "
                f"path is only supported for single-source datasets; install "
                f"WeightedShuffledCycleStatefulSampler for multi-source mixing."
            )
        if source_size <= 0:
            raise RuntimeError(
                f"Source {source_idx} ({source['name']!r}) has zero entries."
            )
        attempt_offset = int(offset_idx) % source_size
        last_err: Exception | None = None

        for _ in range(_MAX_BAD_SAMPLE_RETRIES):
            attempt_encoded = source_idx * self.stride + attempt_offset
            if attempt_encoded in self._bad_indices:
                attempt_offset = (attempt_offset + 1) % source_size
                continue
            entry: dict | None = None
            try:
                entry = self._read_entry(source_idx, attempt_offset)
                if entry is None:
                    self._record_bad_index(attempt_encoded)
                    attempt_offset = (attempt_offset + 1) % source_size
                    continue

                raw_path = self._resolve_mp4_path(entry, source["path_field"])
                if not raw_path:
                    self._record_bad_index(attempt_encoded)
                    attempt_offset = (attempt_offset + 1) % source_size
                    continue

                caption = self._resolve_caption(entry, source["prompt_field"])
                if not caption:
                    self._record_bad_index(attempt_encoded)
                    attempt_offset = (attempt_offset + 1) % source_size
                    continue

                mp4_path = Path(str(raw_path)).expanduser()
                if not mp4_path.is_absolute():
                    # Resolve relative to the source's jsonl directory.
                    mp4_path = (Path(source["path"]).parent / mp4_path).resolve()

                # ---- Strict AV alignment guard (T2AV requires it) ----
                # We rely on the post-decode ``valid_num_samples`` check
                # below to enforce the minimum-duration contract. The
                # earlier upfront probe AudioDecoder was removed because
                # opening a third torchcodec decoder per sample triples
                # the native ffmpeg open/close churn; under
                # ``persistent_workers=True`` that compounded into a slow
                # RSS leak that OOM-killed dataloader workers after
                # ~10h of training. The post-decode guard gives the
                # same correctness with one fewer decoder per sample.

                video = self._load_video(mp4_path)
                audio, duration_s, valid_num_samples = self._load_audio(mp4_path, entry)

                # Defence-in-depth: even if the header lied, the post-decode
                # ``valid_num_samples`` tells us exactly how much real audio
                # we got. If the audio came back materially shorter than the
                # video window, treat it as a bad sample.
                if (
                    self.min_clip_duration_ratio > 0
                    and valid_num_samples < int(self.num_audio_samples * self.min_clip_duration_ratio)
                ):
                    self._record_bad_index(attempt_encoded)
                    logger.debug(
                        "AVPairedJsonlDataset: skip source=%s idx=%d path=%s decoded "
                        "valid_num_samples=%d < required=%d",
                        source["name"], attempt_encoded, mp4_path, valid_num_samples,
                        int(self.num_audio_samples * self.min_clip_duration_ratio),
                    )
                    attempt_offset = (attempt_offset + 1) % source_size
                    continue

                prompt = self._build_prompt(caption, duration_s)
                sample: dict[str, Any] = {
                    "video": video,
                    "audio": audio,
                    "prompt": prompt,
                    "empty_prompt": self.empty_prompt,
                    "video_path": str(mp4_path),
                    "audio_path": str(mp4_path),
                    "duration": float(duration_s),
                    "valid_num_samples": int(valid_num_samples),
                    "kind": KIND_T2AV,
                    "source_index": int(source_idx),
                    "source_name": str(source["name"]),
                }
                add_tokenized_prompt_fields(
                    sample,
                    prompt=prompt,
                    empty_prompt=self.empty_prompt,
                    tokenizer=self.tokenizer,
                    max_sequence_length=self.prompt_max_sequence_length,
                    empty_prompt_tokens=self.empty_prompt_tokens,
                )
                return sample
            except Exception as exc:
                last_err = exc
                if attempt_encoded not in self._bad_indices:
                    logger.warning(
                        "AVPairedJsonlDataset: skipping bad sample source=%s idx=%d "
                        "path=%s reason=%r",
                        source["name"], attempt_encoded,
                        (entry.get(source["path_field"], "<unknown>")
                         if entry is not None else "<unknown>"),
                        exc,
                    )
                    self._record_bad_index(attempt_encoded)
                attempt_offset = (attempt_offset + 1) % source_size
                continue

        raise RuntimeError(
            f"AVPairedJsonlDataset: failed to load a valid sample from "
            f"source {source['name']!r} after {_MAX_BAD_SAMPLE_RETRIES} retries. "
            f"Last error: {last_err!r}"
        )


def collate_av_paired_samples(batch: list[dict]) -> dict:
    """Standard collator for ``AVPairedJsonlDataset``.

    The video tensor is stacked along dim 0 ([B, C, T, H, W]); the
    audio tensor is stacked along dim 0 ([B, 1, num_samples]); prompt
    fields go through the existing tokenized-prompt collator.
    """
    collated: dict[str, Any] = {
        "pixel_values": torch.stack([item["video"] for item in batch]),
        "audio": torch.stack([item["audio"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "empty_prompts": [item["empty_prompt"] for item in batch],
        "video_paths": [item["video_path"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
        "image_paths": [item["video_path"] for item in batch],         # back-compat for loss-spike debugger
        "image_ids": [Path(item["video_path"]).name for item in batch],
        "durations": torch.tensor([item["duration"] for item in batch], dtype=torch.float32),
        "valid_num_samples": torch.tensor(
            [item["valid_num_samples"] for item in batch], dtype=torch.long
        ),
        "kinds": [item.get("kind", "") for item in batch],
        "source_indices": torch.tensor(
            [int(item.get("source_index", 0)) for item in batch], dtype=torch.long
        ),
        "source_names": [str(item.get("source_name", "")) for item in batch],
    }
    collate_tokenized_prompt_fields(collated, batch)
    return collated
