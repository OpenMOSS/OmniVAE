from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from omnivae_generation.trainer.audio_task_prefix import apply_task_prefix
from omnivae_generation.trainer.data import maybe_format_chat_prompt


logger = logging.getLogger(__name__)

_MAX_BAD_SAMPLE_RETRIES = 16

_KIND_TTS = "tts"
_KIND_TTA = "tta"
_KIND_LEGACY = "legacy"
_VALID_KINDS = {_KIND_TTS, _KIND_TTA, _KIND_LEGACY}

_DEFAULT_TTS_TEXT_FIELD = "text"
_DEFAULT_TTA_PROMPT_FIELD = "prompt_en"
_DEFAULT_AUDIO_FIELD = "audio_path"
_LEGACY_AUDIO_FIELDS = ("audio", "audio_path", "wav_path")
_LEGACY_TEXT_FIELDS = ("prompt", "text")

# Per-source encoding stride for `__getitem__`. We pack
# ``encoded = source_idx * STRIDE + offset_idx`` into a single int so the
# Sampler interface (which yields ints) keeps working without changes.
# 2**40 supports up to ~1 trillion lines per source, well beyond any
# realistic jsonl shard.
_ENCODING_STRIDE = 1 << 40

# 4 MB chunks: a good trade-off between syscall overhead and L3 cache pressure
# during the offset scan (sequential read of multi-GB jsonl files).
_OFFSET_SCAN_CHUNK_SIZE = 1 << 22


def _as_path(base_path: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_path / path


def _coerce_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_global_main_process() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _scan_jsonl_offsets(path: Path, *, show_progress: bool) -> np.ndarray:
    """Scan a jsonl file once and return the start byte offset of every line.

    No JSON parsing happens here, only newline scanning via numpy. A 41 GB
    file takes ~30-60 seconds with warm page cache, vs. ~5-30 minutes for a
    full ``json.loads`` pass.
    """
    path = Path(path)
    file_size = path.stat().st_size
    if file_size == 0:
        return np.zeros(0, dtype=np.int64)

    pieces: list[np.ndarray] = [np.array([0], dtype=np.int64)]
    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"[scan offsets] {path.name}",
                leave=False,
                dynamic_ncols=True,
            )
        except ImportError:
            progress = None

    pos = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_OFFSET_SCAN_CHUNK_SIZE)
                if not chunk:
                    break
                data = np.frombuffer(chunk, dtype=np.uint8)
                local = np.flatnonzero(data == 0x0A)  # '\n'
                if local.size:
                    # Each newline at local position p marks the end of a line
                    # whose successor begins at byte (pos + p + 1).
                    pieces.append(local.astype(np.int64) + (pos + 1))
                pos += len(chunk)
                if progress is not None:
                    progress.update(len(chunk))
    finally:
        if progress is not None:
            progress.close()

    offsets = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    # If the file ends in '\n', the last recorded offset points just past EOF;
    # drop it so every retained offset is the start of a real line.
    if offsets.size > 0 and int(offsets[-1]) >= file_size:
        offsets = offsets[:-1]
    return offsets


def _normalize_sources(
    sources: Sequence[dict] | None,
    metadata_paths: Iterable[str] | None,
) -> list[dict]:
    """Normalize the user-facing dataset spec into a uniform list of source dicts.

    Each returned source has at minimum: ``name, kind, path, weight``.
    Kind-specific defaults (text_field / prompt_field / audio_field) are filled in.
    """
    if sources:
        normalized: list[dict] = []
        for raw in sources:
            if not isinstance(raw, dict):
                raise ValueError(f"dataset.sources entries must be dicts, got: {raw!r}")
            kind = str(raw.get("kind") or _KIND_LEGACY).strip().lower()
            if kind not in _VALID_KINDS:
                raise ValueError(
                    f"dataset.sources[*].kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
                )
            path = raw.get("path")
            if not path:
                raise ValueError("dataset.sources[*].path is required.")
            weight = float(raw.get("weight", 1.0))
            if weight < 0:
                raise ValueError(f"dataset.sources[*].weight must be >= 0, got {weight}")
            entry = {
                "name": str(raw.get("name") or Path(str(path)).stem or f"source_{len(normalized)}"),
                "kind": kind,
                "path": str(path),
                "weight": weight,
                "audio_field": str(raw.get("audio_field", _DEFAULT_AUDIO_FIELD)),
            }
            if kind == _KIND_TTS:
                entry["text_field"] = str(raw.get("text_field", _DEFAULT_TTS_TEXT_FIELD))
            elif kind == _KIND_TTA:
                entry["prompt_field"] = str(raw.get("prompt_field", _DEFAULT_TTA_PROMPT_FIELD))
            normalized.append(entry)
        if not normalized:
            raise ValueError("dataset.sources must be a non-empty list.")
        return normalized

    if not metadata_paths:
        raise ValueError(
            "AudioJsonlT2ADataset requires either dataset.sources or dataset.metadata_paths."
        )
    return [
        {
            "name": Path(str(path)).stem or f"legacy_{idx}",
            "kind": _KIND_LEGACY,
            "path": str(path),
            "weight": 1.0,
        }
        for idx, path in enumerate(metadata_paths)
    ]


def _parse_legacy_entry(source: dict, item: dict) -> dict | None:
    audio_path = None
    for field in _LEGACY_AUDIO_FIELDS:
        if item.get(field):
            audio_path = item[field]
            break
    if not audio_path:
        return None
    text = ""
    for field in _LEGACY_TEXT_FIELDS:
        value = item.get(field)
        if value is not None:
            text = str(value)
            break
    return {
        "kind": _KIND_LEGACY,
        "audio": str(audio_path),
        "text": text,
        "duration": item.get("duration"),
        "start_time": item.get("start_time"),
        "end_time": item.get("end_time"),
    }


def _parse_tts_entry(source: dict, item: dict) -> dict | None:
    audio_field = source["audio_field"]
    text_field = source["text_field"]
    audio_path = item.get(audio_field) or item.get("audio")
    if not audio_path:
        return None
    text = item.get(text_field)
    if text is None or not str(text).strip():
        return None
    return {
        "kind": _KIND_TTS,
        "audio": str(audio_path),
        "text": str(text),
        "duration": item.get("duration"),
        "start_time": item.get("start_time"),
        "end_time": item.get("end_time"),
    }


def _parse_tta_entry(source: dict, item: dict) -> dict | None:
    audio_field = source["audio_field"]
    prompt_field = source["prompt_field"]
    audio_path = item.get(audio_field) or item.get("audio")
    if not audio_path:
        return None
    prompt_value = item.get(prompt_field)
    if not isinstance(prompt_value, list):
        return None
    prompt_list = [str(p).strip() for p in prompt_value if p is not None and str(p).strip()]
    if not prompt_list:
        return None
    return {
        "kind": _KIND_TTA,
        "audio": str(audio_path),
        "prompt_list": prompt_list,
        "duration": item.get("duration"),
        "start_time": item.get("start_time"),
        "end_time": item.get("end_time"),
    }


def _parse_for_kind(source: dict, item: dict) -> dict | None:
    kind = source["kind"]
    if kind == _KIND_TTS:
        return _parse_tts_entry(source, item)
    if kind == _KIND_TTA:
        return _parse_tta_entry(source, item)
    return _parse_legacy_entry(source, item)


class AudioJsonlT2ADataset(Dataset):
    """Streaming audio jsonl dataset with multi-source weighted sampling.

    On construction we only scan each source for newline byte offsets
    (``_scan_jsonl_offsets``) so an N-GB jsonl is ready in seconds without
    parsing JSON. JSON parsing happens lazily at ``__getitem__`` time on the
    actually-drawn samples.

    Source spec semantics (see ``_normalize_sources``):
      - ``kind=tts``: ``text_field`` (default ``text``) + ``audio_field`` (default ``audio_path``).
      - ``kind=tta``: ``prompt_field`` (default ``prompt_en``, must be a non-empty
        list) + ``audio_field``. A random element of the list is picked at each
        ``__getitem__`` call.
      - ``kind=legacy``: auto-detect ``audio``/``audio_path``/``wav_path`` and
        ``prompt``/``text`` (back-compat with the old single-source schema).

    Sampler-friendly index encoding: for an item ``i`` of source ``s`` we
    return data via ``encoded = s * _ENCODING_STRIDE + i``. The companion
    ``WeightedShuffledCycleStatefulSampler`` is what produces these encoded
    indices; it owns the per-source shuffle-on-exhaust logic.

    Audio decoding goes through ``torchcodec.decoders.AudioDecoder`` which
    handles wav/flac/mp3/mp4 (mp4 audio stream is extracted via ffmpeg). We
    always crop to the first ``num_audio_samples`` samples; shorter clips are
    zero-padded with ``valid_num_samples`` recording the real length.
    """

    def __init__(
        self,
        *,
        sources: Sequence[dict] | None = None,
        metadata_paths: Iterable[str] | None = None,
        dataset_base_path: str = "/",
        sample_rate: int = 48000,
        num_audio_samples: int = 1440000,
        max_num_audio_samples: int | None = 1440000,
        mono: bool = True,
        append_duration_suffix: bool = True,
        duration_precision: int = 1,
        max_samples: int | None = None,
        tokenizer=None,
        task_prefix_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.sources = _normalize_sources(sources, metadata_paths)
        self.dataset_base_path = Path(dataset_base_path).expanduser()
        self.sample_rate = int(sample_rate)
        self.num_audio_samples = int(num_audio_samples)
        self.max_num_audio_samples = None if max_num_audio_samples is None else int(max_num_audio_samples)
        self.mono = bool(mono)
        self.append_duration_suffix = bool(append_duration_suffix)
        self.duration_precision = int(duration_precision)
        self.task_prefix_enabled = bool(task_prefix_enabled)
        self.tokenizer = tokenizer
        self.empty_prompt = maybe_format_chat_prompt("", tokenizer)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.max_duration_s = (
            None
            if self.max_num_audio_samples is None
            else float(self.max_num_audio_samples) / float(self.sample_rate)
        )

        self.stride = _ENCODING_STRIDE
        self.source_offsets: list[np.ndarray] = []
        self.source_sizes: list[int] = []
        self.source_weights: list[float] = []
        self._scan_all_sources()
        self._total_len = sum(self.source_sizes)
        if self.max_samples is not None and self.max_samples < self._total_len:
            self._total_len = self.max_samples

        self._summarize_sources()

        # Lazy per-process file handle cache; reset on fork (each DataLoader
        # worker is a forked subprocess and must NOT reuse the parent's fds
        # because they share the file position).
        self._handles: dict[int, "object"] = {}
        self._handles_pid: int | None = None
        # Track encoded indices that consistently fail to decode so we don't
        # spam the same warning every retry.
        self._bad_indices: set[int] = set()

    def _scan_all_sources(self) -> None:
        show_progress = _is_global_main_process()
        for src_idx, source in enumerate(self.sources):
            metadata_path = Path(source["path"]).expanduser()
            if not metadata_path.exists():
                raise FileNotFoundError(f"Audio metadata path does not exist: {metadata_path}")
            offsets = _scan_jsonl_offsets(metadata_path, show_progress=show_progress)
            if offsets.size == 0:
                raise ValueError(
                    f"Audio source {source['name']!r} ({metadata_path}) is empty (zero lines)."
                )
            if offsets.size > self.stride:
                raise ValueError(
                    f"Audio source {source['name']!r} has {offsets.size} lines which exceeds "
                    f"the per-source stride of {self.stride}; rebuild with a larger _ENCODING_STRIDE."
                )
            self.source_offsets.append(offsets)
            self.source_sizes.append(int(offsets.size))
            self.source_weights.append(float(source["weight"]))

    def _summarize_sources(self) -> None:
        for source, line_count in zip(self.sources, self.source_sizes):
            logger.info(
                "AudioJsonlT2ADataset source name=%s kind=%s weight=%.4f line_count=%d path=%s",
                source["name"],
                source["kind"],
                source["weight"],
                line_count,
                source["path"],
            )
        # Rank-0 only: render a couple of example prompts so it's obvious from
        # the boot log whether the task-prefix wrap is on and what the model
        # actually sees. Bypassing logger here because logger output may be
        # suppressed at INFO depending on launcher config.
        if self.task_prefix_enabled and _is_global_main_process() and self.sources:
            from omnivae_generation.trainer.audio_task_prefix import apply_task_prefix as _atp

            shown_kinds: set[str] = set()
            samples: list[str] = []
            for source in self.sources:
                kind = source["kind"]
                if kind in shown_kinds:
                    continue
                shown_kinds.add(kind)
                placeholder = (
                    "Hello world."
                    if kind == _KIND_TTS
                    else "heavy rain on a metal roof"
                    if kind == _KIND_TTA
                    else "an example caption"
                )
                rendered = _atp(kind, placeholder)
                samples.append(f"  [{kind}] {rendered}  duration: 8.0s")
            print(
                "AudioJsonlT2ADataset task_prefix_enabled=on; example prompts:\n"
                + "\n".join(samples),
                flush=True,
            )

    def __len__(self) -> int:
        return self._total_len

    def _ensure_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.mono or waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _crop_or_pad(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        valid_num_samples = int(min(waveform.shape[-1], self.num_audio_samples))
        if waveform.shape[-1] > self.num_audio_samples:
            waveform = waveform[..., : self.num_audio_samples]
        elif waveform.shape[-1] < self.num_audio_samples:
            pad = self.num_audio_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform.contiguous(), valid_num_samples

    def _load_audio(self, entry: dict) -> tuple[torch.Tensor, float, int]:
        audio_path = _as_path(self.dataset_base_path, entry["audio"])
        decoder = AudioDecoder(str(audio_path))
        duration_seconds = decoder.metadata.duration_seconds_from_header
        start_time = _coerce_optional_float(entry.get("start_time"))
        end_time = _coerce_optional_float(entry.get("end_time"))

        if start_time is not None and end_time is not None:
            samples = decoder.get_samples_played_in_range(start_time, end_time)
        elif self.max_duration_s is not None and duration_seconds > self.max_duration_s:
            samples = decoder.get_samples_played_in_range(0.0, self.max_duration_s)
        else:
            samples = decoder.get_all_samples()

        waveform = samples.data.float()
        source_sample_rate = int(samples.sample_rate)
        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_sample_rate, self.sample_rate)
        waveform = self._ensure_mono(waveform)
        waveform, valid_num_samples = self._crop_or_pad(waveform)
        nonpad_duration_s = float(valid_num_samples) / float(self.sample_rate)
        return waveform, nonpad_duration_s, valid_num_samples

    def _resolve_text(self, entry: dict) -> str:
        if entry["kind"] == _KIND_TTA:
            prompt_list = entry["prompt_list"]
            return random.choice(prompt_list) if len(prompt_list) > 1 else prompt_list[0]
        return str(entry.get("text") or "")

    def _build_prompt(self, entry: dict, duration_s: float) -> str:
        prompt = self._resolve_text(entry)
        if self.task_prefix_enabled:
            # `apply_task_prefix` for kind == "legacy" returns text unchanged
            # so old single-source ckpts/yamls stay reproducible regardless
            # of this flag.
            prompt = apply_task_prefix(entry["kind"], prompt)
        if self.append_duration_suffix:
            configured_duration = _coerce_optional_float(entry.get("duration"))
            duration = duration_s if configured_duration is None else min(configured_duration, duration_s)
            fmt = f"{{:.{max(0, self.duration_precision)}f}}"
            prompt = f"{prompt} duration: {fmt.format(duration)}s"
        return maybe_format_chat_prompt(prompt, self.tokenizer)

    def _get_handle(self, source_idx: int):
        pid = os.getpid()
        if self._handles_pid != pid:
            # Forked into a worker; parent's fds are unsafe (shared file
            # position). Drop the cached references and re-open in this
            # process on demand.
            self._handles = {}
            self._handles_pid = pid
        handle = self._handles.get(source_idx)
        if handle is None:
            handle = open(self.sources[source_idx]["path"], "rb")
            self._handles[source_idx] = handle
        return handle

    def _read_entry(self, source_idx: int, offset_idx: int) -> dict | None:
        source = self.sources[source_idx]
        offset = int(self.source_offsets[source_idx][offset_idx])
        handle = self._get_handle(source_idx)
        handle.seek(offset)
        line_bytes = handle.readline()
        if not line_bytes or not line_bytes.strip():
            return None
        try:
            item = json.loads(line_bytes)
        except json.JSONDecodeError:
            return None
        parsed = _parse_for_kind(source, item)
        if parsed is None:
            return None
        parsed.update(
            {
                "source_index": source_idx,
                "source_name": source["name"],
                "metadata_path": source["path"],
                "line_number": int(offset_idx) + 1,
            }
        )
        return parsed

    def __getitem__(self, encoded_idx: int):
        encoded = int(encoded_idx)
        source_idx = encoded // self.stride
        offset_idx = encoded % self.stride
        if source_idx < 0 or source_idx >= len(self.sources):
            raise IndexError(
                f"Encoded index {encoded} resolves to source {source_idx} which is out of range "
                f"[0, {len(self.sources)})."
            )

        source_size = self.source_sizes[source_idx]
        if source_size <= 0:
            raise RuntimeError(f"Source {source_idx} has zero entries; cannot serve __getitem__.")

        last_error: Exception | None = None
        attempt_offset = offset_idx % source_size
        for _ in range(_MAX_BAD_SAMPLE_RETRIES):
            attempt_encoded = source_idx * self.stride + attempt_offset
            if attempt_encoded in self._bad_indices:
                attempt_offset = (attempt_offset + 1) % source_size
                continue

            try:
                entry = self._read_entry(source_idx, attempt_offset)
            except Exception as exc:
                last_error = exc
                if attempt_encoded not in self._bad_indices:
                    logger.warning(
                        "AudioJsonlT2ADataset: failed to read line idx=%d source=%s offset=%d: %r",
                        attempt_encoded,
                        self.sources[source_idx]["name"],
                        int(self.source_offsets[source_idx][attempt_offset]),
                        exc,
                    )
                    self._bad_indices.add(attempt_encoded)
                attempt_offset = (attempt_offset + 1) % source_size
                continue

            if entry is None:
                # Parse-time skip (e.g., empty caption_en): not worth a warning,
                # but mark as bad so we don't re-parse the same line on later
                # retries. Advance to the next offset within the same source.
                self._bad_indices.add(attempt_encoded)
                attempt_offset = (attempt_offset + 1) % source_size
                continue

            try:
                waveform, duration_s, valid_num_samples = self._load_audio(entry)
            except Exception as exc:
                last_error = exc
                if attempt_encoded not in self._bad_indices:
                    logger.warning(
                        "AudioJsonlT2ADataset: skipping bad audio idx=%d source=%s path=%s line=%s reason=%r",
                        attempt_encoded,
                        self.sources[source_idx]["name"],
                        _as_path(self.dataset_base_path, entry["audio"]),
                        entry["line_number"],
                        exc,
                    )
                    self._bad_indices.add(attempt_encoded)
                attempt_offset = (attempt_offset + 1) % source_size
                continue

            return {
                "audio": waveform,
                "prompt": self._build_prompt(entry, duration_s),
                "empty_prompt": self.empty_prompt,
                "audio_path": str(_as_path(self.dataset_base_path, entry["audio"])),
                "duration": float(duration_s),
                "valid_num_samples": int(valid_num_samples),
                "metadata_path": entry["metadata_path"],
                "line_number": int(entry["line_number"]),
                "source_name": str(entry.get("source_name", "")),
                "kind": str(entry["kind"]),
            }

        raise RuntimeError(
            f"AudioJsonlT2ADataset: failed to load a valid sample after "
            f"{_MAX_BAD_SAMPLE_RETRIES} attempts in source "
            f"{self.sources[source_idx]['name']!r} starting near offset_idx={offset_idx}; "
            f"last error: {last_error!r}"
        )


def collate_audio_samples(batch):
    return {
        "audio": torch.stack([item["audio"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "empty_prompts": [item["empty_prompt"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
        "durations": torch.tensor([item["duration"] for item in batch], dtype=torch.float32),
        "valid_num_samples": torch.tensor([item["valid_num_samples"] for item in batch], dtype=torch.long),
        "metadata_paths": [item["metadata_path"] for item in batch],
        "line_numbers": [item["line_number"] for item in batch],
        "source_names": [item.get("source_name", "") for item in batch],
        "kinds": [item.get("kind", "") for item in batch],
    }
