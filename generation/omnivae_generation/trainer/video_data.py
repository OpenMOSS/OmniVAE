from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import mmap
import os
import random
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from tqdm.auto import tqdm

from omnivae_generation.trainer.data import (
    _normalize_prompt_max_sequence_length,
    add_tokenized_prompt_fields,
    collate_tokenized_prompt_fields,
    maybe_format_chat_prompt,
    maybe_tokenize_prompt_to_tensors,
)


INDEX_CACHE_VERSION = "video_jsonl_offsets_v1"
JSONL_INDEX_VERSION = 1
DEFAULT_JSONL_PATH_FIELD = "video_path"
DEFAULT_JSONL_PROMPT_FIELD = "prompt_v2"
VALUE_FORMAT_RAW_LINE = "raw_line"
VALUE_FORMAT_JSON_STRING = "json_string"
SUPPORTED_DECODE_BACKENDS = {"auto", "torchcodec"}
DEFAULT_TORCHCODEC_NUM_FFMPEG_THREADS = 1
DEFAULT_MAX_OPEN_DECODERS = 8
DEFAULT_MAX_DECODE_RETRIES = 8


def _build_cache_key(meta_path: Path) -> str:
    stat = meta_path.stat()
    payload = f"{meta_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _load_or_build_offsets(
    meta_path: Path,
    cache_dir: Path,
    *,
    show_progress: bool = False,
) -> np.memmap:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _build_cache_key(meta_path)
    offsets_path = cache_dir / f"video_jsonl_offsets_{INDEX_CACHE_VERSION}_{cache_key}.npy"

    # When several distributed ranks call this concurrently and the cache is
    # cold, naively writing to ``offsets_path`` lets one rank ``np.load`` a
    # half-written file the moment another rank ``np.save`` is mid-flush
    # (observed as ``EOFError: No data left in file``). Build into a per-pid
    # temp file then ``os.replace`` so other ranks only ever see either no
    # file or a fully written one. Even with this safety net callers should
    # gate the build to a single rank (e.g. via ``accelerator.is_main_process``
    # + a barrier) to avoid 8x duplicated jsonl scans.
    if not offsets_path.exists():
        offsets: list[int] = []
        total_bytes = int(meta_path.stat().st_size)
        with meta_path.open("rb") as handle:
            progress = (
                tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"build offsets {meta_path.name}",
                )
                if show_progress
                else None
            )
            try:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(offset)
                    if progress is not None:
                        progress.update(len(line))
            finally:
                if progress is not None:
                    progress.close()
        # NB: ``np.save`` auto-appends ``.npy`` when given a path-like that
        # doesn't already end in ``.npy``. To keep the temp filename and the
        # actual on-disk filename in lockstep (so ``os.replace`` can find
        # it), open the file ourselves and pass the handle.
        tmp_path = offsets_path.with_name(f"{offsets_path.name}.tmp.{os.getpid()}")
        arr = np.asarray(offsets, dtype=np.int64)
        with tmp_path.open("wb") as handle:
            np.save(handle, arr)
        os.replace(tmp_path, offsets_path)

    return np.load(offsets_path, mmap_mode="r")


def _build_sidecar_meta_path(meta_path: Path, cache_dir: Path, sidecar_name: str) -> Path:
    cache_key = _build_cache_key(meta_path)
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(sidecar_name).strip())
    return cache_dir / f"video_jsonl_{safe_name}_{INDEX_CACHE_VERSION}_{cache_key}.index.json"


def _artifact_paths_from_meta(meta_path: Path) -> tuple[Path, Path]:
    meta_name = meta_path.name
    if meta_name.endswith(".index.json"):
        prefix = meta_name[: -len(".index.json")]
    else:
        prefix = meta_path.stem
    return (
        meta_path.parent / f"{prefix}.paths.txt",
        meta_path.parent / f"{prefix}.offsets.u64",
    )


def _build_string_sidecar_index(
    source_jsonl: Path,
    output_meta_path: Path,
    *,
    field_name: str,
    extractor,
    value_format: str = VALUE_FORMAT_RAW_LINE,
    max_records: int | None = None,
    show_progress: bool = False,
) -> Path:
    paths_path, offsets_path = _artifact_paths_from_meta(output_meta_path)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    if value_format not in {VALUE_FORMAT_RAW_LINE, VALUE_FORMAT_JSON_STRING}:
        raise ValueError(f"Unsupported sidecar value_format {value_format!r}.")

    # Same race-safety story as ``_load_or_build_offsets``: when multiple
    # distributed ranks build the same sidecar concurrently we want any
    # observer that sees ``output_meta_path`` to also see fully written
    # blob/offset files. Stage all three artifacts under per-pid temp names
    # then ``os.replace`` them in dependency order (blob and offsets before
    # the meta marker), so a reader that finds ``output_meta_path`` always
    # finds the matching sidecar contents next to it.
    pid = os.getpid()
    paths_tmp = paths_path.with_name(f"{paths_path.name}.tmp.{pid}")
    offsets_tmp = offsets_path.with_name(f"{offsets_path.name}.tmp.{pid}")
    meta_tmp = output_meta_path.with_name(f"{output_meta_path.name}.tmp.{pid}")

    total_bytes = int(source_jsonl.stat().st_size)
    count = 0
    blob_offset = 0
    with (
        source_jsonl.open("rb") as source_handle,
        paths_tmp.open("wb") as values_handle,
        offsets_tmp.open("wb") as offsets_handle,
    ):
        progress = (
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"build {field_name} sidecar",
            )
            if show_progress
            else None
        )
        try:
            while True:
                line = source_handle.readline()
                if not line:
                    break
                if progress is not None:
                    progress.update(len(line))
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise TypeError(
                        f"Expected JSON object while building {field_name!r} sidecar for {source_jsonl}, got {type(record).__name__}."
                    )

                value = extractor(record)
                if value_format == VALUE_FORMAT_JSON_STRING:
                    serialized_value = json.dumps(value, ensure_ascii=False)
                else:
                    serialized_value = value
                value_bytes = serialized_value.encode("utf-8")
                offsets_handle.write(struct.pack("<Q", int(blob_offset)))
                values_handle.write(value_bytes)
                values_handle.write(b"\n")

                blob_offset += len(value_bytes) + 1
                count += 1
                if max_records is not None and count >= max_records:
                    break
        finally:
            if progress is not None:
                progress.close()

    meta_tmp.write_text(
        json.dumps(
            {
                "version": JSONL_INDEX_VERSION,
                "source_jsonl": str(source_jsonl),
                "path_field": str(field_name),
                "value_format": str(value_format),
                "count": int(count),
                "paths_file": paths_path.name,
                "offsets_file": offsets_path.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(paths_tmp, paths_path)
    os.replace(offsets_tmp, offsets_path)
    os.replace(meta_tmp, output_meta_path)
    return output_meta_path


def _load_or_build_prompt_index(
    meta_path: Path,
    cache_dir: Path,
    *,
    prompt_field: str,
    max_records: int | None = None,
    show_progress: bool = False,
) -> "IndexedPathReader":
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = "full" if max_records is None else f"limit_{int(max_records)}"
    output_meta_path = _build_sidecar_meta_path(meta_path, cache_dir, f"prompt_{prompt_field}_{limit_tag}")
    if not output_meta_path.exists():
        _build_string_sidecar_index(
            meta_path,
            output_meta_path,
            field_name=prompt_field,
            extractor=lambda record: _extract_prompt(record, prompt_field),
            value_format=VALUE_FORMAT_JSON_STRING,
            max_records=max_records,
            show_progress=show_progress,
        )
    return IndexedPathReader(output_meta_path)


def _resolve_jsonl_source_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".jsonl":
        return candidate.resolve()
    if candidate.is_dir():
        maybe_jsonl = candidate / "all.jsonl"
        if maybe_jsonl.is_file():
            return maybe_jsonl.resolve()
    raise FileNotFoundError(f"Could not resolve a source JSONL from {candidate}. Expected a .jsonl file or a directory containing all.jsonl.")


def _resolve_video_jsonl_cache_dir(meta_path: Path, cache_dir_raw: Optional[str]) -> Path:
    """Resolve the cache directory for ``meta_path``. When ``cache_dir_raw``
    is empty/None we fall back to a sibling ``<stem>.cache`` directory next
    to the resolved jsonl. Shared by ``VideoJsonlDataset`` and
    ``prebuild_video_jsonl_indexes`` so the two paths agree.
    """
    if cache_dir_raw is None or not str(cache_dir_raw).strip():
        return (meta_path.parent / f"{meta_path.stem}.cache").resolve()
    return Path(cache_dir_raw).expanduser().resolve()


def prebuild_video_jsonl_indexes(dataset_cfg: dict) -> None:
    """Build any missing video_jsonl byte-offset / sidecar artifacts for the
    given ``dataset:`` config block, with a tqdm progress bar. Intended to
    be called once by a single rank (e.g. ``accelerator.is_main_process``)
    behind a barrier; downstream ranks then construct ``VideoJsonlDataset``
    against a warm cache instead of re-scanning the same jsonl 8x and
    racing each other on the same writes.
    """
    meta_path_raw = dataset_cfg.get("meta_path")
    if not meta_path_raw:
        return
    meta_path = _resolve_jsonl_source_path(meta_path_raw)
    cache_dir = _resolve_video_jsonl_cache_dir(meta_path, dataset_cfg.get("cache_dir"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    jsonl_index_path = dataset_cfg.get("jsonl_index_path")
    jsonl_prompt_index_path = dataset_cfg.get("jsonl_prompt_index_path")
    prompt_field_raw = dataset_cfg.get("jsonl_prompt_field") or DEFAULT_JSONL_PROMPT_FIELD
    prompt_field = str(prompt_field_raw).strip() or DEFAULT_JSONL_PROMPT_FIELD
    max_samples = dataset_cfg.get("max_samples")

    # Byte-offset cache is needed whenever any reader is missing (lazy mode
    # or prompt-only sidecar). Build it on the main rank with a progress bar.
    if not jsonl_index_path or not jsonl_prompt_index_path:
        _load_or_build_offsets(meta_path, cache_dir, show_progress=True)

    # Prompt sidecar gets auto-built when only the path sidecar is provided.
    if jsonl_index_path and not jsonl_prompt_index_path:
        _load_or_build_prompt_index(
            meta_path,
            cache_dir,
            prompt_field=prompt_field,
            max_records=None if max_samples is None else int(max_samples),
            show_progress=True,
        )


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _extract_prompt(record: dict[str, Any], prompt_field: str) -> str:
    return _coerce_text(record.get(prompt_field))


def _resolve_video_path(meta_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (meta_path.parent / candidate).resolve()


class IndexedPathReader:
    """Reader compatible with Kei's `*.index.json` path-sidecar format."""

    def __init__(self, meta_path: str | Path) -> None:
        self.meta_path = Path(meta_path).expanduser().resolve()
        if not self.meta_path.is_file():
            raise FileNotFoundError(f"indexed path metadata not found: {self.meta_path}")

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        version = int(meta.get("version", 0))
        if version != JSONL_INDEX_VERSION:
            raise ValueError(
                f"unsupported jsonl path index version: {version} (expected {JSONL_INDEX_VERSION})"
            )

        self.source_jsonl = str(meta.get("source_jsonl", ""))
        self.path_field = str(meta.get("path_field", DEFAULT_JSONL_PATH_FIELD))
        self.value_format = str(meta.get("value_format", VALUE_FORMAT_RAW_LINE))
        if self.value_format not in {VALUE_FORMAT_RAW_LINE, VALUE_FORMAT_JSON_STRING}:
            raise ValueError(
                f"unsupported indexed path value_format: {self.value_format!r} "
                f"(expected one of {[VALUE_FORMAT_RAW_LINE, VALUE_FORMAT_JSON_STRING]!r})"
            )
        self.count = int(meta.get("count", 0))
        self.paths_path = (self.meta_path.parent / str(meta.get("paths_file", ""))).resolve()
        self.offsets_path = (self.meta_path.parent / str(meta.get("offsets_file", ""))).resolve()

        if not self.paths_path.is_file():
            raise FileNotFoundError(f"indexed path blob not found: {self.paths_path}")
        if not self.offsets_path.is_file():
            raise FileNotFoundError(f"indexed path offsets not found: {self.offsets_path}")

        offsets_size = int(self.offsets_path.stat().st_size)
        if offsets_size % 8 != 0:
            raise ValueError(f"offset file size must be a multiple of 8 bytes: {self.offsets_path}")
        if (offsets_size // 8) < self.count:
            raise ValueError(
                f"offset file has fewer entries than metadata count: {self.offsets_path} ({offsets_size // 8} < {self.count})"
            )

        self._paths_fh = None
        self._offsets_fh = None
        self._offsets_mm = None

    def __len__(self) -> int:
        return int(self.count)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_paths_fh"] = None
        state["_offsets_fh"] = None
        state["_offsets_mm"] = None
        return state

    def close(self) -> None:
        if self._offsets_mm is not None:
            self._offsets_mm.close()
            self._offsets_mm = None
        if self._offsets_fh is not None:
            self._offsets_fh.close()
            self._offsets_fh = None
        if self._paths_fh is not None:
            self._paths_fh.close()
            self._paths_fh = None

    def _ensure_open(self) -> None:
        if self._paths_fh is None:
            self._paths_fh = self.paths_path.open("rb")
        if self._offsets_mm is None:
            self._offsets_fh = self.offsets_path.open("rb")
            self._offsets_mm = mmap.mmap(self._offsets_fh.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, idx: int) -> str:
        index = int(idx)
        if index < 0 or index >= self.count:
            raise IndexError(f"indexed path out of range: idx={index}, count={self.count}")

        self._ensure_open()
        assert self._offsets_mm is not None
        assert self._paths_fh is not None

        offset = struct.unpack_from("<Q", self._offsets_mm, index * 8)[0]
        self._paths_fh.seek(int(offset))
        line = self._paths_fh.readline()
        if not line:
            raise RuntimeError(f"failed to read indexed path at idx={index} from {self.paths_path}")
        decoded_line = line.rstrip(b"\r\n").decode("utf-8")
        if self.value_format == VALUE_FORMAT_JSON_STRING:
            value = json.loads(decoded_line)
            if not isinstance(value, str):
                raise TypeError(
                    f"expected JSON string at idx={index} from {self.paths_path}, got {type(value).__name__}."
                )
            return value
        return decoded_line

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _close_decoder(decoder: VideoDecoder) -> None:
    close = getattr(decoder, "close", None)
    if callable(close):
        close()


def _extract_torchcodec_metadata(decoder: VideoDecoder) -> dict[str, Any]:
    metadata = getattr(decoder, "metadata", None)
    return {
        "fps": _coerce_positive_float(getattr(metadata, "average_fps", None))
        or _coerce_positive_float(getattr(metadata, "average_fps_from_header", None)),
        "height": _coerce_positive_int(getattr(metadata, "height", None)),
        "width": _coerce_positive_int(getattr(metadata, "width", None)),
        "num_frames": _coerce_positive_int(getattr(metadata, "num_frames", None)) or int(len(decoder)),
    }


def _sample_video_frame_indices(num_frames_total: int, fps: float, target_fps: float, target_num_frames: int) -> list[int]:
    if target_num_frames <= 0:
        raise ValueError(f"target_num_frames must be positive, got {target_num_frames}.")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}.")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if num_frames_total <= 0:
        raise ValueError(f"num_frames_total must be positive, got {num_frames_total}.")

    frame_sampling_rate = float(fps) / float(target_fps)
    frame_indices: list[int] = []
    for index in range(target_num_frames):
        frame_idx = int(round(index * frame_sampling_rate))
        frame_idx = min(frame_idx, max(0, num_frames_total - 1))
        frame_indices.append(frame_idx)
    return frame_indices


def _decode_torchcodec_frames(decoder: VideoDecoder, frame_indices: list[int]) -> torch.Tensor:
    if len(frame_indices) == 0:
        raise ValueError("frame_indices must be non-empty.")

    if len(frame_indices) == 1:
        return decoder[int(frame_indices[0])].unsqueeze(0)

    start = int(frame_indices[0])
    step = int(frame_indices[1] - frame_indices[0])
    if step > 0 and all(int(frame_indices[idx] - frame_indices[idx - 1]) == step for idx in range(1, len(frame_indices))):
        stop = int(frame_indices[-1]) + step
        return decoder[start:stop:step]

    stop = int(frame_indices[-1]) + 1
    contiguous = decoder[start:stop]
    gather_indices = torch.tensor([int(idx - start) for idx in frame_indices], dtype=torch.long)
    return contiguous.index_select(0, gather_indices)


def _decode_sampled_video_with_torchcodec(
    decoder: VideoDecoder,
    *,
    fps: float,
    target_fps: float,
    target_num_frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    metadata = _extract_torchcodec_metadata(decoder)
    frame_indices = _sample_video_frame_indices(
        int(metadata["num_frames"]),
        fps=fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
    )
    decoded = _decode_torchcodec_frames(decoder, frame_indices)
    if decoded.numel() == 0:
        raise RuntimeError("torchcodec returned an empty clip.")
    return decoded, metadata


def _resize_and_crop_video(
    video: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    center_crop: bool,
    random_flip: bool,
) -> torch.Tensor:
    shorter_side = min(int(source_height), int(source_width))
    if shorter_side <= 0:
        raise ValueError(
            f"Video metadata must provide positive width/height, got height={source_height}, width={source_width}."
        )

    scale = max(
        float(target_height) / float(source_height),
        float(target_width) / float(source_width),
    )
    new_height = max(int(target_height), int(math.ceil(source_height * scale)))
    new_width = max(int(target_width), int(math.ceil(source_width * scale)))

    video_reshaped = video.permute(1, 0, 2, 3).contiguous()
    video_resized = F.interpolate(
        video_reshaped,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    video_resized = video_resized.permute(1, 0, 2, 3).contiguous()

    max_top = max(0, new_height - target_height)
    max_left = max(0, new_width - target_width)
    if center_crop:
        top = max_top // 2
        left = max_left // 2
    else:
        top = 0 if max_top == 0 else random.randint(0, max_top)
        left = 0 if max_left == 0 else random.randint(0, max_left)
    bottom = top + target_height
    right = left + target_width

    video_cropped = video_resized[:, :, top:bottom, left:right]
    if random_flip and random.random() < 0.5:
        video_cropped = video_cropped.flip(-1)
    return video_cropped.contiguous()


class VideoJsonlDataset(Dataset):
    def __init__(
        self,
        *,
        meta_path: str,
        frame_size: tuple[int, int] | list[int],
        num_frames: int,
        target_fps: float,
        center_crop: bool = True,
        random_flip: bool = False,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        decode_backend: str = "auto",
        jsonl_index_path: str | None = None,
        jsonl_path_field: str = DEFAULT_JSONL_PATH_FIELD,
        jsonl_prompt_field: str = DEFAULT_JSONL_PROMPT_FIELD,
        jsonl_prompt_index_path: str | None = None,
        tokenizer=None,
        include_raw_pixel_values: bool = False,
        return_uint8: bool = True,
        prompt_max_sequence_length: int | None = None,
    ) -> None:
        super().__init__()
        self.meta_path = _resolve_jsonl_source_path(meta_path)
        cache_dir_path = _resolve_video_jsonl_cache_dir(self.meta_path, cache_dir)
        if cache_dir is None or not str(cache_dir).strip():
            print(
                f"[VideoJsonlDataset] cache_dir not set; defaulting to {cache_dir_path}",
                flush=True,
            )
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        frame_size_values = tuple(int(item) for item in frame_size)
        if len(frame_size_values) != 2:
            raise ValueError(f"frame_size must have exactly 2 elements [height, width], got {frame_size!r}.")
        if any(item <= 0 for item in frame_size_values):
            raise ValueError(f"frame_size values must be positive, got {frame_size_values!r}.")

        self.frame_size = frame_size_values
        self.num_frames = int(num_frames)
        self.target_fps = float(target_fps)
        self.center_crop = bool(center_crop)
        self.random_flip = bool(random_flip)
        requested_backend = str(decode_backend).strip().lower()
        if requested_backend not in SUPPORTED_DECODE_BACKENDS:
            raise ValueError(
                f"Unsupported video decode backend {decode_backend!r}. Expected one of: {sorted(SUPPORTED_DECODE_BACKENDS)}."
            )
        self.decode_backend = "torchcodec" if requested_backend == "auto" else requested_backend
        self.path_field = str(jsonl_path_field).strip() or DEFAULT_JSONL_PATH_FIELD
        self.prompt_field = str(jsonl_prompt_field).strip()
        if not self.prompt_field:
            raise ValueError("jsonl_prompt_field must be a non-empty string.")
        self.tokenizer = tokenizer
        self.include_raw_pixel_values = bool(include_raw_pixel_values)
        self.return_uint8 = bool(return_uint8)
        self.empty_prompt = maybe_format_chat_prompt("", tokenizer)
        self.prompt_max_sequence_length = _normalize_prompt_max_sequence_length(prompt_max_sequence_length)
        self.empty_prompt_tokens = maybe_tokenize_prompt_to_tensors(
            self.empty_prompt,
            tokenizer,
            self.prompt_max_sequence_length,
        )
        self.num_ffmpeg_threads = DEFAULT_TORCHCODEC_NUM_FFMPEG_THREADS
        self.max_open_decoders = DEFAULT_MAX_OPEN_DECODERS
        self.max_decode_retries = DEFAULT_MAX_DECODE_RETRIES
        self.path_reader = None
        self.prompt_reader = None
        self._decoder_cache: OrderedDict[str, VideoDecoder] = OrderedDict()
        if jsonl_index_path is not None:
            self.path_reader = IndexedPathReader(jsonl_index_path)
            if self.path_reader.source_jsonl:
                indexed_source = Path(self.path_reader.source_jsonl).expanduser().resolve()
                if indexed_source != self.meta_path:
                    raise ValueError(
                        f"jsonl_index_path source_jsonl {indexed_source} does not match meta_path {self.meta_path}."
                    )

        if jsonl_prompt_index_path is not None:
            self.prompt_reader = IndexedPathReader(jsonl_prompt_index_path)
        elif self.path_reader is not None:
            self.prompt_reader = _load_or_build_prompt_index(
                self.meta_path,
                cache_dir_path,
                prompt_field=self.prompt_field,
                max_records=None if max_samples is None else int(max_samples),
            )

        for reader_name, reader in (("jsonl_index_path", self.path_reader), ("jsonl_prompt_index_path", self.prompt_reader)):
            if reader is None:
                continue
            if reader.source_jsonl:
                indexed_source = Path(reader.source_jsonl).expanduser().resolve()
                if indexed_source != self.meta_path:
                    raise ValueError(f"{reader_name} source_jsonl {indexed_source} does not match meta_path {self.meta_path}.")
            if reader_name == "jsonl_prompt_index_path" and reader.path_field and reader.path_field != self.prompt_field:
                raise ValueError(
                    f"{reader_name} path_field {reader.path_field!r} does not match dataset.jsonl_prompt_field {self.prompt_field!r}."
                )

        self.offsets = None
        if self.path_reader is None or self.prompt_reader is None:
            self.offsets = _load_or_build_offsets(self.meta_path, cache_dir_path)

        available_counts = []
        if self.path_reader is not None:
            available_counts.append(len(self.path_reader))
        if self.prompt_reader is not None:
            available_counts.append(len(self.prompt_reader))
        if self.offsets is not None:
            available_counts.append(int(self.offsets.shape[0]))
        if not available_counts:
            raise RuntimeError(f"VideoJsonlDataset could not determine sample count for {self.meta_path}.")
        if max_samples is None and len(set(available_counts)) > 1:
            raise ValueError(
                f"Sidecar sample counts do not agree for {self.meta_path}: {available_counts}. "
                "Provide matching full indices or set max_samples for a truncated smoke run."
            )

        self.num_samples = min(available_counts)
        if max_samples is not None:
            self.num_samples = min(self.num_samples, int(max_samples))
        self._meta_fh = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_meta_fh"] = None
        state["_decoder_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return self.num_samples

    def close(self) -> None:
        if self._meta_fh is not None:
            self._meta_fh.close()
            self._meta_fh = None
        if self.path_reader is not None:
            self.path_reader.close()
        if self.prompt_reader is not None:
            self.prompt_reader.close()
        for decoder in self._decoder_cache.values():
            _close_decoder(decoder)
        self._decoder_cache.clear()

    def _ensure_meta_open(self) -> None:
        if self._meta_fh is None:
            self._meta_fh = self.meta_path.open("rb")

    def _read_record(self, index: int) -> dict[str, Any]:
        if self.offsets is None:
            raise RuntimeError(f"JSONL record access is unavailable for {self.meta_path}; configure sidecar indices instead.")
        offset = int(self.offsets[index])
        self._ensure_meta_open()
        assert self._meta_fh is not None
        self._meta_fh.seek(offset)
        line = self._meta_fh.readline()
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object at line {index} in {self.meta_path}, got {type(payload).__name__}.")
        return payload

    def _get_raw_path(self, index: int, record: dict[str, Any]) -> str:
        if self.path_reader is not None:
            raw_path = self.path_reader.get(index)
        else:
            raw_path = record.get(self.path_field) or record.get("path")
        if not raw_path:
            raise KeyError(
                f"Video sample {index} in {self.meta_path} is missing `{self.path_field}`/`path`."
            )
        return str(raw_path)

    def _get_decoder(self, path: Path) -> VideoDecoder:
        key = str(path)
        decoder = self._decoder_cache.get(key)
        if decoder is not None:
            self._decoder_cache.move_to_end(key, last=True)
            return decoder

        decoder = VideoDecoder(
            path,
            dimension_order="NCHW",
            num_ffmpeg_threads=int(self.num_ffmpeg_threads),
            device="cpu",
            seek_mode="exact",
        )
        self._decoder_cache[key] = decoder
        self._decoder_cache.move_to_end(key, last=True)
        while len(self._decoder_cache) > int(self.max_open_decoders):
            _, stale_decoder = self._decoder_cache.popitem(last=False)
            _close_decoder(stale_decoder)
        return decoder

    def _drop_decoder(self, path: Path) -> None:
        decoder = self._decoder_cache.pop(str(path), None)
        if decoder is not None:
            _close_decoder(decoder)

    def __getitem__(self, index: int):
        current_index = int(index) % int(self.num_samples)
        last_err: Exception | None = None

        for _attempt in range(max(1, int(self.max_decode_retries) + 1)):
            record = self._read_record(current_index) if self.offsets is not None else {}
            raw_path = self._get_raw_path(current_index, record)
            prompt = (
                self.prompt_reader.get(current_index)
                if self.prompt_reader is not None
                else _extract_prompt(record, self.prompt_field)
            )
            video_path = _resolve_video_path(self.meta_path, raw_path)

            try:
                decoder = self._get_decoder(video_path)
                decoder_metadata = _extract_torchcodec_metadata(decoder)
                fps = (
                    _coerce_positive_float(decoder_metadata.get("fps"))
                    or _coerce_positive_float(record.get("fps"))
                    or float(self.target_fps)
                )
                decoded_video, decoded_metadata = _decode_sampled_video_with_torchcodec(
                    decoder,
                    fps=fps,
                    target_fps=self.target_fps,
                    target_num_frames=self.num_frames,
                )
                width = (
                    _coerce_positive_int(decoded_metadata.get("width"))
                    or _coerce_positive_int(record.get("width"))
                    or int(decoded_video.shape[-1])
                )
                height = (
                    _coerce_positive_int(decoded_metadata.get("height"))
                    or _coerce_positive_int(record.get("height"))
                    or int(decoded_video.shape[-2])
                )

                sampled_video = decoded_video.permute(1, 0, 2, 3).contiguous()
                if self.return_uint8:
                    pixel_values = _resize_and_crop_video(
                        sampled_video,
                        source_height=height,
                        source_width=width,
                        target_height=self.frame_size[0],
                        target_width=self.frame_size[1],
                        center_crop=self.center_crop,
                        random_flip=self.random_flip,
                    )
                    if pixel_values.dtype != torch.uint8:
                        pixel_values = pixel_values.clamp(0, 255).to(torch.uint8)
                    raw_pixel_values = pixel_values.float() / 255.0 if self.include_raw_pixel_values else None
                else:
                    pixel_values = sampled_video.float() / 127.5 - 1.0
                    pixel_values = _resize_and_crop_video(
                        pixel_values,
                        source_height=height,
                        source_width=width,
                        target_height=self.frame_size[0],
                        target_width=self.frame_size[1],
                        center_crop=self.center_crop,
                        random_flip=self.random_flip,
                    )
                    raw_pixel_values = (pixel_values + 1.0) / 2.0 if self.include_raw_pixel_values else None

                prompt = maybe_format_chat_prompt(prompt, self.tokenizer)
                sample = {
                    "pixel_values": pixel_values,
                    "prompt": prompt,
                    "empty_prompt": self.empty_prompt,
                    "image_id": record.get("id") or f"{video_path.name}:{current_index}",
                    "image_path": str(video_path),
                    "label_text": "",
                    "synset": "",
                }
                add_tokenized_prompt_fields(
                    sample,
                    prompt=prompt,
                    empty_prompt=self.empty_prompt,
                    tokenizer=self.tokenizer,
                    max_sequence_length=self.prompt_max_sequence_length,
                    empty_prompt_tokens=self.empty_prompt_tokens,
                )
                if self.include_raw_pixel_values:
                    assert raw_pixel_values is not None
                    sample["raw_pixel_values"] = raw_pixel_values
                return sample
            except Exception as exc:
                last_err = exc
                self._drop_decoder(video_path)
                if int(self.num_samples) <= 1:
                    break
                current_index = random.randrange(int(self.num_samples))

        raise RuntimeError(f"Failed to decode video sample after retries (index={index}, last_err={last_err!r})")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def collate_video_samples(batch):
    collated = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "prompts": [item["prompt"] for item in batch],
        "empty_prompts": [item["empty_prompt"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "label_texts": [item["label_text"] for item in batch],
        "synsets": [item["synset"] for item in batch],
    }
    if "raw_pixel_values" in batch[0]:
        collated["raw_pixel_values"] = torch.stack([item["raw_pixel_values"] for item in batch])
    collate_tokenized_prompt_fields(collated, batch)
    return collated
