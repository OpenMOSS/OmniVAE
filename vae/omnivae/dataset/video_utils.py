"""
Utilities for dumping raw video bytes to a local MP4 file for inspection.

Example:
    from omnivae.dataset.video_bytes_debug import save_video_bytes_to_mp4

    save_video_bytes_to_mp4(video_bytes, "debug/sample.mp4")
"""

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Union

import torch
from torchvision.io import write_video

from omnivae.dataset.audio_video_streaming_dataset import read_bytes_from_bin


PathLike = Union[str, Path]


def save_video_bytes_to_mp4(video_bytes: bytes, output_path: PathLike, overwrite: bool = True) -> Path:
    """
    Save raw video bytes directly to a local .mp4 file.

    Args:
        video_bytes: Raw bytes for one encoded video clip.
        output_path: Target file path. If suffix is missing, `.mp4` is appended.
        overwrite: Whether to overwrite an existing file.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(video_bytes)
    return output_path.resolve()


def save_video_from_source_identifier(
    source_identifier: str,
    output_path: PathLike,
    overwrite: bool = True,
) -> Path:
    """
    Save one video to local MP4 from either:

    1. a direct video file path, e.g. `/path/to/sample.mp4`
    2. a packed-bin identifier, e.g. `/path/to/data.bin#video_offset#video_length`

    If extra fields exist after `video_length` (for example audio offsets), they are ignored here.
    """
    source_identifier = str(source_identifier).strip()
    direct_path = Path(source_identifier).expanduser()
    if direct_path.is_file():
        return save_video_bytes_to_mp4(direct_path.read_bytes(), output_path, overwrite=overwrite)

    if "#" in source_identifier:
        parts = source_identifier.split("#")
        if len(parts) < 3:
            raise ValueError(
                f"Packed source identifier must be `bin_path#video_offset#video_length`, got: {source_identifier}"
            )
        bin_path = Path(parts[0]).expanduser()
        if not bin_path.is_file():
            raise FileNotFoundError(f"Bin file not found: {bin_path}")
        video_offset = int(parts[1])
        video_length = int(parts[2])
        video_bytes = read_bytes_from_bin(str(bin_path), video_offset, video_length)
        return save_video_bytes_to_mp4(video_bytes, output_path, overwrite=overwrite)

    raise FileNotFoundError(
        f"Unsupported source identifier: {source_identifier}. "
        "Expected an existing video path or `bin_path#video_offset#video_length`."
    )


def _normalize_video_tensor_to_thwc_uint8(video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a video tensor from CTHW to THWC uint8 for video writing.

    Expected input shape:
        - (C, T, H, W)

    Channel dimension must be 1 or 3.
    """
    if not isinstance(video_tensor, torch.Tensor):
        raise TypeError(f"video_tensor must be a torch.Tensor, got {type(video_tensor)}")

    if video_tensor.ndim != 4:
        raise ValueError(
            f"Expected a 4D video tensor with shape (C,T,H,W), got {tuple(video_tensor.shape)}"
        )

    if video_tensor.shape[0] not in (1, 3):
        raise ValueError(
            f"Expected channel-first video tensor with shape (C,T,H,W) and C in (1, 3), got {tuple(video_tensor.shape)}"
        )
    video_tensor = video_tensor.permute(1, 2, 3, 0)

    video_tensor = video_tensor.detach().cpu().contiguous()

    if torch.is_floating_point(video_tensor):
        video_min = float(video_tensor.min())
        video_max = float(video_tensor.max())
        if video_min < 0.0:
            video_tensor = (video_tensor.clamp(-1.0, 1.0) + 1.0) * 127.5
        elif video_max <= 1.0:
            video_tensor = video_tensor.clamp(0.0, 1.0) * 255.0
        else:
            video_tensor = video_tensor.clamp(0.0, 255.0)
        video_tensor = video_tensor.round().to(torch.uint8)
    else:
        if video_tensor.dtype != torch.uint8:
            video_tensor = video_tensor.clamp(0, 255).to(torch.uint8)

    return video_tensor


def save_video_tensor_to_mp4(
    video_tensor: torch.Tensor,
    output_path: PathLike,
    fps: int = 8,
    overwrite: bool = True,
) -> Path:
    """
    Save a video tensor to a local MP4 file for debugging.

    Expected input shape:
        - (C, T, H, W): common training format in this repo

    Expected value ranges:
        - integer tensor: values are used directly (clamped to [0, 255] if needed)
        - float tensor in [-1, 1]: converted to [0, 255]
        - float tensor in [0, 1]: converted to [0, 255]
        - float tensor in [0, 255]: cast to uint8
    """
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_tensor_thwc = _normalize_video_tensor_to_thwc_uint8(video_tensor)
    fps_value = float(fps)
    if fps_value <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    video_fps = int(fps_value) if fps_value.is_integer() else Fraction(str(fps_value)).limit_denominator(1000)
    write_video(str(output_path), video_tensor_thwc, fps=video_fps)
    return output_path.resolve()


def save_video_from_bin_record(record: Dict[str, Any], output_path: PathLike, overwrite: bool = True) -> Path:
    """
    Read one video's bytes from a training metadata record and dump it as a local MP4.

    Expected keys in `record`:
        - bin_path
        - video_bin_offset
        - video_bin_length
    """
    required_keys = ("bin_path", "video_bin_offset", "video_bin_length")
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise KeyError(f"Missing required keys in record: {missing_keys}")

    video_bytes = read_bytes_from_bin(
        record["bin_path"],
        int(record["video_bin_offset"]),
        int(record["video_bin_length"]),
    )
    return save_video_bytes_to_mp4(video_bytes, output_path, overwrite=overwrite)


def load_jsonl_record(jsonl_path: PathLike, line_number: int = 0) -> Dict[str, Any]:
    """
    Load one JSONL record by zero-based line number.
    """
    jsonl_path = Path(jsonl_path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for current_idx, line in enumerate(f):
            if current_idx == line_number:
                return json.loads(line)
    raise IndexError(f"line_number={line_number} is out of range for {jsonl_path}")


def save_video_from_jsonl(
    jsonl_path: PathLike,
    output_path: PathLike,
    line_number: int = 0,
    overwrite: bool = True,
) -> Path:
    """
    Convenience helper: load one JSONL record and dump its raw video bytes to MP4.
    """
    record = load_jsonl_record(jsonl_path, line_number=line_number)
    return save_video_from_bin_record(record, output_path, overwrite=overwrite)
