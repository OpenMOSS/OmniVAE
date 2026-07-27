"""
Audio-Video Streaming Dataset for Joint VAE Training

从 bin 文件同时加载视频和音频数据，支持：
- 流式读取 jsonl 元数据
- 视频帧随机裁剪 (temporal crop)
- 音频重采样和对齐
- 多数据集加权采样
- 精确的 state_dict 用于训练恢复

Shape Conventions:
    B = batch_size
    C = channels (3 for RGB video, 1 for mono audio)
    T = temporal frames (video)
    H = height
    W = width
    T_a = audio samples (audio_sample_rate * duration)

References:
- streaming_bin_video_dataset.py (Video processing)
- AudioProcessorForSpeechTokenizer (Audio processing)
"""

import os
import io
import json
import re
import yaml
import mmap
import logging
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import soundfile as sf
from typing import Dict, Any, List, Optional, Union, Iterator, Callable, Tuple
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from abc import ABC, abstractmethod
from torch.utils.data import IterableDataset, default_collate
from torchvision import transforms
from omnivae.dataset.transform import center_crop as _center_crop_fn

# Try to import torchcodec for video decoding
try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except Exception as e:
    logging.warning(
        "torchcodec unavailable, falling back to decord if possible: "
        f"{type(e).__name__}: {str(e).splitlines()[0]}"
    )
    HAS_TORCHCODEC = False

# Try to import decord as fallback
try:
    import decord
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


# =====================
# Long-video ID parsing
# =====================
# 命名约定 (与 scripts/data/reorg/reorg_jsonl.py 保持一致):
#   <long_video_id>_<clip_idx>_<start>_<end>.mp4
# 其中 long_video_id 可能包含点号/括号/连字符/中文等字符，只要求尾部
# 形如 _<int>_<float>_<float>.mp4。
_LONG_VID_PATTERN = re.compile(r"^(?P<vid>.+)_\d+_[0-9.]+_[0-9.]+\.mp4$")


def parse_long_video_id(video_path: Optional[str]) -> Optional[str]:
    """从视频路径解析长视频 ID。解析失败返回 None（样本自然没有 sibling）。"""
    if not video_path:
        return None
    basename = os.path.basename(str(video_path))
    m = _LONG_VID_PATTERN.match(basename)
    if m is None:
        return None
    return m.group("vid")


# =====================
# Video Loading Utils
# =====================
def read_bytes_from_bin(bin_path: str, offset: int, length: int) -> bytes:
    """从 bin 文件的指定位置读取字节"""
    with open(bin_path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def normalize_frame_indices(frame_indices: List[int]) -> List[int]:
    """确保传给解码器的 frame indices 是原生 Python int。"""
    return [int(idx) for idx in frame_indices]


def build_source_identifier(meta: Dict[str, Any]) -> Optional[str]:
    """
    构造可追溯的数据源标识。

    - 普通文件: 返回绝对路径
    - bin 打包样本: 返回 `bin_path#video_offset#video_length[#audio_offset#audio_length]`
    """
    if "bin_path" in meta:
        bin_path = str(Path(meta["bin_path"]).expanduser().resolve(strict=False))
        parts = [bin_path]
        if "video_bin_offset" in meta and "video_bin_length" in meta:
            parts.extend([str(int(meta["video_bin_offset"])), str(int(meta["video_bin_length"]))])
        if "audio_bin_offset" in meta and "audio_bin_length" in meta:
            parts.extend([str(int(meta["audio_bin_offset"])), str(int(meta["audio_bin_length"]))])
        if len(parts) > 1:
            return "#".join(parts)

    file_path = meta.get("video_path") or meta.get("audio_path") or meta.get("path")
    if file_path:
        return str(Path(file_path).expanduser().resolve(strict=False))

    if "file_name" in meta:
        return str(meta["file_name"])
    if "name" in meta:
        return str(meta["name"])
    return None


def resolve_metadata_media_path(
    raw_path: str,
    *,
    root: Optional[str] = None,
    metadata_dir: Optional[str] = None,
) -> str:
    """Resolve a media path from a JSONL record."""
    raw = os.path.expanduser(os.path.expandvars(str(raw_path)))
    if os.path.isabs(raw):
        return raw
    for base in (root, metadata_dir):
        if base:
            candidate = os.path.abspath(os.path.join(str(base), raw))
            if os.path.exists(candidate):
                return candidate
    if root:
        return os.path.abspath(os.path.join(str(root), raw))
    if metadata_dir:
        return os.path.abspath(os.path.join(str(metadata_dir), raw))
    return raw


def compute_frame_sampling_plan(
    total_frames: int,
    source_fps: float,
    num_frames: int,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    random_start: bool = False,
) -> Tuple[List[int], Dict[str, float]]:
    """
    计算视频采样索引，并返回用于音视频同步裁剪的时间信息。

    当指定 target_fps 时，采样时间跨度由 target_fps 决定，而不是原视频 fps。
    例如 num_frames=41, target_fps=8, sample_rate=1 时，对应 5.0 秒音频窗口。

    Args:
        random_start: 是否随机选择起始位置。False 时从头开始采样。
    """
    if total_frames <= 0:
        raise ValueError(f"Video has no decodable frames: total_frames={total_frames}")

    # target_fps 生效时，按时间间隔采样，保证不同源 fps 下的时间窗口一致。
    if target_fps is not None and target_fps > 0 and source_fps > 0:
        frame_step = float(sample_rate) * float(source_fps) / float(target_fps)
        requested_duration = num_frames * float(sample_rate) / float(target_fps)
        requested_span_frames = requested_duration * float(source_fps)

        if total_frames < requested_span_frames:
            video_duration = total_frames / float(source_fps)
            raise ValueError(
                f"Video too short: has {total_frames} frames ({video_duration:.2f}s at {source_fps}fps), "
                f"but need {requested_span_frames:.0f} frames ({requested_duration:.2f}s) "
                f"for num_frames={num_frames}, sample_rate={sample_rate}, target_fps={target_fps}"
            )

        max_start = max(0.0, total_frames - requested_span_frames)
        if random_start and max_start > 0:
            start_pos = float(np.random.uniform(0.0, max_start))
        else:
            start_pos = 0.0

        frame_positions = start_pos + np.arange(num_frames, dtype=np.float64) * frame_step
        frame_indices = normalize_frame_indices(
            np.clip(np.round(frame_positions).astype(np.int64), 0, total_frames - 1).tolist()
        )

        start_frame = int(frame_indices[0])
        duration = requested_duration
        clip_fps = float(target_fps) / float(sample_rate)
        time_info = {
            "start_time": start_frame / float(source_fps),
            "duration": duration,
            "fps": float(source_fps),
            "clip_fps": clip_fps,
            "start_frame": start_frame,
            "total_frames": total_frames,
        }
        return frame_indices, time_info

    required_frames = num_frames * sample_rate
    if total_frames < required_frames:
        video_duration = total_frames / float(source_fps) if source_fps > 0 else -1
        raise ValueError(
            f"Video too short: has {total_frames} frames ({video_duration:.2f}s), "
            f"but need {required_frames} frames "
            f"for num_frames={num_frames}, sample_rate={sample_rate}"
        )

    if not random_start or total_frames == required_frames:
        start_frame = 0
    else:
        start_frame = int(np.random.randint(0, total_frames - required_frames + 1))

    frame_indices = list(range(start_frame, start_frame + required_frames, sample_rate))
    frame_indices = normalize_frame_indices(frame_indices)

    if source_fps > 0:
        duration = num_frames * float(sample_rate) / float(source_fps)
        start_time = start_frame / float(source_fps)
    else:
        duration = num_frames * float(sample_rate) / 25.0
        start_time = 0.0

    time_info = {
        "start_time": start_time,
        "duration": duration,
        "fps": float(source_fps) if source_fps > 0 else 25.0,
        "clip_fps": (float(source_fps) / float(sample_rate)) if source_fps > 0 else (25.0 / float(sample_rate)),
        "start_frame": start_frame,
        "total_frames": total_frames,
    }
    return frame_indices, time_info


def load_video_from_bytes_torchcodec(
    video_bytes: bytes,
    num_frames: int,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    return_time_info: bool = False,
    random_start: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    使用 torchcodec 从字节加载视频
    
    Args:
        video_bytes: 视频字节
        num_frames: 目标帧数 T
        sample_rate: 视频帧采样间隔 (1=每帧, 2=隔帧)
        return_time_info: 是否返回时间信息（用于音频同步）
        random_start: 是否随机选择起始位置
        
    Returns:
        video: (T, H, W, C) uint8 tensor
        time_info (optional): dict with start_time, duration, fps
    """
    stream = io.BytesIO(video_bytes)
    decoder = VideoDecoder(stream)
    
    total_frames = decoder.metadata.num_frames
    fps = decoder.metadata.average_fps
    frame_indices, time_info = compute_frame_sampling_plan(
        total_frames=total_frames,
        source_fps=fps,
        num_frames=num_frames,
        sample_rate=sample_rate,
        target_fps=target_fps,
        random_start=random_start,
    )
    
    # 获取帧: torchcodec returns (T, C, H, W)
    frame_indices = normalize_frame_indices(frame_indices)
    frames = decoder.get_frames_at(frame_indices)  # frames.data: (T, C, H, W)
    # 转换为 (T, H, W, C) 格式
    video = frames.data.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
    
    if return_time_info:
        return video, time_info  # video: (T, H, W, C)
    
    return video  # (T, H, W, C)


def load_video_from_bytes_decord(
    video_bytes: bytes,
    num_frames: int,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    return_time_info: bool = False,
    random_start: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    使用 decord 从字节加载视频
    
    Args:
        video_bytes: 视频字节
        num_frames: 目标帧数 T
        sample_rate: 视频帧采样间隔 (1=每帧, 2=隔帧)
        return_time_info: 是否返回时间信息（用于音频同步）
        random_start: 是否随机选择起始位置
        
    Returns:
        video: (T, H, W, C) uint8 tensor
        time_info (optional): dict with start_time, duration, fps
    """
    stream = io.BytesIO(video_bytes)
    vr = decord.VideoReader(stream)
    
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    frame_indices, time_info = compute_frame_sampling_plan(
        total_frames=total_frames,
        source_fps=fps,
        num_frames=num_frames,
        sample_rate=sample_rate,
        target_fps=target_fps,
        random_start=random_start,
    )
    
    frame_indices = normalize_frame_indices(frame_indices)
    frames = vr.get_batch(frame_indices)  # (T, H, W, C)
    
    if return_time_info:
        return frames, time_info  # frames: (T, H, W, C)
    
    return frames  # (T, H, W, C)


def load_video_from_bytes(
    video_bytes: bytes,
    num_frames: int,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    use_torchcodec: bool = True,
    return_time_info: bool = False,
    random_start: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    加载视频，自动选择后端
    
    Returns:
        video: (T, H, W, C) uint8 tensor
    """
    if use_torchcodec and HAS_TORCHCODEC:
        return load_video_from_bytes_torchcodec(
            video_bytes,
            num_frames,
            sample_rate,
            target_fps,
            return_time_info,
            random_start=random_start,
        )
    elif HAS_DECORD:
        return load_video_from_bytes_decord(
            video_bytes,
            num_frames,
            sample_rate,
            target_fps,
            return_time_info,
            random_start=random_start,
        )
    else:
        raise RuntimeError("No video decoder available. Please install torchcodec or decord.")


def load_video_from_path_torchcodec(
    video_path: str,
    num_frames: int,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    return_time_info: bool = False,
    random_start: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    使用 torchcodec 从文件路径加载视频
    
    Returns:
        video: (T, H, W, C) uint8 tensor
        time_info (optional): dict with start_time, duration, fps
    """
    decoder = VideoDecoder(video_path)
    
    total_frames = decoder.metadata.num_frames
    fps = decoder.metadata.average_fps
    frame_indices, time_info = compute_frame_sampling_plan(
        total_frames=total_frames,
        source_fps=fps,
        num_frames=num_frames,
        sample_rate=sample_rate,
        target_fps=target_fps,
        random_start=random_start,
    )
    
    frame_indices = normalize_frame_indices(frame_indices)
    frames = decoder.get_frames_at(frame_indices)  # frames.data: (T, C, H, W)
    video = frames.data.permute(0, 2, 3, 1)  # (T, H, W, C)
    
    if return_time_info:
        return video, time_info
    return video


def load_video_from_path_decord(
    video_path: str,
    num_frames: int,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    return_time_info: bool = False,
    random_start: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    使用 decord 从文件路径加载视频
    
    Returns:
        video: (T, H, W, C) uint8 tensor
        time_info (optional): dict with start_time, duration, fps
    """
    vr = decord.VideoReader(video_path)
    
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    frame_indices, time_info = compute_frame_sampling_plan(
        total_frames=total_frames,
        source_fps=fps,
        num_frames=num_frames,
        sample_rate=sample_rate,
        target_fps=target_fps,
        random_start=random_start,
    )
    
    frame_indices = normalize_frame_indices(frame_indices)
    frames = vr.get_batch(frame_indices)  # (T, H, W, C)
    
    if return_time_info:
        return frames, time_info
    return frames


def load_video_from_path(
    video_path: str,
    num_frames: int,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    use_torchcodec: bool = True,
    return_time_info: bool = False,
    random_start: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """
    从文件路径加载视频，自动选择后端
    
    Returns:
        video: (T, H, W, C) uint8 tensor
        time_info (optional): dict with start_time, duration, fps
    """
    if use_torchcodec and HAS_TORCHCODEC:
        return load_video_from_path_torchcodec(video_path, num_frames, sample_rate, target_fps, return_time_info, random_start=random_start)
    elif HAS_DECORD:
        return load_video_from_path_decord(video_path, num_frames, sample_rate, target_fps, return_time_info, random_start=random_start)
    else:
        raise RuntimeError("No video decoder available. Please install torchcodec or decord.")


# =====================
# Audio Loading Utils
# =====================
def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sample_rate: int = 24000,
    max_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> torch.Tensor:
    """
    从字节加载音频
    
    Args:
        audio_bytes: 音频字节
        target_sample_rate: 目标采样率
        max_duration: 最大时长 (秒)，None 表示不限制
        start_time: 音频起始时间 (秒)，用于与视频同步裁剪
        duration: 音频时长 (秒)，用于与视频同步裁剪
        
    Returns:
        audio: (1, T_a) tensor, 范围 [-1, 1]
    """
    buf = io.BytesIO(audio_bytes)
    
    # soundfile 解码: wav_np shape is [T_a] or [T_a, C]
    wav_np, sr = sf.read(buf, dtype="float32")
    
    if wav_np.ndim == 1:
        waveform = torch.from_numpy(wav_np).unsqueeze(0)  # (T_a,) -> (1, T_a)
    else:
        # soundfile 默认 [T_a, C]，转成 [C, T_a]，再转单声道
        waveform = torch.from_numpy(wav_np).transpose(0, 1)  # (T_a, C) -> (C, T_a)
        waveform = waveform.mean(dim=0, keepdim=True)  # (C, T_a) -> (1, T_a)
    
    # waveform: (1, T_a)
    
    # 如果指定了 start_time 和 duration，先按时间裁剪（在原始采样率下）
    if start_time is not None and duration is not None:
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        # 确保不越界
        start_sample = max(0, min(start_sample, waveform.shape[1]))
        end_sample = max(start_sample, min(end_sample, waveform.shape[1]))
        waveform = waveform[:, start_sample:end_sample]  # (1, T_a) -> (1, T_crop)
        
        # 如果裁剪后长度不足，用零填充
        expected_samples = int(duration * sr)
        if waveform.shape[1] < expected_samples:
            pad_size = expected_samples - waveform.shape[1]
            pad = torch.zeros(1, pad_size)  # (1, pad_size)
            waveform = torch.cat([waveform, pad], dim=1)  # (1, T_crop) + (1, pad_size) -> (1, expected_samples)
    
    # 重采样: (1, T_a_old) -> (1, T_a_new)
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,  # (1, T_a_old)
            orig_freq=sr,
            new_freq=target_sample_rate,
        )  # (1, T_a_new)
    
    # 截断到最大时长（如果没有指定 duration）
    if max_duration is not None and start_time is None:
        max_samples = int(max_duration * target_sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]  # (1, T_a) -> (1, max_samples)
    
    return waveform  # (1, T_a)


def load_audio_from_path(
    audio_path: str,
    target_sample_rate: int = 24000,
    max_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> torch.Tensor:
    """
    从文件路径加载音频
    
    Args:
        audio_path: 音频文件路径
        target_sample_rate: 目标采样率
        max_duration: 最大时长 (秒)，None 表示不限制
        start_time: 音频起始时间 (秒)，用于与视频同步裁剪
        duration: 音频时长 (秒)，用于与视频同步裁剪
    
    Returns:
        audio: (1, T_a) tensor, 范围 [-1, 1]
    """
    try:
        waveform, sr = torchaudio.load(audio_path)  # (C, T_a)
    except Exception:
        wav_np, sr = sf.read(audio_path, dtype="float32")  # (T_a,) or (T_a, C)
        if wav_np.ndim == 1:
            waveform = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T_a)
        else:
            waveform = torch.from_numpy(wav_np).transpose(0, 1)  # (C, T_a)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T_a)
    
    if start_time is not None and duration is not None:
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        start_sample = max(0, min(start_sample, waveform.shape[1]))
        end_sample = max(start_sample, min(end_sample, waveform.shape[1]))
        waveform = waveform[:, start_sample:end_sample]
        expected_samples = int(duration * sr)
        if waveform.shape[1] < expected_samples:
            pad_size = expected_samples - waveform.shape[1]
            waveform = torch.cat([waveform, torch.zeros(1, pad_size)], dim=1)
    
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sample_rate,
        )
    
    if max_duration is not None and start_time is None:
        max_samples = int(max_duration * target_sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
    
    return waveform  # (1, T_a)


def load_audio_from_video_bytes(
    video_bytes: bytes,
    target_sample_rate: int = 24000,
    max_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> torch.Tensor:
    """
    从视频字节中提取音频
    
    Returns:
        audio: (1, T_a) tensor
    """
    buf = io.BytesIO(video_bytes)
    
    try:
        # torchaudio 可以从视频中提取音频
        waveform, sr = torchaudio.load(buf)  # (C, T_a)
    except Exception as e:
        logging.warning(f"Failed to load audio from video bytes: {e}")
        # 返回静音: (1, T_a)
        target_duration = duration or max_duration or 1.0
        return torch.zeros(1, int(target_sample_rate * target_duration))  # (1, T_a)
    
    # 转单声道: (C, T_a) -> (1, T_a)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T_a)
    
    # 如果指定了 start_time 和 duration，先按时间裁剪（在原始采样率下）
    if start_time is not None and duration is not None:
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        # 确保不越界
        start_sample = max(0, min(start_sample, waveform.shape[1]))
        end_sample = max(start_sample, min(end_sample, waveform.shape[1]))
        waveform = waveform[:, start_sample:end_sample]  # (1, T_a) -> (1, T_crop)
        
        # 如果裁剪后长度不足，用零填充
        expected_samples = int(duration * sr)
        if waveform.shape[1] < expected_samples:
            pad_size = expected_samples - waveform.shape[1]
            pad = torch.zeros(1, pad_size)  # (1, pad_size)
            waveform = torch.cat([waveform, pad], dim=1)  # (1, T_crop) + (1, pad_size) -> (1, expected)
    
    # 重采样: (1, T_a_old) -> (1, T_a_new)
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,  # (1, T_a_old)
            orig_freq=sr,
            new_freq=target_sample_rate,
        )  # (1, T_a_new)
    
    # 截断到最大时长（如果没有指定 duration）
    if max_duration is not None and start_time is None:
        max_samples = int(max_duration * target_sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]  # (1, T_a) -> (1, max_samples)
    
    return waveform  # (1, T_a)


def load_audio_from_video_path(
    video_path: str,
    target_sample_rate: int = 24000,
    max_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> torch.Tensor:
    """
    从视频文件路径中提取音频
    
    Args:
        video_path: 视频文件路径
        target_sample_rate: 目标采样率
        max_duration: 最大时长 (秒)，None 表示不限制
        start_time: 音频起始时间 (秒)，用于与视频同步裁剪
        duration: 音频时长 (秒)，用于与视频同步裁剪
    
    Returns:
        audio: (1, T_a) tensor
    """
    try:
        waveform, sr = torchaudio.load(video_path)  # (C, T_a)
    except Exception as e:
        logging.warning(f"Failed to load audio from video path {video_path}: {e}")
        target_duration = duration or max_duration or 1.0
        return torch.zeros(1, int(target_sample_rate * target_duration))
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T_a)
    
    if start_time is not None and duration is not None:
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        start_sample = max(0, min(start_sample, waveform.shape[1]))
        end_sample = max(start_sample, min(end_sample, waveform.shape[1]))
        waveform = waveform[:, start_sample:end_sample]
        expected_samples = int(duration * sr)
        if waveform.shape[1] < expected_samples:
            pad_size = expected_samples - waveform.shape[1]
            waveform = torch.cat([waveform, torch.zeros(1, pad_size)], dim=1)
    
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sample_rate,
        )
    
    if max_duration is not None and start_time is None:
        max_samples = int(max_duration * target_sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
    
    return waveform  # (1, T_a)


# =====================
# Video Transform Builder
# =====================
_VALID_SPATIAL_MODES = ("resize_center_crop", "resize")


def build_video_transform(
    resolution: Union[int, tuple],
    spatial_transform_mode: str = "resize_center_crop",
    spatial_roundtrip_short_edge: Optional[int] = None,
) -> transforms.Compose:
    """
    构建视频空间变换 pipeline。

    Args:
        resolution: 目标分辨率，int 或 (H, W) 元组
        spatial_transform_mode:
            "resize_center_crop" — Resize 短边 + CenterCrop，保持宽高比（默认）
            "resize" — 直接 Resize 到 (H, W)，不保持宽高比
        spatial_roundtrip_short_edge:
            可选，若给定整数 N，则在常规 transform 之前先做一次
            ``Resize(短边=N, antialias=True)`` 以构成 bilinear round-trip 低通，
            抹掉高频指纹（用作 A 路 = 正方形已 crop 数据的正则化实验）。
            默认 None 关闭。典型值：对 256x256 磁盘数据设 224（先降到 224 再升到 256）。

    Input:  (C, T, H, W) uint8
    Output: (C, T, H', W') float in [-1, 1]
    """
    if spatial_transform_mode not in _VALID_SPATIAL_MODES:
        raise ValueError(
            f"spatial_transform_mode must be one of {_VALID_SPATIAL_MODES}, "
            f"got '{spatial_transform_mode}'"
        )

    if isinstance(resolution, int):
        resolution_tuple = (resolution, resolution)
    else:
        resolution_tuple = tuple(resolution)

    steps: list = []
    if spatial_roundtrip_short_edge is not None and int(spatial_roundtrip_short_edge) > 0:
        steps.append(transforms.Resize(int(spatial_roundtrip_short_edge), antialias=True))

    if spatial_transform_mode == "resize_center_crop":
        target_short_edge = min(resolution_tuple)
        crop_size = resolution_tuple
        steps.extend([
            transforms.Resize(target_short_edge, antialias=True),
            transforms.Lambda(lambda x: _center_crop_fn(x, crop_size)),
            transforms.Lambda(lambda x: x.float() / 127.5 - 1.0),
        ])
    else:
        steps.extend([
            transforms.Resize(resolution_tuple, antialias=True),
            transforms.Lambda(lambda x: x.float() / 127.5 - 1.0),
        ])

    return transforms.Compose(steps)


# =====================
# Processor Classes
# =====================
class AudioVideoProcessor:
    """
    处理器：从 bin 文件加载视频和音频
    
    Output Shapes:
        video: (C, T, H, W) tensor in [-1, 1], C=3, T=num_frames, H=W=resolution
        audio: (1, T_a) tensor, T_a = audio_sample_rate * duration
    """
    
    def __init__(
        self,
        num_frames: int = 25,
        resolution: Union[int, tuple] = 256,
        sample_rate: int = 1,
        target_fps: Optional[float] = None,
        audio_sample_rate: int = 24000,
        max_audio_duration: Optional[float] = None,
        use_torchcodec: bool = True,
        random_start: bool = False,
        spatial_transform_mode: str = "resize_center_crop",
        spatial_roundtrip_short_edge: Optional[int] = None,
    ):
        self.num_frames = num_frames
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = tuple(resolution)
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_duration = max_audio_duration
        self.use_torchcodec = use_torchcodec
        self.random_start = random_start
        
        self.video_transform = build_video_transform(
            self.resolution, spatial_transform_mode, spatial_roundtrip_short_edge,
        )
    
    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理一条数据，视频和音频会同步随机裁剪到相同的时间位置
        
        Returns:
            dict with:
                video: (C, T, H, W), audio: (1, T_a), valid: bool, time_info: dict
        """
        try:
            bin_path = item["bin_path"]
            
            # 加载视频字节
            video_offset = item["video_bin_offset"]
            video_length = item["video_bin_length"]
            video_bytes = read_bytes_from_bin(bin_path, video_offset, video_length)
            
            # 加载视频并获取时间信息: video (T, H, W, C), time_info dict
            video, time_info = load_video_from_bytes(
                video_bytes,
                num_frames=self.num_frames,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
                use_torchcodec=self.use_torchcodec,
                return_time_info=True,
                random_start=self.random_start,
            )  # video: (T, H, W, C) uint8
            
            # 转换格式: (T, H, W, C) -> (C, T, H, W)
            video = video.permute(3, 0, 1, 2)  # (C, T, H, W)
            # 归一化: (C, T, H, W) uint8 -> (C, T, H, W) float [-1, 1]
            video = self.video_transform(video)  # (C, T, H, W)
            
            # 获取视频裁剪的时间信息
            video_start_time = time_info["start_time"]
            video_duration = time_info["duration"]
            
            # 加载音频（与视频同步裁剪）
            if "audio_bin_offset" in item and "audio_bin_length" in item:
                # 从单独的音频 bin 加载
                audio_offset = item["audio_bin_offset"]
                audio_length = item["audio_bin_length"]
                audio_bytes = read_bytes_from_bin(bin_path, audio_offset, audio_length)
                audio = load_audio_from_bytes(
                    audio_bytes,
                    target_sample_rate=self.audio_sample_rate,
                    max_duration=self.max_audio_duration,
                    start_time=video_start_time,
                    duration=video_duration,
                )  # (1, T_a)
            else:
                # 从视频字节中提取音频（使用相同的时间裁剪）
                audio = load_audio_from_video_bytes(
                    video_bytes,
                    target_sample_rate=self.audio_sample_rate,
                    max_duration=self.max_audio_duration,
                    start_time=video_start_time,
                    duration=video_duration,
                )  # (1, T_a)
            
            result = {
                "video": video,  # (C, T, H, W)
                "audio": audio,  # (1, T_a)
                "valid": True,
                "time_info": time_info,
            }
            caption_text = item.get("prompt_v2") or item.get("prompt_v1") or item.get("prompt")
            if caption_text:
                result["caption"] = str(caption_text)
            video_desc = item.get("video_description")
            if video_desc:
                result["video_description"] = str(video_desc)
            audio_desc = item.get("audio_description")
            if audio_desc:
                result["audio_description"] = str(audio_desc)
            return result
            
        except Exception as e:
            logging.error(f"Error processing item {item}: {e}")
            return None


class VideoOnlyProcessor:
    """
    仅处理视频的处理器（用于从 bin 加载的视频验证集）
    
    Output: video (C, T, H, W) tensor in [-1, 1]
    """
    
    def __init__(
        self,
        num_frames: int = 25,
        resolution: Union[int, tuple] = 256,
        sample_rate: int = 1,
        target_fps: Optional[float] = None,
        use_torchcodec: bool = True,
        spatial_transform_mode: str = "resize_center_crop",
        spatial_roundtrip_short_edge: Optional[int] = None,
    ):
        self.num_frames = num_frames
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = tuple(resolution)
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.use_torchcodec = use_torchcodec
        
        self.video_transform = build_video_transform(
            self.resolution, spatial_transform_mode, spatial_roundtrip_short_edge,
        )
    
    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            bin_path = item["bin_path"]
            video_offset = item["video_bin_offset"]
            video_length = item["video_bin_length"]
            video_bytes = read_bytes_from_bin(bin_path, video_offset, video_length)
            
            video = load_video_from_bytes(
                video_bytes,
                num_frames=self.num_frames,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
                use_torchcodec=self.use_torchcodec,
            )  # (T, H, W, C)
            
            video = video.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video = self.video_transform(video)  # (C, T, H, W) in [-1, 1]
            
            return {
                "video": video,  # (C, T, H, W)
                "valid": True,
            }
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            return None


class VideoFileProcessor:
    """
    从文件路径加载视频的处理器（用于评测数据）
    
    Output: video (C, T, H, W) tensor in [-1, 1]
    """
    
    def __init__(
        self,
        num_frames: int = 25,
        resolution: Union[int, tuple] = 256,
        sample_rate: int = 1,
        target_fps: Optional[float] = None,
        use_torchcodec: bool = True,
        video_root: Optional[str] = None,
        spatial_transform_mode: str = "resize_center_crop",
        spatial_roundtrip_short_edge: Optional[int] = None,
    ):
        self.num_frames = num_frames
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = tuple(resolution)
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.use_torchcodec = use_torchcodec
        self.video_root = video_root
        
        self.video_transform = build_video_transform(
            self.resolution, spatial_transform_mode, spatial_roundtrip_short_edge,
        )
    
    def _resolve_path(self, raw_path: str, metadata_dir: Optional[str] = None) -> str:
        return resolve_metadata_media_path(
            raw_path, root=self.video_root, metadata_dir=metadata_dir,
        )
    
    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            video_path = item.get("video_path") or item.get("path")
            if not video_path:
                logging.error(f"No video_path found in item: {item}")
                return None
            
            video_path = self._resolve_path(video_path, item.get("_jsonl_dir"))
            
            if not os.path.exists(video_path):
                logging.error(f"Video file not found: {video_path}")
                return None
            
            video = load_video_from_path(
                video_path,
                num_frames=self.num_frames,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
                use_torchcodec=self.use_torchcodec,
            )  # (T, H, W, C)
            
            video = video.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video = self.video_transform(video)  # (C, T, H, W) in [-1, 1]
            
            return {
                "video": video,  # (C, T, H, W)
                "video_path": video_path,
                "valid": True,
            }
        except Exception as e:
            logging.error(f"Error processing video file {item}: {e}")
            return None


class AudioOnlyProcessor:
    """
    仅处理音频的处理器（用于从 bin 加载的音频验证集）
    
    Output: audio (1, T_a) tensor
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        max_duration: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration

    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            bin_path = item["bin_path"]
            
            if "audio_bin_offset" in item and "audio_bin_length" in item:
                audio_offset = item["audio_bin_offset"]
                audio_length = item["audio_bin_length"]
                audio_bytes = read_bytes_from_bin(bin_path, audio_offset, audio_length)
                audio = load_audio_from_bytes(
                    audio_bytes,
                    target_sample_rate=self.sample_rate,
                    max_duration=self.max_duration,
                )  # (1, T_a)
            else:
                # 从视频中提取
                video_offset = item["video_bin_offset"]
                video_length = item["video_bin_length"]
                video_bytes = read_bytes_from_bin(bin_path, video_offset, video_length)
                audio = load_audio_from_video_bytes(
                    video_bytes,
                    target_sample_rate=self.sample_rate,
                    max_duration=self.max_duration,
                )  # (1, T_a)
            
            return {
                "audio": audio,  # (1, T_a)
                "valid": True,
            }
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return None


class AudioFileProcessor:
    """
    从文件路径加载音频的处理器（用于评测数据）
    
    Output: audio (1, T_a) tensor
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        max_duration: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration

    def _resolve_path(self, raw_path: str, metadata_dir: Optional[str] = None) -> str:
        return resolve_metadata_media_path(raw_path, metadata_dir=metadata_dir)
    
    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            audio_path = item.get("audio_path") or item.get("path")
            if not audio_path:
                logging.error(f"No audio_path found in item: {item}")
                return None
            
            audio_path = self._resolve_path(audio_path, item.get("_jsonl_dir"))

            if not os.path.exists(audio_path):
                logging.error(f"Audio file not found: {audio_path}")
                return None
            
            audio = load_audio_from_path(
                audio_path,
                target_sample_rate=self.sample_rate,
                max_duration=self.max_duration,
            )  # (1, T_a)
            
            return {
                "audio": audio,  # (1, T_a)
                "audio_path": audio_path,
                "valid": True,
            }
        except Exception as e:
            logging.error(f"Error processing audio file {item}: {e}")
            return None


class AudioVideoFileProcessor:
    """
    从文件路径加载音视频的处理器（训练用，文件路径格式）
    
    支持的 jsonl 字段：
        - video_path (必须): 视频文件路径
        - audio_path (可选): 单独的音频文件路径，不提供则从视频中提取音轨
    
    Output Shapes:
        video: (C, T, H, W) tensor in [-1, 1], C=3, T=num_frames, H=W=resolution
        audio: (1, T_a) tensor, T_a = audio_sample_rate * duration
    """
    
    def __init__(
        self,
        num_frames: int = 25,
        resolution: Union[int, tuple] = 256,
        sample_rate: int = 1,
        target_fps: Optional[float] = None,
        audio_sample_rate: int = 24000,
        max_audio_duration: Optional[float] = None,
        use_torchcodec: bool = True,
        video_root: Optional[str] = None,
        random_start: bool = False,
        spatial_transform_mode: str = "resize_center_crop",
        spatial_roundtrip_short_edge: Optional[int] = None,
        distill_encoder_fps: Optional[float] = None,
        distill_audio_target_sr: Optional[int] = None,
    ):
        self.num_frames = num_frames
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = tuple(resolution)
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_duration = max_audio_duration
        self.use_torchcodec = use_torchcodec
        self.video_root = video_root
        self.random_start = random_start
        self.distill_encoder_fps = distill_encoder_fps
        self.distill_audio_target_sr = distill_audio_target_sr
        
        self.video_transform = build_video_transform(
            self.resolution, spatial_transform_mode, spatial_roundtrip_short_edge,
        )
    
    def _resolve_path(self, raw_path: str, metadata_dir: Optional[str] = None) -> str:
        return resolve_metadata_media_path(
            raw_path, root=self.video_root, metadata_dir=metadata_dir,
        )
    
    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理一条数据，视频和音频会同步随机裁剪到相同的时间位置
        
        Returns:
            dict with:
                video: (C, T, H, W), audio: (1, T_a), valid: bool, time_info: dict
        """
        try:
            video_path = item.get("video_path") or item.get("path")
            if not video_path:
                logging.error(f"No video_path found in item: {item}")
                return None
            
            video_path = self._resolve_path(video_path, item.get("_jsonl_dir"))
            
            if not os.path.exists(video_path):
                logging.error(f"Video file not found: {video_path}")
                return None
            
            video, time_info = load_video_from_path(
                video_path,
                num_frames=self.num_frames,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
                use_torchcodec=self.use_torchcodec,
                return_time_info=True,
                random_start=self.random_start,
            )  # video: (T, H, W, C) uint8
            
            video = video.permute(3, 0, 1, 2)  # (C, T, H, W) uint8
            video = self.video_transform(video)  # (C, T, H, W) in [-1, 1]

            distill_first_frame = None
            distill_video_frames = None
            if self.distill_encoder_fps is not None:
                from PIL import Image as PILImage
                T_total = video.shape[1]
                data_fps = self.target_fps if self.target_fps else 24.0

                def _frame_to_pil(frame_float):
                    return PILImage.fromarray(
                        ((frame_float.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    )

                distill_first_frame = _frame_to_pil(video[:, 0])
                T_rem = T_total - 1
                group_size = max(1, round(data_fps / self.distill_encoder_fps))
                center = group_size // 2
                indices = list(range(center, T_rem, group_size))
                if len(indices) % 2 != 0:
                    indices = indices[:-1]
                if len(indices) < 2:
                    indices = [0, min(1, T_rem - 1)]
                distill_video_frames = [_frame_to_pil(video[:, 1 + i]) for i in indices]
            
            video_start_time = time_info["start_time"]
            video_duration = time_info["duration"]
            
            audio_path = item.get("audio_path")
            if audio_path:
                audio_path = self._resolve_path(audio_path, item.get("_jsonl_dir"))
            if audio_path and os.path.exists(audio_path):
                audio = load_audio_from_path(
                    audio_path,
                    target_sample_rate=self.audio_sample_rate,
                    max_duration=self.max_audio_duration,
                    start_time=video_start_time,
                    duration=video_duration,
                )
            else:
                audio = load_audio_from_video_path(
                    video_path,
                    target_sample_rate=self.audio_sample_rate,
                    max_duration=self.max_audio_duration,
                    start_time=video_start_time,
                    duration=video_duration,
                )

            distill_audio_16k = None
            if self.distill_audio_target_sr is not None and audio is not None:
                import librosa
                wav_np = audio[0].float().numpy()  # (T_a,) — already on CPU
                if self.audio_sample_rate != self.distill_audio_target_sr:
                    wav_np = librosa.resample(
                        wav_np,
                        orig_sr=self.audio_sample_rate,
                        target_sr=self.distill_audio_target_sr,
                    )
                distill_audio_16k = wav_np
            
            result = {
                "video": video,       # (C, T, H, W)
                "audio": audio,       # (1, T_a)
                "video_path": video_path,
                "valid": True,
                "time_info": time_info,
                "long_video_id": parse_long_video_id(video_path),
            }
            if distill_first_frame is not None:
                result["distill_first_frame"] = distill_first_frame
            if distill_video_frames is not None:
                result["distill_video_frames"] = distill_video_frames
            if distill_audio_16k is not None:
                result["distill_audio_16k"] = distill_audio_16k
            caption_text = item.get("prompt_v2") or item.get("prompt_v1") or item.get("prompt")
            if caption_text:
                result["caption"] = str(caption_text)
            video_desc = item.get("video_description")
            if video_desc:
                result["video_description"] = str(video_desc)
            audio_desc = item.get("audio_description")
            if audio_desc:
                result["audio_description"] = str(audio_desc)
            return result
            
        except Exception as e:
            logging.error(f"Error processing item {item}: {e}")
            return None


# Image extensions that ImageFileProcessor accepts. Lower-case match.
_IMAGE_FILE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _is_image_path(path: Optional[str]) -> bool:
    if not path:
        return False
    return str(path).lower().endswith(_IMAGE_FILE_EXTS)


class ImageFileProcessor:
    """
    从文件路径加载图片并伪装成单帧视频 (C, 1, H, W) 用于和视频 pipeline 共用。

    支持的 jsonl 字段:
        - image_path / video_path / path: 图片文件路径 (.jpg/.png/.webp/.bmp/.jpeg)

    Output Shapes:
        video: (C, 1, H, W) tensor in [-1, 1]，T 维 = 1 表示单帧伪视频
        不返回 audio 字段 (图片没有音轨)；distill 路径仅产 image_feat。
    """

    def __init__(
        self,
        resolution: Union[int, tuple] = 256,
        spatial_transform_mode: str = "resize_center_crop",
        spatial_roundtrip_short_edge: Optional[int] = None,
        video_root: Optional[str] = None,
        distill_encoder_fps: Optional[float] = None,
        # Accept (and silently ignore) the remaining kwargs that come from
        # ``build_audio_video_streaming_dataset`` so this processor is a
        # drop-in replacement.
        num_frames: Optional[int] = None,
        sample_rate: Optional[int] = None,
        target_fps: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
        max_audio_duration: Optional[float] = None,
        use_torchcodec: Optional[bool] = None,
        random_start: bool = False,
        distill_audio_target_sr: Optional[int] = None,
    ):
        if isinstance(resolution, int):
            self.resolution = (resolution, resolution)
        else:
            self.resolution = tuple(resolution)
        self.video_root = video_root
        self.distill_encoder_fps = distill_encoder_fps
        # Stored for completeness but unused (no temporal axis to sample).
        self.num_frames = num_frames
        self.target_fps = target_fps

        self.video_transform = build_video_transform(
            self.resolution, spatial_transform_mode, spatial_roundtrip_short_edge,
        )

    def _resolve_path(self, raw_path: str) -> str:
        if self.video_root and not os.path.isabs(raw_path):
            return os.path.join(self.video_root, raw_path)
        return raw_path

    def __call__(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            from PIL import Image as PILImage

            image_path = (
                item.get("image_path")
                or item.get("video_path")
                or item.get("path")
            )
            if not image_path:
                logging.error(f"No image_path found in item: {item}")
                return None

            image_path = self._resolve_path(image_path)
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                return None

            with PILImage.open(image_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                np_img = np.asarray(pil_img, dtype=np.uint8)  # (H, W, 3)

            # (H, W, 3) -> (3, H, W) -> (3, 1, H, W) so it matches video shape.
            video = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(1).contiguous()
            video = self.video_transform(video)  # (C, 1, H', W') in [-1, 1]

            distill_first_frame = None
            if self.distill_encoder_fps is not None:
                # Re-derive the PIL frame from the (already normalised) tensor
                # so it stays consistent with the video pipeline (same crop/
                # resize already applied).
                frame = video[:, 0]
                distill_first_frame = PILImage.fromarray(
                    ((frame.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                )

            result = {
                "video": video,            # (C, 1, H, W)
                "video_path": image_path,  # reuse the same key so downstream code is happy
                "image_path": image_path,
                "is_image": True,
                "valid": True,
            }
            if distill_first_frame is not None:
                result["distill_first_frame"] = distill_first_frame

            caption_text = item.get("prompt_v2") or item.get("prompt_v1") or item.get("prompt")
            if caption_text:
                result["caption"] = str(caption_text)
            return result

        except Exception as e:
            logging.error(f"Error processing image file {item}: {e}")
            return None


# =====================
# Dataset Classes
# =====================
class StreamingJsonlDataset(IterableDataset):
    """流式 JSONL 数据集，使用 mmap 读取"""
    
    def __init__(
        self,
        name: str,
        jsonl_path: Union[str, Path],
        data_rank: int = 0,
        data_world_size: int = 1,
    ):
        super().__init__()
        self.name = name
        self.jsonl_path = Path(jsonl_path)
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        
        self.handles = None
        self.state_dict = {"bytes_offset": 0, "line_shift": 0}
    
    def _open_file(self):
        if self.handles is None:
            f = open(self.jsonl_path, "rb")
            if os.path.getsize(self.jsonl_path) == 0:
                logging.warning(f"Skipping empty jsonl file: {self.jsonl_path}")
                f.close()
                return None
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.handles = (f, mm)
        return self.handles[1]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        mm = self._open_file()
        if mm is None:
            return
        mm.seek(self.state_dict["bytes_offset"])
        
        line_idx = 0
        while True:
            line = mm.readline()
            if not line:
                break
            
            self.state_dict["bytes_offset"] = mm.tell()
            self.state_dict["line_shift"] = line_idx
            line_idx += 1
            
            # 分布式数据划分
            if (line_idx - 1) % self.data_world_size != self.data_rank:
                continue
            
            try:
                data = json.loads(line.decode("utf-8").strip())
                yield {
                    "data": data,
                    "state_dict": deepcopy(self.state_dict),
                }
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {line_idx} in {self.jsonl_path}: {e}")
                continue
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.state_dict["bytes_offset"] = state_dict.get("bytes_offset", 0)
        self.state_dict["line_shift"] = state_dict.get("line_shift", 0)
        logging.info(
            f"[{self.name}] Resumed from bytes_offset={self.state_dict['bytes_offset']}, "
            f"line_shift={self.state_dict['line_shift']}"
        )
    
    def __del__(self):
        if self.handles is not None:
            try:
                self.handles[0].close()
                self.handles[1].close()
            except Exception:
                pass


class StreamingAggregationDataset(IterableDataset):
    """聚合多个 StreamingJsonlDataset"""
    
    def __init__(
        self,
        name: str,
        dataset_dir_or_path: Union[str, Path],
        data_rank: int = 0,
        data_world_size: int = 1,
    ):
        super().__init__()
        dataset_dir_or_path = Path(dataset_dir_or_path)
        
        if dataset_dir_or_path.is_dir():
            jsonl_paths = sorted(dataset_dir_or_path.rglob("*.jsonl"))
        elif dataset_dir_or_path.is_file() and dataset_dir_or_path.suffix == ".jsonl":
            jsonl_paths = [dataset_dir_or_path]
        else:
            raise TypeError("`dataset_dir_or_path` must be a directory or a jsonl file.")
        
        if not jsonl_paths:
            raise ValueError(f"No jsonl files found in {dataset_dir_or_path}")
        
        self.datasets = [
            StreamingJsonlDataset(name, jsonl_path, data_rank, data_world_size)
            for jsonl_path in jsonl_paths
        ]
        
        self.name = name
        self.dataset_dir_or_path = dataset_dir_or_path
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.state_dict = {"file_shift": 0}
        self._jsonl_path = None
    
    @property
    def jsonl_path(self):
        return self._jsonl_path
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx, dataset in enumerate(self.datasets):
            if idx < self.state_dict["file_shift"] % len(self.datasets):
                continue
            for item in iter(dataset):
                self._jsonl_path = dataset.jsonl_path
                data = item["data"]
                if isinstance(data, dict):
                    data = dict(data)
                    data.setdefault("_jsonl_dir", str(Path(dataset.jsonl_path).parent))
                yield {
                    "data": data,
                    "state_dict": {"file_shift": self.state_dict["file_shift"], **item["state_dict"]},
                }
            self.state_dict["file_shift"] += 1
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.state_dict["file_shift"] = state_dict.pop("file_shift", 0)
        idx = self.state_dict["file_shift"] % len(self.datasets)
        self.datasets[idx].load_state_dict(state_dict)
        logging.info(f"[{self.name}] Resumed from file_shift={self.state_dict['file_shift']}")


class AudioVideoStreamingDataset(IterableDataset):
    """音视频联合流式数据集，支持多数据集加权采样"""
    
    def __init__(
        self,
        datasets: List[StreamingAggregationDataset],
        processors: List,
        weights: Optional[Union[List[float], Dict[str, float]]] = None,
        raise_processor_error: bool = False,
        raise_stop_iteration: bool = False,
        filter_fn: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        data_rank: int = 0,
        seed: int = 1024,
    ):
        super().__init__()
        if len(datasets) != len(processors):
            raise ValueError("Length of processors list must match number of datasets.")
        
        self.datasets = datasets
        self.processors = processors
        
        # 归一化权重
        if weights is None:
            self.weights = [1.0 / len(datasets)] * len(datasets)
        else:
            if isinstance(weights, dict):
                self.weights = [weights.get(d.name, 1.0) for d in datasets]
            else:
                self.weights = list(weights)
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        
        self.raise_processor_error = raise_processor_error
        self.raise_stop_iteration = raise_stop_iteration
        self.filter_fn = filter_fn if filter_fn else (lambda *args, **kwargs: True)
        
        self.iterators: Optional[List[Iterator]] = None
        self.rng_state = np.random.default_rng(seed + data_rank)
        self.consumed_samples = defaultdict(int)
        self.used_epochs = defaultdict(int)
        self.empty_pass_times = defaultdict(int)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.iterators is None:
            self.iterators = [iter(d) for d in self.datasets]
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            worker_id = worker_info.id % worker_info.num_workers
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        cur_states = {"dataset_state_dict": {}}
        
        while True:
            cur_rng_state = self.rng_state.bit_generator.state
            next_dataset_idx = self.rng_state.choice(len(self.datasets), p=self.weights)
            next_dataset_name = self.datasets[next_dataset_idx].name
            
            try:
                item: Dict[str, Any] = next(self.iterators[next_dataset_idx])
                
                cur_states["consumed_samples"] = dict(self.consumed_samples)
                cur_states["rng_state"] = self.rng_state.bit_generator.state
                cur_states["used_epochs"] = dict(self.used_epochs)
                cur_states["dataset_state_dict"][next_dataset_name] = item["state_dict"]
                
                meta = item["data"]
                
                if self.filter_fn(next_dataset_name, meta) and sum(self.consumed_samples.values()) % num_workers == worker_id:
                    processed_item = None
                    try:
                        processed_item = self.processors[next_dataset_idx](meta)
                    except Exception as e:
                        logging.error(f"Load data wrong from meta `{meta}` in {self.datasets[next_dataset_idx].jsonl_path}.")
                        if self.raise_processor_error:
                            raise e
                    
                    if processed_item is not None:
                        source_identifier = build_source_identifier(meta)
                        if source_identifier:
                            file_name = source_identifier
                        else:
                            file_name = f"{next_dataset_name}_{self.consumed_samples[next_dataset_name]}"
                        
                        yield_data = {
                            "data": {
                                "name": next_dataset_name,
                                "dataset_id": next_dataset_idx,
                                "file_name": file_name,
                                "source_path": source_identifier,
                                **processed_item,
                            },
                            "state_dict": deepcopy(cur_states),
                        }
                        yield yield_data
                
                self.consumed_samples[next_dataset_name] += 1
                
            except StopIteration:
                if self.raise_stop_iteration:
                    break
                
                self.rng_state.bit_generator.state = cur_rng_state
                self.used_epochs[next_dataset_name] += 1
                
                if self.consumed_samples[next_dataset_name] <= 0:
                    self.empty_pass_times[next_dataset_name] += 1
                
                if self.empty_pass_times[next_dataset_name] >= 2:
                    raise RuntimeError(
                        f"Dataset {next_dataset_name} has not spit out any data for two rounds."
                    )
                
                logging.info(
                    f"AudioVideoStreamingDataset: {next_dataset_name} have cycled "
                    f"{self.used_epochs[next_dataset_name]} times!"
                )
                agg_dataset = self.datasets[next_dataset_idx]
                agg_dataset.state_dict["file_shift"] = 0
                for inner_ds in agg_dataset.datasets:
                    inner_ds.state_dict["bytes_offset"] = 0
                    inner_ds.state_dict["line_shift"] = 0
                self.iterators[next_dataset_idx] = iter(agg_dataset)
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.consumed_samples.update(state_dict.get("consumed_samples", {}))
        if "used_epochs" in state_dict:
            self.used_epochs.update(state_dict["used_epochs"])
        if state_dict.get("rng_state") is not None:
            self.rng_state.bit_generator.state = state_dict["rng_state"]
        
        logging.info(
            f"Loaded AudioVideoStreamingDataset state: consumed_samples={dict(self.consumed_samples)}, "
            f"used_epochs={dict(self.used_epochs)}"
        )
        
        for dataset in self.datasets:
            if (dataset_state_dict := state_dict.get("dataset_state_dict", {}).get(dataset.name)) is not None:
                dataset.load_state_dict(dataset_state_dict)
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "consumed_samples": dict(self.consumed_samples),
            "used_epochs": dict(self.used_epochs),
            "rng_state": self.rng_state.bit_generator.state,
            "dataset_state_dict": {d.name: d.state_dict for d in self.datasets},
        }


class IVAlterstepStreamingDataset(IterableDataset):
    """图片/视频双流交替数据集。

    每次 ``__iter__`` 同时从一个 ``image_dataset``（图片 stream，per-sample
    shape ``(C, 1, H, W)``，无音频）和一个 ``video_dataset``（视频 stream，per-sample
    shape ``(C, T, H, W)``，可能含音频）中各取一份样本，分别打包成
    ``image_batch`` 和 ``video_batch``，最后 yield 一个嵌套字典:

        {
            "image_batch": <AudioVideoCollator output>,  # (B_img, C, 1, H, W)
            "video_batch": <AudioVideoCollator output>,  # (B_vid, C, T, H, W)
        }

    具体的 modality 抽样（图片 step / 视频 step）放在 trainer 入口完成 (参考
    SSVAE ``VideoAutoencodingEngine.get_input``)，这样保证不同 rank 的 DataLoader
    永远是产 image+video 双 batch，由 trainer 内部 RNG + DDP broadcast 保证
    全局 modality 一致。
    """

    def __init__(
        self,
        image_dataset: IterableDataset,
        video_dataset: IterableDataset,
        image_batch_size: int = 1,
        video_batch_size: int = 1,
        image_collator: Optional["AudioVideoCollator"] = None,
        video_collator: Optional["AudioVideoCollator"] = None,
    ):
        super().__init__()
        if image_batch_size <= 0:
            raise ValueError(f"image_batch_size must be > 0, got {image_batch_size}")
        if video_batch_size <= 0:
            raise ValueError(f"video_batch_size must be > 0, got {video_batch_size}")

        self.image_dataset = image_dataset
        self.video_dataset = video_dataset
        self.image_batch_size = int(image_batch_size)
        self.video_batch_size = int(video_batch_size)

        # Lazy-default to module-level AudioVideoCollator. We can't reference
        # the class by name yet (it's defined below), so accept the instance
        # to be constructed in the builder.
        self.image_collator = image_collator
        self.video_collator = video_collator

    @property
    def name(self) -> str:
        return "iv_alterstep"

    def _take_n(self, it: Iterator, n: int) -> List[Dict[str, Any]]:
        out = []
        for _ in range(n):
            try:
                out.append(next(it))
            except StopIteration:
                break
        return out

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.image_collator is None or self.video_collator is None:
            raise RuntimeError(
                "IVAlterstepStreamingDataset requires `image_collator` and "
                "`video_collator` to be provided (typically via builder)."
            )

        img_iter = iter(self.image_dataset)
        vid_iter = iter(self.video_dataset)

        while True:
            img_samples = self._take_n(img_iter, self.image_batch_size)
            vid_samples = self._take_n(vid_iter, self.video_batch_size)

            # If either side runs out, stop. AudioVideoStreamingDataset
            # already loops infinitely by default (raise_stop_iteration=False)
            # so this typically only happens when both streams are exhausted.
            if not img_samples or not vid_samples:
                logging.warning(
                    "[IVAlterstep] one of image/video streams produced 0 samples "
                    f"(img={len(img_samples)}, vid={len(vid_samples)}); stopping"
                )
                return

            img_batch = self.image_collator(img_samples)
            vid_batch = self.video_collator(vid_samples)

            # Merge state_dicts so the trainer's resume machinery can save
            # the cursor for both streams. Each sub-state_dict is a dict
            # like {"dataset_state_dict": {...}, "consumed_samples": ...,
            # "rng_state": ..., "used_epochs": ...}. We tag them under
            # ``image`` / ``video`` namespaces so they don't collide.
            img_sd = img_batch.get("state_dict") or {}
            vid_sd = vid_batch.get("state_dict") or {}
            merged_state_dict = {
                "iv_alterstep": True,
                "image_state_dict": img_sd,
                "video_state_dict": vid_sd,
                # Mirror whichever was last consumed at the top level so
                # downstream code reading ``state_dict["dataset_state_dict"]``
                # can still find a value.
                "dataset_state_dict": {
                    **(img_sd.get("dataset_state_dict") or {}),
                    **(vid_sd.get("dataset_state_dict") or {}),
                },
            }

            yield {
                "image_batch": img_batch,
                "video_batch": vid_batch,
                "state_dict": merged_state_dict,
            }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if state_dict is None:
            return
        # Accept either the format yielded by __iter__ (image_state_dict/
        # video_state_dict) or the snapshot format from .state_dict().
        img_sd = state_dict.get("image_state_dict") or state_dict.get("image_dataset")
        vid_sd = state_dict.get("video_state_dict") or state_dict.get("video_dataset")
        if img_sd is not None:
            self.image_dataset.load_state_dict(img_sd)
        if vid_sd is not None:
            self.video_dataset.load_state_dict(vid_sd)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "iv_alterstep": True,
            "image_dataset": self.image_dataset.state_dict(),
            "video_dataset": self.video_dataset.state_dict(),
        }


# =====================
# Collator
# =====================
class AudioVideoCollator:
    """
    Collate 函数，支持状态保存
    
    Input: list of dict with video (C, T, H, W) and audio (1, T_a)
    Output: 
        video: (B, C, T, H, W) stacked
        audio: (B, 1, T_a) padded and stacked
    """
    
    def __init__(self, collate_fn: Callable = None):
        self.collate_fn = collate_fn if collate_fn else default_collate
    
    @staticmethod
    def state_dict_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        state_dict = {"dataset_state_dict": {}}
        
        for key in ("consumed_samples", "used_epochs", "rng_state"):
            state_dict[key] = batch[-1]["state_dict"].get(key)
        
        for sample in batch:
            state_dict["dataset_state_dict"].update(sample["state_dict"]["dataset_state_dict"])
        
        return state_dict
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        state_dict = self.state_dict_collate_fn(batch)
        
        # 自定义 collate 处理 video 和 audio
        data_batch = [item["data"] for item in batch]
        
        # 分离不同字段: each video is (C, T, H, W), each audio is (1, T_a)
        videos = [d["video"] for d in data_batch if "video" in d]
        audios = [d["audio"] for d in data_batch if "audio" in d]
        
        result = {}
        
        if videos:
            # Stack videos: list of (C, T, H, W) -> (B, C, T, H, W)
            result["video"] = torch.stack(videos, dim=0)  # (B, C, T, H, W)
        
        if audios:
            # 音频可能长度不同，需要 padding
            max_len = max(a.shape[1] for a in audios)  # max T_a across batch
            result["audio_lengths"] = torch.tensor([a.shape[1] for a in audios], dtype=torch.long)
            padded_audios = []
            for a in audios:  # a: (1, T_a)
                if a.shape[1] < max_len:
                    pad = torch.zeros(1, max_len - a.shape[1])  # (1, pad_size)
                    a = torch.cat([a, pad], dim=1)  # (1, T_a) + (1, pad) -> (1, max_len)
                padded_audios.append(a)  # (1, max_len)
            # Stack audios: list of (1, max_len) -> (B, 1, max_len)
            result["audio"] = torch.stack(padded_audios, dim=0)  # (B, 1, T_a)
        
        # Captions (from jsonl 'prompt' field, if present)
        captions = [d.get("caption", "") for d in data_batch]
        if any(c for c in captions):
            result["captions"] = captions
        video_descriptions = [d.get("video_description", "") for d in data_batch]
        if any(v for v in video_descriptions):
            result["video_descriptions"] = video_descriptions
        audio_descriptions = [d.get("audio_description", "") for d in data_batch]
        if any(a for a in audio_descriptions):
            result["audio_descriptions"] = audio_descriptions

        # 其他元数据
        result["file_names"] = [d.get("file_name", "") for d in data_batch]
        result["source_paths"] = [d.get("source_path", "") for d in data_batch]
        result["valid"] = torch.tensor([d.get("valid", True) for d in data_batch])  # (B,)

        # 长视频 ID（用于 sibling-aware 对比学习负采样）
        # 注意：必须**永远**输出该字段（哪怕全部为 None），否则不同 rank
        # 的 batch 在该 key 上结构不对称（有的 rank 有此 key、有的没有），
        # 会导致 model.forward 中 dist.all_gather 的进入条件因 batch 内容
        # 差异而不一致 → 集合通信死锁 → NCCL 600s watchdog timeout。
        result["long_video_ids"] = [d.get("long_video_id") for d in data_batch]

        # Modality flag — set by ImageFileProcessor. ``True`` 表示这是个图片
        # 单帧 batch (T=1)，trainer 会据此屏蔽 audio/contrastive/video_disc 等
        # 视频专属分支。
        is_image_flags = [bool(d.get("is_image", False)) for d in data_batch]
        if any(is_image_flags):
            # 对一个 batch 而言要么全是图片要么全是视频 (双流由 IVAlterstep 保证)
            result["is_image"] = all(is_image_flags)

        # Distill preprocessing cache (PIL frames / numpy audio, kept as lists)
        for distill_key in ("distill_first_frame", "distill_video_frames", "distill_audio_16k"):
            vals = [d.get(distill_key) for d in data_batch]
            if any(v is not None for v in vals):
                result[distill_key] = vals
        
        return {
            "data": result,
            "state_dict": state_dict,
        }


# =====================
# Builder Functions
# =====================
def _extract_count_from_filename(filename: str) -> int:
    """从 jsonl 文件名尾部提取样本条数，如 nonspeech_multi_shots_184917.jsonl -> 184917"""
    match = re.search(r'_(\d+)\.jsonl$', filename)
    if match:
        return int(match.group(1))
    return 0


def parse_data_mixture_yaml(
    yaml_path: Union[str, Path],
    data_root_override: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, str], Dict[str, float], Optional[str]]:
    """
    解析层级式数据混合配置 yaml，返回扁平化的 (dataset_paths, dataset_weights)。

    yaml 格式:
        data_root: /path/to/sub_data
        nonspeech:
          weight: 0.75
          data:
            main_data:
              weight: 0.8
              data: ["youtube04", "youtube05"]
        speech:
          weight: 0.25
          data: ...

    对于每个顶层类别 (如 nonspeech)，扫描子目录中以该类别名开头的非空 jsonl 文件，
    根据文件名尾部数字（样本条数）按比例分配组内权重。

    Returns:
        dataset_paths:   {qualified_name: jsonl_path}
        dataset_weights: {qualified_name: weight}
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    data_root = Path(data_root_override or cfg.get('data_root', '.'))
    if not data_root.is_absolute():
        data_root = (yaml_path.parent / data_root).resolve()

    train_video_root = cfg.get('file_root')

    dataset_paths: Dict[str, str] = {}
    dataset_weights: Dict[str, float] = {}

    _reserved_keys = {'data_root', 'file_root'}
    for category_name, category_cfg in cfg.items():
        if category_name in _reserved_keys:
            continue
        if not isinstance(category_cfg, dict) or 'weight' not in category_cfg:
            continue

        category_weight = float(category_cfg['weight'])
        category_data = category_cfg.get('data', {})

        for group_name, group_cfg in category_data.items():
            if not isinstance(group_cfg, dict) or 'weight' not in group_cfg:
                continue

            group_weight = category_weight * float(group_cfg['weight'])
            dir_list = group_cfg.get('data', [])

            group_entries: List[Tuple[str, str, int]] = []

            for dir_name in dir_list:
                dir_path = data_root / dir_name
                if not dir_path.is_dir():
                    logging.warning(f"[parse_data_mixture] Directory not found, skipping: {dir_path}")
                    continue

                for jsonl_file in sorted(dir_path.glob(f"{category_name}_*.jsonl")):
                    if jsonl_file.stat().st_size == 0:
                        continue

                    count = _extract_count_from_filename(jsonl_file.name)
                    if count <= 0:
                        logging.warning(
                            f"[parse_data_mixture] Cannot extract count from filename: "
                            f"{jsonl_file.name}, skipping"
                        )
                        continue

                    ds_name = f"{category_name}.{group_name}.{dir_name}.{jsonl_file.stem}"
                    group_entries.append((ds_name, str(jsonl_file), count))

            total_count = sum(e[2] for e in group_entries)
            if total_count > 0:
                for ds_name, jsonl_path, count in group_entries:
                    w = group_weight * (count / total_count)
                    dataset_paths[ds_name] = jsonl_path
                    dataset_weights[ds_name] = w

    total_weight = sum(dataset_weights.values())
    logging.info(
        f"[parse_data_mixture] Parsed {yaml_path}: "
        f"{len(dataset_paths)} datasets, total_weight={total_weight:.6f}"
    )
    for ds_name in sorted(dataset_paths.keys()):
        logging.info(
            f"  {ds_name}: weight={dataset_weights[ds_name]:.6f}, "
            f"path={dataset_paths[ds_name]}"
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"# Parsed from: {yaml_path}\n")
            f.write(f"# Total datasets: {len(dataset_paths)}\n")
            f.write(f"# Total weight: {total_weight:.6f}\n")
            f.write(f"# Format: dataset_name\\tweight\\tpath\n\n")
            for ds_name in sorted(dataset_paths.keys()):
                f.write(
                    f"{ds_name}\t{dataset_weights[ds_name]:.6f}\t"
                    f"{dataset_paths[ds_name]}\n"
                )
        logging.info(f"[parse_data_mixture] Saved triplets to {save_path}")

    return dataset_paths, dataset_weights, train_video_root


def build_audio_video_streaming_dataset(
    dataset_dir_or_paths: Union[str, List[str], Dict[str, str], None] = None,
    num_frames: int = 25,
    resolution: int = 256,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    audio_sample_rate: int = 24000,
    max_audio_duration: Optional[float] = None,
    use_torchcodec: bool = True,
    data_rank: int = 0,
    data_world_size: int = 1,
    raise_processor_error: bool = False,
    raise_stop_iteration: bool = False,
    weights: Optional[Union[List[float], Dict[str, float]]] = None,
    seed: int = 1024,
    name: str = "AudioVideoDataset",
    use_file_processor: bool = False,
    data_mixture_yaml: Optional[Union[str, Path]] = None,
    mixture_save_path: Optional[Union[str, Path]] = None,
    video_root: Optional[str] = None,
    random_start: bool = False,
    spatial_transform_mode: str = "resize_center_crop",
    spatial_roundtrip_short_edge: Optional[int] = None,
    distill_encoder_fps: Optional[float] = None,
    distill_audio_target_sr: Optional[int] = None,
    image_only: bool = False,
) -> AudioVideoStreamingDataset:
    """
    构建音视频流式数据集（训练用）
    
    Args:
        dataset_dir_or_paths: 数据集目录或路径（与 data_mixture_yaml 二选一）
        use_file_processor: True 时使用文件路径处理器 (jsonl 中存 video_path)，
                            False 时使用 bin 处理器 (jsonl 中存 bin_path + offset + length)
        data_mixture_yaml: 层级式数据混合配置 yaml 路径，提供时忽略
                           dataset_dir_or_paths 和 weights
        mixture_save_path: 解析结果保存路径，写出 (name, weight, path) 三元组
        spatial_transform_mode: "resize_center_crop" (保持宽高比) 或 "resize" (直接拉伸)
    
    Output shapes per sample:
        video: (C, T, H, W) = (3, num_frames, resolution, resolution)
        audio: (1, T_a) = (1, audio_sample_rate * duration)
    """
    if data_mixture_yaml is not None:
        dataset_dict, weights, yaml_video_root = parse_data_mixture_yaml(
            data_mixture_yaml, save_path=mixture_save_path,
        )
        if video_root is None:
            video_root = yaml_video_root
    elif dataset_dir_or_paths is not None:
        if isinstance(dataset_dir_or_paths, str):
            dataset_dict = {Path(dataset_dir_or_paths).stem: dataset_dir_or_paths}
        elif isinstance(dataset_dir_or_paths, list):
            dataset_dict = {Path(p).stem: p for p in dataset_dir_or_paths}
        elif isinstance(dataset_dir_or_paths, dict):
            dataset_dict = dataset_dir_or_paths
        else:
            raise TypeError(f"Unsupported type: {type(dataset_dir_or_paths)}")
    else:
        raise ValueError("Either dataset_dir_or_paths or data_mixture_yaml must be provided")
    
    if image_only:
        if not use_file_processor:
            raise ValueError(
                "image_only=True requires use_file_processor=True (image data must "
                "be loaded from file paths, not bin files)."
            )
        ProcessorClass = ImageFileProcessor
    else:
        ProcessorClass = AudioVideoFileProcessor if use_file_processor else AudioVideoProcessor
    
    datasets = []
    processors = []
    
    processor_kwargs = dict(
        num_frames=num_frames,
        resolution=resolution,
        sample_rate=sample_rate,
        target_fps=target_fps,
        audio_sample_rate=audio_sample_rate,
        max_audio_duration=max_audio_duration,
        use_torchcodec=use_torchcodec,
        random_start=random_start,
        spatial_transform_mode=spatial_transform_mode,
        spatial_roundtrip_short_edge=spatial_roundtrip_short_edge,
    )
    if (use_file_processor or image_only) and video_root is not None:
        processor_kwargs["video_root"] = video_root
    if (use_file_processor or image_only) and distill_encoder_fps is not None:
        processor_kwargs["distill_encoder_fps"] = distill_encoder_fps
        if distill_audio_target_sr is not None and not image_only:
            processor_kwargs["distill_audio_target_sr"] = distill_audio_target_sr

    for ds_name, ds_path in dataset_dict.items():
        logging.info(f"Building dataset: {ds_name} from {ds_path}")
        datasets.append(
            StreamingAggregationDataset(ds_name, ds_path, data_rank, data_world_size)
        )
        processors.append(ProcessorClass(**processor_kwargs))
    
    return AudioVideoStreamingDataset(
        datasets=datasets,
        processors=processors,
        weights=weights,
        raise_processor_error=raise_processor_error,
        raise_stop_iteration=raise_stop_iteration,
        data_rank=data_rank,
        seed=seed,
    )


def build_iv_alterstep_streaming_dataset(
    *,
    # ---- Image source dispatch ----
    image_loader: str = "jsonl",
    # ---- Image stream (jsonl loader) ----
    image_dataset_dir_or_paths: Union[str, List[str], Dict[str, str], None] = None,
    image_data_mixture_yaml: Optional[Union[str, Path]] = None,
    image_weights: Optional[Union[List[float], Dict[str, float]]] = None,
    image_video_root: Optional[str] = None,
    image_mixture_save_path: Optional[Union[str, Path]] = None,
    image_batch_size: int = 1,
    # ---- Image stream (relaion loader) ----
    relaion_root: Optional[str] = None,
    relaion_slave_path: Optional[str] = None,
    relaion_base_image_path: Optional[str] = None,
    relaion_split: str = "train",
    relaion_image_size: Optional[int] = None,
    relaion_center_crop: bool = False,
    relaion_random_flip: bool = False,
    relaion_recaption_prob: float = 0.0,
    relaion_cache_dir: Optional[str] = None,
    relaion_max_samples: Optional[int] = None,
    relaion_repeat: int = 1,
    # ---- Video stream config ----
    video_dataset_dir_or_paths: Union[str, List[str], Dict[str, str], None] = None,
    video_data_mixture_yaml: Optional[Union[str, Path]] = None,
    video_weights: Optional[Union[List[float], Dict[str, float]]] = None,
    video_root: Optional[str] = None,
    video_mixture_save_path: Optional[Union[str, Path]] = None,
    video_batch_size: int = 1,
    # ---- Shared config ----
    num_frames: int = 25,
    resolution: int = 256,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    audio_sample_rate: int = 24000,
    max_audio_duration: Optional[float] = None,
    use_torchcodec: bool = True,
    data_rank: int = 0,
    data_world_size: int = 1,
    raise_processor_error: bool = False,
    seed: int = 1024,
    random_start: bool = False,
    spatial_transform_mode: str = "resize_center_crop",
    spatial_roundtrip_short_edge: Optional[int] = None,
    distill_encoder_fps: Optional[float] = None,
    distill_audio_target_sr: Optional[int] = None,
) -> "IVAlterstepStreamingDataset":
    """构建 image+video 双流交替数据集。

    image 源由 ``image_loader`` 决定:
        - ``"jsonl"`` (默认): 走 ``ImageFileProcessor``, 从 jsonl 中拿 image_path
          直接读 jpg/png/webp; 输出 (C, 1, H, W)。
        - ``"relaion"``: 走 ``RelaionStreamingImageDataset``, 从 master/slave
          jsonl + 二进制 package 读图; 输出 (C, 1, H, W) + caption。

    video 源固定走 ``AudioVideoFileProcessor`` (mp4/wav), 输出 (C, T, H, W) + audio。

    最后用 ``IVAlterstepStreamingDataset`` 把两路 stream 包成一层, 每个 step yield
    ``{"image_batch": ..., "video_batch": ...}``。

    image 与 video 各自的 batch 由 IVAlterstepStreamingDataset 内部 collate 完成;
    DataLoader 侧应使用 ``passthrough_collate_fn`` 当 collate_fn (batch_size=1)。
    """
    image_loader = (image_loader or "jsonl").lower()
    if image_loader == "jsonl":
        image_ds = build_audio_video_streaming_dataset(
            dataset_dir_or_paths=image_dataset_dir_or_paths,
            data_mixture_yaml=image_data_mixture_yaml,
            weights=image_weights,
            video_root=image_video_root,
            mixture_save_path=image_mixture_save_path,
            num_frames=1,                 # images are single-frame
            resolution=resolution,
            sample_rate=1,
            target_fps=None,
            audio_sample_rate=audio_sample_rate,
            max_audio_duration=max_audio_duration,
            use_torchcodec=False,         # images don't need video decoder
            data_rank=data_rank,
            data_world_size=data_world_size,
            raise_processor_error=raise_processor_error,
            raise_stop_iteration=False,   # image stream cycles forever
            seed=seed,
            name="ImageDataset",
            use_file_processor=True,
            random_start=False,
            spatial_transform_mode=spatial_transform_mode,
            spatial_roundtrip_short_edge=spatial_roundtrip_short_edge,
            distill_encoder_fps=distill_encoder_fps,
            distill_audio_target_sr=None,
            image_only=True,
        )
    elif image_loader == "relaion":
        if not relaion_root:
            raise ValueError("image_loader='relaion' requires relaion_root")
        if not relaion_base_image_path:
            raise ValueError("image_loader='relaion' requires relaion_base_image_path")

        from omnivae.dataset.relaion_dataset import (
            RelaionDataset,
            RelaionStreamingImageDataset,
        )

        if distill_encoder_fps is not None:
            logging.warning(
                "[iv-alterstep] image_loader='relaion' currently does NOT emit "
                "distill_first_frame; image-step distillation will only run on "
                "the encoder image_feat path."
            )

        relaion_ds = RelaionDataset(
            root=relaion_root,
            slave_path=relaion_slave_path,
            base_image_path=relaion_base_image_path,
            split=relaion_split,
            image_size=int(relaion_image_size or resolution),
            center_crop=bool(relaion_center_crop),
            random_flip=bool(relaion_random_flip),
            recaption_prob=float(relaion_recaption_prob),
            cache_dir=relaion_cache_dir or "./cache/relaion",
            max_samples=relaion_max_samples,
            repeat=int(relaion_repeat),
        )
        image_ds = RelaionStreamingImageDataset(
            relaion_ds=relaion_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            seed=seed,
            name="relaion",
        )
    else:
        raise ValueError(
            f"unknown image_loader={image_loader!r}; expected 'jsonl' or 'relaion'"
        )

    video_ds = build_audio_video_streaming_dataset(
        dataset_dir_or_paths=video_dataset_dir_or_paths,
        data_mixture_yaml=video_data_mixture_yaml,
        weights=video_weights,
        video_root=video_root,
        mixture_save_path=video_mixture_save_path,
        num_frames=num_frames,
        resolution=resolution,
        sample_rate=sample_rate,
        target_fps=target_fps,
        audio_sample_rate=audio_sample_rate,
        max_audio_duration=max_audio_duration,
        use_torchcodec=use_torchcodec,
        data_rank=data_rank,
        data_world_size=data_world_size,
        raise_processor_error=raise_processor_error,
        raise_stop_iteration=False,
        seed=seed + 1,                # decorrelate from image stream
        name="VideoDataset",
        use_file_processor=True,
        random_start=random_start,
        spatial_transform_mode=spatial_transform_mode,
        spatial_roundtrip_short_edge=spatial_roundtrip_short_edge,
        distill_encoder_fps=distill_encoder_fps,
        distill_audio_target_sr=distill_audio_target_sr,
        image_only=False,
    )

    return IVAlterstepStreamingDataset(
        image_dataset=image_ds,
        video_dataset=video_ds,
        image_batch_size=image_batch_size,
        video_batch_size=video_batch_size,
        image_collator=AudioVideoCollator(),
        video_collator=AudioVideoCollator(),
    )


def passthrough_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """与 ``IVAlterstepStreamingDataset`` 配套的 collate_fn (DataLoader 侧)。

    IVAlterstepStreamingDataset 已经在内部完成了双流的 collate，所以 DataLoader
    必须用 ``batch_size=1`` 并用此 passthrough 函数把 ``[batch_dict]`` 解包成
    ``batch_dict``。
    """
    if len(batch) != 1:
        raise ValueError(
            f"passthrough_collate_fn requires DataLoader batch_size=1, "
            f"got {len(batch)} (image+video batch sizes are configured inside "
            f"IVAlterstepStreamingDataset)."
        )
    return batch[0]


def scan_jsonl_files(directory: Union[str, Path]) -> Dict[str, str]:
    """扫描目录下的所有 jsonl 文件，返回 {数据集名: jsonl路径} 字典"""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    
    jsonl_files = sorted(directory.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No jsonl files found in {directory}")
    
    return {f.stem: str(f) for f in jsonl_files}


def build_video_only_dataset(
    dataset_dir_or_paths: Union[str, List[str], Dict[str, str]],
    num_frames: int = 25,
    resolution: int = 256,
    sample_rate: int = 1,
    target_fps: Optional[float] = None,
    use_torchcodec: bool = True,
    data_rank: int = 0,
    data_world_size: int = 1,
    seed: int = 1024,
    use_file_processor: bool = False,
    video_root: Optional[str] = None,
    spatial_transform_mode: str = "resize_center_crop",
    spatial_roundtrip_short_edge: Optional[int] = None,
) -> AudioVideoStreamingDataset:
    """
    构建仅视频的数据集（用于验证）
    
    Output: video (C, T, H, W) per sample
    """
    if isinstance(dataset_dir_or_paths, str):
        dataset_dict = {Path(dataset_dir_or_paths).stem: dataset_dir_or_paths}
    elif isinstance(dataset_dir_or_paths, list):
        dataset_dict = {Path(p).stem: p for p in dataset_dir_or_paths}
    else:
        dataset_dict = dataset_dir_or_paths
    
    datasets = []
    processors = []
    
    ProcessorClass = VideoFileProcessor if use_file_processor else VideoOnlyProcessor
    
    processor_kwargs = dict(
        num_frames=num_frames,
        resolution=resolution,
        sample_rate=sample_rate,
        target_fps=target_fps,
        use_torchcodec=use_torchcodec,
        spatial_transform_mode=spatial_transform_mode,
        spatial_roundtrip_short_edge=spatial_roundtrip_short_edge,
    )
    if use_file_processor and video_root is not None:
        processor_kwargs["video_root"] = video_root

    for ds_name, ds_path in dataset_dict.items():
        datasets.append(
            StreamingAggregationDataset(ds_name, ds_path, data_rank, data_world_size)
        )
        processors.append(ProcessorClass(**processor_kwargs))
    
    return AudioVideoStreamingDataset(
        datasets=datasets,
        processors=processors,
        raise_stop_iteration=True,
        data_rank=data_rank,
        seed=seed,
    )


def build_audio_only_dataset(
    dataset_dir_or_paths: Union[str, List[str], Dict[str, str]],
    sample_rate: int = 24000,
    max_duration: Optional[float] = None,
    data_rank: int = 0,
    data_world_size: int = 1,
    seed: int = 1024,
    use_file_processor: bool = False,
) -> AudioVideoStreamingDataset:
    """
    构建仅音频的数据集（用于验证）
    
    Output: audio (1, T_a) per sample
    """
    if isinstance(dataset_dir_or_paths, str):
        dataset_dict = {Path(dataset_dir_or_paths).stem: dataset_dir_or_paths}
    elif isinstance(dataset_dir_or_paths, list):
        dataset_dict = {Path(p).stem: p for p in dataset_dir_or_paths}
    else:
        dataset_dict = dataset_dir_or_paths
    
    datasets = []
    processors = []
    
    ProcessorClass = AudioFileProcessor if use_file_processor else AudioOnlyProcessor
    
    for ds_name, ds_path in dataset_dict.items():
        datasets.append(
            StreamingAggregationDataset(ds_name, ds_path, data_rank, data_world_size)
        )
        processors.append(
            ProcessorClass(
                sample_rate=sample_rate,
                max_duration=max_duration,
            )
        )
    
    return AudioVideoStreamingDataset(
        datasets=datasets,
        processors=processors,
        raise_stop_iteration=True,
        data_rank=data_rank,
        seed=seed,
    )


if __name__ == "__main__":
    # 简单测试
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    
    dataset = build_audio_video_streaming_dataset(
        dataset_dir_or_paths=args.jsonl_path,
        num_frames=args.num_frames,
        resolution=args.resolution,
    )
    
    collator = AudioVideoCollator()
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    
    for i, batch in enumerate(loader):
        data = batch["data"]
        print(f"Batch {i}:")
        if "video" in data:
            print(f"  video shape: {data['video'].shape}")  # (B, C, T, H, W)
        if "audio" in data:
            print(f"  audio shape: {data['audio'].shape}")  # (B, 1, T_a)
        if i >= 2:
            break
