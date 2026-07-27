"""Audio/video data prep shared by multiple tasks.

The joint_av layout ships ``sample-versebench-NNNN-setX.{mp4, wav}`` as separate
files. Some downstream tools (SyncNet pipeline, AV-Align ffmpeg pipeline,
Synchformer official wrapper) expect an mp4 with an embedded audio track. Those
tools call this module to materialise a temporary muxed mp4 next to a rank-owned
tmp directory; subsequent invocations on the same (video, audio, output_path)
short-circuit.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PREPROCESS_CACHE = REPO_ROOT / "eval" / "t2av" / "preprocess_cache"
_AUDIO_MEM: "OrderedDict[str, tuple[np.ndarray, int]]" = OrderedDict()
_VIDEO_MEM: "OrderedDict[str, np.ndarray]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def ffmpeg_bin() -> str:
    bin_ = _which("ffmpeg")
    if bin_ is None:
        raise RuntimeError("ffmpeg not found on PATH")
    return bin_


def mux_av(video_path: str, audio_path: str, output_path: str, *, audio_codec: str = "aac") -> str:
    """Combine video+audio into one mp4. Returns output_path.

    Skips the work when output already exists and is newer than both inputs.
    """
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        out_mtime = os.path.getmtime(output_path)
        if out_mtime >= os.path.getmtime(video_path) and out_mtime >= os.path.getmtime(audio_path):
            return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


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


def _cache_root() -> Path:
    raw = os.environ.get("MY_EVAL_PREPROCESS_CACHE")
    return Path(raw).expanduser() if raw else DEFAULT_PREPROCESS_CACHE


def _preprocess_cache_enabled() -> bool:
    return not _env_flag("MY_EVAL_DISABLE_PREPROCESS_CACHE", False)


def _file_cache_key(path: str, options: str) -> str:
    p = Path(path).expanduser().resolve()
    st = p.stat()
    payload = f"{p}|{st.st_size}|{st.st_mtime_ns}|{options}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _lru_get(cache: "OrderedDict[str, object]", key: str):
    with _CACHE_LOCK:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value


def _lru_put(cache: "OrderedDict[str, object]", key: str, value: object, max_items: int) -> None:
    if max_items <= 0:
        return
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)


def _read_wav_mono_uncached(audio_path: str, target_sr: int) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32, copy=False), sr


def load_wav_mono(audio_path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Read a wav file as float32 mono at target_sr, with shared memory/disk cache.

    Audio cache is on by default because resampled waveforms are small compared
    with RGB video frames. Disable with ``MY_EVAL_AUDIO_DISK_CACHE=0`` or all
    preprocessing cache with ``MY_EVAL_DISABLE_PREPROCESS_CACHE=1``.
    """
    options = f"audio_mono_sr={int(target_sr)}"
    if not _preprocess_cache_enabled():
        return _read_wav_mono_uncached(audio_path, target_sr)
    key = _file_cache_key(audio_path, options)
    mem = _lru_get(_AUDIO_MEM, key)
    if mem is not None:
        audio, sr = mem
        return np.asarray(audio, dtype=np.float32), int(sr)

    cache_path = _cache_root() / "audio" / str(int(target_sr)) / f"{key}.npy"
    if _env_flag("MY_EVAL_AUDIO_DISK_CACHE", True) and cache_path.is_file():
        audio = np.load(cache_path, allow_pickle=False).astype(np.float32, copy=False)
        _lru_put(_AUDIO_MEM, key, (audio, int(target_sr)), _env_int("MY_EVAL_AUDIO_MEMORY_CACHE_SIZE", 128))
        return audio, int(target_sr)

    audio, sr = _read_wav_mono_uncached(audio_path, target_sr)
    if _env_flag("MY_EVAL_AUDIO_DISK_CACHE", True):
        try:
            _atomic_save_npy(cache_path, audio)
        except Exception:
            pass
    _lru_put(_AUDIO_MEM, key, (audio, sr), _env_int("MY_EVAL_AUDIO_MEMORY_CACHE_SIZE", 128))
    return audio, sr


def _decode_video_rgb_uncached(video_path: str) -> np.ndarray:
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(video_path)
    try:
        frames = [np.asarray(f, dtype=np.uint8) for f in clip.iter_frames()]
    finally:
        clip.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def load_video_rgb_array(video_path: str) -> np.ndarray:
    """Decode all RGB frames to ``uint8`` (T,H,W,3), with small memory cache.

    Video disk caching is off by default because full-frame caches can be large.
    Enable it explicitly with ``MY_EVAL_VIDEO_DISK_CACHE=1`` for small profiling
    runs or when the cache filesystem has enough space.
    """
    options = "video_rgb_all_moviepy_v1"
    if not _preprocess_cache_enabled():
        return _decode_video_rgb_uncached(video_path)
    key = _file_cache_key(video_path, options)
    mem = _lru_get(_VIDEO_MEM, key)
    if mem is not None:
        return np.asarray(mem, dtype=np.uint8)

    cache_path = _cache_root() / "video_rgb" / f"{key}.npy"
    if _env_flag("MY_EVAL_VIDEO_DISK_CACHE", False) and cache_path.is_file():
        frames = np.load(cache_path, allow_pickle=False).astype(np.uint8, copy=False)
        _lru_put(_VIDEO_MEM, key, frames, _env_int("MY_EVAL_VIDEO_MEMORY_CACHE_SIZE", 2))
        return frames

    frames = _decode_video_rgb_uncached(video_path)
    if _env_flag("MY_EVAL_VIDEO_DISK_CACHE", False):
        try:
            _atomic_save_npy(cache_path, frames)
        except Exception:
            pass
    _lru_put(_VIDEO_MEM, key, frames, _env_int("MY_EVAL_VIDEO_MEMORY_CACHE_SIZE", 2))
    return frames


def load_video_rgb_pil(video_path: str, *, convert_rgb: bool = True):
    from PIL import Image
    frames = load_video_rgb_array(video_path)
    out = [Image.fromarray(frame) for frame in frames]
    if convert_rgb:
        out = [frame.convert("RGB") for frame in out]
    return out


def rank_tmp_dir(target_dir: Path, kind: str, rank: int) -> Path:
    d = target_dir / "tmp" / kind / f"rank{rank}"
    d.mkdir(parents=True, exist_ok=True)
    return d
