"""Audio amplitude (RMS) and loudness (LUFS).

Reads the .wav from ``record["audio_path"]`` directly with soundfile + librosa
(via ``utils/audio_video.load_wav_mono``); does NOT touch MOVA's
``eval_audio_amplitude.py`` because that module hard-imports ``torchcodec`` at
module load time, which is not installed in the verse-bench env. The RMS /
LUFS math is identical -- see the original MOVA functions for reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from my_eval.utils.audio_video import load_wav_mono
from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample


_SR = 48000


def _compute_amplitude(waveform: np.ndarray) -> float:
    """RMS amplitude. Same formula as MOVA's compute_audio_amplitude."""
    return float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2)))


def _compute_loudness(waveform: np.ndarray, sr: int) -> float:
    """LUFS loudness (BS.1770). Prefers audiotools (if installed), falls back
    to pyloudnorm. Returns NaN if both are missing."""
    # Try audiotools first (preferred by MOVA).
    try:
        from audiotools import AudioSignal  # type: ignore
        if waveform.ndim == 1:
            waveform_2d = waveform[np.newaxis, :]
        else:
            waveform_2d = waveform if waveform.shape[0] < waveform.shape[1] else waveform.T
        return float(AudioSignal(waveform_2d, sr).loudness())
    except Exception:
        pass
    try:
        import pyloudnorm as pyln  # type: ignore
        flat = waveform.flatten() if waveform.ndim > 1 else waveform
        return float(pyln.Meter(sr).integrated_loudness(flat))
    except Exception as exc:
        print(f"[audio_amplitude] no LUFS backend available: {exc}", flush=True)
        return float("nan")


def run_task(
    rank: int,
    local_rank: int,
    world_size: int,
    target_dir: Path,
    manifest: Dict[str, Any],
    skip_completed: bool = True,
    metric_keys: list[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    metric_keys = metric_keys or ["amplitude_rms", "loudness_lufs"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[audio_amplitude] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": 0.0}

    for idx, rec in enumerate(my_records):
        stem = rec["file_stem"]
        if skip_completed and already_done(target_dir, "audio_amplitude", stem, metric_keys):
            continue
        payload: Dict[str, Any] = {
            "amplitude_rms": float("nan"),
            "loudness_lufs": float("nan"),
            "audio_path": rec["audio_path"],
        }
        try:
            audio, sr = load_wav_mono(rec["audio_path"], _SR)
            payload["amplitude_rms"] = _compute_amplitude(audio)
            payload["loudness_lufs"] = _compute_loudness(audio, sr)
        except Exception as exc:
            log(rank, f"[audio_amplitude] failed for {stem}: {exc}")
        write_per_sample(target_dir, "audio_amplitude", stem, payload)
        if (idx + 1) % 100 == 0:
            log(rank, f"  audio_amplitude {idx + 1}/{len(my_records)}")
    return {"model_load_elapsed_sec": 0.0}
