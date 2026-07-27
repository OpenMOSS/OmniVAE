"""DAC-style full-reference metrics for audio reconstruction evaluation.

The implementation follows the public Descript Audio Codec evaluation recipe:

* multi-scale log-mel L1 distance at 44.1 kHz by default;
* multi-scale STFT log-magnitude + magnitude distance at 44.1 kHz by default;
* ViSQOL Audio mode, preserving DAC's published argument order by default.

References:
  https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/evaluate.py
  https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


MEL_WINDOWS = (32, 64, 128, 256, 512, 1024, 2048)
MEL_BINS = (5, 10, 20, 40, 80, 160, 320)
STFT_WINDOWS = (2048, 512)
DEFAULT_SAMPLE_RATES = (44100,)
EPS = 1.0e-5

_VISQOL_APIS: Dict[str, Tuple[object, int]] = {}


def _make_signal(audio: np.ndarray, sample_rate: int, target_rate: int):
    try:
        from audiotools import AudioSignal
    except ImportError as exc:
        raise RuntimeError(
            "DAC Mel/STFT metrics require descript-audiotools. "
            "Install the OmniVAE metrics extra or run "
            "`python -m pip install descript-audiotools`."
        ) from exc

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # Accept either channels-first or samples-first arrays and follow the
        # evaluation protocol's mono convention.
        channel_axis = 0 if audio.shape[0] <= audio.shape[1] else 1
        audio = audio.mean(axis=channel_axis)
    elif audio.ndim != 1:
        raise ValueError(f"expected mono/stereo audio, got shape={audio.shape}")
    signal = AudioSignal(
        torch.from_numpy(audio).reshape(1, 1, -1), int(sample_rate)
    )
    if int(sample_rate) != int(target_rate):
        signal = signal.resample(int(target_rate))
    return signal


def _align(reference, estimate):
    length = min(reference.signal_length, estimate.signal_length)
    if length <= 0:
        raise ValueError("aligned audio has zero samples")
    return reference.truncate_samples(length), estimate.truncate_samples(length)


def _mel_distance(reference, estimate) -> float:
    loss = 0.0
    for window, n_mels in zip(MEL_WINDOWS, MEL_BINS):
        kwargs = {
            "window_length": window,
            "hop_length": window // 4,
            "window_type": None,
        }
        reference_mel = reference.mel_spectrogram(
            n_mels, mel_fmin=0.0, mel_fmax=None, **kwargs
        )
        estimate_mel = estimate.mel_spectrogram(
            n_mels, mel_fmin=0.0, mel_fmax=None, **kwargs
        )
        # DAC defaults: pow=1, clamp_eps=1e-5, mag_weight=0, log_weight=1.
        loss += float(
            F.l1_loss(
                estimate_mel.clamp(EPS).log10(),
                reference_mel.clamp(EPS).log10(),
            )
        )
    return loss


def _stft_distance(reference, estimate) -> float:
    loss = 0.0
    # audiotools warns whenever a second STFT resolution replaces its cached
    # tensor. This is expected for DAC's multi-resolution metric.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="stft_data changed shape")
        for window in STFT_WINDOWS:
            reference.stft(window, window // 4, None)
            estimate.stft(window, window // 4, None)
            reference_magnitude = reference.magnitude
            estimate_magnitude = estimate.magnitude
            # DAC defaults: pow=2, mag_weight=1, log_weight=1.
            loss += float(
                F.l1_loss(
                    estimate_magnitude.clamp(EPS).pow(2.0).log10(),
                    reference_magnitude.clamp(EPS).pow(2.0).log10(),
                )
            )
            loss += float(F.l1_loss(estimate_magnitude, reference_magnitude))
    return loss


def _get_visqol_api(mode: str):
    if mode in _VISQOL_APIS:
        return _VISQOL_APIS[mode]
    try:
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
    except ImportError as exc:
        raise RuntimeError(
            "ViSQOL requires Google's compiled Python binding. See "
            "docs/installation.md for installation instructions, or pass "
            "--no-compute_visqol."
        ) from exc

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        model_name = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        model_name = (
            "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_"
            "learn.005_ep2400_train1_7_raw.tflite"
        )
    else:
        raise ValueError(f"unsupported ViSQOL mode: {mode}")

    config.options.svr_model_path = str(
        Path(visqol_lib_py.__file__).resolve().parent / "model" / model_name
    )
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    _VISQOL_APIS[mode] = (api, int(config.audio.sample_rate))
    return _VISQOL_APIS[mode]


def _visqol(reference, estimate, mode: str, argument_order: str) -> float:
    api, target_rate = _get_visqol_api(mode)
    reference = reference.clone().to_mono().resample(target_rate)
    estimate = estimate.clone().to_mono().resample(target_rate)
    if argument_order == "dac":
        # DAC evaluate.py passes ground truth as audiotools' `estimates`
        # argument and reconstruction as `references`. Preserve that public
        # recipe for exact comparison with DAC numbers.
        api_reference, api_degraded = estimate, reference
    elif argument_order == "standard":
        api_reference, api_degraded = reference, estimate
    else:
        raise ValueError(f"unsupported ViSQOL argument order: {argument_order}")
    result = api.Measure(
        api_reference.audio_data[0, 0].detach().cpu().numpy().astype(float),
        api_degraded.audio_data[0, 0].detach().cpu().numpy().astype(float),
    )
    return float(result.moslqo)


def evaluate_dac_metrics(
    reference_audio: np.ndarray,
    estimate_audio: np.ndarray,
    sample_rate: int,
    *,
    sample_rates: Iterable[int] = DEFAULT_SAMPLE_RATES,
    compute_visqol: bool = True,
    visqol_mode: str = "audio",
    visqol_argument_order: str = "dac",
    metric_errors: Dict[str, str] | None = None,
) -> Dict[str, float]:
    """Evaluate one aligned mono pair and return named DAC-style metrics."""
    output: Dict[str, float] = {}
    # AudioSignal caches STFT tensors and warns when the resolution changes.
    # DAC intentionally uses many resolutions, so the shape changes are
    # expected and should not flood long evaluation logs.
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="stft_data changed shape")
        for metric_sample_rate in sample_rates:
            suffix = str(metric_sample_rate)
            try:
                reference = _make_signal(reference_audio, sample_rate, metric_sample_rate)
                estimate = _make_signal(estimate_audio, sample_rate, metric_sample_rate)
                reference, estimate = _align(reference, estimate)
            except Exception as exc:
                if metric_errors is None:
                    raise
                metric_errors[f"dac_setup_{suffix}"] = str(exc)
                continue
            for name, metric_fn in (
                (f"dac_mel_distance_{suffix}", _mel_distance),
                (f"dac_stft_distance_{suffix}", _stft_distance),
            ):
                try:
                    output[name] = metric_fn(reference, estimate)
                except Exception as exc:
                    if metric_errors is None:
                        raise
                    metric_errors[name] = str(exc)
            if compute_visqol:
                name = f"dac_visqol_{visqol_mode}_{suffix}"
                try:
                    output[name] = _visqol(
                        reference,
                        estimate,
                        visqol_mode,
                        visqol_argument_order,
                    )
                except Exception as exc:
                    if metric_errors is None:
                        raise
                    metric_errors[name] = str(exc)
    return output
