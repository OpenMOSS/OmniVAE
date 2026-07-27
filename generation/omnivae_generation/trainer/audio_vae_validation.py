"""Audio VAE encode -> decode reconstruction sanity check.

Counterpart to :mod:`omnivae_generation.trainer.vae_validation` (visual). Every
``train.vae_validation_steps`` steps (when ``dataset.type=audio_jsonl``):

1. Take the first ``train.vae_validation_num_samples`` (default 4) waveforms
   from the current training batch.
2. Run them through the *frozen* audio VAE: ``preprocess`` (pad to hop-length
   boundary if available) -> ``encode`` -> ``posterior.mode()`` (or ``sample``,
   controlled by ``train.vae_validation_sample_mode``) -> ``decode`` -> trim.
3. Log to trackers (wandb / tensorboard):

   * Scalars under ``validation/audio_vae/{l1, snr_db, mel_l1, latent_mean,
     latent_std}``.
   * Audio previews:
     - wandb: a list of ``wandb.Audio`` for GT and Recon under
       ``validation/audio_vae/preview/{gt,recon}``.
     - tensorboard: per-sample ``add_audio`` under
       ``validation/audio_vae/preview/{i}_{gt,recon}``.

4. Mirror the reconstructions to disk under
   ``<output_dir>/audio_vae_samples/step-XXXXXXXX/{i}_gt.wav`` and
   ``{i}_recon.wav`` plus a ``metrics.json`` manifest.

Only the main process performs the encode / decode / log work; other ranks
no-op so we don't run the encoder N times.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator

from omnivae_generation.trainer.utils import ensure_dir, save_json


logger = logging.getLogger(__name__)


def _positive_int(value: Any, *, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return int(default)
    return v if v > 0 else int(default)


def _resolve_sample_rate(config: dict, vae_model) -> int:
    sr = (
        config.get("audio_vae", {}).get("sample_rate")
        or config.get("dataset", {}).get("sample_rate")
        or getattr(vae_model, "sample_rate", None)
        or 48000
    )
    return int(sr)


@torch.no_grad()
def _compute_l1(recon: torch.Tensor, gt: torch.Tensor) -> float:
    return float(F.l1_loss(recon, gt).item())


@torch.no_grad()
def _compute_snr_db(recon: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> float:
    """Per-batch SNR in dB. Mirrors infer_audio_video_vae.run_audio_reconstruction.

    Silent inputs (signal_power < eps) return 0 so the metric stays bounded;
    silent samples are also skipped in the SNR aggregator above the call site
    when present in batches with mixed loud/silent items.
    """
    signal_power = float((gt ** 2).mean().item())
    noise_power = float(((gt - recon) ** 2).mean().item())
    if signal_power < eps:
        return 0.0
    return 10.0 * math.log10(signal_power / (noise_power + eps))


@torch.no_grad()
def _compute_mel_l1(
    recon: torch.Tensor,
    gt: torch.Tensor,
    *,
    sample_rate: int,
    device: torch.device,
) -> float:
    """log-Mel L1 distance, averaged across batch / time / mel.

    Uses 80-band Mel with 1024-pt FFT and 256 hop. Independent of the audio
    VAE's own STFT params (it's just a sanity metric, not a loss).
    """
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(sample_rate),
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    ).to(device=device, dtype=torch.float32)
    eps = 1e-5
    g = torch.log(mel(gt.float().squeeze(1)) + eps)
    r = torch.log(mel(recon.float().squeeze(1)) + eps)
    return float(F.l1_loss(r, g).item())


@torch.no_grad()
def run_audio_vae_validation(
    *,
    accelerator: Accelerator,
    config: dict,
    step: int,
    batch: dict,
    vae,
    vae_dtype: torch.dtype,
) -> None:
    """Run audio VAE reconstruction validation. Main process only."""
    interval = int(config.get("train", {}).get("vae_validation_steps") or 0)
    if interval <= 0 or not accelerator.is_main_process:
        return

    audio = batch.get("audio")
    if not torch.is_tensor(audio):
        logger.warning("audio_vae validation: batch has no 'audio' tensor; skipping.")
        return

    n_preview = _positive_int(config["train"].get("vae_validation_num_samples"), default=4)
    audio = audio[:n_preview]
    if audio.ndim == 2:
        audio = audio.unsqueeze(1)  # [B, T] -> [B, 1, T]
    audio = audio.to(device=accelerator.device, dtype=vae_dtype, non_blocking=True)
    original_len = int(audio.shape[-1])

    sample_mode = str(config["train"].get("vae_validation_sample_mode", "argmax")).strip().lower()
    if sample_mode not in {"sample", "argmax"}:
        raise ValueError(
            f"Unsupported train.vae_validation_sample_mode={sample_mode!r}; "
            "expected 'sample' or 'argmax'."
        )

    audio_paths = list(batch.get("audio_paths") or [])

    vae_model = accelerator.unwrap_model(vae, keep_torch_compile=False)
    vae_was_training = bool(vae_model.training)
    vae_model.eval()

    try:
        sample_rate = _resolve_sample_rate(config, vae_model)
        if hasattr(vae_model, "preprocess"):
            try:
                audio_pp = vae_model.preprocess(audio, sample_rate)
            except TypeError:
                # Some VAE wrappers expose preprocess(audio) without sample_rate.
                audio_pp = vae_model.preprocess(audio)
        else:
            audio_pp = audio

        encoded = vae_model.encode(audio_pp)
        if isinstance(encoded, (tuple, list)):
            posterior = encoded[0]
        else:
            posterior = encoded
        if sample_mode == "argmax" and hasattr(posterior, "mode"):
            latents = posterior.mode()
        elif hasattr(posterior, "sample"):
            latents = posterior.sample()
        elif torch.is_tensor(posterior):
            latents = posterior
        else:
            raise TypeError(
                f"Audio VAE encode returned unsupported object: {type(posterior)!r}"
            )

        recon = vae_model.decode(latents.to(dtype=vae_dtype))
        if isinstance(recon, (tuple, list)):
            recon = recon[0]
        recon = recon[..., :original_len]

        gt_f = audio.float()
        recon_f = recon.float()

        l1 = _compute_l1(recon_f, gt_f)
        snr_db = _compute_snr_db(recon_f, gt_f)
        try:
            mel_l1 = _compute_mel_l1(
                recon_f, gt_f, sample_rate=sample_rate, device=accelerator.device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio_vae validation: mel L1 failed (%s); skipping.", exc)
            mel_l1 = float("nan")

        latents_float = latents.detach().float()
        latent_mean = float(latents_float.mean().item())
        latent_std = float(latents_float.std().item())

        metrics: dict[str, float] = {
            "validation/audio_vae/l1": l1,
            "validation/audio_vae/snr_db": snr_db,
            "validation/audio_vae/mel_l1": mel_l1,
            "validation/audio_vae/latent_mean": latent_mean,
            "validation/audio_vae/latent_std": latent_std,
        }

        # ----- Local .wav backups + manifest ----- #
        sample_dir = ensure_dir(
            Path(config["experiment"]["output_dir"]) / "audio_vae_samples" / f"step-{int(step):08d}"
        )
        gt_cpu = gt_f.cpu()
        recon_cpu = recon_f.cpu()
        n = int(audio.shape[0])
        for i in range(n):
            torchaudio.save(
                str(sample_dir / f"{i:02d}_gt.wav"),
                gt_cpu[i],
                sample_rate=int(sample_rate),
            )
            torchaudio.save(
                str(sample_dir / f"{i:02d}_recon.wav"),
                recon_cpu[i],
                sample_rate=int(sample_rate),
            )

        save_json(
            sample_dir / "metrics.json",
            {
                "step": int(step),
                "sample_mode": sample_mode,
                "sample_rate": int(sample_rate),
                "num_samples": n,
                "audio_shape": list(audio.shape),
                "latent_shape": list(latents.shape),
                "audio_paths": audio_paths[:n],
                "metrics": metrics,
            },
        )

        # ----- Tracker logging ----- #
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                import wandb

                payload: dict[str, Any] = dict(metrics)
                payload["validation/audio_vae/preview/gt"] = [
                    wandb.Audio(
                        gt_cpu[i].squeeze(0).numpy(),
                        sample_rate=int(sample_rate),
                        caption=(
                            f"#{i} | {audio_paths[i]}"
                            if i < len(audio_paths)
                            else f"#{i}"
                        ),
                    )
                    for i in range(n)
                ]
                payload["validation/audio_vae/preview/recon"] = [
                    wandb.Audio(
                        recon_cpu[i].squeeze(0).numpy(),
                        sample_rate=int(sample_rate),
                        caption=(
                            f"recon #{i} | {audio_paths[i]}"
                            if i < len(audio_paths)
                            else f"recon #{i}"
                        ),
                    )
                    for i in range(n)
                ]
                tracker.log(payload, step=int(step))
            elif tracker.name == "tensorboard":
                for k, v in metrics.items():
                    fv = float(v)
                    if math.isnan(fv):
                        continue
                    tracker.writer.add_scalar(k, fv, global_step=int(step))
                for i in range(n):
                    tracker.writer.add_audio(
                        f"validation/audio_vae/preview/{i:02d}_gt",
                        gt_cpu[i],
                        global_step=int(step),
                        sample_rate=int(sample_rate),
                    )
                    tracker.writer.add_audio(
                        f"validation/audio_vae/preview/{i:02d}_recon",
                        recon_cpu[i],
                        global_step=int(step),
                        sample_rate=int(sample_rate),
                    )

        logger.info(
            "audio_vae validation @ step %d: L1=%.4f, SNR=%.2f dB, mel_L1=%.4f, "
            "latents=%s mean=%.3f std=%.3f",
            int(step), l1, snr_db, mel_l1, tuple(latents.shape), latent_mean, latent_std,
        )
    finally:
        if vae_was_training:
            vae_model.train()
        else:
            vae_model.eval()


__all__ = ["run_audio_vae_validation"]
