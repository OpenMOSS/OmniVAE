"""Dual sigma-shift + flow-matching batch prep for joint AV training.

Each modality samples its own ``t`` independently and uses its own
``shift`` to bend the timestep onto a per-modality sigma schedule:

  sigma(t, shift) = shift * t / (1 + (shift - 1) * t)

This is the standard Flow-Matching shifted schedule. We bypass the
discrete ``noise_scheduler.sigmas`` table because mixing two separate
shifts inside a single trainer is otherwise awkward and leaves us
permanently coupled to the scheduler's ``set_timesteps`` cache.
``compute_density_for_timestep_sampling`` still shapes the *t*
distribution so existing yaml knobs (``logit_mean``, ``logit_std``,
``mode_scale``, ``weighting_scheme``) keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


@dataclass
class DualBranchBatch:
    """Per-branch flow-matching tensors. Mirrors ``DiffusionBatch``
    fields but is dimension-agnostic so it works for both video
    ``[B, C, T, H, W]`` and audio ``[B, C, T]`` latents."""
    sigmas: torch.Tensor               # [B, 1, ...] broadcast-shaped
    noisy_latents: torch.Tensor        # same shape as latents
    model_timesteps: torch.Tensor      # [B] float
    target: torch.Tensor               # v-prediction target = (latents - noise)
    weighting: torch.Tensor            # [B, 1, ...] broadcast-shaped


@dataclass
class DualDiffusionBatch:
    video: Optional[DualBranchBatch]
    audio: Optional[DualBranchBatch]
    batch_size: int


def apply_sigma_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Standard FM-shifted schedule.

    ``t`` is in ``[0, 1]``. Larger ``shift`` (e.g. ``5.0`` for video)
    pushes density toward larger ``sigma`` (high-noise / coarse), which
    is what the t2v setup wants for long-horizon generation. ``shift =
    1.0`` is the identity (matches t2a's plain logit-normal).

    Returns a tensor with the same shape as ``t``.
    """
    if shift <= 0:
        raise ValueError(f"sigma shift must be > 0, got {shift}.")
    t = t.float()
    shift_t = float(shift) * t / (1.0 + (float(shift) - 1.0) * t)
    return shift_t.clamp(0.0, 1.0)


def _make_broadcast_view(values: torch.Tensor, n_dim: int) -> torch.Tensor:
    """Reshape ``[B]`` -> ``[B, 1, 1, ...]`` so it broadcasts against an
    ``n_dim``-dimensional latent."""
    if n_dim < 1:
        return values
    shape = (-1,) + (1,) * (n_dim - 1)
    return values.view(*shape)


def _prepare_branch_batch(
    *,
    config_train: dict,
    latents: torch.Tensor,
    shift: float,
    detach_target_latents: bool,
    weighting_scheme_override: str | None = None,
) -> DualBranchBatch:
    """Sample ``t``, compute shifted ``sigma``, build noisy latents and v-target.

    Polarity / convention notes (matched against
    ``omnivae_generation.trainer.zimage_training.prepare_diffusion_batch`` so loss curves
    stay comparable when ``shift == scheduler.config.shift``):

    * ``compute_density_for_timestep_sampling`` returns ``u`` in
      ``[0, 1]`` with the convention ``u → 0`` means "noisy end of the
      schedule" and ``u → 1`` means "clean end" (matches the existing
      trainer's indexing of the high→low ``scheduler.sigmas`` table).
    * ``sigma_base = 1 - u`` therefore lives in the same orientation as
      ``scheduler.sigmas`` (high when noisy, low when clean).
    * ``apply_sigma_shift`` then bends ``sigma_base`` into the shifted
      schedule. With ``shift = 1.0`` it is the identity, recovering the
      single-modality trainer's behaviour byte-equivalently.
    * ``model_timesteps = 1 - sigma`` matches
      ``(num_train_timesteps - timesteps) / num_train_timesteps`` from
      the existing trainer (where ``timesteps = sigma * N``), so the
      ``t_embedder`` sees the same numeric range it always has.

    The discrete ``noise_scheduler.sigmas`` table is intentionally not
    consulted here so per-branch shifts can diverge cleanly.
    """
    batch_size = int(latents.shape[0])
    device = latents.device
    dtype = latents.dtype

    u = compute_density_for_timestep_sampling(
        weighting_scheme=config_train["weighting_scheme"],
        batch_size=batch_size,
        logit_mean=float(config_train["logit_mean"]),
        logit_std=float(config_train["logit_std"]),
        mode_scale=float(config_train["mode_scale"]),
        device=device,
    )
    u = u.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    sigma_base = 1.0 - u

    sigma_1d = apply_sigma_shift(sigma_base, shift)                # [B] float32
    sigmas = _make_broadcast_view(sigma_1d, latents.ndim).to(dtype=dtype)

    noise = torch.randn_like(latents)
    noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
    model_timesteps = (1.0 - sigma_1d).to(device=device, dtype=torch.float32)

    target_latents = latents.detach() if detach_target_latents else latents
    target = target_latents.float() - noise.float()

    weighting_1d = compute_loss_weighting_for_sd3(
        weighting_scheme=weighting_scheme_override or config_train["weighting_scheme"],
        sigmas=sigma_1d,
    )
    weighting = _make_broadcast_view(weighting_1d.to(dtype=dtype), latents.ndim)

    return DualBranchBatch(
        sigmas=sigmas,
        noisy_latents=noisy_latents,
        model_timesteps=model_timesteps,
        target=target,
        weighting=weighting,
    )


def prepare_dual_diffusion_batch(
    *,
    config: dict,
    video_latents: Optional[torch.Tensor],
    audio_latents: Optional[torch.Tensor],
    shift_v: float,
    shift_a: float,
    detach_target_latents: bool = False,
) -> DualDiffusionBatch:
    """Sample independent ``t_v``, ``t_a`` and produce v-prediction targets.

    Each branch keeps its own sigma schedule (``shift_v``, ``shift_a``)
    so a single batch can train high-noise video chunks alongside
    low-noise audio chunks (and vice versa). The video and audio
    timesteps are uncorrelated by construction; this matches the joint
    AV diffusion paper's "dual sigma shift" recipe.
    """
    train_cfg = config["train"]
    if video_latents is None and audio_latents is None:
        raise ValueError("At least one of video_latents / audio_latents must be provided.")

    if video_latents is not None and audio_latents is not None:
        if int(video_latents.shape[0]) != int(audio_latents.shape[0]):
            raise ValueError(
                "Video and audio latents must share the leading batch dim, got "
                f"video={tuple(video_latents.shape)} vs audio={tuple(audio_latents.shape)}."
            )

    video_branch = (
        _prepare_branch_batch(
            config_train=train_cfg,
            latents=video_latents,
            shift=shift_v,
            detach_target_latents=detach_target_latents,
        )
        if video_latents is not None else None
    )
    audio_branch = (
        _prepare_branch_batch(
            config_train=train_cfg,
            latents=audio_latents,
            shift=shift_a,
            detach_target_latents=detach_target_latents,
        )
        if audio_latents is not None else None
    )

    batch_size = int(
        (video_latents if video_latents is not None else audio_latents).shape[0]
    )
    return DualDiffusionBatch(
        video=video_branch,
        audio=audio_branch,
        batch_size=batch_size,
    )
