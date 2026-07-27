from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan

from .wan_parallel_autoencoderkl import WanParallelOps, normalize_wan_chunk_mode


DEFAULT_WAN22_DIFFUSERS_CONFIG = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class _LatentDistAdapter:
    def __init__(self, latent_dist, owner: "Wan2_2_VAE", squeeze_temporal_dim: bool) -> None:
        self._latent_dist = latent_dist
        self._owner = owner
        self._squeeze_temporal_dim = squeeze_temporal_dim

    def sample(self, generator=None) -> torch.Tensor:
        latents = self._owner._normalize_latents(self._latent_dist.sample(generator=generator))
        if self._squeeze_temporal_dim and latents.ndim == 5 and latents.shape[2] == 1:
            return latents.squeeze(2)
        return latents

    def mode(self) -> torch.Tensor:
        latents = self._owner._normalize_latents(self._latent_dist.mode())
        if self._squeeze_temporal_dim and latents.ndim == 5 and latents.shape[2] == 1:
            return latents.squeeze(2)
        return latents

    def __getattr__(self, name: str) -> Any:
        return getattr(self._latent_dist, name)


class _EncoderOutputAdapter:
    def __init__(self, encoded, owner: "Wan2_2_VAE", squeeze_temporal_dim: bool) -> None:
        self._encoded = encoded
        self.latents = None
        if hasattr(encoded, "latent_dist"):
            self.latent_dist = _LatentDistAdapter(
                encoded.latent_dist,
                owner=owner,
                squeeze_temporal_dim=squeeze_temporal_dim,
            )
        if hasattr(encoded, "latents"):
            latents = owner._normalize_latents(encoded.latents)
            if squeeze_temporal_dim and latents.ndim == 5 and latents.shape[2] == 1:
                latents = latents.squeeze(2)
            self.latents = latents

    def __getattr__(self, name: str) -> Any:
        return getattr(self._encoded, name)


class _DecoderOutputAdapter:
    def __init__(self, decoded, squeeze_temporal_dim: bool) -> None:
        self._decoded = decoded
        if hasattr(decoded, "sample"):
            sample = decoded.sample
            if squeeze_temporal_dim and sample.ndim == 5 and sample.shape[2] == 1:
                sample = sample.squeeze(2)
            self.sample = sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self._decoded, name)


class Wan2_2_VAE(nn.Module):
    def __init__(self, inner: AutoencoderKLWan, *, wan_chunk_mode: str = "cache") -> None:
        super().__init__()
        self.inner = inner
        self.dtype = inner.dtype
        self.config = self._build_compat_config(inner)
        self.wan_chunk_mode = normalize_wan_chunk_mode(wan_chunk_mode)
        self._wan_parallel_ops: WanParallelOps | None = None
        if self.wan_chunk_mode == "parallel":
            self._wan_parallel_ops = WanParallelOps(self.inner)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        subfolder: str | None = "vae",
        torch_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        wan_chunk_mode: str = "cache",
        **kwargs,
    ) -> "Wan2_2_VAE":
        inner = AutoencoderKLWan.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(inner, wan_chunk_mode=wan_chunk_mode)

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path: str,
        *,
        config: str | None = None,
        subfolder: str | None = "vae",
        torch_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        wan_chunk_mode: str = "cache",
        **kwargs,
    ) -> "Wan2_2_VAE":
        config = config or DEFAULT_WAN22_DIFFUSERS_CONFIG
        inner = AutoencoderKLWan.from_single_file(
            pretrained_model_link_or_path,
            config=config,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(inner, wan_chunk_mode=wan_chunk_mode)

    @staticmethod
    def _build_compat_config(inner: AutoencoderKLWan):
        config_dict = dict(inner.config)
        # Wan2.2's `patch_size` is an internal encoder/decoder patchify step.
        # External latent feature maps are still compressed by `scale_factor_spatial`.
        # Downstream image pipelines infer latent H/W from `len(block_out_channels)`,
        # so using `scale_factor_spatial * patch_size` incorrectly halves the sampled
        # latent resolution (for Wan2.2-TI2V this would produce 8x8 instead of 16x16
        # latents at 256px).
        spatial_scale = int(config_dict.get("scale_factor_spatial") or 8)
        if spatial_scale <= 0 or spatial_scale & (spatial_scale - 1):
            raise ValueError(f"Wan2.2 VAE expects a power-of-two effective spatial scale, got {spatial_scale}.")

        if "block_out_channels" not in config_dict:
            block_depth = int(math.log2(spatial_scale)) + 1
            config_dict["block_out_channels"] = [1] * block_depth
        config_dict["scaling_factor"] = 1.0
        config_dict["shift_factor"] = 0.0
        return SimpleNamespace(**config_dict)

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.inner.save_pretrained(*args, **kwargs)

    def _latent_stats(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents_mean = torch.tensor(self.inner.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.inner.config.z_dim, 1, 1, 1
        )
        latents_std = torch.tensor(self.inner.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.inner.config.z_dim, 1, 1, 1
        )
        return latents_mean, latents_std.reciprocal()

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean, latents_std = self._latent_stats(latents)
        return (latents - latents_mean) * latents_std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean, latents_std = self._latent_stats(latents)
        return latents / latents_std + latents_mean

    def _validate_parallel_input_frames(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            return
        try:
            temporal_scale = getattr(self.inner.config, "scale_factor_temporal")
        except (AttributeError, KeyError):
            temporal_scale = 4
        temporal_scale = int(temporal_scale or 4)
        frames = int(x.shape[2])
        if temporal_scale > 1 and (frames - 1) % temporal_scale != 0:
            raise ValueError(
                "Wan2.2 VAE wan_chunk_mode='parallel' requires input frames T=scale_factor_temporal*k+1 "
                f"(scale_factor_temporal={temporal_scale}, got T={frames}). Use a Kei-style clip length "
                "such as 49, or set vae.wan_chunk_mode='cache'."
            )

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        squeeze_temporal_dim = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            squeeze_temporal_dim = True
        if self.wan_chunk_mode == "parallel":
            self._validate_parallel_input_frames(x)
            if self._wan_parallel_ops is None:
                self._wan_parallel_ops = WanParallelOps(self.inner)
            latent_dist = self._wan_parallel_ops.encode_latent_dist(x)
            if not return_dict:
                return (latent_dist,)
            encoded = SimpleNamespace(latent_dist=latent_dist)
        else:
            encoded = self.inner.encode(x, return_dict=return_dict)
        if not return_dict:
            return encoded
        return _EncoderOutputAdapter(encoded, owner=self, squeeze_temporal_dim=squeeze_temporal_dim)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        squeeze_temporal_dim = False
        if z.ndim == 4:
            z = z.unsqueeze(2)
            squeeze_temporal_dim = True
        z = self._denormalize_latents(z)
        if self.wan_chunk_mode == "parallel":
            if self._wan_parallel_ops is None:
                self._wan_parallel_ops = WanParallelOps(self.inner)
            decoded_tensor = self._wan_parallel_ops.decode_tensor(z)
            decoded = (decoded_tensor,) if not return_dict else SimpleNamespace(sample=decoded_tensor)
        else:
            decoded = self.inner.decode(z, return_dict=return_dict)
        if not return_dict:
            if (
                squeeze_temporal_dim
                and isinstance(decoded, tuple)
                and decoded
                and torch.is_tensor(decoded[0])
                and decoded[0].ndim == 5
                and decoded[0].shape[2] == 1
            ):
                return (decoded[0].squeeze(2), *decoded[1:])
            return decoded
        return _DecoderOutputAdapter(decoded, squeeze_temporal_dim=squeeze_temporal_dim)

    def clear_cache(self):
        if hasattr(self.inner, "clear_cache"):
            return self.inner.clear_cache()
        return None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)
