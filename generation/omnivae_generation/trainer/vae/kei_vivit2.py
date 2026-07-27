from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from .kei import ViViT2HF


class _LatentDistAdapter:
    def __init__(self, latent_dist, owner: "KeiVivit2VAE") -> None:
        self._latent_dist = latent_dist
        self._owner = owner

    def sample(self, generator=None) -> torch.Tensor:
        latents = self._latent_dist.sample(generator=generator)
        return self._owner.raw_latents_to_training_layout(latents)

    def mode(self) -> torch.Tensor:
        latents = self._latent_dist.mode()
        return self._owner.raw_latents_to_training_layout(latents)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._latent_dist, name)


class _EncoderOutputAdapter:
    def __init__(self, encoded, owner: "KeiVivit2VAE") -> None:
        self._encoded = encoded
        if hasattr(encoded, "latent_dist"):
            self.latent_dist = _LatentDistAdapter(encoded.latent_dist, owner=owner)
        if hasattr(encoded, "latents"):
            self.latents = owner.raw_latents_to_training_layout(encoded.latents)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._encoded, name)


class KeiVivit2VAE(nn.Module):
    def __init__(self, inner: ViViT2HF) -> None:
        super().__init__()
        self.inner = inner
        self.config = self._build_compat_config(inner)
        self._laion_uses_training_layout = True
        self._laion_encode_returns_training_latents = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        subfolder: str | None = None,
        torch_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        **kwargs,
    ) -> "KeiVivit2VAE":
        inner = ViViT2HF.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(inner)

    @staticmethod
    def _build_compat_config(inner: ViViT2HF):
        config_dict = dict(inner.config)
        config_dict["scaling_factor"] = 1.0
        config_dict["shift_factor"] = 0.0
        if config_dict.get("latent_channels") is None:
            config_dict["latent_channels"] = int(inner.latent_channels)
        if config_dict.get("scale_factor_spatial") is None:
            config_dict["scale_factor_spatial"] = int(inner.scale_factor_spatial)
        if config_dict.get("scale_factor_temporal") is None:
            config_dict["scale_factor_temporal"] = int(inner.scale_factor_temporal)
        if config_dict.get("block_out_channels") is None:
            spatial_scale = int(config_dict["scale_factor_spatial"])
            if spatial_scale <= 0 or spatial_scale & (spatial_scale - 1):
                raise ValueError(f"Kei ViViT2 VAE expects a power-of-two spatial scale, got {spatial_scale}.")
            config_dict["block_out_channels"] = [1] * int(spatial_scale.bit_length())
        return SimpleNamespace(**config_dict)

    @property
    def dtype(self):
        return self.inner.dtype

    def _latent_stats(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        latents_mean = getattr(self.config, "latents_mean", None)
        latents_std = getattr(self.config, "latents_std", None)
        if latents_mean is None and latents_std is None:
            return None
        if latents_mean is None or latents_std is None:
            raise ValueError("Kei ViViT2 VAE config must define both latents_mean and latents_std.")
        if latents.ndim < 3:
            raise ValueError(f"Expected Kei ViViT2 latents with at least 3 dims, got {tuple(latents.shape)}.")

        channels = int(latents.shape[1])
        view_shape = (1, channels, *([1] * (latents.ndim - 2)))
        mean = torch.as_tensor(latents_mean, device=latents.device, dtype=latents.dtype).flatten()
        std = torch.as_tensor(latents_std, device=latents.device, dtype=latents.dtype).flatten()
        if mean.numel() == 1:
            mean = mean.repeat(channels)
        if std.numel() == 1:
            std = std.repeat(channels)
        if mean.numel() != channels or std.numel() != channels:
            raise ValueError(
                "Kei ViViT2 latent stats do not match latent channels: "
                f"mean={mean.numel()}, std={std.numel()}, channels={channels}."
            )
        return mean.view(view_shape), std.view(view_shape)

    def raw_latents_to_training_layout(self, latents: torch.Tensor, *, update_stats: bool = False) -> torch.Tensor:
        del update_stats
        stats = self._latent_stats(latents)
        if stats is None:
            return latents
        latents_mean, latents_std = stats
        return (latents - latents_mean) / latents_std

    def training_latents_to_raw_layout(self, latents: torch.Tensor) -> torch.Tensor:
        stats = self._latent_stats(latents)
        if stats is None:
            return latents
        latents_mean, latents_std = stats
        return latents * latents_std + latents_mean

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        encoded = self.inner.encode(x, return_dict=return_dict)
        if not return_dict:
            if isinstance(encoded, tuple) and encoded and hasattr(encoded[0], "sample"):
                return (_LatentDistAdapter(encoded[0], owner=self), *encoded[1:])
            return encoded
        return _EncoderOutputAdapter(encoded, owner=self)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        return self.inner.decode(self.training_latents_to_raw_layout(z), return_dict=return_dict)

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.inner.save_pretrained(*args, **kwargs)

    def clear_cache(self):
        if hasattr(self.inner, "clear_cache"):
            return self.inner.clear_cache()
        return None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)
