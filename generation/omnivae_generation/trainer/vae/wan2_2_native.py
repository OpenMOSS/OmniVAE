"""Wan2.2 VAE loader for the **native open-source `.pth`** format.

This is a drop-in alternative to `Wan2_2_VAE` (which wraps
`diffusers.AutoencoderKLWan` and only loads diffusers-format weights).
``Wan2_2_NativeVAE`` instead loads the standalone ``WanVAE22Model`` from
``opensora.infer.wan2_2vae``, so checkpoints like the open-source
``Wan2.2_VAE.pth`` (with a sibling ``config.json``) can be used directly
without any prior conversion to diffusers layout.

Contract mirrors ``Wan2_2_VAE`` so it is fully drop-in for the rest of the
trainer pipeline:

* ``encode(x)`` returns an object exposing ``.latent_dist`` whose
  ``.sample()`` / ``.mode()`` already apply ``(z - mean) / std`` normalization
  using the model's own ``mean_tensor`` / ``std_tensor`` buffers.
* ``decode(z)`` accepts normalized latents and denormalizes them internally
  before invoking the underlying decoder.
* ``config.scaling_factor = 1.0`` and ``config.shift_factor = 0.0`` so the
  outer ``encode_images_to_latents`` no-ops on the final scale step (the
  same trick used by ``Wan2_2_VAE``).
* 4D ``[B, C, H, W]`` inputs (single-frame T2I) are auto-unsqueezed on the
  temporal axis and re-squeezed on the way out.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lazy import of the standalone WanVAE22Model. This optional path is used only
# when a config selects ``vae.type=wan2_2_native_vae``.
# --------------------------------------------------------------------------- #
def _ensure_video_tokenizer_on_path() -> None:
    try:
        import opensora.infer.wan2_2vae  # noqa: F401
        return
    except ImportError:
        pass

    env_path = os.environ.get("OMNIGEN_WAN_VAE_REPO")
    if env_path:
        env_root = Path(env_path).expanduser().resolve()
        if (env_root / "opensora" / "infer" / "wan2_2vae" / "model.py").is_file():
            if str(env_root) not in sys.path:
                sys.path.insert(0, str(env_root))
            return

    raise ImportError(
        "Cannot import opensora.infer.wan2_2vae (needed for "
        "vae.type='wan2_2_native_vae'). Install a package that provides it or "
        "set OMNIGEN_WAN_VAE_REPO to a repo root containing opensora/infer/wan2_2vae."
    )


def _import_wan_vae22_pieces():
    _ensure_video_tokenizer_on_path()
    from opensora.infer.wan2_2vae import WanVAE22Model
    return WanVAE22Model


# --------------------------------------------------------------------------- #
# config.json + qk_norm resolution                                            #
# Mirrors infer.py's `_resolve_model_config` / `_resolve_qk_norm` but kept    #
# here so we don't depend on infer.py's private helpers.                      #
# --------------------------------------------------------------------------- #
def _resolve_native_config(
    pretrained_path: Path, model_config: str | None
) -> str:
    WanVAE22Model = _import_wan_vae22_pieces()

    if model_config and os.path.exists(model_config):
        return model_config

    side = (
        pretrained_path.parent / "config.json"
        if pretrained_path.is_file()
        else pretrained_path / "config.json"
    )
    if side.exists():
        try:
            cfg_peek = WanVAE22Model.load_config(str(side))
        except Exception as exc:
            logger.warning("Wan2.2 native VAE: failed to read %s (%s); falling back.", side, exc)
            cfg_peek = None
        if cfg_peek is not None and WanVAE22Model.looks_like_vae_config(cfg_peek):
            return str(side)
        if cfg_peek is not None:
            logger.warning(
                "Wan2.2 native VAE: %s does not look like a WanVAE22 config "
                "(missing 'dim_mult'); ignoring it and falling back to the bundled default.",
                side,
            )

    import opensora.infer.wan2_2vae as _pkg
    bundled = Path(_pkg.__file__).resolve().parent / "config.json"
    if bundled.exists():
        return str(bundled)

    raise FileNotFoundError(
        "Wan2.2 native VAE: cannot resolve a config.json. "
        f"Tried model_config={model_config}, sibling={side}, bundled={bundled}."
    )


def _resolve_native_qk_norm(
    arg_value: str, ckpt_path: Path, cfg_dict: dict
) -> bool:
    arg_value = (arg_value or "auto").strip().lower()
    if arg_value == "true":
        return True
    if arg_value == "false":
        return False

    if ckpt_path.is_file():
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning("Wan2.2 native VAE: failed to peek at ckpt for qk_norm (%s)", exc)
            ckpt = None
        if isinstance(ckpt, dict):
            meta = ckpt.get("metadata")
            if isinstance(meta, dict) and "qk_norm_filtered" in meta:
                qk = not bool(meta["qk_norm_filtered"])
                logger.info(
                    "Wan2.2 native VAE: qk_norm inferred from ckpt metadata "
                    "(filtered=%s -> qk_norm=%s)",
                    meta["qk_norm_filtered"], qk,
                )
                return qk

    if isinstance(cfg_dict, dict) and "qk_norm" in cfg_dict:
        qk = bool(cfg_dict["qk_norm"])
        logger.info("Wan2.2 native VAE: qk_norm taken from config.json (%s)", qk)
        return qk

    logger.info(
        "Wan2.2 native VAE: no qk_norm signal from ckpt/config; "
        "defaulting to False (matches open-source Wan2.2 release)."
    )
    return False


# --------------------------------------------------------------------------- #
# Adapters: same shape as the diffusers-side wrappers in wan2_2.py            #
# --------------------------------------------------------------------------- #
class _NativeLatentDistAdapter:
    def __init__(self, posterior, owner: "Wan2_2_NativeVAE", squeeze_temporal_dim: bool) -> None:
        self._posterior = posterior
        self._owner = owner
        self._squeeze = squeeze_temporal_dim

    def _maybe_squeeze(self, z: torch.Tensor) -> torch.Tensor:
        if self._squeeze and z.ndim == 5 and z.shape[2] == 1:
            return z.squeeze(2)
        return z

    def sample(self, generator=None) -> torch.Tensor:
        # The native DiagonalGaussianDistribution.sample() does not accept a
        # generator argument; trainer side never passes one for VAE encode.
        if generator is not None:
            raise NotImplementedError(
                "Wan2_2_NativeVAE: posterior.sample(generator=...) is not "
                "supported by the native DiagonalGaussianDistribution."
            )
        z = self._posterior.sample()
        z = self._owner._normalize_latents(z)
        return self._maybe_squeeze(z)

    def mode(self) -> torch.Tensor:
        z = self._posterior.mode()
        z = self._owner._normalize_latents(z)
        return self._maybe_squeeze(z)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._posterior, name)


class _NativeEncoderOutputAdapter:
    def __init__(self, posterior, owner: "Wan2_2_NativeVAE", squeeze_temporal_dim: bool) -> None:
        self.latent_dist = _NativeLatentDistAdapter(posterior, owner=owner, squeeze_temporal_dim=squeeze_temporal_dim)
        self.latents = None  # trainer pipeline uses .latent_dist for VAEs that have one


class _NativeDecoderOutputAdapter:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


# --------------------------------------------------------------------------- #
# The actual nn.Module                                                         #
# --------------------------------------------------------------------------- #
class Wan2_2_NativeVAE(nn.Module):
    """Drop-in replacement for ``Wan2_2_VAE`` that loads the open-source
    Wan2.2 VAE in its native ``.pth`` format (``WanVAE22Model``)."""

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner  # WanVAE22Model
        self.dtype = next(inner.parameters()).dtype

        spatial_compress = int(getattr(inner, "spatial_compress_factor", 16))
        # diffusers' ZImagePipeline derives `vae_scale_factor` as
        #   2 ** (len(vae.config.block_out_channels) - 1)
        # and uses it to pad/align inputs and shape latents. Wan2.2 VAE
        # spatially downsamples 16x (256 -> 16 latent), so we need a 5-element
        # placeholder list so that 2**(5-1) == 16 == spatial_compress.
        # The actual channel values are unused by ZImagePipeline; only `len()`.
        num_blocks = max(1, int(math.log2(max(2, spatial_compress))) + 1)
        z_dim = int(getattr(inner, "z_dim", 48))
        block_out_channels = tuple(z_dim * (2 ** min(i, 3)) for i in range(num_blocks))

        # Shape a diffusers-ish config so consumers that read scaling_factor /
        # shift_factor / _class_name / block_out_channels keep working
        # (encode_images_to_latents treats shift_factor != None as the no-shift
        # identity branch when both are 0/1, which is exactly what we want here).
        self.config = SimpleNamespace(
            _class_name="WanVAE22Model",
            scaling_factor=1.0,
            shift_factor=0.0,
            z_dim=z_dim,
            patch_size=int(getattr(inner, "patch_size", 2)),
            scale_factor_spatial=spatial_compress,
            scale_factor_temporal=int(getattr(inner, "temporal_compress_factor", 4)),
            block_out_channels=block_out_channels,
            latent_channels=z_dim,
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_native_ckpt(
        cls,
        pretrained_path: str,
        *,
        model_config: str | None = None,
        qk_norm: str = "auto",
        torch_dtype: torch.dtype = torch.float32,
        deterministic_posterior: bool = False,
    ) -> "Wan2_2_NativeVAE":
        WanVAE22Model = _import_wan_vae22_pieces()

        pp = Path(pretrained_path).expanduser().resolve()
        cfg_path = _resolve_native_config(pp, model_config)
        cfg_dict = WanVAE22Model.load_config(cfg_path)
        cfg_dict["qk_norm"] = _resolve_native_qk_norm(qk_norm, pp, cfg_dict)
        cfg_dict.setdefault("deterministic_posterior", deterministic_posterior)

        logger.info(
            "Wan2.2 native VAE: building WanVAE22Model from config=%s, qk_norm=%s",
            cfg_path, cfg_dict["qk_norm"],
        )
        model = WanVAE22Model.from_config(cfg_dict)

        if pp.is_file():
            weight_file: Path | None = pp
        else:
            weight_file = None
            for pattern in ("*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.bin"):
                files = sorted(pp.glob(pattern))
                if files:
                    weight_file = files[-1]
                    break
            if weight_file is None:
                raise FileNotFoundError(
                    f"Wan2.2 native VAE: no weight file (.pth/.pt/.ckpt/.safetensors/.bin) found under {pp}"
                )

        logger.info("Wan2.2 native VAE: loading weights from %s", weight_file)
        model.init_from_ckpt(str(weight_file))
        model = model.eval().requires_grad_(False)
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)
        return cls(model)

    # ------------------------------------------------------------------ #
    def _broadcast_stats(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.inner.mean_tensor.to(device=like.device, dtype=like.dtype)
        std = self.inner.std_tensor.to(device=like.device, dtype=like.dtype)
        # mean/std are 1D (z_dim,); broadcast to (1, C, 1, 1, 1) for 5D latents.
        view_shape = [1] * like.ndim
        view_shape[1] = -1
        return mean.view(*view_shape), std.view(*view_shape)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast_stats(latents)
        return (latents - mean) / std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._broadcast_stats(latents)
        return latents * std + mean

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        squeeze_temporal_dim = False
        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
            squeeze_temporal_dim = True
        posterior = self.inner.encode(x)
        adapter = _NativeEncoderOutputAdapter(
            posterior, owner=self, squeeze_temporal_dim=squeeze_temporal_dim
        )
        if not return_dict:
            return (adapter.latent_dist,)
        return adapter

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        squeeze_temporal_dim = False
        if z.ndim == 4:
            z = z.unsqueeze(2)
            squeeze_temporal_dim = True
        z = self._denormalize_latents(z)
        sample = self.inner.decode(z)
        if squeeze_temporal_dim and sample.ndim == 5 and sample.shape[2] == 1:
            sample = sample.squeeze(2)
        if not return_dict:
            return (sample,)
        return _NativeDecoderOutputAdapter(sample)

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def clear_cache(self):
        if hasattr(self.inner, "clear_cache"):
            return self.inner.clear_cache()
        return None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)
