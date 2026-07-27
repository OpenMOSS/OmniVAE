"""
Self-contained WanVAE2.2 model wrapper.

This is a standalone copy of `WanVAE22Model` that depends only on torch +
einops (via `modules.py`). The original implementation in
`opensora/models/causalvideovae/model/vae/modeling_wanvae22.py` inherits from
`diffusers.ModelMixin` / `ConfigMixin`; here we replace it with plain
`nn.Module` and provide minimal `from_config / from_pretrained / init_from_ckpt`
helpers so that the directory has no project-internal dependencies.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .distributions import DiagonalGaussianDistribution
from .modules import (
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    count_conv3d,
    patchify,
    unpatchify,
)


_DEFAULT_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,
     0.1838,  0.1557, -0.1382,  0.0542,  0.2813,  0.0891,
     0.1570, -0.0098,  0.0375, -0.1825, -0.2246, -0.1207,
    -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899,
    -0.2799, -0.1230, -0.0313, -0.1649,  0.0117,  0.0723,
    -0.2839, -0.2083, -0.0520,  0.3748,  0.0152,  0.1957,
     0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
]

_DEFAULT_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990,
    0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523,
    0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999,
    0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600,
    0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


class WanVAE22Model(nn.Module):
    """Wan2.2 VAE wrapper (self-contained, no diffusers/opensora deps).

    Key behavior identical to
    `opensora/models/causalvideovae/model/vae/modeling_wanvae22.py`:
      - `patchify / unpatchify` (patch_size=2) around encoder/decoder
      - Asymmetric encoder (`dim`) / decoder (`dec_dim`)
      - z_dim=48 (default), spatial compress = 16x, temporal = 4x
      - Encoder input 12ch (patchified 3ch), Decoder output 12ch
    """

    temporal_compress_factor = 4
    spatial_compress_factor = 16

    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 48,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        dropout: float = 0.0,
        patch_size: int = 2,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        deterministic_posterior: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if mean is None:
            mean = list(_DEFAULT_MEAN)
        if std is None:
            std = list(_DEFAULT_STD)

        self._init_kwargs = dict(
            dim=dim,
            dec_dim=dec_dim,
            z_dim=z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=list(attn_scales),
            temperal_downsample=list(temperal_downsample),
            dropout=dropout,
            patch_size=patch_size,
            mean=list(mean),
            std=list(std),
            deterministic_posterior=deterministic_posterior,
            qk_norm=qk_norm,
        )

        self.dim = dim
        self.dec_dim = dec_dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)
        self.dropout = dropout
        self.patch_size = patch_size
        self.qk_norm = qk_norm

        self.encoder = Encoder3d(
            dim=dim,
            z_dim=z_dim * 2,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            temperal_downsample=self.temperal_downsample,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim=dec_dim,
            z_dim=z_dim,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            temperal_upsample=self.temperal_downsample[::-1],
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.register_buffer(
            "mean_tensor", torch.tensor(mean, dtype=torch.float32)
        )
        self.register_buffer(
            "std_tensor", torch.tensor(std, dtype=torch.float32)
        )

        self.deterministic_posterior = deterministic_posterior
        self.clear_cache()

    @property
    def config(self) -> dict:
        return dict(self._init_kwargs)

    def get_encoder(self):
        return [self.encoder, self.conv1]

    def get_decoder(self):
        return [self.conv2, self.decoder]

    def clear_cache(self) -> None:
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode_with_logvar(
        self, x: torch.Tensor, streaming_inference: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = patchify(x, patch_size=self.patch_size)

        if streaming_inference:
            self.clear_cache()
            t = x.shape[2]
            iter_ = 1 + (t - 1) // 4
            for i in range(iter_):
                self._enc_conv_idx = [0]
                if i == 0:
                    out = self.encoder(
                        x[:, :, :1, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                else:
                    out_ = self.encoder(
                        x[:, :, 1 + 4 * (i - 1): 1 + 4 * i, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                    out = torch.cat([out, out_], 2)
            mu, log_var = self.conv1(out).chunk(2, dim=1)
            self.clear_cache()
        else:
            out = self.encoder(x)
            mu, log_var = self.conv1(out).chunk(2, dim=1)

        return mu, log_var

    def encode(
        self, x: torch.Tensor, streaming_inference: bool = False
    ) -> DiagonalGaussianDistribution:
        mu, log_var = self._encode_with_logvar(x, streaming_inference)
        moments = torch.cat([mu, log_var], dim=1)
        posterior = DiagonalGaussianDistribution(
            moments, deterministic=self.deterministic_posterior
        )
        return posterior

    def decode(
        self, z: torch.Tensor, streaming_inference: bool = False
    ) -> torch.Tensor:
        if streaming_inference:
            self.clear_cache()
            iter_ = z.shape[2]
            x = self.conv2(z)
            for i in range(iter_):
                self._conv_idx = [0]
                if i == 0:
                    out = self.decoder(
                        x[:, :, i: i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        first_chunk=True,
                    )
                else:
                    out_ = self.decoder(
                        x[:, :, i: i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                    )
                    out = torch.cat([out, out_], 2)
            self.clear_cache()
        else:
            x = self.conv2(z)
            out = self.decoder(x, first_chunk=True)

        out = unpatchify(out, patch_size=self.patch_size)
        return out

    def forward(
        self,
        input: torch.Tensor,
        sample_posterior: bool = True,
        streaming_inference: bool = False,
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(input, streaming_inference)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z, streaming_inference)
        return dec, posterior

    def get_last_layer(self) -> torch.Tensor:
        return self.decoder.head[-1].weight

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.enable_gradient_checkpointing()
        self.decoder.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.encoder.disable_gradient_checkpointing()
        self.decoder.disable_gradient_checkpointing()

    # ------------------------------------------------------------------ #
    # Loading helpers (no diffusers dependency)                          #
    # ------------------------------------------------------------------ #
    def init_from_ckpt(
        self,
        path: Union[str, os.PathLike],
        ignore_keys: Sequence[str] = (),
        strict: bool = False,
        verbose: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Load weights from a `.pt / .pth / .ckpt` file.

        Supports any of the layouts produced by the project's training /
        extraction scripts:
          - bare flat state_dict
          - {"state_dict": sd}
          - {"state_dict": {"gen_model": sd}}
          - {"ema_state_dict": sd}                      (EMA, prefers EMA
                                                         unless `NOT_USE_EMA_MODEL=1`)
          - {"state_dict": sd, "metadata": {...}}       (BaseModel-style)
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        # UniVAE / AudioVideoVAE training bundles: never feed them through this
        # native Wan loader (they carry ``model_state_dict`` with ``video_vae.*``).
        if isinstance(sd, dict):
            msd = sd.get("model_state_dict")
            if isinstance(msd, dict) and any(
                isinstance(k, str) and k.startswith("video_vae.") for k in msd.keys()
            ):
                raise ValueError(
                    f"The checkpoint at {path!r} looks like a UniVAE / AudioVideoVAE "
                    "training bundle (``model_state_dict`` with ``video_vae.*`` keys). "
                    "It cannot be loaded as a raw Wan2.2 ``.pth`` here. For OmniVAE generation "
                    "use ``vae.type=univae`` and ``vae.model_name_or_path`` pointing at this "
                    "file (CLI: ``--vae_type univae --vae_path ...``). "
                    "Note: ``--audio_vae_type`` only overrides ``audio_vae``; image runs "
                    "must set ``--vae_type univae`` (or yaml ``vae.type``), not ``audio_vae_type``."
                )
        ema_sd = sd.get("ema_state_dict") if isinstance(sd, dict) else None
        if (
            isinstance(sd, dict)
            and isinstance(ema_sd, dict)
            and len(ema_sd) > 0
            and int(os.environ.get("NOT_USE_EMA_MODEL", 0)) == 0
        ):
            sd = ema_sd
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        elif isinstance(sd, dict) and "state_dict" in sd:
            inner = sd["state_dict"]
            if isinstance(inner, dict) and "gen_model" in inner:
                sd = inner["gen_model"]
            else:
                sd = inner

        if not isinstance(sd, dict):
            raise RuntimeError(
                f"Unrecognized checkpoint structure at {path}: {type(sd)}"
            )

        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    sd.pop(k, None)
                    break

        missing, unexpected = self.load_state_dict(sd, strict=strict)
        if verbose:
            print(
                f"[WanVAE22Model] loaded {len(sd)} tensors from {path}\n"
                f"  missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print(f"  first missing keys: {missing[:5]}")
            if unexpected:
                print(f"  first unexpected keys: {unexpected[:5]}")
        return missing, unexpected

    @classmethod
    def load_config(
        cls, config_path: Union[str, os.PathLike]
    ) -> dict:
        """Read a json config and strip diffusers-internal keys."""
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return {k: v for k, v in cfg.items() if not k.startswith("_")}

    @classmethod
    def looks_like_vae_config(cls, cfg: dict) -> bool:
        """Heuristic: a real WanVAE22 config must carry `dim_mult` and
        either `z_dim` or `temperal_downsample`."""
        return (
            isinstance(cfg, dict)
            and "dim_mult" in cfg
            and ("z_dim" in cfg or "temperal_downsample" in cfg)
        )

    @classmethod
    def from_config(
        cls,
        config: Union[str, os.PathLike, dict],
        **overrides,
    ) -> "WanVAE22Model":
        import inspect

        if isinstance(config, (str, os.PathLike)):
            cfg = cls.load_config(config)
        else:
            cfg = {k: v for k, v in dict(config).items() if not k.startswith("_")}
        cfg.update(overrides)

        # Drop kwargs that the constructor does not accept (e.g. an
        # unrelated config from a diffusion transformer that happened to
        # be picked up).
        sig = inspect.signature(cls.__init__)
        accepted = {k for k in sig.parameters.keys() if k != "self"}
        dropped = [k for k in list(cfg.keys()) if k not in accepted]
        for k in dropped:
            cfg.pop(k)
        if dropped:
            print(
                f"[WanVAE22Model.from_config] ignoring unknown kwargs "
                f"{dropped} (not in __init__ signature)."
            )
        return cls(**cfg)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        config_path: Optional[Union[str, os.PathLike]] = None,
        weight_path: Optional[Union[str, os.PathLike]] = None,
        strict: bool = False,
        **overrides,
    ) -> "WanVAE22Model":
        """Load a model from a directory (containing `config.json` and a
        weight file) or directly from a single weight file.

        Resolution rules:
          - If `pretrained_model_name_or_path` is a file: it is treated as the
            weight file. `config_path` (or sibling `config.json`) supplies the
            architecture; falls back to default 48-dim config bundled here.
          - If it is a directory: looks for `config.json` and the first
            `*.pth / *.pt / *.ckpt / *.safetensors / *.bin` inside it.
        """
        path = Path(pretrained_model_name_or_path)

        # Resolve config first.
        cfg_file: Optional[Path] = None
        if config_path is not None:
            cfg_file = Path(config_path)
        elif path.is_dir() and (path / "config.json").exists():
            cfg_file = path / "config.json"
        elif path.is_file() and (path.parent / "config.json").exists():
            cfg_file = path.parent / "config.json"
        else:
            bundled = Path(__file__).resolve().parent / "config.json"
            if bundled.exists():
                cfg_file = bundled

        if cfg_file is None:
            raise FileNotFoundError(
                "Could not locate a config.json. Pass --config_path explicitly."
            )

        model = cls.from_config(cfg_file, **overrides)

        # Resolve weight file.
        if weight_path is not None:
            weight_file = Path(weight_path)
        elif path.is_file():
            weight_file = path
        else:
            weight_file = _find_weight_file(path)

        if weight_file is None:
            raise FileNotFoundError(
                f"Could not locate a weight file under {path}"
            )

        model.init_from_ckpt(weight_file, strict=strict)
        return model


def _find_weight_file(directory: Path) -> Optional[Path]:
    """Pick the first matching weight file in a directory."""
    for pattern in ("*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.bin"):
        matches = sorted(glob.glob(str(directory / pattern)))
        if matches:
            return Path(matches[-1])
    return None
