import os
from typing import List, Tuple

import torch
from diffusers.configuration_utils import register_to_config

from ..modeling_videobase import VideoBaseAE
from ..registry import ModelRegistry
from ..utils.distrib_utils import DiagonalGaussianDistribution
from .wan22.vae import (
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    count_conv3d,
    patchify,
    unpatchify,
)


@ModelRegistry.register("WanVAE22")
class WanVAE22Model(VideoBaseAE):
    """
    Wan2.2 VAE wrapper for training integration.

    Key differences from WanVAE (Wan2.1):
      - patchify/unpatchify (patch_size=2) around encoder/decoder
      - Asymmetric encoder (dim) / decoder (dec_dim)
      - z_dim=48 (vs 16)
      - Down_ResidualBlock / Up_ResidualBlock with skip connections
      - Encoder input: 12ch (patchified 3ch), Decoder output: 12ch
      - spatial_compress_factor=16 (8x downsample + 2x patchify)
    """

    # Compression factors exposed for the trainer.
    temporal_compress_factor = 4
    spatial_compress_factor = 16

    @register_to_config
    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 48,
        dim_mult: Tuple[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Tuple[float] = (),
        temperal_downsample: Tuple[bool] = (False, True, True),
        dropout: float = 0.0,
        patch_size: int = 2,
        mean: List[float] = None,
        std: List[float] = None,
        deterministic_posterior: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if mean is None:
            mean = [
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,
                 0.1838,  0.1557, -0.1382,  0.0542,  0.2813,  0.0891,
                 0.1570, -0.0098,  0.0375, -0.1825, -0.2246, -0.1207,
                -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
                -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899,
                -0.2799, -0.1230, -0.0313, -0.1649,  0.0117,  0.0723,
                -0.2839, -0.2083, -0.0520,  0.3748,  0.0152,  0.1957,
                 0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
            ]
        if std is None:
            std = [
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990,
                0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
                0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523,
                0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999,
                0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
                0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600,
                0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
            ]

        self.dim = dim
        self.dec_dim = dec_dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)
        self.dropout = dropout
        self.patch_size = patch_size

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

        self.register_buffer("mean_tensor", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std_tensor", torch.tensor(std, dtype=torch.float32))

        self.deterministic_posterior = deterministic_posterior

        self.clear_cache()

    @property
    def _scale(self):
        return [torch.zeros_like(self.mean_tensor), torch.ones_like(self.std_tensor)]

    def get_encoder(self):
        return [self.encoder, self.conv1]

    def get_decoder(self):
        return [self.conv2, self.decoder]

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode_with_logvar(self, x: torch.Tensor, streaming_inference: bool):
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
                        x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
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

    def encode(self, x: torch.Tensor, streaming_inference: bool = False):
        mu, log_var = self._encode_with_logvar(x, streaming_inference)
        moments = torch.cat([mu, log_var], dim=1)
        posterior = DiagonalGaussianDistribution(
            moments, deterministic=self.deterministic_posterior
        )
        return posterior

    def decode(self, z: torch.Tensor, streaming_inference: bool = False):
        if streaming_inference:
            self.clear_cache()
            iter_ = z.shape[2]
            x = self.conv2(z)
            for i in range(iter_):
                self._conv_idx = [0]
                if i == 0:
                    out = self.decoder(
                        x[:, :, i : i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        first_chunk=True,
                    )
                else:
                    out_ = self.decoder(
                        x[:, :, i : i + 1, :, :],
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

    def forward(self, input: torch.Tensor, sample_posterior: bool = True, streaming_inference: bool = False):
        posterior = self.encode(input, streaming_inference)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, streaming_inference)
        return dec, posterior

    def get_last_layer(self):
        return self.decoder.head[-1].weight

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.decoder.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.encoder.disable_gradient_checkpointing()
        self.decoder.disable_gradient_checkpointing()

    def init_from_ckpt(self, path, ignore_keys=list()):
        if os.path.isdir(path):
            candidate = os.path.join(path, "state_dict.pt")
            if os.path.exists(candidate):
                path = candidate
        sd = torch.load(path, map_location="cpu")
        if (
            "ema_state_dict" in sd
            and sd["ema_state_dict"] is not None
            and len(sd["ema_state_dict"]) > 0
            and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0
        ):
            sd = sd["ema_state_dict"]
            if "shadow" in sd and isinstance(sd["shadow"], dict):
                sd = sd["shadow"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        has_video_prefix = any(k.startswith("video_vae.") for k in sd.keys())
        if has_video_prefix:
            sd = {
                k[len("video_vae."):]: v
                for k, v in sd.items()
                if k.startswith("video_vae.") and torch.is_tensor(v)
            }
        else:
            sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(missing_keys, unexpected_keys)
