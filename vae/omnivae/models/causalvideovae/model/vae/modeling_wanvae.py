import os
from typing import List, Tuple

import torch
from diffusers.configuration_utils import register_to_config

from ..modeling_videobase import VideoBaseAE
from ..registry import ModelRegistry
from ..utils.distrib_utils import DiagonalGaussianDistribution
from .wan.vae import (
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    count_conv3d,
)


@ModelRegistry.register("WanVAE")
class WanVAEModel(VideoBaseAE):
    @register_to_config
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Tuple[float] = (),
        temperal_downsample: Tuple[bool] = (False, True, True),
        dropout: float = 0.0,
        mean: List[float] = None,
        std: List[float] = None,
        deterministic_posterior: bool = False,
    ) -> None:
        super().__init__()
        if mean is None:
            mean = [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]
        if std is None:
            std = [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ]

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)
        self.dropout = dropout

        self.encoder = Encoder3d(
            dim=dim,
            z_dim=z_dim * 2,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            temperal_downsample=self.temperal_downsample,
            dropout=dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim=dim,
            z_dim=z_dim,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=self.attn_scales,
            temperal_upsample=self.temperal_downsample[::-1],
            dropout=dropout,
        )

        self.register_buffer("mean_tensor", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std_tensor", torch.tensor(std, dtype=torch.float32))

        self.deterministic_posterior = deterministic_posterior

        self.clear_cache()

    @property
    def _scale(self):
        # From-scratch training: disable fixed latent affine normalization.
        return [torch.zeros_like(self.mean_tensor), torch.ones_like(self.std_tensor)]

    def get_encoder(self):
        # Include posterior projection conv so mu/logvar head is trainable.
        return [self.encoder, self.conv1]

    def get_decoder(self):
        # Include latent projection conv so decoder input mapping is trainable.
        return [self.conv2, self.decoder]

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode_with_logvar(self, x: torch.Tensor, streaming_inference: bool):
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
            mu_raw, log_var_raw = self.conv1(out).chunk(2, dim=1)

            # Disable fixed latent affine normalization for from-scratch stability.
            mu = mu_raw
            log_var = log_var_raw
            self.clear_cache()
        else:
            out = self.encoder(x)
            mu_raw, log_var_raw = self.conv1(out).chunk(2, dim=1)

            # Disable fixed latent affine normalization for from-scratch stability.
            mu = mu_raw
            log_var = log_var_raw

        return mu, log_var

    def encode(self, x: torch.Tensor, streaming_inference: bool):
        mu, log_var = self._encode_with_logvar(x, streaming_inference)
        moments = torch.cat([mu, log_var], dim=1)
        posterior = DiagonalGaussianDistribution(
            moments, deterministic=self.deterministic_posterior
        )
        return posterior

    def decode(self, z: torch.Tensor, streaming_inference: bool):
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
            out = self.decoder(x)
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
