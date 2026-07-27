from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_wan_chunk_mode(value: Any) -> str:
    mode = str(value).lower().strip()
    if mode in {"", "none", "null"}:
        return "cache"
    if mode not in {"cache", "parallel"}:
        raise ValueError(f"Unsupported wan_chunk_mode={value!r}; expected one of: cache, parallel.")
    return mode


class WanParallelOps:
    """Kei-style parallel Wan VAE ops that avoid diffusers' temporal cache loop."""

    def __init__(self, vae: nn.Module) -> None:
        from diffusers.models.autoencoders.autoencoder_kl_wan import (
            DiagonalGaussianDistribution,
            patchify,
            unpatchify,
        )

        self.vae = vae
        self._patchify = patchify
        self._unpatchify = unpatchify
        self._diagonal_gaussian_cls = DiagonalGaussianDistribution

    def encode_latent_dist(self, x: torch.Tensor) -> Any:
        x_enc = self._maybe_channels_last_3d(x)
        patch_size = getattr(getattr(self.vae, "config", None), "patch_size", None)
        if patch_size is not None:
            x_enc = self._patchify(x_enc, patch_size=patch_size)
            x_enc = self._maybe_channels_last_3d(x_enc)
        h = self._parallel_encoder_forward(x_enc)
        h = self._parallel_causal_conv(self.vae.quant_conv, h)
        return self._diagonal_gaussian_cls(h)

    def decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        x = self._parallel_causal_conv(self.vae.post_quant_conv, z)
        decoded = self._parallel_decoder_forward(x, first_chunk=True)
        patch_size = getattr(getattr(self.vae, "config", None), "patch_size", None)
        if patch_size is not None:
            decoded = self._unpatchify(decoded.contiguous(), patch_size=patch_size)
        return torch.clamp(decoded, min=-1.0, max=1.0)

    @staticmethod
    def _maybe_channels_last_3d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5 and x.is_cuda and not x.is_contiguous(memory_format=torch.channels_last_3d):
            return x.contiguous(memory_format=torch.channels_last_3d)
        return x

    @staticmethod
    def _maybe_channels_last_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.is_cuda and not x.is_contiguous(memory_format=torch.channels_last):
            return x.contiguous(memory_format=torch.channels_last)
        return x

    def _parallel_causal_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(conv, nn.Identity):
            return x
        x = self._maybe_channels_last_3d(x)
        causal_padding = getattr(conv, "_padding", None)
        if isinstance(conv, nn.Conv3d) and isinstance(causal_padding, tuple) and len(causal_padding) == 6:
            t_left = int(causal_padding[4])
            pad_h = int(causal_padding[2])
            pad_w = int(causal_padding[0])
            if t_left > 0:
                x = F.pad(x, (0, 0, 0, 0, t_left, 0))
                x = self._maybe_channels_last_3d(x)
            old_padding = conv.padding
            conv.padding = (0, pad_h, pad_w)
            try:
                x = nn.Conv3d.forward(conv, x)
            finally:
                conv.padding = old_padding
            return self._maybe_channels_last_3d(x)
        return self._maybe_channels_last_3d(conv(x))

    def _parallel_residual_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        residual = self._parallel_causal_conv(block.conv_shortcut, x)

        x = block.norm1(x)
        x = block.nonlinearity(x)
        x = self._parallel_causal_conv(block.conv1, x)

        x = block.norm2(x)
        x = block.nonlinearity(x)
        x = block.dropout(x)
        x = self._parallel_causal_conv(block.conv2, x)
        return x + residual

    def _parallel_mid_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = self._parallel_residual_block(block.resnets[0], x)
        for attention, resnet in zip(block.attentions, block.resnets[1:]):
            if attention is not None:
                x = attention(x)
            x = self._parallel_residual_block(resnet, x)
        return x

    @staticmethod
    def _apply_2d_resample(resample: nn.Module, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.size()
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        if x_2d.is_cuda and not x_2d.is_contiguous(memory_format=torch.channels_last):
            x_2d = x_2d.contiguous(memory_format=torch.channels_last)
        x_2d = resample(x_2d)
        x_3d = x_2d.reshape(batch, frames, x_2d.size(1), x_2d.size(2), x_2d.size(3)).permute(0, 2, 1, 3, 4)
        if x_3d.is_cuda and not x_3d.is_contiguous(memory_format=torch.channels_last_3d):
            x_3d = x_3d.contiguous(memory_format=torch.channels_last_3d)
        return x_3d

    def _parallel_resample(self, layer: nn.Module, x: torch.Tensor, *, first_chunk: bool = False) -> torch.Tensor:
        mode = str(getattr(layer, "mode", "")).lower().strip()
        if mode in {"none", "upsample2d", "downsample2d"}:
            return layer(x)

        batch, channels, frames, height, width = x.size()
        if mode == "downsample3d":
            x = self._apply_2d_resample(layer.resample, x)
            if int(x.shape[2]) <= 1:
                return x[:, :, :1, :, :]
            x_rest = layer.time_conv(x)
            return torch.cat([x[:, :, :1, :, :], x_rest], dim=2)

        if mode == "upsample3d":
            if first_chunk:
                first = self._apply_2d_resample(layer.resample, x[:, :, :1, :, :])
                if int(frames) <= 1:
                    return first
                x_rest = layer.time_conv(x[:, :, 1:, :, :])
                _, _, frames_rest, _, _ = x_rest.size()
                x_rest = x_rest.reshape(batch, 2, channels, frames_rest, height, width)
                x_rest = torch.stack((x_rest[:, 0, :, :, :, :], x_rest[:, 1, :, :, :, :]), dim=3)
                x_rest = x_rest.reshape(batch, channels, frames_rest * 2, height, width)
                x_rest = self._apply_2d_resample(layer.resample, x_rest)
                return torch.cat([first, x_rest], dim=2)

            x = layer.time_conv(x)
            x = x.reshape(batch, 2, channels, frames, height, width)
            x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), dim=3)
            x = x.reshape(batch, channels, frames * 2, height, width)
            return self._apply_2d_resample(layer.resample, x)

        raise ValueError(f"Unsupported Wan resample mode={mode!r}.")

    def _parallel_encoder_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        class_name = layer.__class__.__name__
        if class_name == "WanResidualDownBlock":
            residual = x
            for resnet in layer.resnets:
                x = self._parallel_residual_block(resnet, x)
            if layer.downsampler is not None:
                x = self._parallel_resample(layer.downsampler, x)
            return x + layer.avg_shortcut(residual)
        if class_name == "WanResample":
            return self._parallel_resample(layer, x)
        if class_name == "WanResidualBlock":
            return self._parallel_residual_block(layer, x)
        return layer(x)

    def _parallel_up_block(self, block: nn.Module, x: torch.Tensor, *, first_chunk: bool) -> torch.Tensor:
        class_name = block.__class__.__name__
        if class_name == "WanResidualUpBlock":
            residual = x
            for resnet in block.resnets:
                x = self._parallel_residual_block(resnet, x)
            if block.upsampler is not None:
                x = self._parallel_resample(block.upsampler, x, first_chunk=first_chunk)
            if block.avg_shortcut is not None:
                x = x + block.avg_shortcut(residual, first_chunk=first_chunk)
            return x
        if class_name == "WanUpBlock":
            for resnet in block.resnets:
                x = self._parallel_residual_block(resnet, x)
            if block.upsamplers is not None:
                x = self._parallel_resample(block.upsamplers[0], x, first_chunk=first_chunk)
            return x
        if class_name == "WanResample":
            return self._parallel_resample(block, x, first_chunk=first_chunk)
        if class_name == "WanResidualBlock":
            return self._parallel_residual_block(block, x)
        return block(x)

    def _parallel_encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder = self.vae.encoder
        x = self._parallel_causal_conv(encoder.conv_in, x)
        for layer in encoder.down_blocks:
            x = self._parallel_encoder_layer(layer, x)
        x = self._parallel_mid_block(encoder.mid_block, x)
        x = encoder.norm_out(x)
        x = encoder.nonlinearity(x)
        x = self._parallel_causal_conv(encoder.conv_out, x)
        return x

    def _parallel_decoder_forward(self, x: torch.Tensor, *, first_chunk: bool) -> torch.Tensor:
        decoder = self.vae.decoder
        x = self._parallel_causal_conv(decoder.conv_in, x)
        x = self._parallel_mid_block(decoder.mid_block, x)
        for up_block in decoder.up_blocks:
            x = self._parallel_up_block(up_block, x, first_chunk=first_chunk)
        x = decoder.norm_out(x)
        x = decoder.nonlinearity(x)
        x = self._parallel_causal_conv(decoder.conv_out, x)
        return x
