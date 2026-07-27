from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.autoencoders.autoencoder_kl import DiagonalGaussianDistribution
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.checkpoint import checkpoint

from .flex_video_blocks import GridShape3D, PositionEmbeddings3D, TransformerFlexBlock3D, rms_norm_preserve_input_dtype


_SUPPORTED_ATTENTION_BACKENDS = {
    "auto",
    "flash",
    "triton",
    "natten_auto",
    "natten_cutlass-fna",
    "natten_hopper-fna",
    "natten_blackwell-fna",
    "natten_flex-fna",
}


@dataclass(frozen=True)
class _RuntimeShapes:
    input_t: int
    input_h: int
    input_w: int
    model_t: int
    stage_grids: tuple[GridShape3D, ...]

    @property
    def fine_grid(self) -> GridShape3D:
        return self.stage_grids[0]

    @property
    def coarse_grid(self) -> GridShape3D:
        return self.stage_grids[-1]

    @property
    def fine_t(self) -> int:
        return int(self.fine_grid[0])

    @property
    def fine_h(self) -> int:
        return int(self.fine_grid[1])

    @property
    def fine_w(self) -> int:
        return int(self.fine_grid[2])

    @property
    def coarse_t(self) -> int:
        return int(self.coarse_grid[0])

    @property
    def coarse_h(self) -> int:
        return int(self.coarse_grid[1])

    @property
    def coarse_w(self) -> int:
        return int(self.coarse_grid[2])


class ViViT2HF(ModelMixin, AutoencoderMixin, ConfigMixin, FromOriginalModelMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        video_shape: tuple[int, int, int] | list[int] | None = None,
        patch_shape: tuple[int, int, int] | list[int] = (4, 16, 16),
        embed_dim: int = 1024,
        stage_embed_dims: tuple[int, ...] | list[int] | None = None,
        latent_dim: int = 32,
        enc_depth: int = 8,
        dec_depth: int = 8,
        enc_stage_depths: tuple[int, ...] | list[int] | None = None,
        dec_stage_depths: tuple[int, ...] | list[int] | None = None,
        num_heads: int | tuple[int, ...] | list[int] = 16,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        encoder_attention_mode: str = "sparse_local",
        decoder_attention_mode: str = "sparse_local",
        attention_backend: str = "auto",
        encoder_causal: bool = True,
        decoder_causal: bool = False,
        window_t: int = 4,
        window_h: int = 1,
        window_w: int = 1,
        attn_block_size: int = 128,
        share_mask_across_batch_heads: bool = True,
        repeat_first_frame_to_patch: bool = False,
        spatial_shuffle_factor: int = 2,
        temporal_shuffle_factor: int = 1,
        temporal_stage_shuffle: tuple[bool, ...] | list[bool] | None = None,
        block_activation_checkpointing: bool = False,
        temporal_block_chunk_size: int | None = None,
        temporal_chunk_checkpointing: bool = False,
        out_act: str | None = None,
        latent_channels: int | None = None,
        scale_factor_spatial: int | None = None,
        scale_factor_temporal: int | None = None,
        latent_layout: str = "bcthw",
        latents_mean: tuple[float, ...] | list[float] | None = None,
        latents_std: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(video_shape, list):
            video_shape = tuple(int(x) for x in video_shape)
        if isinstance(patch_shape, list):
            patch_shape = tuple(int(x) for x in patch_shape)
        if isinstance(stage_embed_dims, list):
            stage_embed_dims = tuple(int(x) for x in stage_embed_dims)
        if isinstance(enc_stage_depths, list):
            enc_stage_depths = tuple(int(x) for x in enc_stage_depths)
        if isinstance(dec_stage_depths, list):
            dec_stage_depths = tuple(int(x) for x in dec_stage_depths)
        if isinstance(num_heads, list):
            num_heads = tuple(int(x) for x in num_heads)
        if isinstance(temporal_stage_shuffle, list):
            temporal_stage_shuffle = tuple(bool(x) for x in temporal_stage_shuffle)
        if video_shape is not None and (not isinstance(video_shape, tuple) or len(video_shape) != 3):
            raise ValueError(f"video_shape must be None or a 3-tuple/list, got {video_shape!r}")
        if not isinstance(patch_shape, tuple) or len(patch_shape) != 3:
            raise ValueError(f"patch_shape must be a 3-tuple/list, got {patch_shape!r}")

        self.in_channels = int(in_channels)
        self.video_shape = video_shape
        self.patch_t = int(patch_shape[0])
        self.patch_h = int(patch_shape[1])
        self.patch_w = int(patch_shape[2])
        if stage_embed_dims is None:
            stage_embed_dims = (int(embed_dim), int(embed_dim))
        self.stage_embed_dims = tuple(int(x) for x in stage_embed_dims)
        self.num_stages = int(len(self.stage_embed_dims))
        self.embed_dim = int(self.stage_embed_dims[0])
        self.deepest_embed_dim = int(self.stage_embed_dims[-1])
        self.latent_dim = int(latent_dim)
        self.enc_depth = int(enc_depth)
        self.dec_depth = int(dec_depth)
        self.enc_stage_depths_input = None if enc_stage_depths is None else tuple(int(x) for x in enc_stage_depths)
        self.dec_stage_depths_input = None if dec_stage_depths is None else tuple(int(x) for x in dec_stage_depths)
        self.temporal_stage_shuffle_input = (
            None if temporal_stage_shuffle is None else tuple(bool(x) for x in temporal_stage_shuffle)
        )
        self.num_heads_input = num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.rope_base = float(rope_base)
        self.qk_norm = bool(qk_norm)
        self.encoder_attention_mode = str(encoder_attention_mode).strip().lower()
        self.decoder_attention_mode = str(decoder_attention_mode).strip().lower()
        self.attention_backend = str(attention_backend).strip().lower()
        self.encoder_causal = bool(encoder_causal)
        self.decoder_causal = bool(decoder_causal)
        self.window_t = int(window_t)
        self.window_h = int(window_h)
        self.window_w = int(window_w)
        self.attn_block_size = int(attn_block_size)
        self.share_mask_across_batch_heads = bool(share_mask_across_batch_heads)
        self.repeat_first_frame_to_patch = bool(repeat_first_frame_to_patch)
        self.spatial_shuffle_factor = int(spatial_shuffle_factor)
        self.temporal_shuffle_factor = int(temporal_shuffle_factor)
        self.block_activation_checkpointing = bool(block_activation_checkpointing)
        if temporal_block_chunk_size is not None and int(temporal_block_chunk_size) <= 0:
            raise ValueError(f"temporal_block_chunk_size must be > 0 when set, got {temporal_block_chunk_size!r}")
        self.temporal_block_chunk_size = None if temporal_block_chunk_size is None else int(temporal_block_chunk_size)
        self.temporal_chunk_checkpointing = bool(temporal_chunk_checkpointing)
        if self.temporal_chunk_checkpointing and self.temporal_block_chunk_size is None:
            raise ValueError("temporal_chunk_checkpointing requires temporal_block_chunk_size to be set")
        self.patch_dim = int(self.in_channels * self.patch_t * self.patch_h * self.patch_w)
        self.token_limit_affects_encoding = False
        self.encode_decode_full_budget_only = True
        self.latent_token_axes = 3
        self.out_act_name = self._canonical_out_act(out_act)

        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.patch_t <= 0 or self.patch_h <= 0 or self.patch_w <= 0:
            raise ValueError("patch_shape entries must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.num_stages < 2:
            raise ValueError("ViViT2 requires at least 2 stages")
        if self.window_t < 0 or self.window_h < 0 or self.window_w < 0:
            raise ValueError("window_t/window_h/window_w must be >= 0")
        if self.attn_block_size <= 0:
            raise ValueError("attn_block_size must be > 0")
        if self.attention_backend not in _SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(f"Unsupported attention_backend: {attention_backend!r}")
        if self.attention_backend.startswith("natten_"):
            if self.encoder_attention_mode != "sparse_local" or self.decoder_attention_mode != "sparse_local":
                raise ValueError(
                    "NATTEN backends only support sparse_local attention_mode for both encoder and decoder, "
                    f"got encoder_attention_mode={self.encoder_attention_mode!r}, "
                    f"decoder_attention_mode={self.decoder_attention_mode!r}"
                )
        if self.spatial_shuffle_factor <= 0:
            raise ValueError("spatial_shuffle_factor must be >= 1")
        if self.temporal_shuffle_factor <= 0:
            raise ValueError("temporal_shuffle_factor must be >= 1")

        self.enc_stage_depths = tuple(self._resolve_stage_depths(self.enc_depth, self.enc_stage_depths_input, "enc_stage_depths"))
        self.dec_stage_depths = tuple(self._resolve_stage_depths(self.dec_depth, self.dec_stage_depths_input, "dec_stage_depths"))
        self.enc_depth = int(sum(self.enc_stage_depths))
        self.dec_depth = int(sum(self.dec_stage_depths))
        self.stage_num_heads = tuple(self._resolve_stage_num_heads(self.num_heads_input))
        self.num_heads = self.stage_num_heads if len(set(self.stage_num_heads)) > 1 else int(self.stage_num_heads[0])
        self.temporal_stage_shuffle = tuple(self._resolve_temporal_stage_shuffle(self.temporal_stage_shuffle_input))
        self.temporal_transition_factors = tuple(
            int(self.temporal_shuffle_factor) if enabled else 1 for enabled in self.temporal_stage_shuffle
        )
        self.first_frame_temporal_span = int(math.prod(self.temporal_transition_factors))
        self.first_frame_prefix_t = (
            int(self.patch_t * self.first_frame_temporal_span - 1) if self.repeat_first_frame_to_patch else 0
        )
        self.latent_layout = self._validate_latent_layout(latent_layout)
        self.latent_channels = int(self.latent_dim) if latent_channels is None else int(latent_channels)
        self.scale_factor_spatial = (
            int(self.patch_h * (self.spatial_shuffle_factor ** (self.num_stages - 1)))
            if scale_factor_spatial is None
            else int(scale_factor_spatial)
        )
        self.scale_factor_temporal = (
            int(self.patch_t * self.first_frame_temporal_span)
            if scale_factor_temporal is None
            else int(scale_factor_temporal)
        )
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        if self.num_stages == 2:
            self.enc_fine_depth = int(self.enc_stage_depths[0])
            self.enc_coarse_depth = int(self.enc_stage_depths[1])
            self.dec_fine_depth = int(self.dec_stage_depths[0])
            self.dec_coarse_depth = int(self.dec_stage_depths[1])

        nominal_stage_grids: tuple[GridShape3D, ...] = tuple((1, 1, 1) for _ in range(self.num_stages))
        if self.video_shape is not None:
            nominal_shapes = self._runtime_shapes_from_dims(*self.video_shape)
            nominal_stage_grids = nominal_shapes.stage_grids

        self.patch_embed = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=(self.patch_t, self.patch_h, self.patch_w),
            stride=(self.patch_t, self.patch_h, self.patch_w),
        )
        self.transition_channel_multiplier = int(
            self.temporal_shuffle_factor * self.spatial_shuffle_factor * self.spatial_shuffle_factor
        )
        self.transition_channel_multipliers = tuple(
            int(temporal_factor * self.spatial_shuffle_factor * self.spatial_shuffle_factor)
            for temporal_factor in self.temporal_transition_factors
        )

        self.enc_stage_blocks = nn.ModuleList(
            [
                self._build_blocks(
                    dim=int(self.stage_embed_dims[stage_idx]),
                    num_heads=int(self.stage_num_heads[stage_idx]),
                    depth=int(self.enc_stage_depths[stage_idx]),
                    nominal_grid=nominal_stage_grids[stage_idx],
                    attention_mode=self.encoder_attention_mode,
                    causal=self.encoder_causal,
                )
                for stage_idx in range(self.num_stages)
            ]
        )
        self.dec_stage_blocks = nn.ModuleList(
            [
                self._build_blocks(
                    dim=int(self.stage_embed_dims[stage_idx]),
                    num_heads=int(self.stage_num_heads[stage_idx]),
                    depth=int(self.dec_stage_depths[stage_idx]),
                    nominal_grid=nominal_stage_grids[stage_idx],
                    attention_mode=self.decoder_attention_mode,
                    causal=self.decoder_causal,
                )
                for stage_idx in range(self.num_stages)
            ]
        )
        self.downsample_projs = nn.ModuleList(
            [
                nn.Linear(
                    int(self.stage_embed_dims[stage_idx] * self.transition_channel_multipliers[stage_idx]),
                    int(self.stage_embed_dims[stage_idx + 1]),
                    bias=True,
                )
                for stage_idx in range(self.num_stages - 1)
            ]
        )
        self.upsample_projs = nn.ModuleList(
            [
                nn.Linear(
                    int(self.stage_embed_dims[stage_idx + 1]),
                    int(self.stage_embed_dims[stage_idx] * self.transition_channel_multipliers[stage_idx]),
                    bias=True,
                )
                for stage_idx in range(self.num_stages - 1)
            ]
        )

        self.enc_norm = nn.RMSNorm(self.deepest_embed_dim, eps=1e-6, elementwise_affine=True)
        self.fc_mu = nn.Linear(self.deepest_embed_dim, self.latent_dim, bias=True)
        self.fc_logvar = nn.Linear(self.deepest_embed_dim, self.latent_dim, bias=True)

        self.latent_proj = nn.Linear(self.latent_dim, self.deepest_embed_dim, bias=True)
        self.dec_norm = nn.RMSNorm(self.embed_dim, eps=1e-6, elementwise_affine=True)
        self.to_patch = nn.Linear(self.embed_dim, self.patch_dim, bias=True)
        self.out_act = self._build_out_act(self.out_act_name)

        self._position_embeddings_by_key: dict[tuple[int, str, str, int, int, int], PositionEmbeddings3D] = {}
        self._init_weights()

    @staticmethod
    def _split_stage_depths(total_depth: int, num_stages: int) -> list[int]:
        base = int(total_depth // num_stages)
        out = [base for _ in range(num_stages)]
        remainder = int(total_depth % num_stages)
        for idx in range(remainder):
            out[num_stages - 1 - idx] += 1
        return out

    def _resolve_stage_depths(
        self,
        total_depth: int,
        explicit_stage_depths: tuple[int, ...] | None,
        field_name: str,
    ) -> list[int]:
        if explicit_stage_depths is not None:
            if len(explicit_stage_depths) != self.num_stages:
                raise ValueError(
                    f"{field_name} length must match number of stages, "
                    f"got {explicit_stage_depths}, stages={self.num_stages}"
                )
            if any(int(x) <= 0 for x in explicit_stage_depths):
                raise ValueError(f"{field_name} entries must be >= 1, got {explicit_stage_depths}")
            return [int(x) for x in explicit_stage_depths]
        if total_depth < self.num_stages:
            raise ValueError(
                "enc_depth/dec_depth must be >= number of stages, "
                f"got total_depth={total_depth}, stages={self.num_stages}"
            )
        return self._split_stage_depths(total_depth, self.num_stages)

    def _resolve_stage_num_heads(self, num_heads: int | tuple[int, ...] | list[int]) -> list[int]:
        if isinstance(num_heads, tuple):
            stage_num_heads = [int(x) for x in num_heads]
        elif isinstance(num_heads, list):
            stage_num_heads = [int(x) for x in num_heads]
        else:
            stage_num_heads = [int(num_heads) for _ in range(self.num_stages)]
        if len(stage_num_heads) != self.num_stages:
            raise ValueError(
                "num_heads length must match number of stages, "
                f"got num_heads={num_heads}, stages={self.num_stages}"
            )
        if any(int(x) <= 0 for x in stage_num_heads):
            raise ValueError(f"num_heads entries must be >= 1, got {stage_num_heads}")
        for idx, dim in enumerate(self.stage_embed_dims):
            if dim % int(stage_num_heads[idx]) != 0:
                raise ValueError(
                    "each stage hidden size must be divisible by num_heads, "
                    f"got stage_embed_dims[{idx}]={dim}, num_heads[{idx}]={stage_num_heads[idx]}"
                )
        return stage_num_heads

    def _resolve_temporal_stage_shuffle(
        self,
        explicit_temporal_stage_shuffle: tuple[bool, ...] | None,
    ) -> list[bool]:
        num_transitions = int(self.num_stages - 1)
        if explicit_temporal_stage_shuffle is not None:
            if len(explicit_temporal_stage_shuffle) != num_transitions:
                raise ValueError(
                    "temporal_stage_shuffle length must match number of stage transitions, "
                    f"got {explicit_temporal_stage_shuffle}, transitions={num_transitions}"
                )
            return [bool(x) for x in explicit_temporal_stage_shuffle]
        return [True for _ in range(num_transitions)]

    def _build_blocks(
        self,
        *,
        dim: int,
        num_heads: int,
        depth: int,
        nominal_grid: GridShape3D,
        attention_mode: str,
        causal: bool,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                TransformerFlexBlock3D(
                    dim,
                    int(num_heads),
                    grid_t=int(nominal_grid[0]),
                    grid_h=int(nominal_grid[1]),
                    grid_w=int(nominal_grid[2]),
                    attention_mode=attention_mode,
                    causal=causal,
                    window_t=self.window_t,
                    window_h=self.window_h,
                    window_w=self.window_w,
                    block_size=self.attn_block_size,
                    mlp_ratio=self.mlp_ratio,
                    rope_base=self.rope_base,
                    qk_norm=self.qk_norm,
                    share_mask_across_batch_heads=self.share_mask_across_batch_heads,
                    attention_backend=self.attention_backend,
                    temporal_block_chunk_size=self.temporal_block_chunk_size,
                    temporal_chunk_checkpointing=self.temporal_chunk_checkpointing,
                )
                for _ in range(int(depth))
            ]
        )

    @staticmethod
    def _canonical_out_act(out_act: str | None) -> str | None:
        if out_act is None:
            return None
        s = str(out_act).lower().strip()
        if s in {"", "none", "identity"}:
            return None
        if s == "sigmoid":
            return "sigmoid"
        raise ValueError(f"Unsupported out_act: {out_act!r} (supported: None/identity/sigmoid)")

    @staticmethod
    def _build_out_act(out_act: str | None) -> nn.Module:
        s = ViViT2HF._canonical_out_act(out_act)
        if s is None:
            return nn.Identity()
        if s == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported out_act: {out_act!r} (supported: None/identity/sigmoid)")

    @staticmethod
    def _validate_latent_layout(latent_layout: str | None) -> str:
        layout = "bcthw" if latent_layout is None else str(latent_layout).strip().lower()
        if layout != "bcthw":
            raise ValueError(f"latent_layout must be 'bcthw' for the HF-facing ViViT2 API, got {latent_layout!r}")
        return layout

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _runtime_shapes_from_dims(self, t: int, h: int, w: int) -> _RuntimeShapes:
        input_t = int(t)
        input_h = int(h)
        input_w = int(w)
        if input_t <= 0 or input_h <= 0 or input_w <= 0:
            raise ValueError(f"input video dims must be positive, got {(input_t, input_h, input_w)!r}")
        if input_h % self.patch_h != 0:
            raise ValueError(f"input H({input_h}) must be divisible by patch_h({self.patch_h})")
        if input_w % self.patch_w != 0:
            raise ValueError(f"input W({input_w}) must be divisible by patch_w({self.patch_w})")
        model_t = int(input_t + self.first_frame_prefix_t)
        if model_t % self.patch_t != 0:
            if self.repeat_first_frame_to_patch:
                raise ValueError(
                    "input T must satisfy repeat_first_frame_to_patch alignment for the configured temporal hierarchy, "
                    f"got T={input_t}, patch_t={self.patch_t}, first_frame_prefix_t={self.first_frame_prefix_t}"
                )
            raise ValueError(f"input T({input_t}) must be divisible by patch_t({self.patch_t})")
        fine_t = int(model_t // self.patch_t)
        fine_h = int(input_h // self.patch_h)
        fine_w = int(input_w // self.patch_w)

        stage_grids: list[GridShape3D] = []
        stage_t = int(fine_t)
        stage_h = int(fine_h)
        stage_w = int(fine_w)
        for stage_idx in range(self.num_stages):
            stage_grids.append((int(stage_t), int(stage_h), int(stage_w)))
            if stage_idx + 1 >= self.num_stages:
                continue
            if (
                stage_t % self.temporal_transition_factors[stage_idx] != 0
                or stage_h % self.spatial_shuffle_factor != 0
                or stage_w % self.spatial_shuffle_factor != 0
            ):
                raise ValueError(
                    "fine patch grid must be divisible by the full stage hierarchy, "
                    f"got stage={stage_idx}, grid={(stage_t, stage_h, stage_w)!r}, "
                    f"temporal_shuffle_factor={self.temporal_transition_factors[stage_idx]}, "
                    f"spatial_shuffle_factor={self.spatial_shuffle_factor}, stages={self.num_stages}"
                )
            stage_t = int(stage_t // self.temporal_transition_factors[stage_idx])
            stage_h = int(stage_h // self.spatial_shuffle_factor)
            stage_w = int(stage_w // self.spatial_shuffle_factor)
        return _RuntimeShapes(
            input_t=input_t,
            input_h=input_h,
            input_w=input_w,
            model_t=model_t,
            stage_grids=tuple(stage_grids),
        )

    def _prepend_first_frame_prefix(self, x: torch.Tensor) -> torch.Tensor:
        if self.first_frame_prefix_t <= 0:
            return x
        prefix = x[:, :1].expand(-1, int(self.first_frame_prefix_t), -1, -1, -1)
        return torch.cat([prefix, x], dim=1)

    def _rope_for_stage(self, stage_idx: int) -> nn.Module:
        return self.enc_stage_blocks[int(stage_idx)][0].attn.rope

    def _position_embeddings_for(
        self,
        *,
        stage_idx: int,
        grid_shape: GridShape3D,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PositionEmbeddings3D:
        key = (int(stage_idx), str(device), str(dtype), int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
        cached = self._position_embeddings_by_key.get(key, None)
        if cached is None:
            gt, gh, gw = (int(v) for v in grid_shape)
            mesh_t, mesh_h, mesh_w = torch.meshgrid(
                torch.arange(gt, dtype=torch.float32),
                torch.arange(gh, dtype=torch.float32),
                torch.arange(gw, dtype=torch.float32),
                indexing="ij",
            )
            rope = self._rope_for_stage(stage_idx)
            cached = rope.build_position_embeddings(
                pos_t=mesh_t.reshape(-1).to(device=device),
                pos_h=mesh_h.reshape(-1).to(device=device),
                pos_w=mesh_w.reshape(-1).to(device=device),
                dtype=dtype,
                device=device,
            )
            self._position_embeddings_by_key[key] = cached
        return cached

    def warmup_runtime_caches(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.video_shape is None:
            return
        shapes = self._runtime_shapes_from_dims(*self.video_shape)
        for stage_idx, grid_shape in enumerate(shapes.stage_grids):
            _ = self._position_embeddings_for(stage_idx=stage_idx, grid_shape=grid_shape, device=device, dtype=dtype)
            for blk in self.enc_stage_blocks[stage_idx]:
                _ = blk.attn._block_mask_for_device(device, grid_shape=grid_shape)
            for blk in self.dec_stage_blocks[stage_idx]:
                _ = blk.attn._block_mask_for_device(device, grid_shape=grid_shape)

    @staticmethod
    def _grid_to_tokens(x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        return x.view(int(b), int(t) * int(h) * int(w), int(c))

    @staticmethod
    def _tokens_to_grid(x: torch.Tensor, *, grid_shape: GridShape3D) -> torch.Tensor:
        b, n, c = x.shape
        gt, gh, gw = (int(v) for v in grid_shape)
        if int(n) != int(gt * gh * gw):
            raise ValueError(f"token count mismatch: got {n}, expected {gt * gh * gw}")
        return x.view(int(b), int(gt), int(gh), int(gw), int(c))

    def _patchify(self, x: torch.Tensor, *, shapes: _RuntimeShapes) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must have shape (B,T,C,H,W), got {tuple(x.shape)}")
        b, _, c, h, w = x.shape
        if int(c) != self.in_channels:
            raise ValueError(f"input channels mismatch: got {c}, expected {self.in_channels}")
        if int(h) != shapes.input_h or int(w) != shapes.input_w:
            raise ValueError(
                f"runtime shape mismatch: got H/W={(int(h), int(w))}, expected {(shapes.input_h, shapes.input_w)}"
            )
        x_aug = self._prepend_first_frame_prefix(x)
        x_bcthw = x_aug.permute(0, 2, 1, 3, 4).contiguous()
        patches = self._modules["patch_embed"](x_bcthw)
        return patches.permute(0, 2, 3, 4, 1).contiguous().view(
            int(b),
            int(shapes.fine_t),
            int(shapes.fine_h),
            int(shapes.fine_w),
            int(self.embed_dim),
        )

    def _unpatchify(self, patches: torch.Tensor, *, shapes: _RuntimeShapes) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"patches must have shape (B,T,H,W,patch_dim), got {tuple(patches.shape)}")
        b, fine_t, fine_h, fine_w, patch_dim = patches.shape
        if int(fine_t) != int(shapes.fine_t) or int(fine_h) != int(shapes.fine_h) or int(fine_w) != int(shapes.fine_w):
            raise ValueError(
                "patch grid mismatch: "
                f"got {(int(fine_t), int(fine_h), int(fine_w))}, expected {shapes.fine_grid}"
            )
        if int(patch_dim) != int(self.patch_dim):
            raise ValueError(f"patch dim mismatch: got {patch_dim}, expected {self.patch_dim}")
        x = patches.view(
            int(b),
            int(shapes.fine_t),
            int(shapes.fine_h),
            int(shapes.fine_w),
            int(self.patch_t),
            int(self.patch_h),
            int(self.patch_w),
            int(self.in_channels),
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(
            int(b),
            int(shapes.model_t),
            int(shapes.input_h),
            int(shapes.input_w),
            int(self.in_channels),
        )
        if self.first_frame_prefix_t > 0:
            x = x[:, int(self.first_frame_prefix_t) :, :, :, :]
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def _hierarchical_unshuffle_hidden(
        self,
        x: torch.Tensor,
        *,
        temporal_factor: int | None = None,
        spatial_factor: int | None = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must have shape (B,T,H,W,C), got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        rt = int(self.temporal_shuffle_factor if temporal_factor is None else temporal_factor)
        rh = int(self.spatial_shuffle_factor if spatial_factor is None else spatial_factor)
        rw = int(rh)
        if t % rt != 0 or h % rh != 0 or w % rw != 0:
            raise ValueError(
                "hidden grid must be divisible by shuffle factors, "
                f"got {(t, h, w)}, temporal_shuffle_factor={rt}, spatial_shuffle_factor={rh}"
            )
        x = x.view(int(b), int(t // rt), int(rt), int(h // rh), int(rh), int(w // rw), int(rw), int(c))
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(int(b), int(t // rt), int(h // rh), int(w // rw), int(c * rt * rh * rw))

    def _hierarchical_shuffle_hidden(
        self,
        x: torch.Tensor,
        *,
        temporal_factor: int | None = None,
        spatial_factor: int | None = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must have shape (B,T,H,W,C), got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        rt = int(self.temporal_shuffle_factor if temporal_factor is None else temporal_factor)
        rh = int(self.spatial_shuffle_factor if spatial_factor is None else spatial_factor)
        rw = int(rh)
        factor = int(rt * rh * rw)
        if c % factor != 0:
            raise ValueError(
                "hidden channel dim must be divisible by shuffle volume, "
                f"got C={c}, temporal_shuffle_factor={rt}, spatial_shuffle_factor={rh}"
            )
        base_c = int(c // factor)
        x = x.view(int(b), int(t), int(h), int(w), int(rt), int(rh), int(rw), int(base_c))
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(int(b), int(t * rt), int(h * rh), int(w * rw), int(base_c))

    def _spatial_unshuffle_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self._hierarchical_unshuffle_hidden(x, temporal_factor=1, spatial_factor=self.spatial_shuffle_factor)

    def _spatial_shuffle_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self._hierarchical_shuffle_hidden(x, temporal_factor=1, spatial_factor=self.spatial_shuffle_factor)

    def _run_blocks(
        self,
        x: torch.Tensor,
        *,
        blocks: nn.ModuleList,
        stage_idx: int,
        grid_shape: GridShape3D,
    ) -> torch.Tensor:
        position_embeddings = self._position_embeddings_for(
            stage_idx=stage_idx,
            grid_shape=grid_shape,
            device=x.device,
            dtype=x.dtype,
        )
        for blk in blocks:
            if (
                self.block_activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and not blk.uses_temporal_chunk_checkpointing(grid_shape)
            ):
                x = checkpoint(
                    lambda inp, block=blk: block(
                        inp,
                        position_embeddings=position_embeddings,
                        grid_shape=grid_shape,
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = blk(x, position_embeddings=position_embeddings, grid_shape=grid_shape)
        return x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _prepare_encode_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 4:
            return x[:, None, ...], True
        if x.ndim == 5:
            return x, False
        raise ValueError(f"x must have shape (B,C,H,W) or (B,T,C,H,W), got {tuple(x.shape)}")

    @staticmethod
    def _restore_encoded_tensor(x: torch.Tensor, *, squeeze_time: bool) -> torch.Tensor:
        if not squeeze_time:
            return x
        if x.ndim != 5 or int(x.shape[1]) != 1:
            raise ValueError(f"single-frame image encode expected latent shape (B,1,H,W,C), got {tuple(x.shape)}")
        return x[:, 0].contiguous()

    def _prepare_decode_input(self, z: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if z.ndim == 4:
            return z[:, None, ...], True
        if z.ndim == 5:
            return z, False
        raise ValueError(f"z must have shape (B,H,W,latent_dim) or (B,T,H,W,latent_dim), got {tuple(z.shape)}")

    @staticmethod
    def _restore_decoded_tensor(x: torch.Tensor, *, squeeze_time: bool) -> torch.Tensor:
        if not squeeze_time:
            return x
        if x.ndim != 5 or int(x.shape[1]) != 1:
            raise ValueError(f"single-frame image decode expected output shape (B,1,C,H,W), got {tuple(x.shape)}")
        return x[:, 0].contiguous()

    def _encode_native(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, squeeze_time = self._prepare_encode_input(x)
        shapes = self._runtime_shapes_from_dims(int(x.shape[1]), int(x.shape[3]), int(x.shape[4]))
        stage_hidden = self._patchify(x, shapes=shapes)
        for stage_idx in range(self.num_stages):
            stage_tokens = self._grid_to_tokens(stage_hidden)
            stage_tokens = self._run_blocks(
                stage_tokens,
                blocks=self.enc_stage_blocks[stage_idx],
                stage_idx=stage_idx,
                grid_shape=shapes.stage_grids[stage_idx],
            )
            stage_hidden = self._tokens_to_grid(stage_tokens, grid_shape=shapes.stage_grids[stage_idx])
            if stage_idx + 1 < self.num_stages:
                stage_hidden = self._hierarchical_unshuffle_hidden(
                    stage_hidden,
                    temporal_factor=self.temporal_transition_factors[stage_idx],
                )
                stage_hidden = self.downsample_projs[stage_idx](stage_hidden)

        deepest_tokens = rms_norm_preserve_input_dtype(self._grid_to_tokens(stage_hidden), self.enc_norm)
        deepest_hidden = self._tokens_to_grid(deepest_tokens, grid_shape=shapes.coarse_grid)
        mu = self.fc_mu(deepest_hidden)
        logvar = self.fc_logvar(deepest_hidden)
        z = self.reparameterize(mu, logvar)
        return (
            self._restore_encoded_tensor(z, squeeze_time=squeeze_time),
            self._restore_encoded_tensor(mu, squeeze_time=squeeze_time),
            self._restore_encoded_tensor(logvar, squeeze_time=squeeze_time),
        )

    def _decode_native(self, z: torch.Tensor) -> torch.Tensor:
        z, squeeze_time = self._prepare_decode_input(z)
        _, coarse_t, coarse_h, coarse_w, latent_dim = z.shape
        if int(latent_dim) != int(self.latent_dim):
            raise ValueError(f"latent dim mismatch: got {latent_dim}, expected {self.latent_dim}")

        fine_t = int(coarse_t)
        fine_h = int(coarse_h)
        fine_w = int(coarse_w)
        for temporal_factor in self.temporal_transition_factors:
            fine_t *= int(temporal_factor)
            fine_h *= int(self.spatial_shuffle_factor)
            fine_w *= int(self.spatial_shuffle_factor)
        input_t = int(fine_t * int(self.patch_t) - int(self.first_frame_prefix_t))
        input_h = int(fine_h * self.patch_h)
        input_w = int(fine_w * self.patch_w)
        shapes = self._runtime_shapes_from_dims(input_t, input_h, input_w)

        stage_hidden = self.latent_proj(z)
        for stage_idx in reversed(range(self.num_stages)):
            stage_tokens = self._grid_to_tokens(stage_hidden)
            stage_tokens = self._run_blocks(
                stage_tokens,
                blocks=self.dec_stage_blocks[stage_idx],
                stage_idx=stage_idx,
                grid_shape=shapes.stage_grids[stage_idx],
            )
            stage_hidden = self._tokens_to_grid(stage_tokens, grid_shape=shapes.stage_grids[stage_idx])
            if stage_idx > 0:
                stage_hidden = self.upsample_projs[stage_idx - 1](stage_hidden)
                stage_hidden = self._hierarchical_shuffle_hidden(
                    stage_hidden,
                    temporal_factor=self.temporal_transition_factors[stage_idx - 1],
                )

        fine_tokens = rms_norm_preserve_input_dtype(self._grid_to_tokens(stage_hidden), self.dec_norm)
        patch_grid = self.to_patch(fine_tokens).view(
            int(z.shape[0]),
            int(shapes.fine_t),
            int(shapes.fine_h),
            int(shapes.fine_w),
            int(self.patch_dim),
        )
        x_hat = self._unpatchify(patch_grid, shapes=shapes)
        x_hat = self._modules["out_act"](x_hat)
        return self._restore_decoded_tensor(x_hat, squeeze_time=squeeze_time)

    def _forward_native(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z, mu, logvar = self._encode_native(x)
        x_hat = self._decode_native(z)
        return {
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "ViViT2HF":
        return ModelMixin.from_pretrained.__func__(cls, pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        ModelMixin.save_pretrained(self, save_directory, **kwargs)

    @classmethod
    def from_vivit2(cls, model: Any) -> "ViViT2HF":
        hf_model = cls(**_vivit2_hf_kwargs_from_model(model))
        msg = hf_model.load_state_dict(model.state_dict(), strict=True)
        missing_keys = list(getattr(msg, "missing_keys", []))
        unexpected_keys = list(getattr(msg, "unexpected_keys", []))
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Unexpected ViViT2 -> ViViT2HF state_dict load result: "
                f"missing_keys={missing_keys!r} unexpected_keys={unexpected_keys!r}"
            )
        hf_model.train(model.training)
        return hf_model

    @staticmethod
    def _to_native_video_layout(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.contiguous()
        if x.ndim != 5:
            raise ValueError(f"x must have shape (B,C,H,W) or (B,C,T,H,W), got {tuple(x.shape)}")
        return x.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _to_hf_video_layout(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.contiguous()
        if x.ndim != 5:
            raise ValueError(f"x must have shape (B,C,H,W) or (B,T,C,H,W), got {tuple(x.shape)}")
        return x.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _to_native_latent_layout(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            return z.permute(0, 2, 3, 1).contiguous()
        if z.ndim != 5:
            raise ValueError(f"z must have shape (B,C,H,W) or (B,C,T,H,W), got {tuple(z.shape)}")
        return z.permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def _to_hf_latent_layout(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            return z.permute(0, 3, 1, 2).contiguous()
        if z.ndim != 5:
            raise ValueError(f"latent must have shape (B,H,W,C) or (B,T,H,W,C), got {tuple(z.shape)}")
        return z.permute(0, 4, 1, 2, 3).contiguous()

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        x_native = self._to_native_video_layout(x)
        _, mu_native, logvar_native = self._encode_native(x_native)
        mu = self._to_hf_latent_layout(mu_native)
        logvar = self._to_hf_latent_layout(logvar_native)
        posterior = DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        z_native = self._to_native_latent_layout(z)
        decoded_native = self._decode_native(z_native)
        decoded = self._to_hf_video_layout(decoded_native)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample, return_dict=True).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)


def _vivit2_hf_kwargs_from_model(model: Any) -> dict[str, Any]:
    latent_channels = int(getattr(model, "latent_channels", int(model.latent_dim)))
    default_scale_factor_spatial = int(model.patch_h * (model.spatial_shuffle_factor ** (model.num_stages - 1)))
    default_scale_factor_temporal = int(model.patch_t * model.first_frame_temporal_span)
    scale_factor_spatial = int(
        getattr(model, "scale_factor_spatial", default_scale_factor_spatial)
    )
    scale_factor_temporal = int(getattr(model, "scale_factor_temporal", default_scale_factor_temporal))
    latent_layout = str(getattr(model, "latent_layout", "bcthw"))
    return {
        "in_channels": int(model.in_channels),
        "video_shape": None if model.video_shape is None else [int(v) for v in model.video_shape],
        "patch_shape": [int(model.patch_t), int(model.patch_h), int(model.patch_w)],
        "embed_dim": int(model.embed_dim),
        "stage_embed_dims": [int(v) for v in model.stage_embed_dims],
        "latent_dim": int(model.latent_dim),
        "enc_depth": int(model.enc_depth),
        "dec_depth": int(model.dec_depth),
        "enc_stage_depths": [int(v) for v in model.enc_stage_depths],
        "dec_stage_depths": [int(v) for v in model.dec_stage_depths],
        "num_heads": [int(v) for v in model.stage_num_heads],
        "mlp_ratio": float(model.mlp_ratio),
        "rope_base": float(model.rope_base),
        "qk_norm": bool(model.qk_norm),
        "encoder_attention_mode": str(model.encoder_attention_mode),
        "decoder_attention_mode": str(model.decoder_attention_mode),
        "attention_backend": str(model.attention_backend),
        "encoder_causal": bool(model.encoder_causal),
        "decoder_causal": bool(model.decoder_causal),
        "window_t": int(model.window_t),
        "window_h": int(model.window_h),
        "window_w": int(model.window_w),
        "attn_block_size": int(model.attn_block_size),
        "share_mask_across_batch_heads": bool(model.share_mask_across_batch_heads),
        "repeat_first_frame_to_patch": bool(model.repeat_first_frame_to_patch),
        "spatial_shuffle_factor": int(model.spatial_shuffle_factor),
        "temporal_shuffle_factor": int(model.temporal_shuffle_factor),
        "temporal_stage_shuffle": [bool(v) for v in model.temporal_stage_shuffle],
        "block_activation_checkpointing": bool(model.block_activation_checkpointing),
        "temporal_block_chunk_size": (
            None if model.temporal_block_chunk_size is None else int(model.temporal_block_chunk_size)
        ),
        "temporal_chunk_checkpointing": bool(model.temporal_chunk_checkpointing),
        "out_act": model.out_act_name,
        "latent_channels": latent_channels,
        "scale_factor_spatial": scale_factor_spatial,
        "scale_factor_temporal": scale_factor_temporal,
        "latent_layout": latent_layout,
    }
