import logging
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather as dist_all_gather
from typing import Any, Dict, List, Optional, Tuple, Union


def _normalize_to_list(val, n: int, name: str, allow_none: bool = True):
    """Normalize a scalar, None, or list to a list of length *n*."""
    if val is None:
        return [None] * n if allow_none else [0] * n
    if isinstance(val, (list, tuple)):
        if len(val) != n:
            raise ValueError(
                f"{name} list length {len(val)} != segment_count_list length {n}"
            )
        return list(val)
    return [val] * n


CONTRASTIVE_MODULE_PRESETS = {
    "tiny":   {"dim": 128, "nhead": 8,  "layers": 1, "embed_dim": 128, "hidden_dim": 256},
    "small":  {"dim": 256, "nhead": 8,  "layers": 1, "embed_dim": 256, "hidden_dim": 512},
    "medium": {"dim": 384, "nhead": 12, "layers": 1, "embed_dim": 384, "hidden_dim": 768},
    "large":  {"dim": 512, "nhead": 8,  "layers": 1, "embed_dim": 512, "hidden_dim": 1024},
    "huge":   {"dim": 768, "nhead": 12, "layers": 1, "embed_dim": 768, "hidden_dim": 3072},
}

CNN_INCREASE_PRESETS = {
    "small":  {"num_blocks_per_stage": 1, "kernel_size": 5},
    "medium": {"num_blocks_per_stage": 1, "kernel_size": 7},
    "large":  {"num_blocks_per_stage": 1, "kernel_size": 7},
    "huge":   {"num_blocks_per_stage": 2, "kernel_size": 7},
}


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None or hidden_dim <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttentionPool(nn.Module):
    """Learnable spatial attention pooling over (H, W) positions."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, HW, D)
        Returns:
            (N, D)
        """
        weights = self.attn(self.norm(x)).softmax(dim=1)  # (N, HW, 1)
        return (x * weights).sum(dim=1)                    # (N, D)


class SDPASelfAttention(nn.Module):
    """Self-attention using ``F.scaled_dot_product_attention``.

    Parameter names (``in_proj_weight``, ``in_proj_bias``, ``out_proj``)
    mirror ``nn.MultiheadAttention`` so that state dicts are interchangeable.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.qk_norm = qk_norm

        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.reshape(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, N, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, N, self.d_model)
        return self.out_proj(out)


class SDPAEncoderLayer(nn.Module):
    """Drop-in replacement for ``nn.TransformerEncoderLayer(norm_first=True)``
    that explicitly calls ``F.scaled_dot_product_attention``.

    State-dict keys are identical to the PyTorch original so existing
    checkpoints load without any key mapping.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.self_attn = SDPASelfAttention(d_model, nhead, dropout, qk_norm=qk_norm)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        src = src + self._sa_block(self.norm1(src), src_key_padding_mask)
        src = src + self._ff_block(self.norm2(src))
        return src

    def _sa_block(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.dropout(self.self_attn(x, key_padding_mask=key_padding_mask))

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class CLSPoolTransformerLayer(nn.Module):
    """CLS-token based Transformer aggregation: (B, N, D_in) -> (B, d_model).

    Optionally projects from *input_dim* to *d_model* before the transformer.
    Prepends a learnable CLS token, applies one or more TransformerEncoderLayers,
    and returns the CLS token output.  Optionally adds positional embeddings
    (useful for ordered sequences like temporal aggregation).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 12,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        add_pos_emb: bool = False,
        pos_max_len: int = 128,
        input_dim: Optional[int] = None,
        num_layers: int = 1,
        use_sdpa: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        if input_dim is not None and input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if use_sdpa or qk_norm:
            encoder_layer = SDPAEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                layer_norm_eps=1e-6,
                qk_norm=qk_norm,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                layer_norm_eps=1e-6,
            )
        if num_layers == 1:
            self.encoder = encoder_layer
        else:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.num_layers = num_layers

        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_max_len = 1 + pos_max_len  # +1 for CLS
            self.pos_emb = nn.Parameter(torch.zeros(1, self.pos_max_len, d_model))
            self.pos_drop = nn.Dropout(dropout)
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D_in)
            key_padding_mask: (B, N) bool — True for positions to IGNORE.
        Returns:
            (B, d_model)
        """
        if self.input_proj is not None:
            x = self.input_proj(x)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, d_model)

        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        if self.add_pos_emb:
            seq_len = x.shape[1]
            assert seq_len <= self.pos_max_len, (
                f"Sequence length ({seq_len}) exceeds pos_max_len ({self.pos_max_len})"
            )
            x = x + self.pos_emb[:, :seq_len, :]
            x = self.pos_drop(x)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x[:, 0, :]  # CLS token output


class ConvNeXt1dBlock(nn.Module):
    """ConvNeXt-style 1-D block: depthwise conv -> inverted bottleneck -> residual."""

    def __init__(self, dim: int, kernel_size: int = 7, expand_ratio: int = 4):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = nn.LayerNorm(dim)
        hidden = dim * expand_ratio
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        residual = x
        x = self.dwconv(x)                  # (B, C, T)
        x = x.transpose(1, 2)               # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)               # (B, C, T)
        return x + residual


class Conv1dTemporalPool(nn.Module):
    """1-D temporal pooling via ConvNeXt blocks + adaptive average pool.

    Optionally projects input channels to an internal dimension, applies
    ``num_blocks`` ConvNeXt-1D blocks, then adaptive-avg-pools to the
    desired number of output time-steps.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.output_dim = output_dim

        if input_dim != output_dim:
            self.input_proj = nn.Conv1d(input_dim, output_dim, 1)
        else:
            self.input_proj = None

        self.blocks = nn.Sequential(
            *[ConvNeXt1dBlock(output_dim, kernel_size=kernel_size) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T)
            target_len: desired output temporal length S.
        Returns:
            (B, T_out, output_dim) where T_out == target_len.
        """
        if self.input_proj is not None:
            x = self.input_proj(x)           # (B, output_dim, T)
        x = self.blocks(x)                   # (B, output_dim, T)
        if x.shape[-1] != target_len:
            x = F.adaptive_avg_pool1d(x, target_len)
        x = x.transpose(1, 2)               # (B, S, output_dim)
        x = self.norm(x)
        return x


class GradualConv1dTemporalPool(nn.Module):
    """Progressive channel expansion with wide-kernel temporal convolutions.

    Doubles channels at each stage (e.g. 128->256->512) using kernel_size>1
    convolutions so that channel widening co-occurs with local temporal
    context mixing.  Each stage: Conv1d(k) -> GELU -> ConvNeXt1dBlock(s).
    When ``input_dim >= output_dim`` this degrades to a single-stage module.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks_per_stage: int = 1,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.output_dim = output_dim

        dims = [input_dim]
        d = input_dim
        while d < output_dim:
            d = min(d * 2, output_dim)
            dims.append(d)
        if dims[-1] != output_dim:
            dims.append(output_dim)

        stages: list = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            stages.append(nn.Conv1d(in_d, out_d, kernel_size, padding=kernel_size // 2))
            stages.append(nn.GELU())
            for _ in range(num_blocks_per_stage):
                stages.append(ConvNeXt1dBlock(out_d, kernel_size=kernel_size))
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T)
            target_len: desired output temporal length S.
        Returns:
            (B, T_out, output_dim) where T_out == target_len.
        """
        x = self.stages(x)                  # (B, output_dim, T)
        if x.shape[-1] != target_len:
            x = F.adaptive_avg_pool1d(x, target_len)
        x = x.transpose(1, 2)               # (B, S, output_dim)
        x = self.norm(x)
        return x


class LatentAVContrastiveHead(nn.Module):
    """
    Contrastive head on top of VAE latents.

    Segment-level features are built from the video latent temporal axis, while
    the audio latent is adaptively pooled to the same number of temporal segments.
    """

    VALID_SPATIAL_POOL_MODES = ("mean", "max", "attention_pool", "transformer")

    VALID_TEMPORAL_POOL_MODES = ("mean", "transformer", "conv", "cnn_increase")
    VALID_GLOBAL_TEMPORAL_POOL_MODES = ("mean", "transformer")

    def __init__(
        self,
        video_latent_dim: int,
        audio_latent_dim: int,
        embed_dim: int = 512,
        segment_hidden_dim: Optional[int] = None,
        global_hidden_dim: Optional[int] = None,
        segment_count: Union[None, int, List[Optional[int]]] = None,
        use_segment_loss: bool = True,
        use_global_loss: bool = False,
        init_scale: float = 0.07,
        clamp_scale_min: float = 0.001,
        clamp_scale_max: float = 0.5,
        gather_for_loss: bool = True,
        num_negatives: Union[None, int, List[Optional[int]]] = None,
        num_negative_videos: Union[None, int, List[Optional[int]]] = None,
        same_long_video_priority: bool = False,
        same_long_video_num_negatives: Union[None, int, List[Optional[int]]] = None,
        num_negatives_with_sibling: Union[None, int, List[Optional[int]]] = None,
        num_negatives_no_sibling: Union[None, int, List[Optional[int]]] = None,
        spatial_pool_mode: str = "mean",
        segment_temporal_pool_mode: str = "mean",
        global_temporal_pool_mode: str = "mean",
        skip_first_video_latent_frame: bool = False,
        video_temporal_compress_factor: int = 4,
        transformer_dim: Optional[int] = None,
        transformer_nhead: int = 12,
        transformer_layers: int = 1,
        spatial_transformer_layers: Optional[int] = None,
        segment_transformer_layers: Optional[int] = None,
        global_transformer_layers: Optional[int] = None,
        spatial_merge_factor: int = 2,
        spatial_module_size: Optional[str] = None,
        segment_module_size: Optional[str] = None,
        global_module_size: Optional[str] = None,
        cnn_num_blocks_per_stage: Optional[int] = None,
        cnn_kernel_size: Optional[int] = None,
        use_sdpa: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.qk_norm = qk_norm
        if not use_segment_loss and not use_global_loss:
            raise ValueError("At least one of use_segment_loss or use_global_loss must be True.")
        if spatial_pool_mode not in self.VALID_SPATIAL_POOL_MODES:
            raise ValueError(
                f"spatial_pool_mode must be one of {self.VALID_SPATIAL_POOL_MODES}, "
                f"got '{spatial_pool_mode}'."
            )
        if segment_temporal_pool_mode not in self.VALID_TEMPORAL_POOL_MODES:
            raise ValueError(
                f"segment_temporal_pool_mode must be one of {self.VALID_TEMPORAL_POOL_MODES}, "
                f"got '{segment_temporal_pool_mode}'."
            )
        if global_temporal_pool_mode not in self.VALID_GLOBAL_TEMPORAL_POOL_MODES:
            raise ValueError(
                f"global_temporal_pool_mode must be one of {self.VALID_GLOBAL_TEMPORAL_POOL_MODES}, "
                f"got '{global_temporal_pool_mode}'."
            )

        # --- Normalize multi-granularity list params ---
        if isinstance(segment_count, (list, tuple)):
            self.segment_count_list: List[Optional[int]] = list(segment_count)
        else:
            self.segment_count_list = [segment_count]
        n_gran = len(self.segment_count_list)
        self.num_negatives_list: List[Optional[int]] = _normalize_to_list(
            num_negatives, n_gran, "num_negatives",
        )
        self.num_negative_videos_list: List[Optional[int]] = _normalize_to_list(
            num_negative_videos, n_gran, "num_negative_videos",
        )
        # Sibling-aware negative sampling (per-granularity lists / broadcast scalar).
        self.same_long_video_priority: bool = bool(same_long_video_priority)
        self.same_long_video_num_negatives_list: List[Optional[int]] = _normalize_to_list(
            same_long_video_num_negatives, n_gran, "same_long_video_num_negatives",
        )
        self.num_negatives_with_sibling_list: List[Optional[int]] = _normalize_to_list(
            num_negatives_with_sibling, n_gran, "num_negatives_with_sibling",
        )
        # num_negatives_no_sibling is an alternative specification of the total
        # negative count when sibling-aware sampling is active: instead of
        # giving the total C directly (via num_negatives_with_sibling), specify
        # only the "far" (different-long-video) count N. We then derive
        #   effective num_negatives_with_sibling = (S-1) + K_seg + N
        # at forward time. num_negatives_with_sibling and num_negatives_no_sibling
        # are mutually exclusive per granularity.
        self.num_negatives_no_sibling_list: List[Optional[int]] = _normalize_to_list(
            num_negatives_no_sibling, n_gran, "num_negatives_no_sibling",
        )
        for _i, (_w, _n) in enumerate(zip(
            self.num_negatives_with_sibling_list, self.num_negatives_no_sibling_list
        )):
            if _w is not None and _n is not None:
                raise ValueError(
                    f"num_negatives_with_sibling[{_i}]={_w} and "
                    f"num_negatives_no_sibling[{_i}]={_n} are mutually exclusive; "
                    f"please set only one."
                )
        self.n_granularities = n_gran

        self.video_latent_dim = video_latent_dim
        self.audio_latent_dim = audio_latent_dim
        self.embed_dim = embed_dim
        self.segment_count = self.segment_count_list[0]
        self.use_segment_loss = use_segment_loss
        self.use_global_loss = use_global_loss
        self.clamp_scale_min = clamp_scale_min
        self.clamp_scale_max = clamp_scale_max
        self.gather_for_loss = gather_for_loss
        self.num_negatives = self.num_negatives_list[0]
        self.num_negative_videos = self.num_negative_videos_list[0]
        self.spatial_pool_mode = spatial_pool_mode
        self.segment_temporal_pool_mode = segment_temporal_pool_mode
        self.global_temporal_pool_mode = global_temporal_pool_mode
        self.skip_first_video_latent_frame = skip_first_video_latent_frame
        self.video_temporal_compress_factor = video_temporal_compress_factor
        self.transformer_dim = transformer_dim
        self.transformer_nhead = transformer_nhead
        self.transformer_layers = transformer_layers
        self.spatial_merge_factor = spatial_merge_factor

        # ---- Resolve per-stage size presets ----
        for _sz_name, _sz_val in [
            ("spatial_module_size", spatial_module_size),
            ("segment_module_size", segment_module_size),
            ("global_module_size", global_module_size),
        ]:
            if _sz_val is not None and _sz_val not in CONTRASTIVE_MODULE_PRESETS:
                raise ValueError(
                    f"{_sz_name}='{_sz_val}' is invalid. "
                    f"Choose from {list(CONTRASTIVE_MODULE_PRESETS.keys())}"
                )

        _sp_p = CONTRASTIVE_MODULE_PRESETS.get(spatial_module_size, {})
        _seg_p = CONTRASTIVE_MODULE_PRESETS.get(segment_module_size, {})
        _glb_p = CONTRASTIVE_MODULE_PRESETS.get(global_module_size, {})
        _ref_p = _seg_p or _glb_p or _sp_p

        if _ref_p:
            embed_dim = _ref_p["embed_dim"]
            if segment_hidden_dim is None:
                segment_hidden_dim = (_seg_p or _ref_p).get("hidden_dim")
            if global_hidden_dim is None:
                global_hidden_dim = (_glb_p or _ref_p).get("hidden_dim")

        self.embed_dim = embed_dim

        _sp_dim = _sp_p.get("dim", transformer_dim)
        _sp_nh = _sp_p.get("nhead", transformer_nhead)
        _seg_dim = _seg_p.get("dim", transformer_dim)
        _seg_nh = _seg_p.get("nhead", transformer_nhead)
        _glb_dim = _glb_p.get("dim", transformer_dim)
        _glb_nh = _glb_p.get("nhead", transformer_nhead)

        _nl_spatial = spatial_transformer_layers if spatial_transformer_layers is not None else _sp_p.get("layers", transformer_layers)
        _nl_segment = segment_transformer_layers if segment_transformer_layers is not None else _seg_p.get("layers", transformer_layers)
        _nl_global = global_transformer_layers if global_transformer_layers is not None else _glb_p.get("layers", transformer_layers)

        _spatial_input_dim = video_latent_dim * (spatial_merge_factor ** 2)

        logging.info(
            f"Contrastive head config: "
            f"spatial(dim={_sp_dim}, nh={_sp_nh}, nl={_nl_spatial}) "
            f"segment(dim={_seg_dim}, nh={_seg_nh}, nl={_nl_segment}) "
            f"global(dim={_glb_dim}, nh={_glb_nh}, nl={_nl_global}) "
            f"embed_dim={embed_dim} seg_hidden={segment_hidden_dim} glb_hidden={global_hidden_dim} "
            f"spatial_merge_factor={spatial_merge_factor}"
        )

        # --- Spatial aggregation modules ---
        if self.spatial_pool_mode == "attention_pool":
            self.spatial_attn_pool = SpatialAttentionPool(_spatial_input_dim)
        else:
            self.spatial_attn_pool = None
        if self.spatial_pool_mode == "transformer":
            self.spatial_transformer = CLSPoolTransformerLayer(
                d_model=_sp_dim or _spatial_input_dim,
                nhead=_sp_nh,
                input_dim=_spatial_input_dim if _sp_dim else None,
                num_layers=_nl_spatial,
                use_sdpa=use_sdpa,
                qk_norm=qk_norm,
            )
        else:
            self.spatial_transformer = None

        # Effective video feature dim after spatial pooling:
        #   transformer mode → d_model; other modes → _spatial_input_dim (D * k²)
        _v_sp = (self.spatial_transformer.d_model
                 if self.spatial_transformer is not None
                 else _spatial_input_dim)

        # --- Segment temporal aggregation modules ---
        self.video_segment_conv = None
        self.audio_segment_conv = None

        if self.segment_temporal_pool_mode == "transformer":
            _any_sc_not_none = any(sc is not None for sc in self.segment_count_list)
            if _any_sc_not_none:
                self.video_segment_temporal_transformer = CLSPoolTransformerLayer(
                    d_model=_seg_dim or _v_sp,
                    nhead=_seg_nh,
                    input_dim=_v_sp if (_seg_dim and _seg_dim != _v_sp) else None,
                    num_layers=_nl_segment,
                    use_sdpa=use_sdpa,
                    qk_norm=qk_norm,
                )
            else:
                self.video_segment_temporal_transformer = None
            self.audio_segment_temporal_transformer = CLSPoolTransformerLayer(
                d_model=_seg_dim or audio_latent_dim,
                nhead=_seg_nh,
                input_dim=audio_latent_dim if _seg_dim else None,
                num_layers=_nl_segment,
                use_sdpa=use_sdpa,
                qk_norm=qk_norm,
            )
        elif self.segment_temporal_pool_mode == "conv":
            _conv_out_dim = _seg_dim or _v_sp
            self.video_segment_conv = Conv1dTemporalPool(
                input_dim=_v_sp, output_dim=_conv_out_dim,
            )
            self.audio_segment_conv = Conv1dTemporalPool(
                input_dim=audio_latent_dim, output_dim=_conv_out_dim,
            )
            self.video_segment_temporal_transformer = None
            self.audio_segment_temporal_transformer = None
        elif self.segment_temporal_pool_mode == "cnn_increase":
            _conv_out_dim = _seg_dim or _v_sp
            _cnn_inc_p = CNN_INCREASE_PRESETS.get(segment_module_size, {})
            _cnn_bps = cnn_num_blocks_per_stage if cnn_num_blocks_per_stage is not None else _cnn_inc_p.get("num_blocks_per_stage", 1)
            _cnn_ks = cnn_kernel_size if cnn_kernel_size is not None else _cnn_inc_p.get("kernel_size", 7)
            logging.info(
                f"cnn_increase config: num_blocks_per_stage={_cnn_bps}, kernel_size={_cnn_ks}, "
                f"video({_v_sp}->{_conv_out_dim}), audio({audio_latent_dim}->{_conv_out_dim})"
            )
            self.video_segment_conv = GradualConv1dTemporalPool(
                input_dim=_v_sp, output_dim=_conv_out_dim,
                num_blocks_per_stage=_cnn_bps, kernel_size=_cnn_ks,
            )
            self.audio_segment_conv = GradualConv1dTemporalPool(
                input_dim=audio_latent_dim, output_dim=_conv_out_dim,
                num_blocks_per_stage=_cnn_bps, kernel_size=_cnn_ks,
            )
            self.video_segment_temporal_transformer = None
            self.audio_segment_temporal_transformer = None
        else:
            self.video_segment_temporal_transformer = None
            self.audio_segment_temporal_transformer = None

        # --- Determine segment feature dims (needed by global aggregation) ---
        if self.video_segment_temporal_transformer is not None:
            _v_seg_d = self.video_segment_temporal_transformer.d_model
        elif self.video_segment_conv is not None:
            _v_seg_d = self.video_segment_conv.output_dim
        else:
            _v_seg_d = _v_sp

        if self.audio_segment_temporal_transformer is not None:
            _a_seg_d = self.audio_segment_temporal_transformer.d_model
        elif self.audio_segment_conv is not None:
            _a_seg_d = self.audio_segment_conv.output_dim
        else:
            _a_seg_d = audio_latent_dim

        # --- Global temporal aggregation modules ---
        if self.global_temporal_pool_mode == "transformer":
            self.video_global_temporal_transformer = CLSPoolTransformerLayer(
                d_model=_glb_dim or _v_seg_d,
                nhead=_glb_nh,
                input_dim=_v_seg_d if (_glb_dim and _glb_dim != _v_seg_d) else None,
                num_layers=_nl_global,
                add_pos_emb=True, pos_max_len=128,
                use_sdpa=use_sdpa,
                qk_norm=qk_norm,
            )
            self.audio_global_temporal_transformer = CLSPoolTransformerLayer(
                d_model=_glb_dim or _a_seg_d,
                nhead=_glb_nh,
                input_dim=_a_seg_d if (_glb_dim and _glb_dim != _a_seg_d) else None,
                num_layers=_nl_global,
                add_pos_emb=True, pos_max_len=512,
                use_sdpa=use_sdpa,
                qk_norm=qk_norm,
            )
        else:
            self.video_global_temporal_transformer = None
            self.audio_global_temporal_transformer = None

        # --- Projection heads ---
        _v_glob_d = (self.video_global_temporal_transformer.d_model
                     if self.video_global_temporal_transformer is not None
                     else _v_seg_d)
        _a_glob_d = (self.audio_global_temporal_transformer.d_model
                     if self.audio_global_temporal_transformer is not None
                     else _a_seg_d)

        if self.use_segment_loss:
            self.video_segment_proj = ProjectionHead(_v_seg_d, embed_dim, segment_hidden_dim)
            self.audio_segment_proj = ProjectionHead(_a_seg_d, embed_dim, segment_hidden_dim)
            self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)
        else:
            self.video_segment_proj = None
            self.audio_segment_proj = None
            self.logit_scale = None

        if self.use_global_loss:
            self.video_global_proj = ProjectionHead(_v_glob_d, embed_dim, global_hidden_dim)
            self.audio_global_proj = ProjectionHead(_a_glob_d, embed_dim, global_hidden_dim)
            self.global_logit_scale = nn.Parameter(torch.ones([]) * init_scale)
        else:
            self.video_global_proj = None
            self.audio_global_proj = None
            self.global_logit_scale = None

    def _merge_spatial_tokens(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Merge k×k spatial token patches into single tokens by concatenation.

        E.g. with spatial_merge_factor=2, every 2×2 block of tokens (each dim D)
        is concatenated into one token of dim D*4, reducing the sequence length
        from H*W to (H/2)*(W/2).

        Args:
            x: (N, H*W, D)
            H, W: spatial grid dimensions
        Returns:
            (N, (H//k)*(W//k), D*k*k)
        """
        k = self.spatial_merge_factor
        assert H % k == 0 and W % k == 0, (
            f"Spatial dims ({H}, {W}) must be divisible by spatial_merge_factor={k}"
        )
        N, _, D = x.shape
        x = x.view(N, H, W, D)
        x = x.reshape(N, H // k, k, W // k, k, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (N, H//k, W//k, k, k, D)
        x = x.reshape(N, (H // k) * (W // k), k * k * D)
        return x

    def _spatial_pool(self, video_latent: torch.Tensor) -> torch.Tensor:
        """Aggregate spatial dims (H, W) according to *spatial_pool_mode*.

        When ``spatial_merge_factor > 1``, k×k spatial patches are concatenated
        **before** any pooling mode, so the token dim becomes D*k² and the
        sequence length drops from H*W to (H/k)*(W/k).  This applies to all
        modes (mean, max, attention_pool, transformer).

        Args:
            video_latent: (B, D, T, H, W)
        Returns:
            (B, D_out, T)
        """
        B, D, T, H, W = video_latent.shape
        x = video_latent.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, D)

        if self.spatial_merge_factor > 1:
            x = self._merge_spatial_tokens(x, H, W)  # (B*T, H'*W', D*k²)

        if self.spatial_pool_mode == "transformer":
            pooled = self.spatial_transformer(x)
        elif self.spatial_pool_mode == "attention_pool":
            pooled = self.spatial_attn_pool(x)
        elif self.spatial_pool_mode == "max":
            pooled = x.max(dim=1).values
        else:
            pooled = x.mean(dim=1)

        return pooled.reshape(B, T, -1).permute(0, 2, 1)

    def _resolve_segment_count(self, video_latent: torch.Tensor) -> int:
        video_segments = int(video_latent.shape[2])
        if self.segment_count is None:
            return max(video_segments, 1)
        return max(1, min(int(self.segment_count), video_segments))

    def _chunk_temporal_transformer(
        self,
        x: torch.Tensor,
        segment_count: int,
        transformer: CLSPoolTransformerLayer,
    ) -> torch.Tensor:
        """Chunk the time dimension into *segment_count* groups and aggregate
        each group with a CLS-token Transformer.

        Args:
            x: (B, T, D_in)
            segment_count: S — desired number of output segments.
            transformer: ``CLSPoolTransformerLayer`` instance (may be None
                when T == segment_count and no projection is needed).
        Returns:
            (B, S, d_model)  where d_model = transformer.d_model or D_in.
        """
        B, T, D = x.shape
        if T == segment_count:
            if transformer is None:
                return x
            x = x.reshape(B * segment_count, 1, D)
            x = transformer(x)  # (B*S, d_model)
            return x.reshape(B, segment_count, -1)

        remainder = T % segment_count
        if remainder != 0:
            pad_len = segment_count - remainder
            x = F.pad(x, (0, 0, 0, pad_len))  # pad along T
            T = T + pad_len

        chunk_size = T // segment_count
        x = x.reshape(B * segment_count, chunk_size, D)
        x = transformer(x)  # (B*S, d_model)
        return x.reshape(B, segment_count, -1)

    def _pool_video_segments(self, video_latent: torch.Tensor, segment_count: int) -> torch.Tensor:
        # (B, D, T, H, W) -> (B, D, T)
        video_temporal = self._spatial_pool(video_latent)

        if self.segment_temporal_pool_mode == "transformer":
            x = video_temporal.transpose(1, 2).contiguous()  # (B, T, D)
            return self._chunk_temporal_transformer(
                x, segment_count, self.video_segment_temporal_transformer,
            )

        if self.segment_temporal_pool_mode in ("conv", "cnn_increase"):
            return self.video_segment_conv(video_temporal, segment_count)

        if video_temporal.shape[-1] != segment_count:
            video_temporal = F.adaptive_avg_pool1d(video_temporal, segment_count)
        return video_temporal.transpose(1, 2).contiguous()

    def _pool_audio_segments(
        self,
        audio_latent: torch.Tensor,
        segment_count: int,
        audio_latent_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.segment_temporal_pool_mode == "transformer":
            if audio_latent_lengths is None:
                x = audio_latent.transpose(1, 2).contiguous()  # (B, T_l, D_a)
                return self._chunk_temporal_transformer(
                    x, segment_count, self.audio_segment_temporal_transformer,
                )
            pooled_segments = []
            max_audio_len = audio_latent.shape[-1]
            for idx in range(audio_latent.shape[0]):
                valid_len = int(audio_latent_lengths[idx].item())
                valid_len = max(1, min(valid_len, max_audio_len))
                x = audio_latent[idx:idx + 1, :, :valid_len].transpose(1, 2).contiguous()
                pooled = self._chunk_temporal_transformer(
                    x, segment_count, self.audio_segment_temporal_transformer,
                )
                pooled_segments.append(pooled)
            return torch.cat(pooled_segments, dim=0)

        if self.segment_temporal_pool_mode in ("conv", "cnn_increase"):
            if audio_latent_lengths is None:
                return self.audio_segment_conv(audio_latent, segment_count)
            pooled_segments = []
            max_audio_len = audio_latent.shape[-1]
            for idx in range(audio_latent.shape[0]):
                valid_len = int(audio_latent_lengths[idx].item())
                valid_len = max(1, min(valid_len, max_audio_len))
                pooled = self.audio_segment_conv(
                    audio_latent[idx:idx + 1, :, :valid_len], segment_count,
                )
                pooled_segments.append(pooled)
            return torch.cat(pooled_segments, dim=0)

        if audio_latent_lengths is None:
            pooled = F.adaptive_avg_pool1d(audio_latent, segment_count)
            return pooled.transpose(1, 2).contiguous()

        pooled_segments = []
        max_audio_len = audio_latent.shape[-1]
        for idx in range(audio_latent.shape[0]):
            valid_len = int(audio_latent_lengths[idx].item())
            valid_len = max(1, min(valid_len, max_audio_len))
            pooled = F.adaptive_avg_pool1d(audio_latent[idx:idx + 1, :, :valid_len], segment_count)
            pooled_segments.append(pooled)
        pooled = torch.cat(pooled_segments, dim=0)
        return pooled.transpose(1, 2).contiguous()

    def _pool_global_from_segments(
        self,
        segment_feat: torch.Tensor,
        transformer: Optional[nn.Module],
    ) -> torch.Tensor:
        """Aggregate segment features (B, S, D) into global feature (B, D).

        E.g. with 20 segments, feed (B, 20, D) into transformer or mean pool.
        """
        if transformer is not None:
            return transformer(segment_feat)  # (B, D)
        return segment_feat.mean(dim=1)  # (B, D)

    def _pool_video_global(self, segment_video: torch.Tensor) -> torch.Tensor:
        # (B, S, D) -> (B, D)
        return self._pool_global_from_segments(
            segment_video,
            self.video_global_temporal_transformer,
        )

    def _pool_audio_global(self, segment_audio: torch.Tensor) -> torch.Tensor:
        # (B, S, D) -> (B, D)
        return self._pool_global_from_segments(
            segment_audio,
            self.audio_global_temporal_transformer,
        )

    @staticmethod
    def _resolve_segment_count_value(
        segment_count_cfg: Optional[int], video_latent: torch.Tensor,
    ) -> int:
        """Resolve a single granularity's segment count from config and latent shape."""
        video_segments = int(video_latent.shape[2])
        if segment_count_cfg is None:
            return max(video_segments, 1)
        return max(1, min(int(segment_count_cfg), video_segments))

    def _pool_video_segments_from_temporal(
        self, video_temporal: torch.Tensor, segment_count: int,
    ) -> torch.Tensor:
        """Pool pre-computed spatial features into temporal segments.

        Args:
            video_temporal: (B, D', T) — output of ``_spatial_pool``.
            segment_count: desired number of output segments.
        Returns:
            (B, S, D_seg)
        """
        if self.segment_temporal_pool_mode == "transformer":
            x = video_temporal.transpose(1, 2).contiguous()  # (B, T, D)
            return self._chunk_temporal_transformer(
                x, segment_count, self.video_segment_temporal_transformer,
            )
        if self.segment_temporal_pool_mode in ("conv", "cnn_increase"):
            return self.video_segment_conv(video_temporal, segment_count)
        vt = video_temporal
        if vt.shape[-1] != segment_count:
            vt = F.adaptive_avg_pool1d(vt, segment_count)
        return vt.transpose(1, 2).contiguous()

    def _resolve_num_negatives_with_sibling(self, i: int, S: int) -> Optional[int]:
        """Return effective num_negatives_with_sibling for granularity ``i``.

        Priority:
          1. If num_negatives_with_sibling_list[i] is set -> use it verbatim
             (user directly specified the total column count C).
          2. Else if num_negatives_no_sibling_list[i] is set -> derive
             C = (S - 1) + K_seg + N, where
                S - 1  = intra (same-clip) negatives,
                K_seg  = same_long_video_num_negatives_list[i] (0 if None),
                N      = num_negatives_no_sibling_list[i].
          3. Else return None (fall back to num_negatives_list[i]).

        Only meaningful when sibling-aware sampling is active (i.e.
        same_long_video_priority=True and K_seg > 0); otherwise the returned
        value is ignored downstream.
        """
        w = self.num_negatives_with_sibling_list[i]
        if w is not None:
            return int(w)
        n = self.num_negatives_no_sibling_list[i]
        if n is None:
            return None
        k = self.same_long_video_num_negatives_list[i] or 0
        return int(max(S - 1, 0)) + int(k) + int(n)

    def _gather_segment_feats(
        self, world_size: int, B: int, S: int,
        vfeat: torch.Tensor, afeat: torch.Tensor,
    ):
        """All-gather segment features across ranks."""
        if world_size > 1 and self.gather_for_loss:
            rank = dist.get_rank()
            rank_offset = rank * B * S
            B_eff = B * world_size
            vfeat_pool = torch.cat(dist_all_gather(vfeat), dim=0)
            afeat_pool = torch.cat(dist_all_gather(afeat), dim=0)
        else:
            rank_offset = 0
            B_eff = B
            vfeat_pool = vfeat
            afeat_pool = afeat
        return rank_offset, B_eff, vfeat_pool, afeat_pool

    def _gather_global_feats(
        self, world_size: int, B: int,
        global_vfeat: torch.Tensor, global_afeat: torch.Tensor,
    ):
        """All-gather global features across ranks."""
        if world_size > 1 and self.gather_for_loss:
            rank = dist.get_rank()
            rank_offset_g = rank * B
            vfeat_g_pool = torch.cat(dist_all_gather(global_vfeat), dim=0)
            afeat_g_pool = torch.cat(dist_all_gather(global_afeat), dim=0)
        else:
            rank_offset_g = 0
            vfeat_g_pool = global_vfeat
            afeat_g_pool = global_afeat
        return rank_offset_g, vfeat_g_pool, afeat_g_pool

    @torch.no_grad()
    def _build_neg_indices_vectorized(
        self,
        B: int,
        B_eff: int,
        S: int,
        rank_offset: int,
        device: torch.device,
        num_negative_videos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rank_offset % max(S, 1) != 0:
            raise ValueError(
                f"rank_offset ({rank_offset}) must be divisible by S ({S})."
            )

        n_local = B * S

        seg_range = torch.arange(S, device=device)
        not_self = ~torch.eye(S, dtype=torch.bool, device=device)
        intra_rel = seg_range.unsqueeze(0).expand(S, -1)[not_self].view(S, max(S - 1, 0))

        video_base = (torch.arange(B, device=device) * S + rank_offset).view(B, 1, 1)
        intra_idx = (video_base + intra_rel.unsqueeze(0)).reshape(n_local, max(S - 1, 0))

        num_cross_videos = B_eff - 1
        if num_cross_videos <= 0:
            return intra_idx, torch.empty(n_local, 0, dtype=torch.long, device=device)

        need_subsample = (
            num_negative_videos is not None
            and 0 < num_negative_videos < num_cross_videos
        )

        if not need_subsample:
            n_pool = B_eff * S
            all_pool_idx = torch.arange(n_pool, device=device)
            pool_video_of = all_pool_idx // max(S, 1)
            local_pool_video_ids = torch.arange(B, device=device) + rank_offset // max(S, 1)
            cross_mask = pool_video_of.unsqueeze(0) != local_pool_video_ids.unsqueeze(1)

            num_cross = num_cross_videos * S
            cross_per_video = all_pool_idx.unsqueeze(0).expand(B, -1)[cross_mask].view(B, num_cross)
            cross_idx = cross_per_video.unsqueeze(1).expand(-1, S, -1).reshape(n_local, num_cross)
        else:
            all_video_ids = torch.arange(B_eff, device=device)
            local_video_ids = torch.arange(B, device=device) + rank_offset // max(S, 1)
            video_mask = all_video_ids.unsqueeze(0) != local_video_ids.unsqueeze(1)
            cross_video_pool = all_video_ids.unsqueeze(0).expand(B, -1)[video_mask].view(B, num_cross_videos)

            perm = torch.rand(B, num_cross_videos, device=device).argsort(dim=1)[:, :num_negative_videos]
            sampled_videos = cross_video_pool.gather(1, perm)

            seg_offsets = seg_range.unsqueeze(0).unsqueeze(0)
            cross_segments = sampled_videos.unsqueeze(-1) * S + seg_offsets
            cross_per_video = cross_segments.reshape(B, num_negative_videos * S)
            cross_idx = cross_per_video.unsqueeze(1).expand(-1, S, -1).reshape(n_local, num_negative_videos * S)

        return intra_idx, cross_idx

    @torch.no_grad()
    def _subsample_cross_videos(
        self,
        cross_idx: torch.Tensor,
        num_cross_videos: int,
        S: int,
        num_negative_videos: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_cross_videos <= 0 or num_negative_videos >= num_cross_videos:
            return cross_idx

        n_local = cross_idx.shape[0]
        cross_by_video = cross_idx.view(n_local, num_cross_videos, S)
        perm = torch.rand(n_local, num_cross_videos, device=device).argsort(dim=1)[:, :num_negative_videos]
        perm = perm.unsqueeze(-1).expand(-1, -1, S)
        selected = cross_by_video.gather(1, perm)
        return selected.reshape(n_local, num_negative_videos * S)

    def _select_neg_idx_with_k(
        self,
        intra_idx: torch.Tensor,
        cross_idx: torch.Tensor,
        num_negatives: Optional[int],
        device: torch.device,
    ) -> torch.Tensor:
        num_intra = intra_idx.shape[1]
        num_cross = cross_idx.shape[1]

        def _sample_cols(src: torch.Tensor, k: int) -> torch.Tensor:
            n_rows, n_cols = src.shape
            if k <= 0 or n_cols == 0:
                return src[:, :0]
            perm = torch.rand(n_rows, n_cols, device=device).argsort(dim=1)[:, :k]
            return src.gather(1, perm)

        if num_negatives is None:
            return torch.cat([intra_idx, cross_idx], dim=1)
        if num_negatives <= num_intra:
            return _sample_cols(intra_idx, num_negatives)

        num_cross_needed = num_negatives - num_intra
        if num_cross_needed >= num_cross:
            return torch.cat([intra_idx, cross_idx], dim=1)
        return torch.cat([intra_idx, _sample_cols(cross_idx, num_cross_needed)], dim=1)

    @torch.no_grad()
    def _build_neg_indices_sibling_aware(
        self,
        B: int,
        B_eff: int,
        S: int,
        rank_offset: int,
        long_video_ids_pool: torch.Tensor,  # (B_eff,) int64
        K_seg: int,
        num_total: int,
        num_total_with_sibling: Optional[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Sibling-aware negative index sampling.

        Per row priority: intra (S-1 same-clip segments) → sibling (up to
        K_seg segments from other clips sharing long_video_id) → far (random
        from other long videos), filled until column count reaches
        ``C_max = num_total_with_sibling or num_total``.

        Per-row column budget (controls effective negatives per anchor):
          * rows that have at least one sibling in the pool use the full
            ``C_max`` columns.
          * rows without any sibling fall back to ``C_no_sib = num_total``
            effective columns — the remaining ``C_max - C_no_sib`` columns
            are padded with a safe fallback index and marked invalid in
            ``valid_neg_mask`` (they will be masked to -inf before CE).

        Returns:
            neg_idx: (B*S, C_max)  int64 tensor of pool indices (padding
                columns for no-sibling rows carry a safe dummy index).
            num_intra: number of intra (same-clip) columns at the start
                (same for every row).
            sibling_take: (B*S,) int64 tensor, per-row number of sibling
                columns placed immediately after the intra columns (can be
                zero for rows without siblings).
            valid_neg_mask: (B*S, C_max) bool tensor; True for real
                negative columns, False for padding columns.
        """
        if rank_offset % max(S, 1) != 0:
            raise ValueError(
                f"rank_offset ({rank_offset}) must be divisible by S ({S})."
            )

        n_local = B * S

        # ---- intra_idx: (n_local, S-1) ----
        seg_range = torch.arange(S, device=device)
        not_self = ~torch.eye(S, dtype=torch.bool, device=device)
        intra_rel = seg_range.unsqueeze(0).expand(S, -1)[not_self].view(S, max(S - 1, 0))
        video_base = (torch.arange(B, device=device) * S + rank_offset).view(B, 1, 1)
        intra_idx = (video_base + intra_rel.unsqueeze(0)).reshape(n_local, max(S - 1, 0))

        num_intra = intra_idx.shape[1]

        C_max = int(num_total_with_sibling if num_total_with_sibling is not None else num_total)
        # No-sibling rows fall back to ``num_total`` effective columns. Clamp
        # so C_no_sib ∈ [num_intra, C_max] — anchors always keep their intra
        # negatives, and never "use more than C_max" columns.
        C_no_sib = int(num_total)
        C_no_sib = max(min(C_no_sib, C_max), min(num_intra, C_max))
        num_far_needed = max(0, C_max - num_intra)

        if B_eff <= 1 or num_far_needed <= 0:
            # Degenerate case: no far pool or C_max <= num_intra. Treat all
            # rows identically and return a full-True mask (the unique
            # available behaviour).
            zero_sib = torch.zeros(n_local, dtype=torch.long, device=device)
            C_eff = min(num_intra, C_max)
            neg_idx_trunc = intra_idx[:, :C_max]
            valid_neg_mask = torch.zeros(n_local, C_max, dtype=torch.bool, device=device)
            if C_eff > 0:
                valid_neg_mask[:, :C_eff] = True
            return neg_idx_trunc, min(num_intra, C_max), zero_sib, valid_neg_mask

        # ---- Build masks: sibling vs far (cross-long-video) ----
        local_video_ids = torch.arange(B, device=device) + rank_offset // max(S, 1)

        all_seg_pool = torch.arange(B_eff * S, device=device)
        pool_video_per_seg = all_seg_pool // max(S, 1)           # (B_eff*S,)
        pool_lvid_per_seg = long_video_ids_pool.repeat_interleave(S)  # (B_eff*S,)

        own_vidx_per_seg = local_video_ids.repeat_interleave(S)  # (n_local,)
        own_lvid_per_seg = long_video_ids_pool[local_video_ids].repeat_interleave(S)  # (n_local,)

        cross_mask = pool_video_per_seg.unsqueeze(0) != own_vidx_per_seg.unsqueeze(1)
        same_lvid = pool_lvid_per_seg.unsqueeze(0) == own_lvid_per_seg.unsqueeze(1)
        sibling_mask = cross_mask & same_lvid     # (n_local, B_eff*S)
        far_mask = cross_mask & ~same_lvid        # (n_local, B_eff*S)

        # Which rows have at least one sibling in the pool. Used to decide
        # per-row effective column budget.
        has_sibling_row = sibling_mask.any(dim=1)                     # (n_local,)

        neg_inf = torch.finfo(torch.float32).min

        # ---- Pick up to K_seg sibling negatives per row (random w/o replacement) ----
        K_seg_eff = int(min(max(K_seg, 0), num_far_needed))
        if K_seg_eff > 0:
            sib_scores = torch.rand(n_local, B_eff * S, device=device)
            sib_scores = sib_scores.masked_fill(~sibling_mask, neg_inf)
            sib_topk_vals, sib_topk_idx = sib_scores.topk(K_seg_eff, dim=1)
            sib_valid = sib_topk_vals > neg_inf                      # (n_local, K_seg_eff)
            sibling_take = sib_valid.sum(dim=1).clamp_(max=num_far_needed)
        else:
            sib_topk_idx = torch.zeros(n_local, 0, dtype=torch.long, device=device)
            sibling_take = torch.zeros(n_local, dtype=torch.long, device=device)

        # ---- Pick num_far_needed far negatives per row (random w/o replacement) ----
        far_scores = torch.rand(n_local, B_eff * S, device=device)
        far_scores = far_scores.masked_fill(~far_mask, neg_inf)
        far_topk_vals, far_topk_idx = far_scores.topk(num_far_needed, dim=1)
        far_valid = far_topk_vals > neg_inf                          # (n_local, num_far_needed)

        # ---- Merge: first sibling_take_i columns <- sibling picks, rest <- far ----
        col_range = torch.arange(num_far_needed, device=device).unsqueeze(0)
        take_sib_col = col_range < sibling_take.unsqueeze(1)         # (n_local, num_far_needed)

        # Pad/truncate sib_topk_idx to num_far_needed columns so torch.where shapes match.
        if sib_topk_idx.shape[1] < num_far_needed:
            pad_width = num_far_needed - sib_topk_idx.shape[1]
            sib_padded = torch.cat(
                [sib_topk_idx, far_topk_idx[:, :pad_width]], dim=1,
            )
        else:
            sib_padded = sib_topk_idx[:, :num_far_needed]

        merged = torch.where(take_sib_col, sib_padded, far_topk_idx)

        # When far pool is tiny (rare: only siblings exist), some far picks can
        # be invalid (score==-inf). Replace those with intra's first column as
        # a safe fallback (CE handles duplicated negatives gracefully).
        if num_intra > 0:
            fallback = intra_idx[:, 0:1].expand(-1, num_far_needed)
        else:
            fallback = torch.zeros(
                n_local, num_far_needed, dtype=torch.long, device=device,
            )
        need_fallback = (~take_sib_col) & (~far_valid)
        merged = torch.where(need_fallback, fallback, merged)

        neg_idx = torch.cat([intra_idx, merged], dim=1)

        # ---- Per-row valid_neg_mask ----
        # Column layout: [intra(num_intra) | merged(num_far_needed)].
        # * Rows with sibling: all C_max columns are valid.
        # * Rows without sibling: only the first C_no_sib columns are valid;
        #   the trailing (C_max - C_no_sib) columns are padding.
        col_idx = torch.arange(C_max, device=device).unsqueeze(0)        # (1, C_max)
        row_budget = torch.where(
            has_sibling_row, torch.full_like(has_sibling_row, C_max, dtype=torch.long),
            torch.full_like(has_sibling_row, C_no_sib, dtype=torch.long),
        ).unsqueeze(1)                                                   # (n_local, 1)
        valid_neg_mask = col_idx < row_budget                            # (n_local, C_max)

        # Replace the padding columns' indices with a safe in-pool fallback
        # so downstream ``pool[neg_idx]`` never goes out of range. CE will
        # mask these positions to -inf anyway, so the concrete value does
        # not affect the loss — but gather semantics require valid indices.
        if num_intra > 0:
            pad_fill = intra_idx[:, 0:1].expand(-1, C_max)
        else:
            pad_fill = torch.zeros(n_local, C_max, dtype=torch.long, device=device)
        neg_idx = torch.where(valid_neg_mask, neg_idx, pad_fill)

        return neg_idx, num_intra, sibling_take, valid_neg_mask

    def sample_negatives_for_loss(
        self,
        vfeat_local: torch.Tensor,
        afeat_local: torch.Tensor,
        vfeat_pool: torch.Tensor,
        afeat_pool: torch.Tensor,
        B: int,
        B_eff: int,
        S: int,
        scale: torch.Tensor,
        rank_offset: int = 0,
        num_negatives: Optional[int] = None,
        num_negative_videos: Optional[int] = None,
        long_video_ids_pool: Optional[torch.Tensor] = None,
        same_long_video_num_negatives: Optional[int] = None,
        num_negatives_with_sibling: Optional[int] = None,
        return_neg_idx: bool = False,
    ):
        n_local = B * S
        device = vfeat_local.device

        use_sibling = (
            self.same_long_video_priority
            and long_video_ids_pool is not None
            and num_negatives is not None
            and (same_long_video_num_negatives is not None and same_long_video_num_negatives > 0)
        )

        if use_sibling:
            neg_idx, num_intra, sibling_take, valid_neg_mask = self._build_neg_indices_sibling_aware(
                B=B, B_eff=B_eff, S=S,
                rank_offset=rank_offset,
                long_video_ids_pool=long_video_ids_pool,
                K_seg=int(same_long_video_num_negatives),
                num_total=int(num_negatives),
                num_total_with_sibling=(
                    int(num_negatives_with_sibling)
                    if num_negatives_with_sibling is not None else None
                ),
                device=device,
            )
        else:
            intra_idx, cross_idx = self._build_neg_indices_vectorized(
                B, B_eff, S, rank_offset, device,
                num_negative_videos=num_negative_videos,
            )
            neg_idx = self._select_neg_idx_with_k(intra_idx, cross_idx, num_negatives, device)
            num_intra = min(intra_idx.shape[1], neg_idx.shape[1])
            sibling_take = torch.zeros(n_local, dtype=torch.long, device=device)
            # Non-sibling path: every column is a real negative by construction.
            valid_neg_mask = torch.ones(
                n_local, int(neg_idx.shape[1]), dtype=torch.bool, device=device,
            )

        pos_idx = (torch.arange(n_local, device=device) + rank_offset).unsqueeze(1)
        all_idx = torch.cat([pos_idx, neg_idx], dim=1)

        afeat_sampled = afeat_pool[all_idx]
        vfeat_sampled = vfeat_pool[all_idx]
        sim_v2a = torch.einsum("nd,nkd->nk", vfeat_local, afeat_sampled) / scale
        sim_a2v = torch.einsum("nd,nkd->nk", afeat_local, vfeat_sampled) / scale

        # Apply per-row valid mask: padding columns (e.g. no-sibling rows'
        # fallback tail) are set to -inf so they do not contribute to the
        # softmax denominator in F.cross_entropy. The positive column (col 0)
        # is always valid.
        C = int(neg_idx.shape[1])
        pos_valid = torch.ones(n_local, 1, dtype=torch.bool, device=device)
        valid_full = torch.cat([pos_valid, valid_neg_mask.to(torch.bool)], dim=1)  # (n_local, 1+C)
        neg_inf_fill = torch.finfo(sim_v2a.dtype).min
        sim_v2a = sim_v2a.masked_fill(~valid_full, neg_inf_fill)
        sim_a2v = sim_a2v.masked_fill(~valid_full, neg_inf_fill)

        targets = torch.zeros(n_local, dtype=torch.long, device=device)
        if return_neg_idx:
            # near_mask[i, j] = True iff neg_idx[i, j] is either an intra
            # (same-clip) or a sibling (same long-video) negative. The
            # column layout is [intra(num_intra) | sibling(sibling_take_i) |
            # far(rest)], so the near prefix length per row is
            # (num_intra + sibling_take[i]). Padding columns are excluded
            # via the intersection with ``valid_neg_mask``.
            near_len = (num_intra + sibling_take).clamp_(min=0, max=C)  # (n_local,)
            col_range = torch.arange(C, device=device).unsqueeze(0)     # (1, C)
            near_mask = col_range < near_len.unsqueeze(1)               # (n_local, C)
            near_mask = near_mask & valid_neg_mask.to(torch.bool)
            return sim_v2a, sim_a2v, targets, num_intra, neg_idx, near_mask
        return sim_v2a, sim_a2v, targets, num_intra

    def sample_global_negatives_for_loss(
        self,
        vfeat_g: torch.Tensor,
        afeat_g: torch.Tensor,
        vfeat_g_pool: torch.Tensor,
        afeat_g_pool: torch.Tensor,
        rank_offset_global: int,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sim_v2a = vfeat_g @ afeat_g_pool.T / scale
        sim_a2v = afeat_g @ vfeat_g_pool.T / scale
        targets = torch.arange(
            rank_offset_global,
            rank_offset_global + vfeat_g.shape[0],
            device=vfeat_g.device,
            dtype=torch.long,
        )
        return sim_v2a, sim_a2v, targets

    @torch.no_grad()
    def clamp_logit_scales(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.logit_scale is not None:
            self.logit_scale.clamp_(self.clamp_scale_min, self.clamp_scale_max)
        if self.global_logit_scale is not None:
            self.global_logit_scale.clamp_(self.clamp_scale_min, self.clamp_scale_max)
        seg_scale = self.logit_scale.detach().clone() if self.logit_scale is not None else None
        global_scale = self.global_logit_scale.detach().clone() if self.global_logit_scale is not None else None
        return seg_scale, global_scale

    def _gather_feats(
        self,
        world_size: int,
        B: int,
        S: int,
        vfeat: Optional[torch.Tensor],
        afeat: Optional[torch.Tensor],
        global_vfeat: Optional[torch.Tensor],
        global_afeat: Optional[torch.Tensor],
    ):
        vfeat_pool = afeat_pool = vfeat_g_pool = afeat_g_pool = None

        if world_size > 1 and self.gather_for_loss:
            rank = dist.get_rank()
            rank_offset = rank * B * S
            rank_offset_g = rank * B
            B_eff = B * world_size

            if self.use_segment_loss and vfeat is not None and afeat is not None:
                vfeat_pool = torch.cat(dist_all_gather(vfeat), dim=0)
                afeat_pool = torch.cat(dist_all_gather(afeat), dim=0)
            if self.use_global_loss and global_vfeat is not None and global_afeat is not None:
                vfeat_g_pool = torch.cat(dist_all_gather(global_vfeat), dim=0)
                afeat_g_pool = torch.cat(dist_all_gather(global_afeat), dim=0)
        else:
            rank_offset = 0
            rank_offset_g = 0
            B_eff = B
            if self.use_segment_loss:
                vfeat_pool = vfeat
                afeat_pool = afeat
            if self.use_global_loss:
                vfeat_g_pool = global_vfeat
                afeat_g_pool = global_afeat

        return rank_offset, rank_offset_g, B_eff, vfeat_pool, afeat_pool, vfeat_g_pool, afeat_g_pool

    def _align_causal_first_frame(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        audio_latent_lengths: Optional[torch.Tensor],
    ) -> tuple:
        """Drop the first video latent frame (causal padding artifact) and
        the corresponding proportion of audio latent frames so that the
        remaining segments are temporally aligned.

        WanVAE causal structure:
          - latent frame 0  <- 1 input frame
          - latent frame k  <- video_temporal_compress_factor input frames (k>=1)
        """
        T_video = video_latent.shape[2]
        if T_video <= 1:
            return video_latent, audio_latent, audio_latent_lengths

        total_input = 1 + (T_video - 1) * self.video_temporal_compress_factor
        first_frame_ratio = 1.0 / total_input

        video_latent = video_latent[:, :, 1:]

        T_audio = audio_latent.shape[-1]
        audio_skip = round(T_audio * first_frame_ratio)
        audio_latent = audio_latent[:, :, audio_skip:]

        if audio_latent_lengths is not None:
            audio_latent_lengths = (audio_latent_lengths - audio_skip).clamp_(min=1)

        return video_latent, audio_latent, audio_latent_lengths

    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        audio_latent_lengths: Optional[torch.Tensor] = None,
        world_size: int = 1,
        long_video_ids_pool: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.skip_first_video_latent_frame:
            video_latent, audio_latent, audio_latent_lengths = (
                self._align_causal_first_frame(video_latent, audio_latent, audio_latent_lengths)
            )

        logit_scale, global_logit_scale = self.clamp_logit_scales()
        B = int(video_latent.shape[0])

        # Shared spatial pooling (done once for all granularities)
        video_temporal = self._spatial_pool(video_latent)  # (B, D', T)

        # --- Per-granularity segment processing ---
        granularities: List[Dict[str, Any]] = []
        first_segment_video: Optional[torch.Tensor] = None
        first_segment_audio: Optional[torch.Tensor] = None

        for i, sc_raw in enumerate(self.segment_count_list):
            segment_count = self._resolve_segment_count_value(sc_raw, video_latent)
            segment_video = self._pool_video_segments_from_temporal(video_temporal, segment_count)
            segment_audio = self._pool_audio_segments(audio_latent, segment_count, audio_latent_lengths)

            if i == 0:
                first_segment_video = segment_video
                first_segment_audio = segment_audio

            g_result: Dict[str, Any] = {"segment_count": sc_raw, "S": segment_count}

            if self.use_segment_loss:
                segment_vfeat = self.video_segment_proj(segment_video.reshape(-1, segment_video.shape[-1]))
                segment_afeat = self.audio_segment_proj(segment_audio.reshape(-1, segment_audio.shape[-1]))
                segment_vfeat = F.normalize(segment_vfeat, dim=-1)
                segment_afeat = F.normalize(segment_afeat, dim=-1)

                S = int(segment_count)
                rank_offset, B_eff, vfeat_pool, afeat_pool = self._gather_segment_feats(
                    world_size, B, S, segment_vfeat, segment_afeat,
                )

                sim_v2a, sim_a2v, targets, _ = self.sample_negatives_for_loss(
                    vfeat_local=segment_vfeat,
                    afeat_local=segment_afeat,
                    vfeat_pool=vfeat_pool,
                    afeat_pool=afeat_pool,
                    B=B, B_eff=B_eff, S=S,
                    scale=self.logit_scale,
                    rank_offset=rank_offset,
                    num_negatives=self.num_negatives_list[i],
                    num_negative_videos=self.num_negative_videos_list[i],
                    long_video_ids_pool=long_video_ids_pool,
                    same_long_video_num_negatives=self.same_long_video_num_negatives_list[i],
                    num_negatives_with_sibling=self._resolve_num_negatives_with_sibling(i, S),
                )
                g_result["losses"] = {
                    "segment_contrastive_loss": (
                        F.cross_entropy(sim_v2a, targets) + F.cross_entropy(sim_a2v, targets)
                    ) / 2,
                }
                g_result.update({
                    "segment_vfeat": segment_vfeat,
                    "segment_afeat": segment_afeat,
                    "segment_vfeat_pool": vfeat_pool,
                    "segment_afeat_pool": afeat_pool,
                    "B": B, "B_eff": B_eff, "rank_offset": rank_offset,
                })
            else:
                g_result["losses"] = {}

            granularities.append(g_result)

        # --- Global loss (computed once from the first granularity's segments) ---
        result: Dict[str, Any] = {
            "logit_scales": (logit_scale, global_logit_scale),
            "granularities": granularities,
        }

        global_losses: Dict[str, torch.Tensor] = {}
        if (
            self.use_global_loss
            and first_segment_video is not None
            and first_segment_audio is not None
        ):
            global_video = self._pool_video_global(first_segment_video)
            global_audio = self._pool_audio_global(first_segment_audio)
            global_vfeat = F.normalize(self.video_global_proj(global_video), dim=-1)
            global_afeat = F.normalize(self.audio_global_proj(global_audio), dim=-1)

            rank_offset_g, vfeat_g_pool, afeat_g_pool = self._gather_global_feats(
                world_size, B, global_vfeat, global_afeat,
            )

            sim_v2a_g, sim_a2v_g, targets_g = self.sample_global_negatives_for_loss(
                vfeat_g=global_vfeat, afeat_g=global_afeat,
                vfeat_g_pool=vfeat_g_pool, afeat_g_pool=afeat_g_pool,
                rank_offset_global=rank_offset_g, scale=self.global_logit_scale,
            )
            global_losses["global_contrastive_loss"] = (
                F.cross_entropy(sim_v2a_g, targets_g) + F.cross_entropy(sim_a2v_g, targets_g)
            ) / 2

            result.update({
                "global_vfeat": global_vfeat,
                "global_afeat": global_afeat,
                "global_vfeat_pool": vfeat_g_pool,
                "global_afeat_pool": afeat_g_pool,
                "rank_offset_g": rank_offset_g,
            })

        result["losses"] = global_losses
        return result


# =============================================================================
# Intra-segment cross-attention contrastive head (Design C, ALBEF-style)
# =============================================================================


class SinusoidalPE1D(nn.Module):
    """Fixed 1D sinusoidal positional encoding, stored as a non-persistent buffer."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPE1D dim ({dim}) must be even.")
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        self.max_len = int(max_len)
        self.dim = int(dim)

    def forward(self, n: int) -> torch.Tensor:
        if n > self.max_len:
            raise ValueError(
                f"SinusoidalPE1D: requested length {n} exceeds max_len {self.max_len}."
            )
        return self.pe[:n]


class SinusoidalPE2D(nn.Module):
    """Fixed 2D sinusoidal positional encoding; first dim/2 encodes row, last dim/2 encodes column."""

    def __init__(self, max_h: int, max_w: int, dim: int):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(
                f"SinusoidalPE2D dim ({dim}) must be divisible by 4 (split row/col, each sin/cos)."
            )
        half = dim // 2

        row_pos = torch.arange(max_h, dtype=torch.float32).unsqueeze(1)
        div_r = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half)
        )
        row_pe = torch.zeros(max_h, half)
        row_pe[:, 0::2] = torch.sin(row_pos * div_r)
        row_pe[:, 1::2] = torch.cos(row_pos * div_r)

        col_pos = torch.arange(max_w, dtype=torch.float32).unsqueeze(1)
        div_c = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half)
        )
        col_pe = torch.zeros(max_w, half)
        col_pe[:, 0::2] = torch.sin(col_pos * div_c)
        col_pe[:, 1::2] = torch.cos(col_pos * div_c)

        pe = torch.zeros(max_h, max_w, dim)
        pe[..., :half] = row_pe.unsqueeze(1).expand(-1, max_w, -1)
        pe[..., half:] = col_pe.unsqueeze(0).expand(max_h, -1, -1)
        self.register_buffer("pe", pe, persistent=False)
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.dim = int(dim)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """Return (h*w, dim) flattened row-major."""
        if h > self.max_h or w > self.max_w:
            raise ValueError(
                f"SinusoidalPE2D: requested ({h},{w}) exceeds table ({self.max_h},{self.max_w})."
            )
        return self.pe[:h, :w].reshape(h * w, -1)


class SDPACrossAttention(nn.Module):
    """Cross-attention using ``F.scaled_dot_product_attention``.

    Query is projected from ``q_in``; keys and values are projected together
    from ``kv_in`` (fused ``kv_proj``). Mirrors :class:`SDPASelfAttention`.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Nq, _ = q_in.shape
        Nkv = kv_in.shape[1]

        q = self.q_proj(q_in).reshape(B, Nq, self.nhead, self.head_dim).transpose(1, 2)
        kv = (
            self.kv_proj(kv_in)
            .reshape(B, Nkv, 2, self.nhead, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, Nkv, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, Nq, self.d_model)
        return self.out_proj(out)


class IntraSegSelfAttnBlock(nn.Module):
    """Pre-norm Transformer block: self-attn + FFN with GELU."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = SDPASelfAttention(d_model, nhead, dropout=dropout, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.dropout(self.linear2(self.act(self.linear1(self.norm2(x)))))
        return x


class CrossModalBlock(nn.Module):
    """Pre-norm cross-attn + FFN block. Query side is updated; KV side unchanged."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = SDPACrossAttention(d_model, nhead, dropout=dropout, qk_norm=qk_norm)
        self.norm_ff = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_norm = self.norm_q(q_in)
        kv_norm = self.norm_kv(kv_in)
        q_in = q_in + self.dropout(
            self.cross_attn(q_norm, kv_norm, key_padding_mask=key_padding_mask)
        )
        q_in = q_in + self.dropout(
            self.linear2(self.act(self.linear1(self.norm_ff(q_in))))
        )
        return q_in


class IntraSegCrossAttnHead(nn.Module):
    """Intra-segment Self-Attn + Cross-Attn contrastive head (ALBEF-style, "Design C").

    For every video-latent time step (segment) we keep all tokens:
      * video side: ``(H/k) * (W/k)`` spatial tokens after ``spatial_merge_factor``
        k x k merging, with a learnable CLS token and 2D sinusoidal PE.
      * audio side: a fixed number of consecutive audio-latent tokens per
        segment (even split: ``n_a_per_seg = L_valid // S`` capped by
        ``max_audio_tokens_per_seg``), with a learnable CLS token and 1D
        sinusoidal PE.

    Each side runs ``self_attn_layers`` self-attention blocks, then the pair
    exchanges information through ``cross_attn_layers`` cross-attention
    blocks whose KV come from the opposite side's post-self-attention tokens
    (fixed, source of truth for both directions). The final CLS tokens are
    L2-normalized to obtain per-segment video / audio embeddings which are
    fed into the same ``sample_negatives_for_loss`` / InfoNCE pipeline used
    by :class:`LatentAVContrastiveHead`.

    The output dictionary is **structurally identical** to
    :class:`LatentAVContrastiveHead`'s (single granularity), so the trainer
    consumes it without modification.
    """

    # -------- helpers borrowed from LatentAVContrastiveHead --------
    # These operate purely on pooled features / indices / logit scales, so
    # reusing them avoids duplicating ~250 lines and keeps the loss math
    # exactly in sync with the existing head.
    _align_causal_first_frame = LatentAVContrastiveHead._align_causal_first_frame
    clamp_logit_scales = LatentAVContrastiveHead.clamp_logit_scales
    _gather_segment_feats = LatentAVContrastiveHead._gather_segment_feats
    _build_neg_indices_vectorized = LatentAVContrastiveHead._build_neg_indices_vectorized
    _subsample_cross_videos = LatentAVContrastiveHead._subsample_cross_videos
    _select_neg_idx_with_k = LatentAVContrastiveHead._select_neg_idx_with_k
    _build_neg_indices_sibling_aware = LatentAVContrastiveHead._build_neg_indices_sibling_aware
    sample_negatives_for_loss = LatentAVContrastiveHead.sample_negatives_for_loss
    _resolve_num_negatives_with_sibling = LatentAVContrastiveHead._resolve_num_negatives_with_sibling

    def __init__(
        self,
        video_latent_dim: int,
        audio_latent_dim: int,
        embed_dim: int = 512,
        nhead: int = 8,
        self_attn_layers: int = 2,
        cross_attn_layers: int = 2,
        spatial_merge_factor: int = 2,
        max_spatial_h: int = 64,
        max_spatial_w: int = 64,
        max_audio_tokens_per_seg: int = 32,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        init_scale: float = 0.07,
        clamp_scale_min: float = 0.001,
        clamp_scale_max: float = 0.5,
        gather_for_loss: bool = True,
        num_negatives: Union[None, int, List[Optional[int]]] = None,
        num_negative_videos: Union[None, int, List[Optional[int]]] = None,
        same_long_video_priority: bool = False,
        same_long_video_num_negatives: Union[None, int, List[Optional[int]]] = None,
        num_negatives_with_sibling: Union[None, int, List[Optional[int]]] = None,
        num_negatives_no_sibling: Union[None, int, List[Optional[int]]] = None,
        skip_first_video_latent_frame: bool = True,
        video_temporal_compress_factor: int = 4,
        qk_norm: bool = False,
        use_itm: bool = False,
        lambda_itm: float = 1.0,
        itm_neg_per_direction: int = 1,
        itm_sim_temperature: float = 1.0,
        itm_neg_source: str = "near",
        itm_start_step: int = 0,
    ):
        super().__init__()

        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead}).")
        if embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by 4 for 2D sinusoidal PE."
            )

        self.video_latent_dim = int(video_latent_dim)
        self.audio_latent_dim = int(audio_latent_dim)
        self.embed_dim = int(embed_dim)
        self.nhead = int(nhead)
        self.self_attn_layers = int(self_attn_layers)
        self.cross_attn_layers = int(cross_attn_layers)
        self.spatial_merge_factor = int(spatial_merge_factor)
        self.max_spatial_h = int(max_spatial_h)
        self.max_spatial_w = int(max_spatial_w)
        self.max_audio_tokens_per_seg = int(max_audio_tokens_per_seg)
        self.dropout = float(dropout)
        self.qk_norm = bool(qk_norm)

        self.skip_first_video_latent_frame = bool(skip_first_video_latent_frame)
        self.video_temporal_compress_factor = int(video_temporal_compress_factor)

        # --- ITC+ITM design ---
        # ITC (InfoNCE) uses ONLY post-self-attn CLS, i.e. unimodal encoders -
        # this is the standard ALBEF-style fix that prevents cross-modal
        # information leakage (the cross-attn output would otherwise let the
        # anchor's CLS attend to its own partner tokens, making the
        # contrastive task trivial).
        # ITM (binary pair matching) is the only consumer of cross_attn_blocks;
        # it takes self-attn features of a pair, fuses them via cross-attn,
        # then classifies "is this pair real?" with a small MLP head. Hard
        # negatives are sampled from the local ITC similarity matrix.
        self.use_itm = bool(use_itm)
        self.lambda_itm = float(lambda_itm)
        self.itm_neg_per_direction = int(itm_neg_per_direction)
        self.itm_sim_temperature = float(itm_sim_temperature)
        # itm_neg_source selects where ITM hard negatives come from:
        #   * "near"       : intra (same clip) + sibling (same long-video)
        #                    columns of ITC's neg_idx. Robust from step 0
        #                    because these are structural hard negatives that
        #                    don't depend on ITC having converged.
        #   * "hard_itc"   : full C-column neg pool weighted by ITC sim
        #                    (ALBEF original behaviour). Needs ITC to be at
        #                    least partially trained to give useful signal.
        _allowed_sources = {"near", "hard_itc"}
        if itm_neg_source not in _allowed_sources:
            raise ValueError(
                f"itm_neg_source must be one of {_allowed_sources}, "
                f"got {itm_neg_source!r}."
            )
        self.itm_neg_source = str(itm_neg_source)
        # itm_start_step: number of global training steps to wait before the
        # ITM branch is activated. Until then the head returns ITC-only loss
        # (itm_loss_raw / itm_acc are reported as -1 sentinels). Rationale:
        # at step 0 the cross-attn weights are random and ITC has not yet
        # produced useful features, so ITM tends to collapse to the prior
        # (1/(1+2k)). Letting ITC warm up first gives cross-attn a stable
        # starting point. 0 means "on from the start" (legacy behaviour).
        self.itm_start_step = max(int(itm_start_step), 0)
        # current_step is updated externally by the trainer before every
        # model forward (see trainer training_step). Kept as a plain python
        # int so it does not end up in the state_dict.
        self.current_step = 0
        if self.use_itm and int(cross_attn_layers) <= 0:
            raise ValueError(
                "use_itm=True requires cross_attn_layers > 0."
            )
        if self.use_itm and self.itm_neg_per_direction < 1:
            raise ValueError(
                f"itm_neg_per_direction must be >= 1, got {self.itm_neg_per_direction}."
            )
        if (not self.use_itm) and int(cross_attn_layers) > 0:
            logging.info(
                "IntraSegCrossAttnHead: cross_attn_layers > 0 but use_itm=False; "
                "cross-attn weights will be allocated but never used. Set "
                "cross_attn_layers=0 to save memory/params."
            )

        self.clamp_scale_min = float(clamp_scale_min)
        self.clamp_scale_max = float(clamp_scale_max)
        self.gather_for_loss = bool(gather_for_loss)

        # Trainer expects a per-granularity list of these; expose single granularity.
        self.n_granularities = 1
        self.segment_count: Optional[int] = None
        self.segment_count_list: List[Optional[int]] = [None]
        self.num_negatives_list: List[Optional[int]] = _normalize_to_list(
            num_negatives, 1, "num_negatives",
        )
        self.num_negative_videos_list: List[Optional[int]] = _normalize_to_list(
            num_negative_videos, 1, "num_negative_videos",
        )
        self.same_long_video_priority: bool = bool(same_long_video_priority)
        self.same_long_video_num_negatives_list: List[Optional[int]] = _normalize_to_list(
            same_long_video_num_negatives, 1, "same_long_video_num_negatives",
        )
        self.num_negatives_with_sibling_list: List[Optional[int]] = _normalize_to_list(
            num_negatives_with_sibling, 1, "num_negatives_with_sibling",
        )
        # See LatentAVContrastiveHead docstring on num_negatives_no_sibling:
        #   far-only negative count; derives effective C at forward time.
        self.num_negatives_no_sibling_list: List[Optional[int]] = _normalize_to_list(
            num_negatives_no_sibling, 1, "num_negatives_no_sibling",
        )
        if (
            self.num_negatives_with_sibling_list[0] is not None
            and self.num_negatives_no_sibling_list[0] is not None
        ):
            raise ValueError(
                "num_negatives_with_sibling and num_negatives_no_sibling are "
                "mutually exclusive; please set only one."
            )
        self.num_negatives = self.num_negatives_list[0]
        self.num_negative_videos = self.num_negative_videos_list[0]

        # Only segment-level loss is defined for this head (global would require
        # another aggregator; out of scope for Design C).
        self.use_segment_loss = True
        self.use_global_loss = False

        # ---- Input projections ----
        v_in_dim = int(video_latent_dim) * (self.spatial_merge_factor ** 2)
        self.video_in_proj = nn.Linear(v_in_dim, embed_dim)
        self.audio_in_proj = nn.Linear(int(audio_latent_dim), embed_dim)

        # ---- Learnable CLS tokens ----
        self.cls_video = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_audio = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_video, std=0.02)
        nn.init.trunc_normal_(self.cls_audio, std=0.02)

        # ---- Positional encodings (fixed sinusoidal) ----
        pe_max_h = max(self.max_spatial_h // self.spatial_merge_factor, 1)
        pe_max_w = max(self.max_spatial_w // self.spatial_merge_factor, 1)
        self.video_pe_2d = SinusoidalPE2D(max_h=pe_max_h, max_w=pe_max_w, dim=embed_dim)
        self.audio_pe_1d = SinusoidalPE1D(max_len=self.max_audio_tokens_per_seg, dim=embed_dim)

        # ---- Self-attention towers (Ls layers each) ----
        self.video_self_blocks = nn.ModuleList([
            IntraSegSelfAttnBlock(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout, qk_norm=qk_norm,
            )
            for _ in range(self.self_attn_layers)
        ])
        self.audio_self_blocks = nn.ModuleList([
            IntraSegSelfAttnBlock(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout, qk_norm=qk_norm,
            )
            for _ in range(self.self_attn_layers)
        ])

        # ---- Cross-modal towers (Lx layers each). KV of each layer is fixed
        # to the opposite side's post-self-attn features (ALBEF style).
        self.video_cross_blocks = nn.ModuleList([
            CrossModalBlock(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout, qk_norm=qk_norm,
            )
            for _ in range(self.cross_attn_layers)
        ])
        self.audio_cross_blocks = nn.ModuleList([
            CrossModalBlock(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout, qk_norm=qk_norm,
            )
            for _ in range(self.cross_attn_layers)
        ])

        # ---- Output norms applied to CLS before L2 normalize ----
        self.video_out_norm = nn.LayerNorm(embed_dim)
        self.audio_out_norm = nn.LayerNorm(embed_dim)

        # ---- ITM pair-matching head (only used if use_itm=True). Inputs are
        # post-cross-attn [v_cls_x ; a_cls_x], normalised separately so their
        # magnitudes don't dominate, then an MLP -> scalar logit -> BCE.
        if self.use_itm:
            self.itm_head_norm = nn.LayerNorm(2 * embed_dim)
            self.itm_head = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.itm_head_norm = None
            self.itm_head = None

        # ---- Logit scales ----
        self.logit_scale = nn.Parameter(torch.ones([]) * float(init_scale))
        self.global_logit_scale = None

        logging.info(
            f"IntraSegCrossAttnHead: embed_dim={embed_dim}, nhead={nhead}, "
            f"self_attn_layers={self.self_attn_layers}, cross_attn_layers={self.cross_attn_layers}, "
            f"spatial_merge_factor={self.spatial_merge_factor}, "
            f"max_audio_tokens_per_seg={self.max_audio_tokens_per_seg}, "
            f"video_in_dim={v_in_dim}, audio_in_dim={audio_latent_dim}, "
            f"skip_first_video_latent_frame={self.skip_first_video_latent_frame}"
        )

    # ------------------------- tokenization helpers -------------------------

    def _merge_and_project_video(self, video_latent: torch.Tensor) -> torch.Tensor:
        """``(B, Dv, T, H, W)`` -> ``(B, T, N_v, d)`` with 2D PE added.

        Spatial k x k patches are concatenated (not averaged) so the token dim
        becomes ``k**2 * Dv`` before the linear projection to ``embed_dim``.
        """
        B, Dv, T, H, W = video_latent.shape
        k = self.spatial_merge_factor
        if H % k != 0 or W % k != 0:
            raise ValueError(
                f"Spatial dims ({H}, {W}) must be divisible by spatial_merge_factor={k}"
            )
        Hk, Wk = H // k, W // k

        x = video_latent.permute(0, 2, 3, 4, 1)               # (B, T, H, W, Dv)
        x = x.reshape(B, T, Hk, k, Wk, k, Dv)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()       # (B, T, H/k, W/k, k, k, Dv)
        x = x.reshape(B, T, Hk * Wk, k * k * Dv)              # (B, T, N_v, k^2*Dv)
        x = self.video_in_proj(x)                             # (B, T, N_v, d)

        pe = self.video_pe_2d(Hk, Wk).to(dtype=x.dtype)       # (N_v, d)
        x = x + pe.view(1, 1, Hk * Wk, -1)
        return x

    def _split_and_project_audio(
        self,
        audio_latent: torch.Tensor,
        S: int,
        audio_latent_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Even-split (``n_a_per_seg = L_valid // S``) audio tokens per segment.

        The per-segment token count ``n_a`` is taken as ``min`` of
        ``valid_len_i // S`` across the batch (and clipped by
        ``max_audio_tokens_per_seg``). Each sample contributes its first
        ``S * n_a`` valid tokens; any position past ``valid_lens[i]`` is
        marked in the ``key_padding_mask``.

        Returns:
            x: ``(B, S, n_a, d)`` after 1D PE added.
            key_padding_mask: ``(B, S, n_a)`` bool, True for pad positions.
            n_a: int — tokens per segment.
        """
        B, Da, L_total = audio_latent.shape
        device = audio_latent.device

        if audio_latent_lengths is None:
            valid_lens = torch.full((B,), L_total, dtype=torch.long, device=device)
        else:
            valid_lens = audio_latent_lengths.to(device=device, dtype=torch.long)
            valid_lens = valid_lens.clamp(min=1, max=L_total)

        per_seg_lens = (valid_lens // max(S, 1)).clamp(min=1)   # (B,)
        n_a = int(per_seg_lens.min().item())
        n_a = min(n_a, int(self.max_audio_tokens_per_seg))
        n_a = max(n_a, 1)

        take = S * n_a
        if audio_latent.shape[-1] < take:
            # Pad on the right so that every sample has at least ``take`` positions.
            pad_len = take - audio_latent.shape[-1]
            audio_latent = F.pad(audio_latent, (0, pad_len))
        x = audio_latent[:, :, :take].contiguous()              # (B, Da, S*n_a)

        pos = torch.arange(take, device=device).unsqueeze(0).expand(B, -1)
        kpm_flat = pos >= valid_lens.unsqueeze(1)               # (B, S*n_a)

        x = x.transpose(1, 2).contiguous()                      # (B, S*n_a, Da)
        x = self.audio_in_proj(x)                               # (B, S*n_a, d)
        x = x.view(B, S, n_a, -1)                               # (B, S, n_a, d)
        kpm = kpm_flat.view(B, S, n_a)

        pe = self.audio_pe_1d(n_a).to(dtype=x.dtype)            # (n_a, d)
        x = x + pe.view(1, 1, n_a, -1)
        return x, kpm, n_a

    # ------------------------------- forward -------------------------------

    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        audio_latent_lengths: Optional[torch.Tensor] = None,
        world_size: int = 1,
        long_video_ids_pool: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.skip_first_video_latent_frame:
            video_latent, audio_latent, audio_latent_lengths = (
                self._align_causal_first_frame(video_latent, audio_latent, audio_latent_lengths)
            )

        logit_scale, global_logit_scale = self.clamp_logit_scales()
        B, Dv, T, H, W = video_latent.shape
        S = int(T)

        # ---- Per-segment tokenization ----
        v_tokens = self._merge_and_project_video(video_latent)                 # (B, T, N_v, d)
        a_tokens, a_kpm, n_a = self._split_and_project_audio(
            audio_latent, S, audio_latent_lengths,
        )                                                                      # (B, T, n_a, d)

        N_v = v_tokens.shape[2]
        d = v_tokens.shape[-1]
        BT = B * S

        v_seq = v_tokens.reshape(BT, N_v, d)
        a_seq = a_tokens.reshape(BT, n_a, d)
        a_kpm_seq = a_kpm.reshape(BT, n_a)

        # ---- Prepend CLS on each side ----
        cls_v = self.cls_video.to(dtype=v_seq.dtype).expand(BT, -1, -1)        # (BT, 1, d)
        cls_a = self.cls_audio.to(dtype=a_seq.dtype).expand(BT, -1, -1)
        v_seq = torch.cat([cls_v, v_seq], dim=1)                               # (BT, 1+N_v, d)
        a_seq = torch.cat([cls_a, a_seq], dim=1)                               # (BT, 1+n_a, d)

        cls_mask = torch.zeros(BT, 1, dtype=torch.bool, device=a_kpm_seq.device)
        a_kpm_seq = torch.cat([cls_mask, a_kpm_seq], dim=1)                    # (BT, 1+n_a)
        v_kpm_seq: Optional[torch.Tensor] = None  # no padding on the video side

        # ---- Self-attention per modality (unimodal encoders) ----
        for blk in self.video_self_blocks:
            v_seq = blk(v_seq, key_padding_mask=v_kpm_seq)
        for blk in self.audio_self_blocks:
            a_seq = blk(a_seq, key_padding_mask=a_kpm_seq)

        # ====== ITC: extract CLS from post-self-attn (NO cross-attn) ======
        # Using post-self-attn CLS is the standard ALBEF fix that prevents
        # cross-modal leakage in contrastive learning.
        v_cls_itc = self.video_out_norm(v_seq[:, 0])                           # (BT, d)
        a_cls_itc = self.audio_out_norm(a_seq[:, 0])
        segment_vfeat = F.normalize(v_cls_itc, dim=-1)
        segment_afeat = F.normalize(a_cls_itc, dim=-1)

        # ---- Gather across ranks + InfoNCE (same machinery as the base head) ----
        rank_offset, B_eff, vfeat_pool, afeat_pool = self._gather_segment_feats(
            world_size, B, S, segment_vfeat, segment_afeat,
        )

        # itm_active combines the compile-time switch (use_itm) with the
        # runtime step-based warmup gate (itm_start_step). Before the start
        # step the head behaves exactly like use_itm=False: ITC only, no
        # neg_idx materialised, no cross-attn forward. The ITM head weights
        # still receive zero gradient during this window but are created
        # only when use_itm=True, so toggling is cheap.
        itm_active = self.use_itm and (self.current_step >= self.itm_start_step)

        # Ask sample_negatives_for_loss to also expose neg_idx so ITM can reuse
        # the exact same negative pool as ITC (same 3-tier: intra + sibling + far)
        # when ITM is active. Otherwise we keep the 4-tuple return to avoid
        # paying the extra tensor allocation.
        if itm_active:
            (
                sim_v2a, sim_a2v, targets, _, neg_idx, near_mask,
            ) = self.sample_negatives_for_loss(
                vfeat_local=segment_vfeat,
                afeat_local=segment_afeat,
                vfeat_pool=vfeat_pool,
                afeat_pool=afeat_pool,
                B=B, B_eff=B_eff, S=S,
                scale=self.logit_scale,
                rank_offset=rank_offset,
                num_negatives=self.num_negatives_list[0],
                num_negative_videos=self.num_negative_videos_list[0],
                long_video_ids_pool=long_video_ids_pool,
                same_long_video_num_negatives=self.same_long_video_num_negatives_list[0],
                num_negatives_with_sibling=self._resolve_num_negatives_with_sibling(0, S),
                return_neg_idx=True,
            )
        else:
            sim_v2a, sim_a2v, targets, _ = self.sample_negatives_for_loss(
                vfeat_local=segment_vfeat,
                afeat_local=segment_afeat,
                vfeat_pool=vfeat_pool,
                afeat_pool=afeat_pool,
                B=B, B_eff=B_eff, S=S,
                scale=self.logit_scale,
                rank_offset=rank_offset,
                num_negatives=self.num_negatives_list[0],
                num_negative_videos=self.num_negative_videos_list[0],
                long_video_ids_pool=long_video_ids_pool,
                same_long_video_num_negatives=self.same_long_video_num_negatives_list[0],
                num_negatives_with_sibling=self._resolve_num_negatives_with_sibling(0, S),
            )
            neg_idx = None
            near_mask = None

        itc_loss = (
            F.cross_entropy(sim_v2a, targets) + F.cross_entropy(sim_a2v, targets)
        ) / 2

        # ====== ITM: pair matching with cross-attn + BCE (optional) ======
        losses_dict: Dict[str, Any] = {}
        if itm_active:
            itm_loss, itm_acc = self._compute_itm_loss(
                v_seq=v_seq, a_seq=a_seq, a_kpm_seq=a_kpm_seq,
                sim_v2a=sim_v2a, sim_a2v=sim_a2v, neg_idx=neg_idx,
                near_mask=near_mask,
                world_size=world_size,
                B=B, B_eff=B_eff, S=S, rank_offset=rank_offset,
            )
            losses_dict["itc_loss_raw"] = itc_loss.detach()
            losses_dict["itm_loss_raw"] = itm_loss.detach()
            losses_dict["itm_acc"] = itm_acc.detach()
            # The trainer only consumes `segment_contrastive_loss` from each
            # granularity; combine ITC + ITM here so existing loss plumbing
            # (lambda_segment_contrastive, adaptive balancing, etc.) stays
            # unchanged.
            segment_loss = itc_loss + self.lambda_itm * itm_loss
        elif self.use_itm:
            # ITM is configured but not yet active (warmup before
            # itm_start_step). Log ITC raw so the itc/segment curves stay
            # continuous, but skip itm_loss_raw / itm_acc - trainer's
            # None-check will simply not log those keys during warmup,
            # which is the cleanest way to show the gap on dashboards.
            losses_dict["itc_loss_raw"] = itc_loss.detach()
            segment_loss = itc_loss
        else:
            segment_loss = itc_loss

        losses_dict["segment_contrastive_loss"] = segment_loss

        granularity: Dict[str, Any] = {
            "segment_count": None,
            "S": S,
            "losses": losses_dict,
            "segment_vfeat": segment_vfeat,
            "segment_afeat": segment_afeat,
            "segment_vfeat_pool": vfeat_pool,
            "segment_afeat_pool": afeat_pool,
            "B": B, "B_eff": B_eff, "rank_offset": rank_offset,
        }

        return {
            "logit_scales": (logit_scale, global_logit_scale),
            "granularities": [granularity],
            "losses": {},
        }

    def _compute_itm_loss(
        self,
        v_seq: torch.Tensor,        # (BS, 1+N_v, d)   local post-self-attn, with CLS
        a_seq: torch.Tensor,        # (BS, 1+n_a, d)
        a_kpm_seq: torch.Tensor,    # (BS, 1+n_a)      bool
        sim_v2a: torch.Tensor,      # (BS, 1+C)        ITC scores: col 0 = pos, 1..C = neg
        sim_a2v: torch.Tensor,      # (BS, 1+C)
        neg_idx: torch.Tensor,      # (BS, C)          indices into the GATHERED pool
        near_mask: Optional[torch.Tensor],  # (BS, C) bool, True on intra/sibling cols
        world_size: int,
        B: int,
        B_eff: int,
        S: int,
        rank_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ALBEF-style ITM: positive pair + 2*k hard negatives per anchor.

        Negative candidates are drawn from ITC's neg pool (shape ``(BS, C)``)
        indexed by ``neg_idx`` into the gathered cross-rank pool. Which
        subset of the C columns to sample from is controlled by
        ``self.itm_neg_source``:

        - ``"near"``: restrict to intra + sibling columns (structural hard
          negatives that are informative from step 0, independent of ITC
          convergence). Within that subset sampling weight is still
          ``softmax(sim/T)``, so once ITC sim becomes useful we still pick
          the hardest inside the near pool.
        - ``"hard_itc"``: sample from all C columns weighted by ITC sim
          (original ALBEF behaviour; relies on ITC being trained enough).

        Cross-rank token sequences are all-gathered so negatives can span
        the whole pool; non-local-rank shards are detached (standard
        dist.all_gather behaviour), so gradient only flows via the local
        anchor pathway + cross-attn + ITM MLP (ALBEF semantics).
        """
        BS = B * S
        k = int(self.itm_neg_per_direction)
        C = int(neg_idx.shape[1])
        device = v_seq.device

        if BS < 2 or C < 1:
            # Degenerate shapes: no candidate negatives -> return zero so the
            # loss doesn't break grad graph.
            zero = v_seq.sum() * 0.0
            return zero, torch.zeros((), device=device)

        # We can never hard-mine more negatives than C columns (without
        # replacement). When the pool is smaller than k we silently clip.
        k_eff = int(min(k, C))

        # ---------------- 1) Gather sequences across ranks ----------------
        if world_size > 1 and self.gather_for_loss:
            v_seq_pool = torch.cat(dist_all_gather(v_seq), dim=0)
            a_seq_pool = torch.cat(dist_all_gather(a_seq), dim=0)
            a_kpm_pool = torch.cat(dist_all_gather(a_kpm_seq), dim=0)
        else:
            v_seq_pool = v_seq
            a_seq_pool = a_seq
            a_kpm_pool = a_kpm_seq

        # ---------------- 2) Build candidate mask + hard mining ----------
        with torch.no_grad():
            T_temp = max(float(self.itm_sim_temperature), 1e-4)
            scores_v2a = sim_v2a[:, 1:].detach().float() / T_temp             # (BS, C)
            scores_a2v = sim_a2v[:, 1:].detach().float() / T_temp

            use_near = (self.itm_neg_source == "near" and near_mask is not None)
            if use_near:
                # Rows without any near candidates (e.g. degenerate S=1 case)
                # transparently fall back to the full pool, otherwise
                # multinomial would explode on an all-(-inf) row.
                row_has_near = near_mask.any(dim=1, keepdim=True)             # (BS, 1)
                effective_mask = torch.where(
                    row_has_near, near_mask, torch.ones_like(near_mask)
                )
                neg_inf = torch.finfo(scores_v2a.dtype).min
                scores_v2a = scores_v2a.masked_fill(~effective_mask, neg_inf)
                scores_a2v = scores_a2v.masked_fill(~effective_mask, neg_inf)

            p_v2a = F.softmax(scores_v2a, dim=1)
            p_a2v = F.softmax(scores_a2v, dim=1)
            p_v2a = torch.nan_to_num(p_v2a, nan=0.0, posinf=0.0, neginf=0.0)
            p_a2v = torch.nan_to_num(p_a2v, nan=0.0, posinf=0.0, neginf=0.0)
            row_sum_v = p_v2a.sum(dim=1, keepdim=True)
            row_sum_a = p_a2v.sum(dim=1, keepdim=True)
            uniform = torch.full_like(p_v2a, 1.0 / max(C, 1))
            p_v2a = torch.where(row_sum_v > 0, p_v2a / row_sum_v.clamp_min(1e-12), uniform)
            p_a2v = torch.where(row_sum_a > 0, p_a2v / row_sum_a.clamp_min(1e-12), uniform)

            # In "near" mode a row's effective candidate count can be as low
            # as num_intra (~S-1). If that is smaller than k we must sample
            # with replacement to avoid multinomial errors.
            if use_near:
                min_near = int(effective_mask.sum(dim=1).min().item())
                replace = (k_eff > min_near)
            else:
                replace = (k_eff > C)
            hard_cols_v2a = torch.multinomial(p_v2a, k_eff, replacement=replace)  # (BS, k_eff)
            hard_cols_a2v = torch.multinomial(p_a2v, k_eff, replacement=replace)

            hard_a_idx = neg_idx.gather(1, hard_cols_v2a)                     # (BS, k_eff)
            hard_v_idx = neg_idx.gather(1, hard_cols_a2v)

        # ---------------- 3) Build (1 + 2*k_eff)*BS triples --------------
        # Positive: (v_seq[i], a_seq[i])
        # v->a negs: (v_seq[i] broadcast, a_seq_pool[hard_a_idx[i, j]])
        # a->v negs: (v_seq_pool[hard_v_idx[i, j]], a_seq[i] broadcast)
        hard_a_flat = hard_a_idx.reshape(-1)                                  # (BS*k_eff,)
        hard_v_flat = hard_v_idx.reshape(-1)

        # Broadcast local v_seq over k_eff copies for the v->a direction.
        v_rep = v_seq.unsqueeze(1).expand(BS, k_eff, -1, -1).reshape(
            BS * k_eff, v_seq.shape[1], v_seq.shape[2]
        )
        a_rep = a_seq.unsqueeze(1).expand(BS, k_eff, -1, -1).reshape(
            BS * k_eff, a_seq.shape[1], a_seq.shape[2]
        )
        a_kpm_rep = a_kpm_seq.unsqueeze(1).expand(BS, k_eff, -1).reshape(
            BS * k_eff, a_kpm_seq.shape[1]
        )

        a_hard = a_seq_pool[hard_a_flat]                                      # (BS*k, 1+n_a, d)
        a_kpm_hard = a_kpm_pool[hard_a_flat]
        v_hard = v_seq_pool[hard_v_flat]                                      # (BS*k, 1+N_v, d)

        v_stack = torch.cat([v_seq, v_rep, v_hard], dim=0)                    # (BS*(1+2k), 1+N_v, d)
        a_stack = torch.cat([a_seq, a_hard, a_rep], dim=0)                    # (BS*(1+2k), 1+n_a, d)
        a_kpm_stack = torch.cat([a_kpm_seq, a_kpm_hard, a_kpm_rep], dim=0)

        # ---------------- 4) Cross-attn fusion ---------------------------
        # Use the stacked KV of each side (NOT the pre-fusion snapshot, since
        # each triple has its own KV assignment after concat).
        v_x = v_stack
        a_x = a_stack
        for blk in self.video_cross_blocks:
            v_x = blk(v_x, a_stack, key_padding_mask=a_kpm_stack)
        for blk in self.audio_cross_blocks:
            a_x = blk(a_x, v_stack, key_padding_mask=None)

        # ---------------- 5) ITM head + BCE ------------------------------
        v_cls_x = v_x[:, 0]                                                   # (BS*(1+2k), d)
        a_cls_x = a_x[:, 0]
        pair_feat = torch.cat([v_cls_x, a_cls_x], dim=-1)                     # (BS*(1+2k), 2d)
        pair_feat = self.itm_head_norm(pair_feat)
        logits = self.itm_head(pair_feat).squeeze(-1)                         # (BS*(1+2k),)

        labels = torch.cat(
            [torch.ones(BS, device=device, dtype=logits.dtype),
             torch.zeros(BS * k_eff, device=device, dtype=logits.dtype),
             torch.zeros(BS * k_eff, device=device, dtype=logits.dtype)],
            dim=0,
        )
        itm_loss = F.binary_cross_entropy_with_logits(logits, labels)
        with torch.no_grad():
            preds = (logits > 0).to(labels.dtype)
            itm_acc = (preds == labels).float().mean()
        return itm_loss, itm_acc
