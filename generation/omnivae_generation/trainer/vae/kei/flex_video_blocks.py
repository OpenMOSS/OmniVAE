from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from .modern_video_blocks import SwiGLU

try:
    import natten
except ImportError:
    natten = None

Position3D = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
GridShape3D = tuple[int, int, int]
PositionEmbeddings3D = tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]

_COMPILED_CREATE_BLOCK_MASK: Callable[..., BlockMask] | None = None
_COMPILED_FLEX_ATTENTION: Callable[..., torch.Tensor] | None = None
_FLEX_ATTENTION_BACKENDS = {"auto", "triton", "flash"}
_NATTEN_ATTENTION_BACKEND_MAP: dict[str, str | None] = {
    "natten_auto": None,
    "natten_cutlass-fna": "cutlass-fna",
    "natten_hopper-fna": "hopper-fna",
    "natten_blackwell-fna": "blackwell-fna",
    "natten_flex-fna": "flex-fna",
}


@torch.compiler.disable
def _run_natten_na3d_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    kernel_size: GridShape3D,
    is_causal: tuple[bool, bool, bool],
    backend: str | None,
) -> torch.Tensor:
    if natten is None:
        raise ImportError("NATTEN attention backend requires the optional 'natten' package to be installed")
    return natten.na3d(
        query,
        key,
        value,
        kernel_size=kernel_size,
        stride=1,
        dilation=1,
        is_causal=is_causal,
        backend=backend,
    )


def rms_norm_preserve_input_dtype(x: torch.Tensor, norm: nn.RMSNorm) -> torch.Tensor:
    weight = norm.weight
    if weight is not None and weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    return F.rms_norm(x, norm.normalized_shape, weight, norm.eps)


def _effective_block_size(block_size: int, *, seq_len: int) -> int:
    n = max(1, min(int(block_size), int(seq_len)))
    return 1 << int(math.floor(math.log2(float(n))))


def _parse_attention_backend(attention_backend: str) -> tuple[str, dict[str, Any] | None, str | None]:
    backend = str(attention_backend).strip().lower()
    if backend == "auto":
        return "flex", None, None
    if backend == "triton":
        return "flex", {"FORCE_USE_FLEX_ATTENTION": True}, None
    if backend == "flash":
        return "flex", {"BACKEND": "FLASH"}, None
    if backend in _NATTEN_ATTENTION_BACKEND_MAP:
        return "natten", None, _NATTEN_ATTENTION_BACKEND_MAP[backend]
    raise ValueError(f"Unsupported attention_backend: {attention_backend!r}")


def _normalize_grid_shape(grid_shape: GridShape3D | None, *, default: GridShape3D) -> GridShape3D:
    if grid_shape is None:
        return default
    gt, gh, gw = (int(v) for v in grid_shape)
    if gt <= 0 or gh <= 0 or gw <= 0:
        raise ValueError(f"grid_shape entries must be > 0, got {(gt, gh, gw)!r}")
    return (gt, gh, gw)


def _slice_position_embeddings(
    position_embeddings: PositionEmbeddings3D,
    *,
    token_start: int,
    token_end: int,
) -> PositionEmbeddings3D:
    return tuple(
        (cos[:, :, int(token_start) : int(token_end), :], sin[:, :, int(token_start) : int(token_end), :])
        for cos, sin in position_embeddings
    )


def _natten_kernel_size_for_grid(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    window_t: int,
    window_h: int,
    window_w: int,
    causal: bool,
) -> GridShape3D:
    kernel_t = int(window_t + 1) if bool(causal) else int(2 * window_t + 1)
    kernel_h = int(2 * window_h + 1)
    kernel_w = int(2 * window_w + 1)
    return (
        max(1, min(int(kernel_t), int(grid_t))),
        max(1, min(int(kernel_h), int(grid_h))),
        max(1, min(int(kernel_w), int(grid_w))),
    )


def _build_local_3d_mask_mod(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    window_t: int,
    window_h: int,
    window_w: int,
    causal: bool,
):
    tokens_per_t = int(grid_h) * int(grid_w)
    grid_w_int = int(grid_w)
    w_t = int(window_t)
    w_h = int(window_h)
    w_w = int(window_w)
    is_causal = bool(causal)

    def mask_mod(batch: torch.Tensor, head: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        del batch, head
        q_t = torch.div(q_idx, tokens_per_t, rounding_mode="floor")
        kv_t = torch.div(kv_idx, tokens_per_t, rounding_mode="floor")

        q_hw = torch.remainder(q_idx, tokens_per_t)
        kv_hw = torch.remainder(kv_idx, tokens_per_t)
        q_h = torch.div(q_hw, grid_w_int, rounding_mode="floor")
        kv_h = torch.div(kv_hw, grid_w_int, rounding_mode="floor")
        q_w = torch.remainder(q_hw, grid_w_int)
        kv_w = torch.remainder(kv_hw, grid_w_int)

        if is_causal:
            t_ok = (q_t >= kv_t) & ((q_t - kv_t) <= w_t)
        else:
            t_ok = torch.abs(q_t - kv_t) <= w_t
        h_ok = torch.abs(q_h - kv_h) <= w_h
        w_ok = torch.abs(q_w - kv_w) <= w_w
        return t_ok & h_ok & w_ok

    return mask_mod


def _get_compiled_create_block_mask() -> Callable[..., BlockMask]:
    global _COMPILED_CREATE_BLOCK_MASK
    if _COMPILED_CREATE_BLOCK_MASK is None:
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask, dynamic=False)
    return _COMPILED_CREATE_BLOCK_MASK


def _get_compiled_flex_attention() -> Callable[..., torch.Tensor]:
    global _COMPILED_FLEX_ATTENTION
    if torch.compiler.is_dynamo_compiling():
        return flex_attention
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX_ATTENTION


def _build_local_3d_block_mask(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    window_t: int,
    window_h: int,
    window_w: int,
    causal: bool,
    block_size: int,
    share_across_batch_heads: bool,
    heads: int,
    device: torch.device,
) -> BlockMask:
    seq_len = int(grid_t) * int(grid_h) * int(grid_w)
    mask_mod = _build_local_3d_mask_mod(
        grid_t=int(grid_t),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        window_t=int(window_t),
        window_h=int(window_h),
        window_w=int(window_w),
        causal=bool(causal),
    )
    builder = _get_compiled_create_block_mask()
    return builder(
        mask_mod,
        1,
        1 if bool(share_across_batch_heads) else int(heads),
        int(seq_len),
        int(seq_len),
        device=device,
        BLOCK_SIZE=int(_effective_block_size(block_size, seq_len=seq_len)),
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPE3D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE3D requires even head_dim, got head_dim={self.head_dim}")
        if self.head_dim < 6:
            raise ValueError(f"RoPE3D requires head_dim >= 6, got head_dim={self.head_dim}")

        pair_count = self.head_dim // 2
        pairs_t = pair_count // 3
        pairs_h = pair_count // 3
        pairs_w = pair_count // 3
        remainder = pair_count - (pairs_t + pairs_h + pairs_w)
        if remainder >= 1:
            pairs_t += 1
        if remainder >= 2:
            pairs_h += 1

        self.axis_pair_counts = (int(pairs_t), int(pairs_h), int(pairs_w))
        self.axis_dims = tuple(2 * x for x in self.axis_pair_counts)
        max_pairs = max(self.axis_pair_counts)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, max_pairs, dtype=torch.float32) / float(max_pairs)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_axis_position_embeddings(
        self,
        pos: torch.Tensor,
        *,
        axis_pairs: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("l,d->ld", pos.to(device=device, dtype=torch.float32), self.inv_freq[:axis_pairs])
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def build_position_embeddings(
        self,
        *,
        pos_t: torch.Tensor,
        pos_h: torch.Tensor,
        pos_w: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> PositionEmbeddings3D:
        return (
            self._build_axis_position_embeddings(pos_t, axis_pairs=int(self.axis_pair_counts[0]), dtype=dtype, device=device),
            self._build_axis_position_embeddings(pos_h, axis_pairs=int(self.axis_pair_counts[1]), dtype=dtype, device=device),
            self._build_axis_position_embeddings(pos_w, axis_pairs=int(self.axis_pair_counts[2]), dtype=dtype, device=device),
        )

    def _apply_axis(
        self,
        x: torch.Tensor,
        *,
        embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = embeddings
        return (x * cos) + (_rotate_half(x) * sin)

    def apply_position_embeddings(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: PositionEmbeddings3D,
    ) -> torch.Tensor:
        dt, dh, dw = (int(v) for v in self.axis_dims)
        xt, xh, xw = x.split((dt, dh, dw), dim=-1)
        emb_t, emb_h, emb_w = position_embeddings
        xt = self._apply_axis(xt, embeddings=emb_t)
        xh = self._apply_axis(xh, embeddings=emb_h)
        xw = self._apply_axis(xw, embeddings=emb_w)
        return torch.cat([xt, xh, xw], dim=-1)


class FlexSelfAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        attention_mode: str = "sparse_local",
        causal: bool = False,
        window_t: int = 4,
        window_h: int = 1,
        window_w: int = 1,
        block_size: int = 128,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        share_mask_across_batch_heads: bool = True,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim({self.dim}) must be divisible by num_heads({self.num_heads})")
        self.head_dim = self.dim // self.num_heads
        self.grid_t = int(grid_t)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.seq_len = int(self.grid_t * self.grid_h * self.grid_w)
        self.attention_mode = str(attention_mode).strip().lower()
        self.causal = bool(causal)
        self.window_t = int(window_t)
        self.window_h = int(window_h)
        self.window_w = int(window_w)
        self.block_size = int(block_size)
        self.share_mask_across_batch_heads = bool(share_mask_across_batch_heads)
        self.qk_norm = bool(qk_norm)
        self.attention_backend = str(attention_backend).strip().lower()
        self.attention_impl, self.kernel_options, self.natten_backend = _parse_attention_backend(self.attention_backend)

        if self.window_t < 0 or self.window_h < 0 or self.window_w < 0:
            raise ValueError("window_t/window_h/window_w must be >= 0")
        if self.attention_mode not in {"dense", "sparse_local"}:
            raise ValueError(f"Unsupported attention_mode: {attention_mode!r}")
        if self.attention_backend not in (_FLEX_ATTENTION_BACKENDS | set(_NATTEN_ATTENTION_BACKEND_MAP)):
            raise ValueError(f"Unsupported attention_backend: {attention_backend!r}")
        if self.attention_impl == "natten" and self.attention_mode != "sparse_local":
            raise ValueError(
                f"NATTEN backends only support sparse_local attention_mode, got attention_mode={self.attention_mode!r}"
            )
        if self.attention_impl == "natten" and natten is None:
            raise ImportError(
                f"attention_backend={self.attention_backend!r} requires the optional 'natten' package to be installed"
            )

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        self.rope = RoPE3D(self.head_dim, base=float(rope_base))
        if self.qk_norm:
            self.q_norm: nn.RMSNorm | None = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
            self.k_norm: nn.RMSNorm | None = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

        self._block_mask_by_device: dict[tuple[str, int, int, int], BlockMask] = {}
        self._dense_mask_by_device: dict[tuple[str, int, int, int], torch.Tensor] = {}

    def _block_mask_for_device(
        self,
        device: torch.device,
        *,
        grid_shape: GridShape3D | None = None,
    ) -> BlockMask | None:
        if self.attention_mode != "sparse_local":
            return None
        gt, gh, gw = _normalize_grid_shape(grid_shape, default=(self.grid_t, self.grid_h, self.grid_w))
        key = (str(device), int(gt), int(gh), int(gw))
        cached = self._block_mask_by_device.get(key, None)
        if cached is None:
            cached = _build_local_3d_block_mask(
                grid_t=int(gt),
                grid_h=int(gh),
                grid_w=int(gw),
                window_t=int(self.window_t),
                window_h=int(self.window_h),
                window_w=int(self.window_w),
                causal=bool(self.causal),
                block_size=int(self.block_size),
                share_across_batch_heads=bool(self.share_mask_across_batch_heads),
                heads=int(self.num_heads),
                device=device,
            )
            self._block_mask_by_device[key] = cached
        return cached

    def _dense_mask_for_device(
        self,
        device: torch.device,
        *,
        grid_shape: GridShape3D | None = None,
    ) -> torch.Tensor | None:
        if self.attention_mode != "sparse_local":
            return None
        gt, gh, gw = _normalize_grid_shape(grid_shape, default=(self.grid_t, self.grid_h, self.grid_w))
        key = (str(device), int(gt), int(gh), int(gw))
        cached = self._dense_mask_by_device.get(key, None)
        if cached is not None:
            return cached

        seq_len = int(gt * gh * gw)
        tokens_per_t = int(gh * gw)
        q_idx = torch.arange(seq_len, device=device)
        kv_idx = torch.arange(seq_len, device=device)
        q_t = torch.div(q_idx, tokens_per_t, rounding_mode="floor")
        kv_t = torch.div(kv_idx, tokens_per_t, rounding_mode="floor")
        q_hw = torch.remainder(q_idx, tokens_per_t)
        kv_hw = torch.remainder(kv_idx, tokens_per_t)
        q_h = torch.div(q_hw, int(gw), rounding_mode="floor")
        kv_h = torch.div(kv_hw, int(gw), rounding_mode="floor")
        q_w = torch.remainder(q_hw, int(gw))
        kv_w = torch.remainder(kv_hw, int(gw))

        if self.causal:
            t_ok = (q_t[:, None] >= kv_t[None, :]) & ((q_t[:, None] - kv_t[None, :]) <= int(self.window_t))
        else:
            t_ok = torch.abs(q_t[:, None] - kv_t[None, :]) <= int(self.window_t)
        h_ok = torch.abs(q_h[:, None] - kv_h[None, :]) <= int(self.window_h)
        w_ok = torch.abs(q_w[:, None] - kv_w[None, :]) <= int(self.window_w)
        cached = (t_ok & h_ok & w_ok).to(dtype=torch.bool)
        self._dense_mask_by_device[key] = cached
        return cached

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: PositionEmbeddings3D,
        grid_shape: GridShape3D | None = None,
    ) -> torch.Tensor:
        b, l, _ = x.shape
        gt, gh, gw = _normalize_grid_shape(grid_shape, default=(self.grid_t, self.grid_h, self.grid_w))
        seq_len = int(gt * gh * gw)
        if int(l) != seq_len:
            raise ValueError(f"sequence length mismatch: got {l}, expected {seq_len}")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q = rms_norm_preserve_input_dtype(q, self.q_norm)
            k = rms_norm_preserve_input_dtype(k, self.k_norm)

        q = self.rope.apply_position_embeddings(q, position_embeddings=position_embeddings)
        k = self.rope.apply_position_embeddings(k, position_embeddings=position_embeddings)

        if q.device.type == "cpu":
            dense_mask = self._dense_mask_for_device(q.device, grid_shape=(gt, gh, gw))
            if dense_mask is not None:
                dense_mask = dense_mask.view(1, 1, seq_len, seq_len)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask, dropout_p=0.0)
            out = out.transpose(1, 2).contiguous().view(b, l, self.dim)
            return self.proj(out)

        if self.attention_impl == "natten":
            kernel_size = _natten_kernel_size_for_grid(
                grid_t=int(gt),
                grid_h=int(gh),
                grid_w=int(gw),
                window_t=int(self.window_t),
                window_h=int(self.window_h),
                window_w=int(self.window_w),
                causal=bool(self.causal),
            )
            if min(kernel_size) >= 2:
                is_causal = (bool(self.causal), False, False)
                q_natten = q.transpose(1, 2).view(
                    b,
                    int(gt),
                    int(gh),
                    int(gw),
                    self.num_heads,
                    self.head_dim,
                )
                k_natten = k.transpose(1, 2).view(
                    b,
                    int(gt),
                    int(gh),
                    int(gw),
                    self.num_heads,
                    self.head_dim,
                )
                v_natten = v.transpose(1, 2).view(
                    b,
                    int(gt),
                    int(gh),
                    int(gw),
                    self.num_heads,
                    self.head_dim,
                )
                out = _run_natten_na3d_eager(
                    q_natten,
                    k_natten,
                    v_natten,
                    kernel_size=kernel_size,
                    is_causal=is_causal,
                    backend=self.natten_backend,
                )
                out = out.view(b, l, self.dim)
                return self.proj(out)
            dense_mask = self._dense_mask_for_device(q.device, grid_shape=(gt, gh, gw))
            if dense_mask is not None:
                dense_mask = dense_mask.view(1, 1, seq_len, seq_len)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask, dropout_p=0.0)
            out = out.transpose(1, 2).contiguous().view(b, l, self.dim)
            return self.proj(out)

        block_mask = self._block_mask_for_device(q.device, grid_shape=(gt, gh, gw))
        out = _get_compiled_flex_attention()(q, k, v, block_mask=block_mask, kernel_options=self.kernel_options)
        out = out.transpose(1, 2).contiguous().view(b, l, self.dim)
        return self.proj(out)


class TransformerFlexBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        attention_mode: str = "sparse_local",
        causal: bool = False,
        window_t: int = 4,
        window_h: int = 1,
        window_w: int = 1,
        block_size: int = 128,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        share_mask_across_batch_heads: bool = True,
        attention_backend: str = "auto",
        temporal_block_chunk_size: int | None = None,
        temporal_chunk_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        if temporal_block_chunk_size is not None and int(temporal_block_chunk_size) <= 0:
            raise ValueError(f"temporal_block_chunk_size must be > 0 when set, got {temporal_block_chunk_size!r}")
        self.temporal_block_chunk_size = None if temporal_block_chunk_size is None else int(temporal_block_chunk_size)
        self.temporal_chunk_checkpointing = bool(temporal_chunk_checkpointing)
        self.norm1 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = FlexSelfAttention3D(
            dim,
            num_heads,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            attention_mode=attention_mode,
            causal=causal,
            window_t=window_t,
            window_h=window_h,
            window_w=window_w,
            block_size=block_size,
            rope_base=rope_base,
            qk_norm=qk_norm,
            share_mask_across_batch_heads=share_mask_across_batch_heads,
            attention_backend=attention_backend,
        )
        self.norm2 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.mlp = SwiGLU(dim, hidden_dim=hidden)

    def _temporal_chunk_size_for_grid(self, grid_shape: GridShape3D | None) -> int | None:
        if self.temporal_block_chunk_size is None:
            return None
        if self.attn.attention_mode != "sparse_local":
            return None
        gt, _, _ = _normalize_grid_shape(grid_shape, default=(self.attn.grid_t, self.attn.grid_h, self.attn.grid_w))
        chunk_t = int(self.temporal_block_chunk_size)
        if chunk_t >= int(gt):
            return None
        return chunk_t

    def uses_temporal_chunk_checkpointing(self, grid_shape: GridShape3D | None = None) -> bool:
        return (
            bool(self.temporal_chunk_checkpointing)
            and self.training
            and torch.is_grad_enabled()
            and self._temporal_chunk_size_for_grid(grid_shape) is not None
        )

    def _forward_full(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: PositionEmbeddings3D,
        grid_shape: GridShape3D,
    ) -> torch.Tensor:
        x = x + self.attn(
            rms_norm_preserve_input_dtype(x, self.norm1),
            position_embeddings=position_embeddings,
            grid_shape=grid_shape,
        )
        x = x + self.mlp(rms_norm_preserve_input_dtype(x, self.norm2))
        return x

    def _forward_temporal_chunked(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: PositionEmbeddings3D,
        grid_shape: GridShape3D,
        temporal_chunk_size: int,
    ) -> torch.Tensor:
        gt, gh, gw = (int(v) for v in grid_shape)
        tokens_per_t = int(gh * gw)
        chunk_t = max(1, int(temporal_chunk_size))
        left_halo_t = int(self.attn.window_t)
        right_halo_t = 0 if bool(self.attn.causal) else int(self.attn.window_t)
        checkpoint_chunks = (
            bool(self.temporal_chunk_checkpointing) and self.training and torch.is_grad_enabled()
        )
        output = x.new_empty(x.shape) if not torch.is_grad_enabled() else None
        chunks: list[torch.Tensor] = []

        for q_t0 in range(0, gt, chunk_t):
            q_t1 = min(gt, int(q_t0 + chunk_t))
            q_token0 = int(q_t0 * tokens_per_t)
            q_token1 = int(q_t1 * tokens_per_t)

            def compute_chunk(
                source_x: torch.Tensor,
                *,
                q_t0_bound: int = int(q_t0),
                q_t1_bound: int = int(q_t1),
                q_token0_bound: int = int(q_token0),
                q_token1_bound: int = int(q_token1),
            ) -> torch.Tensor:
                ctx_t0 = max(0, int(q_t0_bound - left_halo_t))
                ctx_t1 = min(gt, int(q_t1_bound + right_halo_t))
                ctx_token0 = int(ctx_t0 * tokens_per_t)
                ctx_token1 = int(ctx_t1 * tokens_per_t)
                keep_token0 = int((q_t0_bound - ctx_t0) * tokens_per_t)
                keep_token1 = int(keep_token0 + (q_t1_bound - q_t0_bound) * tokens_per_t)
                x_ctx = source_x[:, ctx_token0:ctx_token1, :]
                pos_ctx = _slice_position_embeddings(
                    position_embeddings,
                    token_start=ctx_token0,
                    token_end=ctx_token1,
                )
                attn_ctx = self.attn(
                    rms_norm_preserve_input_dtype(x_ctx, self.norm1),
                    position_embeddings=pos_ctx,
                    grid_shape=(int(ctx_t1 - ctx_t0), gh, gw),
                )
                attn_q = attn_ctx[:, keep_token0:keep_token1, :]
                x_q = source_x[:, q_token0_bound:q_token1_bound, :] + attn_q
                return x_q + self.mlp(rms_norm_preserve_input_dtype(x_q, self.norm2))

            if checkpoint_chunks:
                y_q = checkpoint(compute_chunk, x, use_reentrant=False)
            else:
                y_q = compute_chunk(x)

            if output is None:
                chunks.append(y_q)
            else:
                output[:, q_token0:q_token1, :].copy_(y_q)

        if output is not None:
            return output
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: PositionEmbeddings3D,
        grid_shape: GridShape3D | None = None,
    ) -> torch.Tensor:
        normalized_grid_shape = _normalize_grid_shape(
            grid_shape,
            default=(self.attn.grid_t, self.attn.grid_h, self.attn.grid_w),
        )
        chunk_t = self._temporal_chunk_size_for_grid(normalized_grid_shape)
        if chunk_t is not None:
            return self._forward_temporal_chunked(
                x,
                position_embeddings=position_embeddings,
                grid_shape=normalized_grid_shape,
                temporal_chunk_size=chunk_t,
            )
        return self._forward_full(x, position_embeddings=position_embeddings, grid_shape=normalized_grid_shape)


__all__ = [
    "Position3D",
    "GridShape3D",
    "PositionEmbeddings3D",
    "RoPE3D",
    "FlexSelfAttention3D",
    "TransformerFlexBlock3D",
    "rms_norm_preserve_input_dtype",
]
