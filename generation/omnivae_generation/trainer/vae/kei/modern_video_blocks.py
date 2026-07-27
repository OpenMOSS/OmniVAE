from __future__ import annotations

from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def sdpa_with_cudnn_preference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> torch.Tensor:
    use_cudnn_preference = bool(q.is_cuda and (not torch.are_deterministic_algorithms_enabled()))
    sdpa_ctx = sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if use_cudnn_preference else nullcontext()
    with sdpa_ctx:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=bool(is_causal),
        )


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


Position4D = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class RoPE4D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        if self.head_dim % 8 != 0:
            raise ValueError(f"RoPE4D requires head_dim % 8 == 0, got head_dim={self.head_dim}")
        self.quarter = self.head_dim // 4

        freq_dim = self.quarter // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, freq_dim, dtype=torch.float32) / float(freq_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def build_position_embeddings(
        self,
        *,
        pos_t: torch.Tensor,
        pos_h: torch.Tensor,
        pos_w: torch.Tensor,
        pos_q: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.stack([pos_t, pos_h, pos_w, pos_q], dim=0).to(device=device, dtype=torch.float32)
        freqs = torch.einsum("al,d->ald", pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        cos = cos.permute(1, 0, 2)[None, None, :, :, :]
        sin = sin.permute(1, 0, 2)[None, None, :, :, :]
        return cos, sin

    def apply_position_embeddings(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        b, h, l, hd = x.shape
        d = int(self.quarter)
        if hd != 4 * d:
            raise ValueError(f"RoPE4D head_dim mismatch: got {hd}, expected {4 * d}")
        x4 = x.view(b, h, l, 4, d)
        x4 = (x4 * cos) + (_rotate_half(x4) * sin)
        return x4.view(b, h, l, hd)


class SelfAttention4D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim({self.dim}) must be divisible by num_heads({self.num_heads})")
        self.head_dim = self.dim // self.num_heads

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        self.rope = RoPE4D(self.head_dim, base=float(rope_base))
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm: nn.RMSNorm | None = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
            self.k_norm: nn.RMSNorm | None = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: Position4D,
        key_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, l, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        pos_t, pos_h, pos_w, pos_q = positions
        embeddings = self.rope.build_position_embeddings(
            pos_t=pos_t,
            pos_h=pos_h,
            pos_w=pos_w,
            pos_q=pos_q,
            dtype=q.dtype,
            device=q.device,
        )
        q = self.rope.apply_position_embeddings(q, position_embeddings=embeddings)
        k = self.rope.apply_position_embeddings(k, position_embeddings=embeddings)
        attn_mask = None if key_keep is None else key_keep[:, None, None, :]
        out = sdpa_with_cudnn_preference(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, l, self.dim)
        return self.proj(out)


class CrossAttention4D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim({self.dim}) must be divisible by num_heads({self.num_heads})")
        self.head_dim = self.dim // self.num_heads

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.kv_proj = nn.Linear(self.dim, 2 * self.dim, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        self.rope = RoPE4D(self.head_dim, base=float(rope_base))
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm: nn.RMSNorm | None = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
            self.k_norm: nn.RMSNorm | None = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: Position4D,
        context: torch.Tensor | None,
        context_positions: Position4D | None,
        context_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None or int(context.shape[1]) == 0:
            return torch.zeros_like(x)

        b, lq, _ = x.shape
        lk = int(context.shape[1])

        q = self.q_proj(x)
        kv = self.kv_proj(context)
        k, v = kv.chunk(2, dim=-1)
        q = q.view(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, lk, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q_t, q_h, q_w, q_q = positions
        kv_t, kv_h, kv_w, kv_q = context_positions if context_positions is not None else positions
        q_embeddings = self.rope.build_position_embeddings(
            pos_t=q_t,
            pos_h=q_h,
            pos_w=q_w,
            pos_q=q_q,
            dtype=q.dtype,
            device=q.device,
        )
        kv_embeddings = self.rope.build_position_embeddings(
            pos_t=kv_t,
            pos_h=kv_h,
            pos_w=kv_w,
            pos_q=kv_q,
            dtype=k.dtype,
            device=k.device,
        )
        q = self.rope.apply_position_embeddings(q, position_embeddings=q_embeddings)
        k = self.rope.apply_position_embeddings(k, position_embeddings=kv_embeddings)

        attn_mask = None if context_keep is None else context_keep[:, None, None, :]
        out = sdpa_with_cudnn_preference(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, lq, self.dim)
        return self.proj(out)


class TransformerSelfCrossBlock4D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.self_attn = SelfAttention4D(dim, num_heads, rope_base=rope_base, qk_norm=qk_norm)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.cross_attn = CrossAttention4D(dim, num_heads, rope_base=rope_base, qk_norm=qk_norm)
        self.norm3 = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.mlp = SwiGLU(dim, hidden_dim=hidden)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: Position4D,
        context: torch.Tensor | None = None,
        context_positions: Position4D | None = None,
        context_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), positions=positions)
        if context is not None and int(context.shape[1]) > 0:
            x = x + self.cross_attn(
                self.norm2(x),
                positions=positions,
                context=context,
                context_positions=context_positions,
                context_keep=context_keep,
            )
        x = x + self.mlp(self.norm3(x))
        return x


__all__ = [
    "CrossAttention4D",
    "Position4D",
    "RoPE4D",
    "SelfAttention4D",
    "SwiGLU",
    "TransformerSelfCrossBlock4D",
    "sdpa_with_cudnn_preference",
]
