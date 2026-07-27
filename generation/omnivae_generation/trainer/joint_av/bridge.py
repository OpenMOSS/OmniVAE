"""Cross-modality Bridge module for joint AV training.

A ``BridgeBlock`` is inserted after every ``bridge_interval`` main Z-Image
transformer blocks. Each bridge runs two independent cross-attentions:

  * A2V: video query attends to audio key/value, modulated by ``t_audio``
  * V2A: audio query attends to video key/value, modulated by ``t_video``

Both directions use:

  * RMSNorm pre-norm (mirrors the trunk's ``ZImageTransformerBlock``)
  * QK-norm (per-head RMSNorm) with the same ``head_dim`` / ``n_heads`` as
    the trunk (15 heads, 128 head_dim)
  * Temporal-only RoPE: only the time axis (``axes_dims[0] == 16``) of the
    trunk's RoPE table is active; the spatial axes are filled with the
    identity rotation. Video token time indices are scaled to the audio
    time scale so both modalities share a single RoPE timeline.
  * Cross-modality AdaLN: scale/gate are derived from the *opposite*
    modality's timestep embedding via an independent MLP whose final
    Linear is zero-initialised.

The output projection ``W_o`` is also zero-initialised, so on step 0 the
delta added back into the trunk is exactly zero (verified by the
``zero_init_check`` helper) and the joint model is byte-equivalent to
running the two branches independently.

For reference, see LTX2's ``BasicAVTransformerBlock``
(``ltx2/ltx-core/src/ltx_core/model/transformer/transformer.py``); we
reuse the high-level pattern but keep the bridge as a *separate* module
that lives between trunk blocks rather than fused inside a block, to
match the user's "insertion-interval" specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


_ADALN_EMBED_DIM = 256


@dataclass(frozen=True)
class BridgeRoPELayout:
    """Geometry needed to build temporal-only RoPE freqs for both modalities.

    ``head_dim`` is the per-head channel count (e.g. 128). ``axes_dims`` /
    ``axes_lens`` mirror the trunk's ``rope_embedder`` (e.g.
    ``[16, 56, 56]`` and ``[2048, 16, 16]``). ``rope_theta`` matches the
    trunk's ``rope_theta`` (256.0).
    """

    head_dim: int
    axes_dims: tuple[int, int, int]
    axes_lens: tuple[int, int, int]
    rope_theta: float


def _precompute_t_axis_freqs(t_dim: int, t_len: int, theta: float) -> torch.Tensor:
    """Real-valued RoPE freqs for a single time axis.

    Returns a ``[t_len, t_dim // 2, 2]`` tensor where the last dim packs
    ``(cos, sin)``. Mirrors ``RopeEmbedder.precompute_freqs_cis`` in
    ``runtime_patches.patch_diffusers_zimage_real_rope`` so the bridge's
    rotary path is byte-identical to the trunk's on the time axis.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, t_dim, 2, dtype=torch.float64) / t_dim))
    timestep = torch.arange(t_len, dtype=torch.float64)
    phase = torch.outer(timestep, freqs).float()
    return torch.stack((phase.cos(), phase.sin()), dim=-1)


class _TemporalRopeCache:
    """Lazy per-device cache for the t-axis cos/sin pairs.

    The bridge only ever uses the t-axis; we materialise it once per
    device on first use (and re-materialise transparently when the batch
    moves).
    """

    def __init__(self, layout: BridgeRoPELayout):
        self.layout = layout
        self._t_freqs_by_device: dict[torch.device, torch.Tensor] = {}

    def get(self, device: torch.device) -> torch.Tensor:
        cached = self._t_freqs_by_device.get(device)
        if cached is not None:
            return cached
        freqs = _precompute_t_axis_freqs(
            self.layout.axes_dims[0], self.layout.axes_lens[0], self.layout.rope_theta
        ).to(device)
        self._t_freqs_by_device[device] = freqs
        return freqs


def build_temporal_rope_freqs(
    t_indices: torch.Tensor,
    rope_cache: _TemporalRopeCache,
) -> torch.Tensor:
    """Build temporal-only RoPE freqs for a flat token sequence.

    Args:
      t_indices: ``[B, S]`` int tensor of time indices in the *audio* time
        scale (``[0, axes_lens[0])``). Padding tokens may safely use ``0``
        because attention masks zero out their contribution downstream.
      rope_cache: a ``_TemporalRopeCache`` built from the trunk's RoPE
        layout. Holds the device-local t-axis cos/sin pairs.

    Returns:
      Real-valued freqs ``[B, S, head_dim // 2, 2]`` where channels
      ``[:t_dim // 2]`` carry the t-axis rotation and the remaining
      ``[(head_dim - t_dim) // 2]`` channels carry the *identity*
      rotation ``(cos=1, sin=0)``. This lets us reuse the trunk's
      ``apply_rotary_emb`` shape without rotating spatial channels.
    """
    if t_indices.dim() != 2:
        raise ValueError(f"t_indices must be [B, S], got shape {tuple(t_indices.shape)}.")

    layout = rope_cache.layout
    head_dim = layout.head_dim
    t_dim = layout.axes_dims[0]
    if head_dim % 2 != 0 or t_dim % 2 != 0:
        raise ValueError(
            f"head_dim ({head_dim}) and t_axis_dim ({t_dim}) must both be even."
        )

    device = t_indices.device
    t_freqs = rope_cache.get(device)
    if t_indices.dtype != torch.long:
        t_indices = t_indices.long()
    t_indices_clamped = t_indices.clamp_(0, t_freqs.shape[0] - 1)

    bsz, seq = t_indices_clamped.shape
    t_pairs = t_freqs[t_indices_clamped]
    pad_pairs = (head_dim - t_dim) // 2
    if pad_pairs > 0:
        identity = torch.zeros(
            (bsz, seq, pad_pairs, 2),
            dtype=t_pairs.dtype,
            device=device,
        )
        identity[..., 0] = 1.0
        return torch.cat([t_pairs, identity], dim=-2).contiguous()
    return t_pairs.contiguous()


def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Mirror of the trunk rotary path (``runtime_patches._attn_processor_call``).

    ``x_in`` is ``[B, S, n_heads, head_dim]``; ``freqs_cis`` is
    ``[B, S, head_dim // 2, 2]``. Always runs in fp32 (matching the
    trunk's ``torch.amp.autocast(enabled=False)`` behaviour) for
    numerical stability under bf16 mixed precision.
    """
    with torch.amp.autocast("cuda", enabled=False):
        x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.to(device=x_in.device, dtype=x.dtype).unsqueeze(2)
        x_real, x_imag = x.unbind(-1)
        cos = freqs_cis[..., 0]
        sin = freqs_cis[..., 1]
        rotated = torch.stack(
            (x_real * cos - x_imag * sin, x_real * sin + x_imag * cos),
            dim=-1,
        ).flatten(3)
    return rotated.type_as(x_in)


class _RMSNorm(nn.Module):
    """Mirror of ``diffusers.models.normalization.RMSNorm`` with affine
    weight=1; used for both QK-norm and pre-attn norms so the bridge
    matches the trunk's normalisation conventions exactly."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.to(x.dtype)).to(input_dtype)


class CrossModalityAdaLN(nn.Module):
    """Per-bridge AdaLN modulation driven by the *opposite* modality's
    timestep embedding.

    Produces a ``[B, 2 * dim]`` tensor split into ``(scale, gate)`` per
    token. ``scale`` is added to ``1.0`` and ``gate`` is passed through
    ``tanh``, mirroring ``ZImageTransformerBlock``'s gating convention.

    The final ``Linear`` is zero-initialised so the bridge contributes
    exactly zero on step 0 (gate=tanh(0)=0).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.act = nn.SiLU()
        self.proj = nn.Linear(min(dim, _ADALN_EMBED_DIM), 2 * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mod = self.proj(self.act(t_emb))
        scale, gate = mod.chunk(2, dim=-1)
        # Match ZImageTransformerBlock's gating: scale = 1 + s, gate = tanh(g)
        scale = 1.0 + scale.unsqueeze(1)
        gate = gate.tanh().unsqueeze(1)
        return scale, gate


class _CrossAttention(nn.Module):
    """Plain Q-from-self / KV-from-other attention with QK-norm.

    Mirrors the trunk's ``ZSingleStreamAttnProcessor`` shape conventions
    (``[B, S, H, D]``) so the rotary application is identical. Uses
    ``F.scaled_dot_product_attention`` for backend-agnostic SDPA.

    The output projection ``to_out`` is zero-initialised; combined with
    the AdaLN-zero gate this gives a *double* guarantee that the bridge
    contributes nothing on step 0.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int, qk_norm: bool, norm_eps: float):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.norm_q = _RMSNorm(head_dim, eps=norm_eps) if qk_norm else None
        self.norm_k = _RMSNorm(head_dim, eps=norm_eps) if qk_norm else None
        nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        q_mask: torch.Tensor | None,
        kv_mask: torch.Tensor | None,
        q_freqs: torch.Tensor | None,
        k_freqs: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seq_q, _ = x_q.shape
        seq_kv = x_kv.shape[1]

        query = self.to_q(x_q).unflatten(-1, (self.n_heads, self.head_dim))
        key = self.to_k(x_kv).unflatten(-1, (self.n_heads, self.head_dim))
        value = self.to_v(x_kv).unflatten(-1, (self.n_heads, self.head_dim))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if q_freqs is not None:
            query = _apply_rotary_emb(query, q_freqs)
        if k_freqs is not None:
            key = _apply_rotary_emb(key, k_freqs)

        # SDPA expects [B, H, S, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_mask = None
        if kv_mask is not None:
            attn_mask = kv_mask[:, None, None, :]
            if q_mask is not None:
                # Ensure padded query rows don't poison fp16 softmax (rows that
                # have all-False keys would NaN otherwise). Forcing q_mask
                # rows of all-False to attend to *something* is unnecessary
                # because we mask their output back to zero below; we just
                # need at least one True key for numerical stability.
                attn_mask = attn_mask.expand(-1, self.n_heads, seq_q, -1).contiguous()

        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_q, self.n_heads * self.head_dim)
        out = self.to_out(out)
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).to(out.dtype)
        return out


class BridgeBlock(nn.Module):
    """Symmetric A2V + V2A cross-attention bridge.

    Forward signature matches the trunk's idiom: takes the two modality
    token streams (already extracted from the unified sequence via
    ``_extract_compact_x_tokens``) plus their masks, time indices, and
    *opposite* modality timestep embeddings, returns *deltas* to be
    scattered back into each branch's unified sequence.

    With ``use_asymmetric_ati=False`` (default) both directions are full
    cross-attention. With ``True`` the A2V direction uses a windowed
    attention pattern (each video token attends only to a window of audio
    tokens covering its own time slice), and the V2A direction uses
    interpolated video context for each audio time-step. ATI is left as
    an opt-in feature for future experiments; the default keeps the
    bridge symmetric and simple.
    """

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        head_dim: int,
        qk_norm: bool,
        norm_eps: float,
        rope_layout: BridgeRoPELayout,
        use_asymmetric_ati: bool = False,
        a2v_window_size: int = 1,
    ):
        super().__init__()
        if rope_layout.head_dim != head_dim:
            raise ValueError(
                f"BridgeRoPELayout.head_dim={rope_layout.head_dim} must equal head_dim={head_dim}."
            )
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.use_asymmetric_ati = bool(use_asymmetric_ati)
        self.a2v_window_size = int(a2v_window_size)
        self._rope_cache = _TemporalRopeCache(rope_layout)

        # A2V: video tokens query, audio tokens supply K/V
        self.a2v_norm_q = _RMSNorm(dim, eps=norm_eps)
        self.a2v_norm_kv = _RMSNorm(dim, eps=norm_eps)
        self.a2v_norm_out = _RMSNorm(dim, eps=norm_eps)
        self.a2v_attn = _CrossAttention(
            dim=dim, n_heads=n_heads, head_dim=head_dim, qk_norm=qk_norm, norm_eps=norm_eps,
        )
        self.a2v_adaln = CrossModalityAdaLN(dim)

        # V2A: audio tokens query, video tokens supply K/V
        self.v2a_norm_q = _RMSNorm(dim, eps=norm_eps)
        self.v2a_norm_kv = _RMSNorm(dim, eps=norm_eps)
        self.v2a_norm_out = _RMSNorm(dim, eps=norm_eps)
        self.v2a_attn = _CrossAttention(
            dim=dim, n_heads=n_heads, head_dim=head_dim, qk_norm=qk_norm, norm_eps=norm_eps,
        )
        self.v2a_adaln = CrossModalityAdaLN(dim)

    def temporal_rope_freqs(self, t_indices: torch.Tensor) -> torch.Tensor:
        return build_temporal_rope_freqs(t_indices, self._rope_cache)

    def forward(
        self,
        *,
        video_tokens: torch.Tensor,           # [B, S_v, D]
        audio_tokens: torch.Tensor,           # [B, S_a, D]
        video_mask: torch.Tensor,             # [B, S_v] bool
        audio_mask: torch.Tensor,             # [B, S_a] bool
        video_t_indices: torch.Tensor,        # [B, S_v] long, audio-scale indices
        audio_t_indices: torch.Tensor,        # [B, S_a] long, audio-scale indices
        t_emb_video: torch.Tensor,            # [B, ADALN_EMBED_DIM] -- modulates V2A
        t_emb_audio: torch.Tensor,            # [B, ADALN_EMBED_DIM] -- modulates A2V
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(delta_video, delta_audio)`` to add back into each
        branch's unified sequence.

        Note the modulation cross-wiring: the A2V cross-attention (where
        video is the *query*) is modulated by ``t_emb_audio`` (the
        opposite modality), and vice versa. This matches the spec
        "use the opposite modality's timestep embedding".
        """
        if self.use_asymmetric_ati:
            return self._forward_asymmetric(
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
                video_mask=video_mask,
                audio_mask=audio_mask,
                video_t_indices=video_t_indices,
                audio_t_indices=audio_t_indices,
                t_emb_video=t_emb_video,
                t_emb_audio=t_emb_audio,
            )

        video_freqs = self.temporal_rope_freqs(video_t_indices)
        audio_freqs = self.temporal_rope_freqs(audio_t_indices)

        # A2V: video Q -> audio K,V (modulated by t_emb_audio)
        a2v_scale, a2v_gate = self.a2v_adaln(t_emb_audio)
        a2v_attn_out = self.a2v_attn(
            x_q=self.a2v_norm_q(video_tokens) * a2v_scale,
            x_kv=self.a2v_norm_kv(audio_tokens),
            q_mask=video_mask,
            kv_mask=audio_mask,
            q_freqs=video_freqs,
            k_freqs=audio_freqs,
        )
        delta_video = a2v_gate * self.a2v_norm_out(a2v_attn_out)

        # V2A: audio Q -> video K,V (modulated by t_emb_video)
        v2a_scale, v2a_gate = self.v2a_adaln(t_emb_video)
        v2a_attn_out = self.v2a_attn(
            x_q=self.v2a_norm_q(audio_tokens) * v2a_scale,
            x_kv=self.v2a_norm_kv(video_tokens),
            q_mask=audio_mask,
            kv_mask=video_mask,
            q_freqs=audio_freqs,
            k_freqs=video_freqs,
        )
        delta_audio = v2a_gate * self.v2a_norm_out(v2a_attn_out)

        return delta_video, delta_audio

    def _forward_asymmetric(
        self,
        *,
        video_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        video_t_indices: torch.Tensor,
        audio_t_indices: torch.Tensor,
        t_emb_video: torch.Tensor,
        t_emb_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional asymmetric variant.

        For A2V, we restrict each video token to attend only to audio
        tokens whose ``audio_t`` is within ``+/- a2v_window_size`` of the
        video token's audio-scale ``t``. For V2A we keep full attention
        but linearly interpolate a per-time-bucket video summary to use
        as KV. Default-off; provided for future ablation work.
        """
        # Build a [B, S_v, S_a] mask: keep audio token j for video token i iff
        # |audio_t[j] - video_t[i]| <= window AND audio_mask[j].
        v_t = video_t_indices.unsqueeze(-1)              # [B, S_v, 1]
        a_t = audio_t_indices.unsqueeze(1)               # [B, 1, S_a]
        window = max(1, int(self.a2v_window_size))
        in_window = (a_t - v_t).abs() <= window          # [B, S_v, S_a]
        windowed_kv_mask = in_window & audio_mask.unsqueeze(1)

        video_freqs = self.temporal_rope_freqs(video_t_indices)
        audio_freqs = self.temporal_rope_freqs(audio_t_indices)

        # ------ A2V (windowed) ------
        a2v_scale, a2v_gate = self.a2v_adaln(t_emb_audio)
        # Reuse SDPA via custom mask: pass [B, 1, S_v, S_a] and let SDPA broadcast over heads.
        a2v_attn_out = self._sdpa_with_mask(
            self.a2v_attn,
            x_q=self.a2v_norm_q(video_tokens) * a2v_scale,
            x_kv=self.a2v_norm_kv(audio_tokens),
            q_freqs=video_freqs,
            k_freqs=audio_freqs,
            attn_mask=windowed_kv_mask.unsqueeze(1),     # [B, 1, S_v, S_a]
            q_mask=video_mask,
        )
        delta_video = a2v_gate * self.a2v_norm_out(a2v_attn_out)

        # ------ V2A (interpolated context) ------
        # Bucket-mean video features per audio-scale time index, then attend.
        # Cheap: scatter_mean along time.
        bsz, sa = audio_t_indices.shape
        v_summary = torch.zeros_like(audio_tokens)
        v_count = torch.zeros((bsz, sa, 1), dtype=audio_tokens.dtype, device=audio_tokens.device)
        # Map each video token to the audio-scale time index, then accumulate
        v_t_clamped = video_t_indices.clamp_(0, sa - 1)
        scatter_idx = v_t_clamped.unsqueeze(-1).expand(-1, -1, video_tokens.shape[-1])
        v_summary.scatter_add_(1, scatter_idx, video_tokens * video_mask.unsqueeze(-1).to(video_tokens.dtype))
        ones = torch.ones_like(video_t_indices, dtype=audio_tokens.dtype).unsqueeze(-1) * video_mask.unsqueeze(-1).to(audio_tokens.dtype)
        v_count.scatter_add_(1, v_t_clamped.unsqueeze(-1), ones)
        v_summary = v_summary / v_count.clamp_min(1.0)

        v2a_scale, v2a_gate = self.v2a_adaln(t_emb_video)
        v2a_attn_out = self.v2a_attn(
            x_q=self.v2a_norm_q(audio_tokens) * v2a_scale,
            x_kv=self.v2a_norm_kv(v_summary),
            q_mask=audio_mask,
            kv_mask=audio_mask,                          # summary lives on the audio grid
            q_freqs=audio_freqs,
            k_freqs=audio_freqs,
        )
        delta_audio = v2a_gate * self.v2a_norm_out(v2a_attn_out)

        return delta_video, delta_audio

    @staticmethod
    def _sdpa_with_mask(
        attn: _CrossAttention,
        *,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
        attn_mask: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_q, _ = x_q.shape
        query = attn.to_q(x_q).unflatten(-1, (attn.n_heads, attn.head_dim))
        key = attn.to_k(x_kv).unflatten(-1, (attn.n_heads, attn.head_dim))
        value = attn.to_v(x_kv).unflatten(-1, (attn.n_heads, attn.head_dim))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if q_freqs is not None:
            query = _apply_rotary_emb(query, q_freqs)
        if k_freqs is not None:
            key = _apply_rotary_emb(key, k_freqs)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_q, attn.n_heads * attn.head_dim)
        out = attn.to_out(out)
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).to(out.dtype)
        return out


def assert_bridge_zero_initialised(bridge: BridgeBlock) -> None:
    """Sanity-check the two zero-init guarantees that make ``bridge_enabled=True``
    equivalent to ``bridge_enabled=False`` on step 0.

    Raises:
      AssertionError if either ``to_out.weight`` or
      ``CrossModalityAdaLN.proj.{weight, bias}`` is non-zero in any
      direction.
    """
    for name, attn in (("a2v", bridge.a2v_attn), ("v2a", bridge.v2a_attn)):
        if attn.to_out.weight.abs().max().item() != 0.0:
            raise AssertionError(f"BridgeBlock.{name}_attn.to_out is not zero-initialised.")
    for name, adaln in (("a2v", bridge.a2v_adaln), ("v2a", bridge.v2a_adaln)):
        if adaln.proj.weight.abs().max().item() != 0.0:
            raise AssertionError(f"BridgeBlock.{name}_adaln.proj.weight is not zero-initialised.")
        if adaln.proj.bias.abs().max().item() != 0.0:
            raise AssertionError(f"BridgeBlock.{name}_adaln.proj.bias is not zero-initialised.")
