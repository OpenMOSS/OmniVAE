"""``BridgedZImageJointModel`` — a thin wrapper that composes two
pretrained ``ZImageTransformer2DModel`` branches (t2v + t2a) plus a stack
of ``BridgeBlock`` cross-attention modules between every
``bridge_interval`` main blocks.

The joint forward reuses the patched helpers from
``omnivae_generation.trainer.runtime_patches`` so each branch behaves *exactly* like the
single-modality trainer does — embedding, refiners, padded unified
sequence, final layer, unpatchify — and only the main-block loop is
replaced to interleave bridges.

Equivalence guarantee: with ``bridge_enabled=False`` (or with all
bridges still at their zero-init state), the joint forward is
byte-equivalent to running each branch through its standalone forward.
This is exercised by the ``tests/test_joint_av.py::test_bridge_disabled_equivalence``
test.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from omnivae_generation.trainer.joint_av.bridge import (
    BridgeBlock,
    BridgeRoPELayout,
    assert_bridge_zero_initialised,
)
from omnivae_generation.trainer.runtime_patches import (
    _build_unified_dense_sequence,
    _extract_compact_x_tokens,
    _unpatchify_compact_x_tokens,
)


_ADALN_EMBED_DIM = 256


@dataclass
class _BranchPacked:
    """Per-branch tensors after embedding + refiners + unification."""
    unified: torch.Tensor          # [B, max_unified_len, D]
    unified_freqs: torch.Tensor    # [B, max_unified_len, head_dim//2, 2]
    unified_mask: torch.Tensor     # [B, max_unified_len] bool
    x_start_offsets: torch.Tensor  # [B] long
    x_lengths: torch.Tensor        # [B] long
    max_x_len: int                 # static int (== x.shape[1])
    x_size: list                   # original [(F,H,W), ...] per sample (for unpatchify)
    adaln_input: Optional[torch.Tensor]  # [B, ADALN_EMBED_DIM] or None
    t_emb: Optional[torch.Tensor]        # alias of adaln_input for the bridge
    patch_size: int
    f_patch_size: int
    audio_scale_t_indices: torch.Tensor  # [B, max_x_len] long, audio-scale t per X token


def _compute_audio_scale_t_indices(
    *,
    x_size: list[tuple[int, int, int]],
    patch_size: int,
    f_patch_size: int,
    max_x_len: int,
    audio_t_lengths_per_sample: torch.Tensor,  # [B] long, audio-scale T_a per sample
    device: torch.device,
) -> torch.Tensor:
    """Build per-X-token audio-scale time indices.

    For each sample with ``x_size = (F, H, W)`` (the original, un-patched
    latent shape), the patched grid is ``(F_t, H_t, W_t) = (F//f_patch,
    H//patch, W//patch)`` and tokens are flat-indexed in lexicographic
    ``(t, h, w)`` order. We return, for each token, the audio-scale time
    index ``round(t * (T_a / F_t))`` so video and audio tokens share a
    single timeline through the bridge's RoPE.

    Tokens past each sample's actual ``x_length`` are padded with ``0``
    (the bridge masks them out, but ``0`` keeps RoPE indexing safe).
    """
    bsz = len(x_size)
    out = torch.zeros((bsz, max_x_len), dtype=torch.long, device=device)
    for i, size in enumerate(x_size):
        if size is None:
            continue
        F, H, W = size
        F_t = max(1, int(F) // max(1, int(f_patch_size)))
        H_t = max(1, int(H) // max(1, int(patch_size)))
        W_t = max(1, int(W) // max(1, int(patch_size)))
        n_tokens = F_t * H_t * W_t
        if n_tokens == 0:
            continue
        # Token index -> latent-time index
        idx = torch.arange(n_tokens, device=device)
        t_local = idx // (H_t * W_t)                    # [n_tokens] long
        # Audio-scale mapping. Use the *real* per-sample audio length so
        # short clips don't accidentally map to indices past the audio's
        # own RoPE table.
        t_a = int(audio_t_lengths_per_sample[i].item())
        if t_a <= 0:
            t_a = 1
        if F_t <= 1:
            scaled = torch.zeros_like(t_local)
        else:
            # round(t_local * (T_a - 1) / (F_t - 1)) so endpoints land at 0 / T_a-1.
            scaled = (t_local.to(torch.float32) * float(t_a - 1) / float(F_t - 1)).round().long()
            scaled = scaled.clamp_(0, max(0, t_a - 1))
        out[i, :n_tokens] = scaled
    return out


def _compute_native_t_indices(
    *,
    x_size: list[tuple[int, int, int]],
    patch_size: int,
    f_patch_size: int,
    max_x_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Native t-axis index of each X token (no audio rescaling).

    Used for the *audio* branch, where ``H = W = 1`` and the native
    t-axis is already the audio scale. Returns ``[B, max_x_len]`` long,
    zero-padded.
    """
    bsz = len(x_size)
    out = torch.zeros((bsz, max_x_len), dtype=torch.long, device=device)
    for i, size in enumerate(x_size):
        if size is None:
            continue
        F, H, W = size
        F_t = max(1, int(F) // max(1, int(f_patch_size)))
        H_t = max(1, int(H) // max(1, int(patch_size)))
        W_t = max(1, int(W) // max(1, int(patch_size)))
        n_tokens = F_t * H_t * W_t
        if n_tokens == 0:
            continue
        idx = torch.arange(n_tokens, device=device)
        t_local = idx // (H_t * W_t)
        out[i, :n_tokens] = t_local
    return out


def _scatter_x_delta_into_unified(
    unified: torch.Tensor,
    delta_x: torch.Tensor,
    x_start_offsets: torch.Tensor,
    x_lengths: torch.Tensor,
    max_x_len: int,
) -> torch.Tensor:
    """Inverse of ``_extract_compact_x_tokens``: add ``delta_x`` back into
    the unified sequence at each sample's X slot.

    Implemented via scatter_add along dim=1 so it stays compile/grad
    friendly (avoids Python-level loops over the batch).
    """
    bsz, max_unified_len, hidden_dim = unified.shape
    local_positions = torch.arange(max_x_len, device=unified.device).unsqueeze(0)
    target_positions = x_start_offsets.unsqueeze(1) + local_positions
    valid = local_positions < x_lengths.unsqueeze(1)            # [B, max_x_len]
    safe_target = target_positions.masked_fill(~valid, 0)        # avoid out-of-range
    contribution = (delta_x * valid.unsqueeze(-1).to(delta_x.dtype)).to(unified.dtype)
    unified = unified.scatter_add(
        1,
        safe_target.unsqueeze(-1).expand(-1, -1, hidden_dim),
        contribution,
    )
    return unified


class BridgedZImageJointModel(nn.Module):
    """Joint AV transformer = video branch + audio branch + bridges.

    The two ``ZImageTransformer2DModel`` instances are *not* mutated; we
    just call their patched helpers on top. This keeps both branches
    fully compatible with the existing single-modality trainer's
    checkpoint format (each branch can be independently loaded /
    exported via ``omnivae_generation.trainer.modeling.load_pretrained_transformer_weights``
    / the diffusers ``save_pretrained`` path).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        *,
        video_transformer,                   # ZImageTransformer2DModel
        audio_transformer,                   # ZImageTransformer2DModel
        bridge_interval: int = 2,
        bridge_enabled: bool = True,
        use_asymmetric_ati: bool = False,
        a2v_window_size: int = 1,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()
        if not hasattr(video_transformer, "layers") or not hasattr(audio_transformer, "layers"):
            raise ValueError(
                "BridgedZImageJointModel requires patched ZImageTransformer2DModel "
                "instances (with .layers, .noise_refiner, .context_refiner)."
            )
        if int(video_transformer.dim) != int(audio_transformer.dim):
            raise ValueError(
                f"Video and audio transformers must share the same hidden dim: "
                f"got video.dim={video_transformer.dim} vs audio.dim={audio_transformer.dim}."
            )
        if int(video_transformer.n_heads) != int(audio_transformer.n_heads):
            raise ValueError(
                f"Video and audio transformers must share the same n_heads: "
                f"got video.n_heads={video_transformer.n_heads} vs audio.n_heads={audio_transformer.n_heads}."
            )
        if int(len(video_transformer.layers)) != int(len(audio_transformer.layers)):
            raise ValueError(
                "Video and audio transformers must have the same number of main "
                f"layers, got {len(video_transformer.layers)} vs {len(audio_transformer.layers)}."
            )

        self.video = video_transformer
        self.audio = audio_transformer
        self.bridge_interval = int(bridge_interval)
        if self.bridge_interval <= 0:
            raise ValueError(f"bridge_interval must be positive, got {bridge_interval}.")
        self.bridge_enabled = bool(bridge_enabled)
        self.use_asymmetric_ati = bool(use_asymmetric_ati)
        self.a2v_window_size = int(a2v_window_size)

        n_main = len(self.video.layers)
        # Insert a bridge after layer (i+1) where (i+1) % bridge_interval == 0.
        # Refiner blocks (noise_refiner / context_refiner) are *never* bridged
        # per spec.
        n_bridges = n_main // self.bridge_interval
        dim = int(self.video.dim)
        n_heads = int(self.video.n_heads)
        head_dim = dim // n_heads

        # Mirror the trunk's RoPE layout (video and audio share axes_dims).
        rope_layout = BridgeRoPELayout(
            head_dim=head_dim,
            axes_dims=tuple(int(x) for x in self.audio.rope_embedder.axes_dims),
            axes_lens=tuple(int(x) for x in self.audio.rope_embedder.axes_lens),
            rope_theta=float(self.audio.rope_embedder.theta),
        )
        self.bridges = nn.ModuleList(
            [
                BridgeBlock(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    qk_norm=qk_norm,
                    norm_eps=norm_eps,
                    rope_layout=rope_layout,
                    use_asymmetric_ati=use_asymmetric_ati,
                    a2v_window_size=a2v_window_size,
                )
                for _ in range(n_bridges)
            ]
        )

        # Cache per-layer-index -> bridge_index for the inner loop. Only
        # layers ``l`` such that ``(l + 1) % bridge_interval == 0`` get a
        # bridge appended after them.
        bridge_after_layer: dict[int, int] = {}
        for layer_idx in range(n_main):
            if (layer_idx + 1) % self.bridge_interval == 0:
                bridge_after_layer[layer_idx] = (layer_idx + 1) // self.bridge_interval - 1
        self._bridge_after_layer = bridge_after_layer
        self.gradient_checkpointing = False

    # --------------------------------------------------------------- helpers
    @property
    def dim(self) -> int:
        return int(self.video.dim)

    @property
    def n_heads(self) -> int:
        return int(self.video.n_heads)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True
        if hasattr(self.video, "enable_gradient_checkpointing"):
            self.video.enable_gradient_checkpointing()
        if hasattr(self.audio, "enable_gradient_checkpointing"):
            self.audio.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        if hasattr(self.video, "disable_gradient_checkpointing"):
            self.video.disable_gradient_checkpointing()
        if hasattr(self.audio, "disable_gradient_checkpointing"):
            self.audio.disable_gradient_checkpointing()

    def set_attention_backend(self, backend: str) -> None:
        for branch in (self.video, self.audio):
            if hasattr(branch, "set_attention_backend"):
                branch.set_attention_backend(backend)

    def materialize_rope_cache(self, device: torch.device) -> None:
        for branch in (self.video, self.audio):
            if hasattr(branch, "materialize_rope_cache"):
                branch.materialize_rope_cache(device)
        # Bridge RoPE caches are populated lazily on first use; force them
        # now so the first training step doesn't pay the materialise cost.
        dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
        for bridge in self.bridges:
            bridge.temporal_rope_freqs(dummy)

    def assert_bridges_zero_initialised(self) -> None:
        for bridge in self.bridges:
            assert_bridge_zero_initialised(bridge)

    def named_branch_parameters(self) -> list[tuple[str, str, nn.Parameter]]:
        """Iterate parameters tagged by ``(tag, name, param)`` for the
        heterogeneous-LR optimizer.

        ``tag`` is one of ``"backbone"`` (video/audio main weights) or
        ``"bridge"`` (bridge cross-attn + AdaLN). The ``name`` is the
        full ``self.named_parameters()`` name so checkpointing keeps
        working unchanged.
        """
        out: list[tuple[str, str, nn.Parameter]] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            tag = "bridge" if name.startswith("bridges.") else "backbone"
            out.append((tag, name, param))
        return out

    # --------------------------------------------------------------- forward
    def _prepare_branch(
        self,
        branch,
        latent_list: list[torch.Tensor],
        prompt_embeds: list[torch.Tensor],
        timestep: torch.Tensor,
        patch_size: int,
        f_patch_size: int,
        audio_t_lengths: torch.Tensor,
        compute_audio_scale_t_indices: bool,
    ) -> _BranchPacked:
        """Run a single branch through embed + refiners + unification."""
        packed_inputs = branch.prepare_dense_inputs(
            latent_list,
            prompt_embeds,
            patch_size,
            f_patch_size,
        )
        x = packed_inputs["x"]
        cap_feats = packed_inputs["cap_feats"]
        x_size = packed_inputs["x_size"]
        x_freqs = packed_inputs["x_freqs"]
        cap_freqs = packed_inputs["cap_freqs"]
        x_mask = packed_inputs["x_mask"]
        cap_mask = packed_inputs["cap_mask"]
        device = x.device
        max_x_len = int(x.shape[1])

        # Mirror the patched _forward's preamble
        use_timestep = bool(getattr(branch, "_laion_use_timestep", True))
        if use_timestep:
            adaln_input = branch.t_embedder(timestep * branch.t_scale).type_as(x)
        else:
            adaln_input = None

        # Embed
        x = branch.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        cap_feats = branch.cap_embedder(cap_feats)

        # Refiners. We inline the iteration here (mirroring the patched
        # ``_run_(noise|context)_refiner_blocks`` body) so we don't depend
        # on the patcher's nested closures, which are not importable.
        use_grad_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing
        for layer in branch.context_refiner:
            if use_grad_ckpt:
                cap_feats = branch._gradient_checkpointing_func(layer, cap_feats, cap_mask, cap_freqs)
            else:
                cap_feats = layer(cap_feats, cap_mask, cap_freqs)
        for layer in branch.noise_refiner:
            if use_grad_ckpt:
                x = branch._gradient_checkpointing_func(
                    layer, x, x_mask, x_freqs, adaln_input, None, None, None,
                )
            else:
                x = layer(x, x_mask, x_freqs, adaln_input, None, None, None)

        # Build unified sequence (basic mode == [x, cap])
        unified, unified_freqs, unified_mask, _, x_start_offsets, x_lengths = _build_unified_dense_sequence(
            x, x_freqs, x_mask, cap_feats, cap_freqs, cap_mask,
            omni_mode=False,
            device=device,
        )

        # Per-token audio-scale time indices for the bridge. For the audio
        # branch (H=W=1) the scale is identity, but we still go through
        # the same code path so the audio branch can ignore the
        # `audio_t_lengths` arg.
        if compute_audio_scale_t_indices:
            t_indices = _compute_audio_scale_t_indices(
                x_size=x_size,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                max_x_len=max_x_len,
                audio_t_lengths_per_sample=audio_t_lengths,
                device=device,
            )
        else:
            t_indices = _compute_native_t_indices(
                x_size=x_size,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                max_x_len=max_x_len,
                device=device,
            )

        return _BranchPacked(
            unified=unified,
            unified_freqs=unified_freqs,
            unified_mask=unified_mask,
            x_start_offsets=x_start_offsets,
            x_lengths=x_lengths,
            max_x_len=max_x_len,
            x_size=x_size,
            adaln_input=adaln_input,
            t_emb=adaln_input,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            audio_scale_t_indices=t_indices,
        )

    def _run_main_blocks_with_bridges(
        self,
        video_packed: Optional[_BranchPacked],
        audio_packed: Optional[_BranchPacked],
        bridge_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Walk ``self.video.layers`` and ``self.audio.layers`` in
        lock-step, inserting bridges after every ``bridge_interval``
        layers.

        ``bridge_mask`` is an optional ``[B] bool`` tensor; samples whose
        entry is ``False`` do not receive the bridge delta (equivalent
        to a single-modality forward for that sample). ``None`` means
        the legacy behaviour (all samples bridged). The bridge module
        itself still runs on the full batch -- only the scatter-add of
        the delta back into the unified sequence is gated. This keeps
        the forward shape-identical regardless of mask, which matters
        for torch.compile.
        """
        n_main = len(self.video.layers)
        v_unified = video_packed.unified if video_packed is not None else None
        a_unified = audio_packed.unified if audio_packed is not None else None

        ckpt = (
            torch.utils.checkpoint.checkpoint
            if torch.is_grad_enabled() and self.gradient_checkpointing
            else None
        )

        def _call_layer(layer, hidden, mask, freqs, adaln_input):
            if ckpt is not None:
                return ckpt(
                    lambda h: layer(h, mask, freqs, adaln_input, None, None, None),
                    hidden,
                    use_reentrant=False,
                )
            return layer(hidden, mask, freqs, adaln_input, None, None, None)

        # Pre-shape the per-sample bridge gate so it broadcasts cleanly
        # against the [B, S, D] delta tensors. Skip the cast entirely
        # when no mask is provided so the compiled graph is identical
        # to the legacy path.
        bridge_gate: Optional[torch.Tensor] = None
        if bridge_mask is not None:
            if bridge_mask.dtype != torch.bool:
                bridge_mask = bridge_mask.to(dtype=torch.bool)
            # Will be cast to delta dtype lazily below.
            bridge_gate = bridge_mask

        for layer_idx in range(n_main):
            if v_unified is not None:
                v_unified = _call_layer(
                    self.video.layers[layer_idx],
                    v_unified,
                    video_packed.unified_mask,
                    video_packed.unified_freqs,
                    video_packed.adaln_input,
                )
            if a_unified is not None:
                a_unified = _call_layer(
                    self.audio.layers[layer_idx],
                    a_unified,
                    audio_packed.unified_mask,
                    audio_packed.unified_freqs,
                    audio_packed.adaln_input,
                )

            bridge_idx = self._bridge_after_layer.get(layer_idx)
            if (
                bridge_idx is None
                or not self.bridge_enabled
                or v_unified is None
                or a_unified is None
            ):
                continue

            # Extract X tokens, run bridge, scatter deltas back. We do
            # not gradient-checkpoint the bridge separately because its
            # forward is small relative to a main block; the inner main
            # blocks are already checkpointed above.
            v_x = _extract_compact_x_tokens(
                v_unified, video_packed.x_start_offsets, video_packed.x_lengths, video_packed.max_x_len,
            )
            a_x = _extract_compact_x_tokens(
                a_unified, audio_packed.x_start_offsets, audio_packed.x_lengths, audio_packed.max_x_len,
            )
            v_x_mask = (
                torch.arange(video_packed.max_x_len, device=v_x.device).unsqueeze(0)
                < video_packed.x_lengths.unsqueeze(1)
            )
            a_x_mask = (
                torch.arange(audio_packed.max_x_len, device=a_x.device).unsqueeze(0)
                < audio_packed.x_lengths.unsqueeze(1)
            )

            delta_v, delta_a = self.bridges[bridge_idx](
                video_tokens=v_x,
                audio_tokens=a_x,
                video_mask=v_x_mask,
                audio_mask=a_x_mask,
                video_t_indices=video_packed.audio_scale_t_indices,
                audio_t_indices=audio_packed.audio_scale_t_indices,
                t_emb_video=video_packed.t_emb,
                t_emb_audio=audio_packed.t_emb,
            )

            if bridge_gate is not None:
                # Gate the per-sample bridge contribution. Padded tokens
                # already have zero contribution from the bridge; this
                # additionally zeroes out samples where bridge_mask is
                # False, leaving them byte-equivalent to a single-
                # modality forward for that sample.
                gate_v = bridge_gate.to(device=delta_v.device, dtype=delta_v.dtype).view(-1, 1, 1)
                gate_a = bridge_gate.to(device=delta_a.device, dtype=delta_a.dtype).view(-1, 1, 1)
                delta_v = delta_v * gate_v
                delta_a = delta_a * gate_a

            v_unified = _scatter_x_delta_into_unified(
                v_unified, delta_v, video_packed.x_start_offsets, video_packed.x_lengths, video_packed.max_x_len,
            )
            a_unified = _scatter_x_delta_into_unified(
                a_unified, delta_a, audio_packed.x_start_offsets, audio_packed.x_lengths, audio_packed.max_x_len,
            )

        return v_unified, a_unified

    def _finalize_branch(
        self,
        branch,
        packed: _BranchPacked,
        unified: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Apply ``final_layer`` and unpatchify, returning a list of
        per-sample latent tensors (matching the trunk's contract)."""
        ps = packed.patch_size
        fps = packed.f_patch_size
        unified = branch.all_final_layer[f"{ps}-{fps}"](unified, c=packed.adaln_input)
        x_tokens = _extract_compact_x_tokens(
            unified, packed.x_start_offsets, packed.x_lengths, packed.max_x_len,
        )
        return _unpatchify_compact_x_tokens(
            x_tokens, packed.x_size, ps, fps, branch.out_channels,
        )

    def forward(
        self,
        *,
        video_x: Optional[torch.Tensor],     # [B, C_v, T, H, W] -- video latent
        video_t: Optional[torch.Tensor],     # [B] float -- model timestep for video
        audio_x: Optional[torch.Tensor],     # [B, C_a, T_a] -- audio latent
        audio_t: Optional[torch.Tensor],     # [B] float -- model timestep for audio
        prompt_embeds_video: list[torch.Tensor],
        prompt_embeds_audio: list[torch.Tensor],
        video_patch_size: int,
        video_f_patch_size: int,
        audio_patch_size: int = 1,
        audio_f_patch_size: int = 1,
        bridge_mask: Optional[torch.Tensor] = None,  # [B] bool; None = all True
    ) -> tuple[Optional[list[torch.Tensor]], Optional[list[torch.Tensor]]]:
        """Joint forward.

        Returns ``(video_pred_list, audio_pred_list)`` where each entry
        is ``None`` if that modality was not provided. This supports the
        three validation modes (joint AV, video-only, audio-only) with a
        single code path.

        ``bridge_mask`` (optional ``[B] bool``) gates the bridge delta
        per-sample: samples with ``False`` skip the cross-modal
        information exchange and behave like single-branch forwards.
        Used by training-time bridge dropout and inference-time
        BridgeDiT-style dual CFG (NFE=3). Default ``None`` keeps the
        legacy "all samples bridged" behaviour byte-equivalent.
        """
        if video_x is None and audio_x is None:
            raise ValueError("At least one of video_x / audio_x must be provided.")

        # Audio-scale time lengths for each sample, needed to map video
        # token t-indices into the audio timeline. When audio is missing
        # we fall back to the audio branch's RoPE table length so the
        # video branch can still run alone.
        if audio_x is not None:
            # audio_x shape is [B, C, T_a, 1, 1] after build_forward_transformer's expansion
            if audio_x.dim() == 3:
                audio_t_per_sample = torch.full(
                    (audio_x.shape[0],), int(audio_x.shape[-1]), dtype=torch.long, device=audio_x.device,
                )
            elif audio_x.dim() == 5:
                audio_t_per_sample = torch.full(
                    (audio_x.shape[0],), int(audio_x.shape[2]), dtype=torch.long, device=audio_x.device,
                )
            else:
                raise ValueError(
                    f"audio_x must have ndim in (3, 5), got shape {tuple(audio_x.shape)}."
                )
        else:
            assert video_x is not None
            audio_t_per_sample = torch.full(
                (video_x.shape[0],),
                int(self.audio.rope_embedder.axes_lens[0]),
                dtype=torch.long,
                device=video_x.device,
            )

        video_packed: Optional[_BranchPacked] = None
        if video_x is not None:
            assert video_t is not None
            if video_x.dim() != 5:
                raise ValueError(f"video_x must be [B, C, T, H, W], got {tuple(video_x.shape)}.")
            video_packed = self._prepare_branch(
                self.video,
                list(video_x.unbind(dim=0)),
                prompt_embeds_video,
                video_t,
                video_patch_size,
                video_f_patch_size,
                audio_t_per_sample,
                compute_audio_scale_t_indices=True,
            )

        audio_packed: Optional[_BranchPacked] = None
        if audio_x is not None:
            assert audio_t is not None
            audio_input = audio_x
            if audio_input.dim() == 3:
                audio_input = audio_input.unsqueeze(-1).unsqueeze(-1)
            audio_packed = self._prepare_branch(
                self.audio,
                list(audio_input.unbind(dim=0)),
                prompt_embeds_audio,
                audio_t,
                audio_patch_size,
                audio_f_patch_size,
                audio_t_per_sample,
                compute_audio_scale_t_indices=False,
            )

        v_unified, a_unified = self._run_main_blocks_with_bridges(
            video_packed, audio_packed, bridge_mask=bridge_mask,
        )

        video_pred = (
            self._finalize_branch(self.video, video_packed, v_unified)
            if video_packed is not None else None
        )
        audio_pred = (
            self._finalize_branch(self.audio, audio_packed, a_unified)
            if audio_packed is not None else None
        )
        return video_pred, audio_pred

    @staticmethod
    def stack_branch_predictions(pred_list: list[torch.Tensor]) -> torch.Tensor:
        """Match ``build_forward_transformer``'s return contract: stack
        per-sample predictions back into a dense tensor in fp32."""
        return torch.stack([item.float() for item in pred_list], dim=0)


def build_joint_model(
    *,
    video_transformer,
    audio_transformer,
    bridge_interval: int = 2,
    bridge_enabled: bool = True,
    use_asymmetric_ati: bool = False,
    a2v_window_size: int = 1,
    qk_norm: bool = True,
    norm_eps: float = 1e-5,
) -> BridgedZImageJointModel:
    """Convenience constructor that mirrors the ``BridgedZImageJointModel``
    keyword args. Kept as a free function so trainer code can stay
    declarative."""
    model = BridgedZImageJointModel(
        video_transformer=video_transformer,
        audio_transformer=audio_transformer,
        bridge_interval=bridge_interval,
        bridge_enabled=bridge_enabled,
        use_asymmetric_ati=use_asymmetric_ati,
        a2v_window_size=a2v_window_size,
        qk_norm=qk_norm,
        norm_eps=norm_eps,
    )
    model.assert_bridges_zero_initialised()
    return model
