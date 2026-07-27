from __future__ import annotations

from functools import lru_cache

import torch

# Match diffusers.models.transformers.transformer_z_image.SEQ_MULTI_OF.
SEQ_MULTI_OF = 64
FLAT_POSITION_IDS_CACHE_SIZE = 512


def retrieve_latents(
    encoder_output: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder output.")


def is_flux2_vae(vae) -> bool:
    return getattr(getattr(vae, "config", None), "_class_name", "") == "AutoencoderKLFlux2"


def vae_uses_training_layout(vae) -> bool:
    return bool(getattr(vae, "_laion_uses_training_layout", False))


def vae_encode_returns_training_latents(vae) -> bool:
    return bool(getattr(vae, "_laion_encode_returns_training_latents", False))


def _flux2_patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"Expected FLUX.2 latents with 4 dims, got {latents.ndim}.")

    batch_size, num_channels_latents, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "FLUX.2 latents require even spatial dimensions before 2x2 patchify, "
            f"got height={height}, width={width}."
        )

    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def _flux2_unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"Expected FLUX.2 latents with 4 dims, got {latents.ndim}.")

    batch_size, num_channels_latents, height, width = latents.shape
    if num_channels_latents % 4 != 0:
        raise ValueError(
            "FLUX.2 patchified latents require the channel dimension to be divisible by 4, "
            f"got channels={num_channels_latents}."
        )

    latents = latents.reshape(batch_size, num_channels_latents // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // 4, height * 2, width * 2)
    return latents


def _flux2_bn_stats(vae, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(vae, "bn"):
        raise ValueError("FLUX.2 VAE is missing the expected batch-norm layer for latent normalization.")

    batch_norm_eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + batch_norm_eps).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    if latents_bn_mean.shape[1] != latents.shape[1]:
        raise ValueError(
            "FLUX.2 VAE BN stats do not match patchified latent channels: "
            f"stats={latents_bn_mean.shape[1]}, latents={latents.shape[1]}."
        )
    return latents_bn_mean, latents_bn_std


def flux2_raw_latents_to_training_layout(latents: torch.Tensor, vae) -> torch.Tensor:
    patchified_latents = _flux2_patchify_latents(latents)
    latents_bn_mean, latents_bn_std = _flux2_bn_stats(vae, patchified_latents)
    patchified_latents = (patchified_latents - latents_bn_mean) / latents_bn_std

    # Z-Image patchifies latent feature maps internally. Keep FLUX.2 latents in the original
    # 32-channel image layout so the model sees the official patchified+normalized representation
    # after its own patchifier runs.
    return _flux2_unpatchify_latents(patchified_latents)


def _flux2_training_latents_to_raw_layout(latents: torch.Tensor, vae) -> torch.Tensor:
    patchified_latents = _flux2_patchify_latents(latents)
    latents_bn_mean, latents_bn_std = _flux2_bn_stats(vae, patchified_latents)
    patchified_latents = patchified_latents * latents_bn_std + latents_bn_mean
    return _flux2_unpatchify_latents(patchified_latents)


def raw_latents_to_training_layout(latents: torch.Tensor, vae, *, update_stats: bool = False) -> torch.Tensor:
    if hasattr(vae, "raw_latents_to_training_layout"):
        return vae.raw_latents_to_training_layout(latents, update_stats=update_stats)
    if is_flux2_vae(vae):
        return flux2_raw_latents_to_training_layout(latents, vae)
    raise ValueError(f"{vae.__class__.__name__} does not expose a training-latent layout adapter.")


def training_latents_to_raw_layout(latents: torch.Tensor, vae) -> torch.Tensor:
    if hasattr(vae, "training_latents_to_raw_layout"):
        return vae.training_latents_to_raw_layout(latents)
    if is_flux2_vae(vae):
        return _flux2_training_latents_to_raw_layout(latents, vae)
    raise ValueError(f"{vae.__class__.__name__} does not expose a raw-latent layout adapter.")


def patch_flux2_vae_for_zimage(vae):
    if not is_flux2_vae(vae) or getattr(vae, "_laion_flux2_zimage_patch_applied", False):
        return vae

    original_decode = vae.decode

    def _decode_with_flux2_training_layout(latents, *args, **kwargs):
        latents = _flux2_training_latents_to_raw_layout(latents, vae)
        return original_decode(latents, *args, **kwargs)

    vae._laion_original_decode = original_decode
    vae.decode = _decode_with_flux2_training_layout
    vae.raw_latents_to_training_layout = lambda latents, update_stats=False: flux2_raw_latents_to_training_layout(
        latents,
        vae,
    )
    vae.training_latents_to_raw_layout = lambda latents: _flux2_training_latents_to_raw_layout(latents, vae)
    vae._laion_uses_training_layout = True
    vae._laion_encode_returns_training_latents = False
    vae._laion_flux2_zimage_patch_applied = True
    return vae


def _round_up_to_multiple(value: int, multiple: int = SEQ_MULTI_OF) -> int:
    if value <= 0:
        return 0
    return value + (-value) % multiple


def _resolve_padded_sequence_length(lengths_list: list[int], pad_to_length: int | None = None) -> int:
    max_seqlen = max(lengths_list) if lengths_list else 0
    if pad_to_length is None:
        return _round_up_to_multiple(max_seqlen)

    target_length = int(pad_to_length)
    if target_length < 0:
        raise ValueError(f"pad_to_length must be >= 0, got {pad_to_length!r}.")
    if target_length < max_seqlen:
        raise ValueError(
            f"pad_to_length={target_length} is smaller than the longest sequence in the batch ({max_seqlen})."
        )
    return target_length


@lru_cache(maxsize=FLAT_POSITION_IDS_CACHE_SIZE)
def _make_flat_position_ids_cached(
    size: tuple[int, int, int],
    start: tuple[int, int, int],
    device_type: str,
    device_index: int | None,
):
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    if any(length == 0 for length in size):
        return torch.zeros((0, len(size)), dtype=torch.int32, device=device)
    # Match diffusers' ZImageTransformer2DModel.create_coordinate_grid implementation.
    axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
    grids = torch.meshgrid(axes, indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, len(size))


def _make_flat_position_ids(_model, size: tuple[int, int, int], start: tuple[int, int, int], device: torch.device):
    normalized_device = torch.device(device)
    return _make_flat_position_ids_cached(
        tuple(int(length) for length in size),
        tuple(int(offset) for offset in start),
        normalized_device.type,
        normalized_device.index,
    )


def _pack_sequence_batch(
    feats: list[torch.Tensor],
    pos_ids: list[torch.Tensor],
    rope_embedder,
    device: torch.device,
    noise_masks: list[list[int]] | None = None,
    pad_to_length: int | None = None,
):
    bsz = len(feats)
    if bsz == 0:
        raise ValueError("Expected at least one sequence to pack.")

    lengths_list = [int(feat.shape[0]) for feat in feats]
    lengths = torch.tensor(lengths_list, dtype=torch.long, device=device)
    max_seqlen = _resolve_padded_sequence_length(lengths_list, pad_to_length=pad_to_length)
    feat_dim = feats[0].shape[-1]
    freq_dim = sum(rope_embedder.axes_dims) // 2

    if max_seqlen == 0:
        feats_batch = feats[0].new_zeros((bsz, 0, feat_dim))
        freqs_batch = torch.zeros((bsz, 0, freq_dim, 2), dtype=torch.float32, device=device)
    else:
        feats_batch = feats[0].new_zeros((bsz, max_seqlen, feat_dim))
        for idx, feat in enumerate(feats):
            seq_len = feat.shape[0]
            if seq_len > 0:
                feats_batch[idx, :seq_len] = feat
        flat_pos_ids = torch.cat([pos[: feat.shape[0]] for feat, pos in zip(feats, pos_ids)], dim=0)
        freqs_list = list(rope_embedder(flat_pos_ids).split(lengths_list, dim=0))
        freqs_batch = torch.zeros((bsz, max_seqlen, freq_dim, 2), dtype=torch.float32, device=device)
        for idx, freqs in enumerate(freqs_list):
            seq_len = freqs.shape[0]
            if seq_len > 0:
                freqs_batch[idx, :seq_len] = freqs
        freqs_batch = freqs_batch.contiguous()

    attn_mask = torch.arange(max_seqlen, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    noise_mask_tensor = None
    if noise_masks is not None:
        if max_seqlen == 0:
            noise_mask_tensor = torch.zeros((bsz, 0), dtype=torch.long, device=device)
        else:
            noise_mask_tensor = torch.zeros((bsz, max_seqlen), dtype=torch.long, device=device)
            for idx, (feat, mask) in enumerate(zip(feats, noise_masks)):
                seq_len = feat.shape[0]
                if seq_len > 0:
                    noise_mask_tensor[idx, :seq_len] = torch.tensor(mask[:seq_len], dtype=torch.long, device=device)

    return feats_batch, freqs_batch, attn_mask, lengths, noise_mask_tensor


def _build_unified_dense_sequence(
    x: torch.Tensor,
    x_freqs: torch.Tensor,
    x_mask: torch.Tensor,
    cap: torch.Tensor,
    cap_freqs: torch.Tensor,
    cap_mask: torch.Tensor,
    omni_mode: bool,
    device: torch.device,
    x_noise_mask: torch.Tensor | None = None,
    cap_noise_mask: torch.Tensor | None = None,
    siglip: torch.Tensor | None = None,
    siglip_freqs: torch.Tensor | None = None,
    siglip_mask: torch.Tensor | None = None,
    siglip_noise_mask: torch.Tensor | None = None,
):
    batch_size = x.shape[0]
    x_lengths = x_mask.sum(dim=1, dtype=torch.long)
    cap_lengths = cap_mask.sum(dim=1, dtype=torch.long)
    zero_offsets = torch.zeros_like(x_lengths)

    if omni_mode:
        siglip_lengths = (
            siglip_mask.sum(dim=1, dtype=torch.long)
            if siglip is not None and siglip_mask is not None
            else torch.zeros_like(x_lengths)
        )
        unified_lengths = cap_lengths + x_lengths + siglip_lengths
        x_start_offsets = cap_lengths
    else:
        siglip_lengths = None
        unified_lengths = x_lengths + cap_lengths
        x_start_offsets = zero_offsets

    max_unified_len = x.shape[1] + cap.shape[1]
    if omni_mode and siglip is not None:
        max_unified_len += siglip.shape[1]
    hidden_dim = x.shape[-1]
    unified = x.new_zeros((batch_size, max_unified_len, hidden_dim))
    unified_freqs = x_freqs.new_zeros((batch_size, max_unified_len, *x_freqs.shape[2:]))
    unified_mask = torch.arange(max_unified_len, device=device).unsqueeze(0) < unified_lengths.unsqueeze(1)
    unified_noise_mask = (
        torch.zeros((batch_size, max_unified_len), dtype=torch.long, device=device) if omni_mode else None
    )

    def _scatter_segment(
        src: torch.Tensor | None,
        src_freqs: torch.Tensor | None,
        src_mask: torch.Tensor | None,
        start_offsets: torch.Tensor,
        src_noise_mask: torch.Tensor | None = None,
    ) -> None:
        if src is None or src_mask is None or src.shape[1] == 0:
            return

        src_positions = src_mask.long().cumsum(dim=1) - 1
        target_positions = start_offsets.unsqueeze(1) + src_positions
        safe_target_positions = target_positions.masked_fill(~src_mask, 0)

        src_values = (src * src_mask.unsqueeze(-1).to(src.dtype)).to(unified.dtype)
        unified.scatter_add_(
            1,
            safe_target_positions.unsqueeze(-1).expand(-1, -1, src.shape[-1]),
            src_values,
        )

        freq_mask = src_mask.unsqueeze(-1).unsqueeze(-1).to(src_freqs.dtype)
        src_freq_values = (src_freqs * freq_mask).to(unified_freqs.dtype)
        unified_freqs.scatter_add_(
            1,
            safe_target_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *src_freqs.shape[2:]),
            src_freq_values,
        )
        if unified_noise_mask is not None and src_noise_mask is not None:
            unified_noise_mask.scatter_add_(
                1,
                safe_target_positions,
                src_noise_mask * src_mask.to(src_noise_mask.dtype),
            )

    if omni_mode:
        _scatter_segment(cap, cap_freqs, cap_mask, zero_offsets, cap_noise_mask)
        _scatter_segment(x, x_freqs, x_mask, cap_lengths, x_noise_mask)
        if siglip is not None and siglip_freqs is not None and siglip_mask is not None:
            _scatter_segment(siglip, siglip_freqs, siglip_mask, cap_lengths + x_lengths, siglip_noise_mask)
    else:
        _scatter_segment(x, x_freqs, x_mask, zero_offsets)
        _scatter_segment(cap, cap_freqs, cap_mask, x_lengths)

    return unified, unified_freqs, unified_mask, unified_noise_mask, x_start_offsets, x_lengths


def _extract_compact_x_tokens(
    unified: torch.Tensor,
    x_start_offsets: torch.Tensor,
    x_lengths: torch.Tensor,
    max_x_len: int,
):
    _, _, hidden_dim = unified.shape
    local_positions = torch.arange(max_x_len, device=unified.device).unsqueeze(0)
    gather_positions = x_start_offsets.unsqueeze(1) + local_positions
    gathered = unified.gather(1, gather_positions.unsqueeze(-1).expand(-1, -1, hidden_dim))
    gathered_mask = local_positions < x_lengths.unsqueeze(1)
    return gathered * gathered_mask.unsqueeze(-1).to(gathered.dtype)


def _unpatchify_compact_x_tokens(
    x_tokens: torch.Tensor,
    size,
    patch_size: int,
    f_patch_size: int,
    out_channels: int,
):
    pH = pW = patch_size
    pF = f_patch_size
    bsz = x_tokens.shape[0]
    omni_mode = isinstance(size[0], list)
    result = []

    if omni_mode:
        for i in range(bsz):
            cu_len = 0
            x_item = None
            for image_size in size[i]:
                if image_size is None:
                    continue
                F, H, W = image_size
                ori_len = (F // pF) * (H // pH) * (W // pW)
                x_item = (
                    x_tokens[i, cu_len : cu_len + ori_len]
                    .view(F // pF, H // pH, W // pW, pF, pH, pW, out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(out_channels, F, H, W)
                )
                cu_len += ori_len
            result.append(x_item)
    else:
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x_item = (
                x_tokens[i, :ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(out_channels, F, H, W)
            )
            result.append(x_item)

    return result


@torch.compile(disable=True)
def _patchify_and_embed_dense(
    self,
    all_image: list[torch.Tensor],
    all_cap_feats: list[torch.Tensor],
    patch_size: int,
    f_patch_size: int,
    cap_target_length: int | None = None,
):
    device = all_image[0].device
    all_img_out, all_img_size, all_img_pos_ids = [], [], []
    all_cap_out, all_cap_pos_ids = [], []

    for image, cap_feat in zip(all_image, all_cap_feats):
        cap_len = int(cap_feat.shape[0])
        cap_pos_ids = _make_flat_position_ids(self, (cap_len, 1, 1), (1, 0, 0), device)
        all_cap_out.append(cap_feat)
        all_cap_pos_ids.append(cap_pos_ids)

        img_patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
        img_pos_ids = _make_flat_position_ids(self, (F_t, H_t, W_t), (cap_len + 1, 0, 0), device)
        all_img_out.append(img_patches)
        all_img_size.append(size)
        all_img_pos_ids.append(img_pos_ids)

    x, x_freqs, x_mask, _, _ = _pack_sequence_batch(all_img_out, all_img_pos_ids, self.rope_embedder, device)
    cap, cap_freqs, cap_mask, _, _ = _pack_sequence_batch(
        all_cap_out,
        all_cap_pos_ids,
        self.rope_embedder,
        device,
        pad_to_length=cap_target_length,
    )

    return x, cap, all_img_size, x_freqs, cap_freqs, x_mask, cap_mask


@torch.compile(disable=True)
def _patchify_and_embed_omni_dense(
    self,
    all_x: list[list[torch.Tensor]],
    all_cap_feats: list[list[torch.Tensor]],
    all_siglip_feats: list[list[torch.Tensor]],
    patch_size: int,
    f_patch_size: int,
    images_noise_mask: list[list[int]],
    cap_target_length: int | None = None,
):
    bsz = len(all_x)
    device = all_x[0][-1].device
    dtype = all_x[0][-1].dtype
    patch_dim = f_patch_size * patch_size * patch_size * self.in_channels
    cap_feat_dim = all_cap_feats[0][0].shape[-1]

    all_x_out, all_x_size, all_x_pos_ids, all_x_noise_mask = [], [], [], []
    all_cap_out, all_cap_pos_ids, all_cap_noise_mask = [], [], []
    all_sig_out, all_sig_pos_ids, all_sig_noise_mask = [], [], []
    has_any_siglip = False

    for i in range(bsz):
        num_images = len(all_x[i])

        cap_feats_list, cap_pos_list, cap_noise = [], [], []
        cap_end_pos = []
        cap_cu_len = 1

        for j, cap_item in enumerate(all_cap_feats[i]):
            noise_val = images_noise_mask[i][j] if j < len(images_noise_mask[i]) else 1
            cap_len = int(cap_item.shape[0])
            cap_pos = _make_flat_position_ids(self, (cap_len, 1, 1), (cap_cu_len, 0, 0), device)
            cap_feats_list.append(cap_item)
            cap_pos_list.append(cap_pos)
            cap_noise.extend([noise_val] * cap_len)
            cap_cu_len += cap_len
            cap_end_pos.append(cap_cu_len)
            cap_cu_len += 2

        if cap_feats_list:
            all_cap_out.append(torch.cat(cap_feats_list, dim=0))
            all_cap_pos_ids.append(torch.cat(cap_pos_list, dim=0))
        else:
            all_cap_out.append(torch.zeros((0, cap_feat_dim), dtype=dtype, device=device))
            all_cap_pos_ids.append(torch.zeros((0, 3), dtype=torch.int32, device=device))
        all_cap_noise_mask.append(cap_noise)

        x_feats_list, x_pos_list, x_size, x_noise = [], [], [], []
        for j, x_item in enumerate(all_x[i]):
            noise_val = images_noise_mask[i][j]
            if x_item is not None:
                x_patches, size, (F_t, H_t, W_t) = self._patchify_image(x_item, patch_size, f_patch_size)
                x_pos = _make_flat_position_ids(self, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), device)
                x_feats_list.append(x_patches)
                x_pos_list.append(x_pos)
                x_noise.extend([noise_val] * int(x_patches.shape[0]))
                x_size.append(size)
            else:
                x_size.append(None)

        if x_feats_list:
            all_x_out.append(torch.cat(x_feats_list, dim=0))
            all_x_pos_ids.append(torch.cat(x_pos_list, dim=0))
        else:
            all_x_out.append(torch.zeros((0, patch_dim), dtype=dtype, device=device))
            all_x_pos_ids.append(torch.zeros((0, 3), dtype=torch.int32, device=device))
        all_x_size.append(x_size)
        all_x_noise_mask.append(x_noise)

        if all_siglip_feats[i] is None:
            all_sig_out.append(None)
        else:
            sig_feats_list, sig_pos_list, sig_noise = [], [], []
            for j, sig_item in enumerate(all_siglip_feats[i]):
                noise_val = images_noise_mask[i][j]
                if sig_item is None:
                    continue

                has_any_siglip = True
                sig_H, sig_W, sig_C = sig_item.size()
                sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                sig_pos = _make_flat_position_ids(self, (1, sig_H, sig_W), (cap_end_pos[j] + 1, 0, 0), device)
                if x_size[j] is not None:
                    sig_pos = sig_pos.float()
                    sig_pos[..., 1] = sig_pos[..., 1] / max(sig_H - 1, 1) * (x_size[j][1] - 1)
                    sig_pos[..., 2] = sig_pos[..., 2] / max(sig_W - 1, 1) * (x_size[j][2] - 1)
                    sig_pos = sig_pos.to(torch.int32)
                sig_feats_list.append(sig_flat)
                sig_pos_list.append(sig_pos)
                sig_noise.extend([noise_val] * int(sig_flat.shape[0]))

            if sig_feats_list:
                all_sig_out.append(torch.cat(sig_feats_list, dim=0))
                all_sig_pos_ids.append(torch.cat(sig_pos_list, dim=0))
            else:
                all_sig_out.append(torch.zeros((0, self.config.siglip_feat_dim), dtype=dtype, device=device))
                all_sig_pos_ids.append(torch.zeros((0, 3), dtype=torch.int32, device=device))
            all_sig_noise_mask.append(sig_noise)

    x, x_freqs, x_mask, _, x_noise_tensor = _pack_sequence_batch(
        all_x_out, all_x_pos_ids, self.rope_embedder, device, all_x_noise_mask
    )
    cap, cap_freqs, cap_mask, _, cap_noise_tensor = _pack_sequence_batch(
        all_cap_out,
        all_cap_pos_ids,
        self.rope_embedder,
        device,
        all_cap_noise_mask,
        pad_to_length=cap_target_length,
    )

    if has_any_siglip:
        siglip, siglip_freqs, siglip_mask, _, siglip_noise_tensor = _pack_sequence_batch(
            all_sig_out, all_sig_pos_ids, self.rope_embedder, device, all_sig_noise_mask
        )
    else:
        siglip = siglip_freqs = siglip_mask = siglip_noise_tensor = None

    return (
        x,
        cap,
        siglip,
        all_x_size,
        x_freqs,
        cap_freqs,
        siglip_freqs,
        x_mask,
        cap_mask,
        siglip_mask,
        x_noise_tensor,
        cap_noise_tensor,
        siglip_noise_tensor,
    )


@torch.compile(disable=True)
def _prepare_dense_inputs(
    self,
    x,
    cap_feats,
    patch_size: int,
    f_patch_size: int,
    siglip_feats=None,
    image_noise_mask=None,
    cap_target_length: int | None = None,
):
    if cap_target_length is None:
        cap_target_length = getattr(self, "_laion_caption_target_length", None)
    omni_mode = isinstance(x[0], list)

    if omni_mode:
        (
            x,
            cap_feats,
            siglip_feats,
            x_size,
            x_freqs,
            cap_freqs,
            siglip_freqs,
            x_mask,
            cap_mask,
            siglip_mask,
            x_noise_tensor,
            cap_noise_tensor,
            siglip_noise_tensor,
        ) = self.patchify_and_embed_omni(
            x,
            cap_feats,
            siglip_feats,
            patch_size,
            f_patch_size,
            image_noise_mask,
            cap_target_length=cap_target_length,
        )
    else:
        (
            x,
            cap_feats,
            x_size,
            x_freqs,
            cap_freqs,
            x_mask,
            cap_mask,
        ) = self.patchify_and_embed(
            x,
            cap_feats,
            patch_size,
            f_patch_size,
            cap_target_length=cap_target_length,
        )
        siglip_feats = siglip_freqs = siglip_mask = None
        x_noise_tensor = cap_noise_tensor = siglip_noise_tensor = None

    return {
        "x": x,
        "cap_feats": cap_feats,
        "siglip_feats": siglip_feats,
        "x_size": x_size,
        "x_freqs": x_freqs,
        "cap_freqs": cap_freqs,
        "siglip_freqs": siglip_freqs,
        "x_mask": x_mask,
        "cap_mask": cap_mask,
        "siglip_mask": siglip_mask,
        "x_noise_tensor": x_noise_tensor,
        "cap_noise_tensor": cap_noise_tensor,
        "siglip_noise_tensor": siglip_noise_tensor,
        "omni_mode": omni_mode,
    }


def patch_transformers_qwen3_5_compile_friendly_linear_attn_mask() -> None:
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    except ImportError:
        return

    if getattr(Qwen3_5TextModel, "_laion_trainer_linear_attn_mask_compile_patched", False):
        return

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        Avoid tensor-value-dependent Python branches in HF's original implementation so
        `torch.compile(..., fullgraph=True)` can trace prompt-encoding forwards.

        For prompt encoding, returning an all-ones mask instead of `None` is numerically
        equivalent because the downstream linear-attention path only multiplies hidden
        states by the mask. For standard cached decoding, HF forwards the full 2D
        attention mask while slicing `cache_position` to the new tokens, so the shape
        mismatch still lets us skip masking as upstream does.
        """
        if attention_mask is None:
            return None
        if cache_position is not None and attention_mask.shape[-1] != cache_position.shape[0]:
            return None
        return attention_mask

    Qwen3_5TextModel._laion_trainer_original_update_linear_attn_mask = Qwen3_5TextModel._update_linear_attn_mask
    Qwen3_5TextModel._update_linear_attn_mask = _update_linear_attn_mask
    Qwen3_5TextModel._laion_trainer_linear_attn_mask_compile_patched = True


def patch_transformers_qwen3_5_disable_fast_path() -> None:
    """
    Force Qwen3.5 text layers to construct their torch fallback path.

    HF gates the fast path in `modeling_qwen3_5.py` through the module globals
    `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`, and
    `fused_recurrent_gated_delta_rule`. Clearing them before model construction
    makes new `Qwen3_5GatedDeltaNet` instances bind to the non-fused torch
    implementations instead.
    """
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling_qwen3_5
    except ImportError:
        return

    target_names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    )

    if not hasattr(modeling_qwen3_5, "_laion_trainer_original_fast_path_symbols"):
        modeling_qwen3_5._laion_trainer_original_fast_path_symbols = {
            name: getattr(modeling_qwen3_5, name, None) for name in target_names
        }

    for name in target_names:
        if hasattr(modeling_qwen3_5, name):
            setattr(modeling_qwen3_5, name, None)

    if hasattr(modeling_qwen3_5, "is_fast_path_available"):
        modeling_qwen3_5.is_fast_path_available = False

    modeling_qwen3_5._laion_trainer_qwen3_5_fast_path_disabled = True


def patch_diffusers_zimage_real_rope() -> None:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_z_image import (
        RopeEmbedder,
        ZImageTransformer2DModel,
        ZSingleStreamAttnProcessor,
    )

    if getattr(ZImageTransformer2DModel, "_laion_trainer_real_rope_patched", False):
        return

    @staticmethod
    def _precompute_freqs_cis(dim, end, theta: float = 256.0):
        freqs_cis = []
        cpu_device = torch.device("cpu")
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device=cpu_device) / d))
            timestep = torch.arange(e, device=cpu_device, dtype=torch.float64)
            phase = torch.outer(timestep, freqs).float()
            # Match diffusers' real-valued RoPE path while keeping Z-Image's per-sample sequence layout.
            freqs_cis_i = torch.stack((phase.cos(), phase.sin()), dim=-1)
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def _rope_embedder_call(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if (
            self.freqs_cis is None
            or self.freqs_cis[0].is_complex()
            or self.freqs_cis[0].shape[-1] != 2
        ):
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-2)

    def _materialize_rope_cache(self, device: torch.device | None = None):
        target_device = device
        if target_device is None:
            freqs_cis = self.rope_embedder.freqs_cis
            if freqs_cis is not None and len(freqs_cis) > 0:
                target_device = freqs_cis[0].device
            else:
                target_device = next(self.parameters()).device

        rope_embedder = self.rope_embedder
        rope_embedder.freqs_cis = rope_embedder.precompute_freqs_cis(
            rope_embedder.axes_dims,
            rope_embedder.axes_lens,
            theta=rope_embedder.theta,
        )
        rope_embedder.freqs_cis = [freqs_cis.to(target_device) for freqs_cis in rope_embedder.freqs_cis]
        return rope_embedder.freqs_cis

    def _attn_processor_call(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
                freqs_cis = freqs_cis.to(device=x_in.device, dtype=x.dtype).unsqueeze(2)
                x_real, x_imag = x.unbind(-1)
                cos = freqs_cis[..., 0]
                sin = freqs_cis[..., 1]
                x_out = torch.stack(
                    (x_real * cos - x_imag * sin, x_real * sin + x_imag * cos),
                    dim=-1,
                ).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output

    RopeEmbedder.precompute_freqs_cis = _precompute_freqs_cis
    RopeEmbedder.__call__ = _rope_embedder_call
    ZSingleStreamAttnProcessor.__call__ = _attn_processor_call
    ZImageTransformer2DModel.materialize_rope_cache = _materialize_rope_cache
    ZImageTransformer2DModel._laion_trainer_real_rope_patched = True


def patch_diffusers_zimage_forward_block_stacks() -> None:
    from diffusers.models.transformers.transformer_z_image import (
        FinalLayer,
        Transformer2DModelOutput,
        ZImageTransformerBlock,
        ZImageTransformer2DModel,
    )

    if getattr(ZImageTransformer2DModel, "_laion_trainer_forward_block_stacks_patched", False):
        return

    if not getattr(ZImageTransformerBlock, "_laion_trainer_optional_modulation_init_patched", False):
        original_block_init = ZImageTransformerBlock.__init__

        def _init_block_with_optional_modulation(
            self,
            layer_id: int,
            dim: int,
            n_heads: int,
            n_kv_heads: int,
            norm_eps: float,
            qk_norm: bool,
            modulation=True,
        ):
            if getattr(ZImageTransformerBlock, "_laion_force_disable_modulation", False):
                modulation = False
            return original_block_init(
                self,
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,
                modulation=modulation,
            )

        ZImageTransformerBlock._laion_trainer_original_init = original_block_init
        ZImageTransformerBlock.__init__ = _init_block_with_optional_modulation
        ZImageTransformerBlock._laion_trainer_optional_modulation_init_patched = True

    if not getattr(FinalLayer, "_laion_trainer_optional_modulation_patched", False):
        original_final_init = FinalLayer.__init__
        original_final_forward = FinalLayer.forward

        def _init_final_with_optional_modulation(self, hidden_size, out_channels, modulation=None):
            if modulation is None:
                modulation = getattr(FinalLayer, "_laion_default_modulation", True)
            modulation = bool(modulation)

            if modulation:
                original_final_init(self, hidden_size, out_channels)
            else:
                super(FinalLayer, self).__init__()
                self.norm_final = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
                self.linear = torch.nn.Linear(hidden_size, out_channels, bias=True)
                self.adaLN_modulation = None

            self.modulation = modulation

        def _forward_final_with_optional_modulation(
            self,
            x,
            c=None,
            noise_mask=None,
            c_noisy=None,
            c_clean=None,
        ):
            if getattr(self, "modulation", True) and not getattr(self, "_laion_disable_modulation", False):
                return original_final_forward(
                    self,
                    x,
                    c=c,
                    noise_mask=noise_mask,
                    c_noisy=c_noisy,
                    c_clean=c_clean,
                )

            x = self.norm_final(x)
            x = self.linear(x)
            return x

        FinalLayer._laion_trainer_original_init = original_final_init
        FinalLayer.__init__ = _init_final_with_optional_modulation
        FinalLayer._laion_trainer_original_forward = original_final_forward
        FinalLayer.forward = _forward_final_with_optional_modulation
        FinalLayer._laion_trainer_optional_modulation_patched = True

    def _run_noise_refiner_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_gradient_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        for layer in self.noise_refiner:
            hidden_states = (
                self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    adaln_input,
                    noise_mask,
                    adaln_noisy,
                    adaln_clean,
                )
                if use_gradient_checkpointing
                else layer(
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    adaln_input,
                    noise_mask,
                    adaln_noisy,
                    adaln_clean,
                )
            )
        return hidden_states

    def _run_context_refiner_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        use_gradient_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        for layer in self.context_refiner:
            hidden_states = (
                self._gradient_checkpointing_func(layer, hidden_states, attention_mask, freqs_cis)
                if use_gradient_checkpointing
                else layer(hidden_states, attention_mask, freqs_cis)
            )
        return hidden_states

    def _run_siglip_refiner_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        use_gradient_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        for layer in self.siglip_refiner:
            hidden_states = (
                self._gradient_checkpointing_func(layer, hidden_states, attention_mask, freqs_cis)
                if use_gradient_checkpointing
                else layer(hidden_states, attention_mask, freqs_cis)
            )
        return hidden_states

    def _run_main_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        adaln_noisy: torch.Tensor | None = None,
        adaln_clean: torch.Tensor | None = None,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        use_gradient_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = (
                self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    adaln_input,
                    noise_mask,
                    adaln_noisy,
                    adaln_clean,
                )
                if use_gradient_checkpointing
                else layer(
                    hidden_states,
                    attention_mask,
                    freqs_cis,
                    adaln_input,
                    noise_mask,
                    adaln_noisy,
                    adaln_clean,
                )
            )
            if controlnet_block_samples is not None and layer_idx in controlnet_block_samples:
                hidden_states = hidden_states + controlnet_block_samples[layer_idx]
        return hidden_states

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        *,
        x_size,
        x_freqs: torch.Tensor,
        cap_freqs: torch.Tensor,
        x_mask: torch.Tensor,
        cap_mask: torch.Tensor,
        return_dict: bool = True,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
        siglip_feats: torch.Tensor | None = None,
        siglip_freqs: torch.Tensor | None = None,
        siglip_mask: torch.Tensor | None = None,
        x_noise_tensor: torch.Tensor | None = None,
        cap_noise_tensor: torch.Tensor | None = None,
        siglip_noise_tensor: torch.Tensor | None = None,
        omni_mode: bool = False,
        patch_size: int | None = None,
        f_patch_size: int | None = None,
    ):
        """
        Flow: x_embed -> x_refine -> cap_embed -> cap_refine
              -> [siglip_embed -> siglip_refine] -> build_unified -> main_layers -> final_layer -> unpatchify
        """
        if patch_size is None:
            patch_size = int(self.all_patch_size[0])
        if f_patch_size is None:
            f_patch_size = int(self.all_f_patch_size[0])
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        device = x.device
        use_timestep = bool(getattr(self, "_laion_use_timestep", True))

        if use_timestep and omni_mode:
            t_noisy = self.t_embedder(t * self.t_scale).type_as(x)
            t_clean = self.t_embedder(torch.ones_like(t) * self.t_scale).type_as(x)
            adaln_input = None
        elif use_timestep:
            adaln_input = self.t_embedder(t * self.t_scale).type_as(x)
            t_noisy = t_clean = None
        else:
            adaln_input = t_noisy = t_clean = None

        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats = _run_context_refiner_blocks(self, cap_feats, cap_mask, cap_freqs)

        if siglip_feats is not None and self.siglip_embedder is not None:
            siglip_feats = self.siglip_embedder(siglip_feats)
            siglip_feats = _run_siglip_refiner_blocks(self, siglip_feats, siglip_mask, siglip_freqs)

        x = _run_noise_refiner_blocks(
            self,
            x,
            x_mask,
            x_freqs,
            adaln_input,
            x_noise_tensor,
            t_noisy,
            t_clean,
        )

        unified, unified_freqs, unified_mask, unified_noise_tensor, x_start_offsets, x_token_lengths = (
            _build_unified_dense_sequence(
                x,
                x_freqs,
                x_mask,
                cap_feats,
                cap_freqs,
                cap_mask,
                omni_mode,
                device,
                x_noise_tensor,
                cap_noise_tensor,
                siglip_feats,
                siglip_freqs,
                siglip_mask,
                siglip_noise_tensor,
            )
        )
        for layer_idx, layer in enumerate(self.layers):
            unified = (
                self._gradient_checkpointing_func(
                    layer,
                    unified,
                    unified_mask,
                    unified_freqs,
                    adaln_input,
                    unified_noise_tensor,
                    t_noisy,
                    t_clean,
                )
                if torch.is_grad_enabled() and self.gradient_checkpointing
                else layer(unified, unified_mask, unified_freqs, adaln_input, unified_noise_tensor, t_noisy, t_clean)
            )
            if controlnet_block_samples is not None and layer_idx in controlnet_block_samples:
                unified = unified + controlnet_block_samples[layer_idx]

        unified = (
            self.all_final_layer[f"{patch_size}-{f_patch_size}"](
                unified, noise_mask=unified_noise_tensor, c_noisy=t_noisy, c_clean=t_clean
            )
            if omni_mode
            else self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        )

        x = _extract_compact_x_tokens(unified, x_start_offsets, x_token_lengths, x.shape[1])
        x = _unpatchify_compact_x_tokens(x, x_size, patch_size, f_patch_size, self.out_channels)
        if not return_dict:
            return (x,)
        output = Transformer2DModelOutput(sample=x)
        return output

    def _forward_compat(
        self,
        x,
        t,
        cap_feats,
        return_dict: bool = True,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
        siglip_feats=None,
        image_noise_mask=None,
        patch_size: int | None = None,
        f_patch_size: int | None = None,
        x_size=None,
        x_freqs: torch.Tensor | None = None,
        cap_freqs: torch.Tensor | None = None,
        x_mask: torch.Tensor | None = None,
        cap_mask: torch.Tensor | None = None,
        siglip_freqs: torch.Tensor | None = None,
        siglip_mask: torch.Tensor | None = None,
        x_noise_tensor: torch.Tensor | None = None,
        cap_noise_tensor: torch.Tensor | None = None,
        siglip_noise_tensor: torch.Tensor | None = None,
        omni_mode: bool | None = None,
        cap_target_length: int | None = None,
    ):
        packed_forward = getattr(self, "_laion_trainer_forward_packed_callable", None)
        if packed_forward is None:
            packed_forward = type(self)._laion_trainer_forward_packed_impl.__get__(self, type(self))

        if patch_size is None:
            patch_size = int(self.all_patch_size[0])
        if f_patch_size is None:
            f_patch_size = int(self.all_f_patch_size[0])

        if not torch.is_tensor(x):
            packed_inputs = self.prepare_dense_inputs(
                x,
                cap_feats,
                patch_size,
                f_patch_size,
                siglip_feats=siglip_feats,
                image_noise_mask=image_noise_mask,
                cap_target_length=cap_target_length,
            )
            return packed_forward(
                packed_inputs["x"],
                t,
                packed_inputs["cap_feats"],
                x_size=packed_inputs["x_size"],
                x_freqs=packed_inputs["x_freqs"],
                cap_freqs=packed_inputs["cap_freqs"],
                x_mask=packed_inputs["x_mask"],
                cap_mask=packed_inputs["cap_mask"],
                return_dict=return_dict,
                controlnet_block_samples=controlnet_block_samples,
                siglip_feats=packed_inputs["siglip_feats"],
                siglip_freqs=packed_inputs["siglip_freqs"],
                siglip_mask=packed_inputs["siglip_mask"],
                x_noise_tensor=packed_inputs["x_noise_tensor"],
                cap_noise_tensor=packed_inputs["cap_noise_tensor"],
                siglip_noise_tensor=packed_inputs["siglip_noise_tensor"],
                omni_mode=packed_inputs["omni_mode"],
                patch_size=patch_size,
                f_patch_size=f_patch_size,
            )

        if x_size is None or x_freqs is None or cap_freqs is None or x_mask is None or cap_mask is None:
            raise ValueError("Packed tensor forward expects x_size, x_freqs, cap_freqs, x_mask, and cap_mask.")

        return packed_forward(
            x,
            t,
            cap_feats,
            x_size=x_size,
            x_freqs=x_freqs,
            cap_freqs=cap_freqs,
            x_mask=x_mask,
            cap_mask=cap_mask,
            return_dict=return_dict,
            controlnet_block_samples=controlnet_block_samples,
            siglip_feats=siglip_feats,
            siglip_freqs=siglip_freqs,
            siglip_mask=siglip_mask,
            x_noise_tensor=x_noise_tensor,
            cap_noise_tensor=cap_noise_tensor,
            siglip_noise_tensor=siglip_noise_tensor,
            omni_mode=bool(omni_mode),
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )

    ZImageTransformer2DModel.patchify_and_embed = _patchify_and_embed_dense
    ZImageTransformer2DModel.patchify_and_embed_omni = _patchify_and_embed_omni_dense
    ZImageTransformer2DModel.prepare_dense_inputs = _prepare_dense_inputs
    ZImageTransformer2DModel._laion_trainer_forward_packed_impl = _forward
    ZImageTransformer2DModel.forward = _forward_compat

    # ZImageTransformer2DModel.patchify_and_embed = torch.compile(ZImageTransformer2DModel.patchify_and_embed, fullgraph=True, mode="reduce-overhead")

    def _set_forward_compilation(self, compile_model: bool) -> None:
        packed_impl = type(self)._laion_trainer_forward_packed_impl.__get__(self, type(self))
        if compile_model:
            compiled_impl = getattr(self, "_laion_trainer_forward_packed_compiled_impl", None)
            if compiled_impl is None:
                compiled_impl = torch.compile(packed_impl, fullgraph=True, mode="reduce-overhead")
                self._laion_trainer_forward_packed_compiled_impl = compiled_impl
            self._laion_trainer_forward_packed_callable = compiled_impl
        else:
            self._laion_trainer_forward_packed_callable = packed_impl
        self._laion_trainer_forward_compiled = bool(compile_model)

    def _is_forward_compilation_enabled(self) -> bool:
        return bool(getattr(self, "_laion_trainer_forward_compiled", False))

    ZImageTransformer2DModel.set_forward_compilation = _set_forward_compilation
    ZImageTransformer2DModel.is_forward_compilation_enabled = _is_forward_compilation_enabled
    ZImageTransformer2DModel._laion_trainer_forward_block_stacks_patched = True
