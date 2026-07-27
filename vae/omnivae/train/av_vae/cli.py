"""CLI argument parser and config merger for AudioVideoVAE training."""

import argparse
import logging
import re
from typing import Any, Dict, List, Optional, Union

from .utils import _resolve_cfg_reference, _parse_positive_int_list


def _parse_segment_count(raw: str) -> Union[None, int, List[Optional[int]]]:
    """Parse segment_count from CLI string: single int, 'null', or comma-separated list."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        p = parts[0].lower()
        if p in ("null", "none", "0"):
            return None
        return int(p)
    result: List[Optional[int]] = []
    for p in parts:
        p = p.strip().lower()
        if p in ("null", "none", "0"):
            result.append(None)
        else:
            result.append(int(p))
    return result


def _parse_int_list(raw: str) -> Union[int, List[int]]:
    """Parse comma-separated int list or single int from CLI string."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        return int(parts[0])
    return [int(p) for p in parts]


def _parse_float_list(raw: str) -> Union[float, List[float]]:
    """Parse comma-separated float list or single float from CLI string."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        return float(parts[0])
    return [float(p) for p in parts]


_UNSUPPORTED_PUBLIC_OPTIONS = {
    "--use_llm_caption",
    "--no-use_llm_caption",
    "--eval_llm_caption",
    "--no-eval_llm_caption",
    "--llm_dtype",
    "--lr_llm_caption_head",
    "--use_image_video_alter",
    "--no-use_image_video_alter",
    "--image_data_mixture_yaml",
    "--image_dataset_path",
    "--image_batch_size",
    "--video_batch_size",
    "--image_video_weights",
    "--image_loader",
    "--image_relaion_root",
    "--image_relaion_slave_path",
    "--image_relaion_base_image_path",
    "--image_relaion_split",
    "--image_relaion_image_size",
    "--image_relaion_center_crop",
    "--no-image_relaion_center_crop",
    "--image_relaion_random_flip",
    "--no-image_relaion_random_flip",
    "--image_relaion_recaption_prob",
    "--image_relaion_cache_dir",
    "--image_relaion_max_samples",
    "--image_relaion_repeat",
}


def _hide_unsupported_public_options(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if any(opt in _UNSUPPORTED_PUBLIC_OPTIONS for opt in action.option_strings):
            action.help = argparse.SUPPRESS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio-Video VAE Joint Trainer")

    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--tag', type=str, default=None, help='Experiment tag (override auto-generated)')
    parser.add_argument('--debug', type=int, default=0, help='Enable debug mode (1=enable)')
    parser.add_argument('--debug_ip', type=str, default='localhost')
    parser.add_argument('--debug_port', type=int, default=32431)
    parser.add_argument('--continue_train', action='store_true', default=False)
    parser.add_argument('--valid_only', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for validation')

    # Loss overrides
    parser.add_argument('--use_video_recon', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use_audio_recon', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use_segment_contrastive', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use_global_contrastive', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use_llm_caption', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--freeze_vae_encoders', action=argparse.BooleanOptionalAction, default=None)

    # Group weights
    parser.add_argument('--lambda_group_video', type=float, default=None)
    parser.add_argument('--lambda_group_audio', type=float, default=None)
    parser.add_argument('--lambda_group_contrastive', type=float, default=None)
    parser.add_argument('--lambda_segment_contrastive', type=float, default=None)
    parser.add_argument('--lambda_global_contrastive', type=float, default=None)
    parser.add_argument('--lambda_video_kl', type=float, default=None)
    parser.add_argument('--lambda_video_lpips', type=float, default=None)
    parser.add_argument('--lambda_audio_kl', type=float, default=None)

    # Contrastive
    parser.add_argument('--spatial_pool_mode', type=str, default=None)
    parser.add_argument('--spatial_merge_factor', type=int, default=None)
    parser.add_argument('--segment_count', default=None)
    parser.add_argument('--num_negatives', default=None)
    parser.add_argument('--num_negative_videos', default=None)
    parser.add_argument('--same_long_video_priority', action=argparse.BooleanOptionalAction, default=None,
                        help='Segment-level contrastive: prefer negatives from other clips of the same long video.')
    parser.add_argument('--same_long_video_num_negatives', default=None,
                        help='K_seg: #sibling segment negatives per anchor (int or comma-list per granularity).')
    parser.add_argument('--num_negatives_with_sibling', default=None,
                        help='Override total #negatives for anchors that have siblings (int or comma-list).')
    parser.add_argument('--num_negatives_no_sibling', default=None,
                        help='Alternative to --num_negatives_with_sibling: specify only '
                             'the "far" (different-long-video) negative count N. The total '
                             'column count is derived as C = (S-1) + K_seg + N, where '
                             'K_seg = --same_long_video_num_negatives and S = segment_count. '
                             'Mutually exclusive with --num_negatives_with_sibling. '
                             'Accepts int or comma-list per granularity.')
    parser.add_argument('--segment_temporal_pool_mode', type=str, default=None)
    parser.add_argument('--global_temporal_pool_mode', type=str, default=None)
    parser.add_argument('--contrastive_transformer_layers', type=int, default=None)
    parser.add_argument('--transformer_nhead', type=int, default=None)
    parser.add_argument('--spatial_transformer_layers', type=int, default=None)
    parser.add_argument('--segment_transformer_layers', type=int, default=None)
    parser.add_argument('--global_transformer_layers', type=int, default=None)
    parser.add_argument('--contrastive_module_size', type=str, default=None)
    parser.add_argument('--spatial_module_size', type=str, default=None)
    parser.add_argument('--segment_module_size', type=str, default=None)
    parser.add_argument('--global_module_size', type=str, default=None)
    parser.add_argument('--cnn_num_blocks_per_stage', type=int, default=None)
    parser.add_argument('--cnn_kernel_size', type=int, default=None)
    parser.add_argument('--use_sdpa', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--contrastive_transformer_dim', type=int, default=None,
                        help='Internal d_model for contrastive temporal transformers (overrides config)')
    parser.add_argument('--contrastive_use_mean', action=argparse.BooleanOptionalAction, default=None)

    # Contrastive head type switch + intra_seg_xattn specific overrides
    parser.add_argument('--contrastive_type', type=str, default=None,
                        choices=['latent_seg', 'intra_seg_xattn'],
                        help='Contrastive head variant. latent_seg=existing '
                             'LatentAVContrastiveHead; intra_seg_xattn=intra-segment '
                             'self-attn + cross-attn (ALBEF-style) head.')
    parser.add_argument('--contrastive_embed_dim', type=int, default=None,
                        help='embed_dim for contrastive head (both variants); '
                             'must be divisible by 4 and by nhead when intra_seg_xattn.')
    parser.add_argument('--contrastive_nhead', type=int, default=None,
                        help='nhead for intra_seg_xattn head (must divide embed_dim).')
    parser.add_argument('--self_attn_layers', type=int, default=None,
                        help='Ls: number of self-attn layers per side (intra_seg_xattn).')
    parser.add_argument('--cross_attn_layers', type=int, default=None,
                        help='Lx: number of cross-attn layers per side (intra_seg_xattn).')
    parser.add_argument('--max_audio_tokens_per_seg', type=int, default=None,
                        help='Cap on audio tokens per segment for intra_seg_xattn '
                             '(determines 1D PE table size).')
    parser.add_argument('--max_spatial_h', type=int, default=None,
                        help='Max spatial H for 2D PE table (intra_seg_xattn); '
                             'set >= original video latent H.')
    parser.add_argument('--max_spatial_w', type=int, default=None,
                        help='Max spatial W for 2D PE table (intra_seg_xattn); '
                             'set >= original video latent W.')
    parser.add_argument('--contrastive_dim_feedforward', type=int, default=None,
                        help='FFN hidden dim inside intra_seg_xattn blocks; '
                             'null/default -> 4 * embed_dim.')
    parser.add_argument('--contrastive_dropout', type=float, default=None,
                        help='Dropout inside intra_seg_xattn blocks.')
    # NOTE: --qk_norm is defined elsewhere (training.qk_norm) and already
    # propagates to model.contrastive.qk_norm via trainer.py; no dedicated
    # intra_seg_xattn flag is needed.

    # Eval switches
    parser.add_argument('--eval_video_recon', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--eval_audio_recon', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--eval_contrastive', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--eval_contrastive_in_all', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--eval_llm_caption', action=argparse.BooleanOptionalAction, default=None)

    # Training
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--video_loss_reduction', type=str, default=None, choices=['sum', 'mean'])
    parser.add_argument('--video_learn_logvar', action=argparse.BooleanOptionalAction, default=None,
                        help='Whether video logvar is learnable (overrides config)')
    parser.add_argument('--video_logvar_init', type=float, default=None,
                        help='Initial value of video logvar (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None)

    # Image+Video alterstep mixed training (SSVAE-style)
    parser.add_argument(
        '--use_image_video_alter', action=argparse.BooleanOptionalAction, default=None,
        help='Enable image+video alterstep mixed training. When set, the '
             'dataloader produces both an image_batch (T=1) and a video_batch '
             '(T>1) per step, and the trainer randomly picks one modality per '
             'step using --image_video_weights.')
    parser.add_argument(
        '--image_data_mixture_yaml', type=str, default=None,
        help='Path to YAML mixture config for image data (same format as the '
             'video data_mixture_yaml). Only honoured when '
             '--use_image_video_alter is enabled.')
    parser.add_argument(
        '--image_dataset_path', type=str, default=None,
        help='Single jsonl path for image data (mutually exclusive with '
             '--image_data_mixture_yaml). Only honoured when '
             '--use_image_video_alter is enabled.')
    parser.add_argument(
        '--image_batch_size', type=int, default=None,
        help='Per-step image batch size when alterstep is enabled. Default '
             'falls back to --batch_size.')
    parser.add_argument(
        '--video_batch_size', type=int, default=None,
        help='Per-step video batch size when alterstep is enabled. Default '
             'falls back to --batch_size.')
    parser.add_argument(
        '--image_video_weights', type=str, default=None,
        help='Sampling weights "image,video" for the alter step (e.g. "1,7" '
             'means 1/8 image + 7/8 video). Default "1,1".')

    # Image-loader dispatch + Relaion-specific args (image+video alterstep)
    parser.add_argument(
        '--image_loader', type=str, default=None, choices=['jsonl', 'relaion'],
        help='Which image source to use when --use_image_video_alter is on. '
             '"jsonl" reads image_path from a jsonl mixture (default); '
             '"relaion" reads from master/slave jsonl + binary package.')
    parser.add_argument(
        '--image_relaion_root', type=str, default=None,
        help='Master jsonl root for the Relaion image dataset.')
    parser.add_argument(
        '--image_relaion_slave_path', type=str, default=None,
        help='Optional slave jsonl root for prompt fallback (Relaion).')
    parser.add_argument(
        '--image_relaion_base_image_path', type=str, default=None,
        help='Base directory holding the binary image packages (Relaion).')
    parser.add_argument(
        '--image_relaion_split', type=str, default=None,
        help='Split tag used for Relaion index cache (default "train").')
    parser.add_argument(
        '--image_relaion_image_size', type=int, default=None,
        help='Image side length for Relaion transforms; defaults to --resolution.')
    parser.add_argument(
        '--image_relaion_center_crop', action=argparse.BooleanOptionalAction, default=None,
        help='Use CenterCrop for Relaion (else RandomCrop).')
    parser.add_argument(
        '--image_relaion_random_flip', action=argparse.BooleanOptionalAction, default=None,
        help='Apply RandomHorizontalFlip on Relaion images.')
    parser.add_argument(
        '--image_relaion_recaption_prob', type=float, default=None,
        help='Probability of using recaption (else falls back to caption).')
    parser.add_argument(
        '--image_relaion_cache_dir', type=str, default=None,
        help='Directory used to store the per-root Relaion byte-offset index.')
    parser.add_argument(
        '--image_relaion_max_samples', type=int, default=None,
        help='Cap the number of Relaion samples (debug / smoke test).')
    parser.add_argument(
        '--image_relaion_repeat', type=int, default=None,
        help='Times to repeat the Relaion epoch (rarely needed; default 1).')

    parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--pretrained_checkpoint', type=str, default=None)
    parser.add_argument('--keep_audio_vae_pretrained', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pretrained_video_checkpoint', type=str, default=None,
                        help='OmniVAE checkpoint path; only video_vae.* weights will be loaded')
    parser.add_argument('--video_model_name', type=str, default=None,
                        choices=['WanVAE', 'WanVAE22'],
                        help='Override model.video.model_name (video VAE backbone class)')
    parser.add_argument('--video_model_config', type=str, default=None,
                        help='Override model.video.model_config (path to backbone JSON config)')
    parser.add_argument('--pretrained_video_model_path', type=str, default=None,
                        help='Override model.video.pretrained_model_name_or_path '
                             '(standalone VAE ckpt dir; distinct from --pretrained_video_checkpoint '
                             'which loads an OmniVAE training checkpoint)')
    parser.add_argument('--pretrained_audio_checkpoint', type=str, default=None,
                        help='OmniVAE checkpoint path; only audio_vae.* weights will be loaded')
    parser.add_argument('--pretrained_contrastive_checkpoint', type=str, default=None,
                        help='OmniVAE checkpoint path; only contrastive_head.* weights will be loaded')
    parser.add_argument('--pretrained_disc_checkpoint', type=str, default=None,
                        help='OmniVAE checkpoint path; loads only disc weights from '
                             "ckpt['discriminators']. Independent of generator pretrained "
                             'flags. Useful for warm-starting disc from a different stage '
                             '(e.g. gen from distill-only stage1, disc from GAN stage2).')
    parser.add_argument('--pretrained_disc_load_optim', action=argparse.BooleanOptionalAction, default=False,
                        help='Together with --pretrained_disc_checkpoint, also restore '
                             'optim_d / scheduler_d / scaler_d. Default False, meaning a '
                             'fresh disc warmup with the new --lr_disc.')
    parser.add_argument('--global_contrastive_start_steps', type=int, default=None)
    parser.add_argument('--video_distill_start_step', type=int, default=None,
                        help='Start step for video semantic distillation loss (default 0).')
    parser.add_argument('--audio_distill_start_step', type=int, default=None,
                        help='Start step for audio semantic distillation loss (default 0).')
    parser.add_argument('--segment_avclip_start_steps', default=None)
    parser.add_argument('--segment_count_weights', default=None)
    parser.add_argument('--spatial_transform_mode', type=str, default=None)
    parser.add_argument('--spatial_roundtrip_short_edge', type=int, default=None,
                        help='Optional bilinear round-trip low-pass before the normal spatial '
                             'transform. If set (e.g. 224), the video is first Resize-d to this '
                             'short edge (antialias=True) and then fed into the regular '
                             'resize+crop pipeline. Acts as an implicit regularizer that attenuates '
                             'high-frequency H.264 encoding fingerprints. Set to None (default) or '
                             '0 / negative to disable.')
    parser.add_argument('--train_metadata_path', type=str, default=None,
                        help='Override data.train.metadata_paths.train_main with a single dataset path. '
                             'Other metadata_paths/metadata_weights entries (if any) are cleared.')
    parser.add_argument('--grad_log_steps', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Number of micro-steps per optimizer update (default from config, typically 1)')
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Max gradient norm for clipping (overrides config)')
    parser.add_argument('--qk_norm', action=argparse.BooleanOptionalAction, default=None,
                        help='Apply RMSNorm to Q/K in all attention layers')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Linear warmup steps before cosine decay (default: 5000)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum optimizer steps to run (overrides training.max_steps)')
    parser.add_argument('--eval_steps', type=int, default=None,
                        help='Run validation every N steps (overrides training.eval_steps)')
    parser.add_argument('--save_steps', type=int, default=None,
                        help='Save checkpoint every N steps (overrides training.save_steps)')
    parser.add_argument('--reset_scheduler_on_resume',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='When loading a checkpoint via --continue_train, '
                             'rebuild optimizer LR and scheduler base_lrs from '
                             'the current CLI/config values instead of the '
                             'values stored in the checkpoint. Optimizer '
                             'momentum / Adam moments are still preserved. '
                             'Useful when changing LR or scheduler shape on '
                             'resume.')

    # Video loss clamp
    parser.add_argument('--video_loss_clamp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--video_recon_clamp_max', type=float, default=None)
    parser.add_argument('--video_lpips_clamp_max', type=float, default=None)
    parser.add_argument('--video_kl_clamp_max', type=float, default=None)

    # Adaptive loss balance
    parser.add_argument('--adaptive_loss_balance', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--adaptive_balance_audio_ratio', type=float, default=None)
    parser.add_argument('--adaptive_balance_contrastive_ratio', type=float, default=None)
    parser.add_argument('--adaptive_loss_balance_by_uncertainty', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--uncertainty_warmup_steps', type=int, default=None)
    parser.add_argument('--adaptive_loss_balance_by_gradient', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--gradient_balance_video_ratio', type=float, default=None)
    parser.add_argument('--gradient_balance_audio_ratio', type=float, default=None)
    parser.add_argument('--gradient_balance_clamp_max', type=float, default=None)
    parser.add_argument('--gradient_balance_interval', type=int, default=None)

    # Adaptive loss balance v2 (EMA + configurable anchor)
    parser.add_argument('--adaptive_loss_balance_v2', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable EMA-anchored group balance (video_vae/audio_vae/contrastive)')
    parser.add_argument('--adaptive_anchor_source', type=str, default=None,
                        choices=['video_vae', 'audio_vae', 'contrastive'],
                        help='Which group total to use as the adaptive anchor')
    parser.add_argument('--adaptive_anchor_ema_decay', type=float, default=None)
    parser.add_argument('--adaptive_anchor_warmup_steps', type=int, default=None)
    parser.add_argument('--adaptive_scale_clamp_min', type=float, default=None)
    parser.add_argument('--adaptive_scale_clamp_max', type=float, default=None)
    parser.add_argument('--adaptive_ratio_video', type=float, default=None)
    parser.add_argument('--adaptive_ratio_audio', type=float, default=None)
    parser.add_argument('--adaptive_ratio_contrastive', type=float, default=None)
    # Stage1 (phase-freeze) overrides; None = fall back to stage2 values above.
    parser.add_argument('--adaptive_anchor_source_stage1', type=str, default=None,
                        choices=['video_vae', 'audio_vae', 'contrastive'],
                        help='Anchor source during phase-freeze stage1 (requires freeze_video_vae)')
    parser.add_argument('--adaptive_ratio_video_stage1', type=float, default=None,
                        help='Stage1 override for adaptive_ratio_video')
    parser.add_argument('--adaptive_ratio_audio_stage1', type=float, default=None,
                        help='Stage1 override for adaptive_ratio_audio')
    parser.add_argument('--adaptive_ratio_contrastive_stage1', type=float, default=None,
                        help='Stage1 override for adaptive_ratio_contrastive')
    # Adaptive v2 stage2 gradient-balance hybrid (video VAE unfrozen -> switch
    # from EMA-anchor to gradient-norm based balancing, with optional linear
    # blend over the first N updates after unfreeze).
    parser.add_argument('--adaptive_v2_stage2_use_gradient',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='After video VAE unfreeze, switch adaptive v2 to '
                             'gradient-based balance (reuses _compute_gradient_balance_weights)')
    parser.add_argument('--adaptive_v2_stage2_blend_steps', type=int, default=None,
                        help='Linear blend window (in optimizer updates) from '
                             'anchor-scale -> gradient-scale right after unfreeze '
                             '(0 = hard switch)')
    parser.add_argument('--gradient_ratio_video_stage2', type=float, default=None,
                        help='Stage2 override for gradient_balance_video_ratio '
                             '(falls back to gradient_balance_video_ratio if unset)')
    parser.add_argument('--gradient_ratio_audio_stage2', type=float, default=None,
                        help='Stage2 override for gradient_balance_audio_ratio '
                             '(falls back to gradient_balance_audio_ratio if unset)')

    # Phase freezing of the video VAE (encoder + decoder)
    parser.add_argument('--freeze_video_vae', action=argparse.BooleanOptionalAction, default=None,
                        help='Freeze video VAE (encoder+decoder) at training start')
    parser.add_argument('--freeze_video_vae_until_step', type=int, default=None,
                        help='Step at which to unfreeze the video VAE (0 = never)')

    # Phase freezing of the audio VAE (encoder + decoder) — symmetric to video.
    parser.add_argument('--freeze_audio_vae', action=argparse.BooleanOptionalAction, default=None,
                        help='Freeze audio VAE (encoder+decoder) at training start')
    parser.add_argument('--freeze_audio_vae_until_step', type=int, default=None,
                        help='Step at which to unfreeze the audio VAE (0 = never)')

    # Encoder-only freeze for the audio VAE (encoder + quant_conv). Decoder
    # side (post_quant_conv + decoder) stays trainable. Useful for decoder /
    # GAN-only finetune on top of an already-trained encoder. Mutually
    # exclusive with --freeze_audio_vae (which freezes everything).
    parser.add_argument('--freeze_audio_encoder', action=argparse.BooleanOptionalAction, default=None,
                        help='Freeze only the audio encoder (encoder + quant_conv); '
                             'post_quant_conv + decoder remain trainable. '
                             'Mutually exclusive with --freeze_audio_vae.')

    # Encoder-only freeze for the video VAE (encoder + conv1). Decoder side
    # (conv2 + decoder) stays trainable. Symmetric to --freeze_audio_encoder.
    # Useful for GAN-only video decoder finetune on top of a pretrained
    # (e.g. distillation-stage) encoder. Mutually exclusive with
    # --freeze_video_vae (which freezes everything).
    parser.add_argument('--freeze_video_encoder', action=argparse.BooleanOptionalAction, default=None,
                        help='Freeze only the video encoder (encoder + conv1); '
                             'conv2 + decoder remain trainable. '
                             'Mutually exclusive with --freeze_video_vae.')

    # Contrastive gradient scaling (limit backprop into encoders from contrastive head)
    parser.add_argument('--contrastive_grad_scale_video', type=float, default=None,
                        help='Scale on gradient flowing from contrastive head into video encoder')
    parser.add_argument('--contrastive_grad_scale_audio', type=float, default=None,
                        help='Scale on gradient flowing from contrastive head into audio encoder')

    # Per-module learning rates (None = fall back to global --lr)
    parser.add_argument('--lr_video_vae', type=float, default=None)
    parser.add_argument('--lr_audio_vae', type=float, default=None)
    parser.add_argument('--lr_contrastive_head', type=float, default=None)
    parser.add_argument('--lr_llm_caption_head', type=float, default=None)
    parser.add_argument('--lr_distill_proj', type=float, default=None)
    parser.add_argument('--lr_video_logvar', type=float, default=None)

    # Dedicated warmup+cosine schedule for the `video_vae` param group.
    # If any of the four is provided, the video_vae group gets an independent
    # schedule; otherwise it follows the global warmup+cosine.
    parser.add_argument('--lr_video_vae_warmup_steps', type=int, default=None,
                        help='Warmup steps for the video_vae param group (default: same as --warmup_steps).')
    parser.add_argument('--lr_video_vae_total_steps', type=int, default=None,
                        help='Cosine-decay end step for the video_vae group '
                             '(default: max_steps - freeze_video_vae_until_step).')
    parser.add_argument('--lr_video_vae_start_step', type=int, default=None,
                        help='Global step at which the video_vae clock starts ticking '
                             '(default: freeze_video_vae_until_step; 0 if not freezing).')
    parser.add_argument('--lr_video_vae_min_ratio', type=float, default=None,
                        help='Cosine lower-bound ratio for the video_vae group (default: 0.0).')

    # Dedicated warmup+cosine schedule for the `audio_vae` param group
    # (symmetric to video_vae). If any of the four is provided, the audio_vae
    # group gets an independent schedule; otherwise it follows the global
    # warmup+cosine.
    parser.add_argument('--lr_audio_vae_warmup_steps', type=int, default=None,
                        help='Warmup steps for the audio_vae param group (default: same as --warmup_steps).')
    parser.add_argument('--lr_audio_vae_total_steps', type=int, default=None,
                        help='Cosine-decay end step for the audio_vae group '
                             '(default: max_steps - freeze_audio_vae_until_step).')
    parser.add_argument('--lr_audio_vae_start_step', type=int, default=None,
                        help='Global step at which the audio_vae clock starts ticking '
                             '(default: freeze_audio_vae_until_step; 0 if not freezing).')
    parser.add_argument('--lr_audio_vae_min_ratio', type=float, default=None,
                        help='Cosine lower-bound ratio for the audio_vae group (default: 0.0).')

    # Dtype
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--video_vae_dtype', type=str, default=None)
    parser.add_argument('--audio_vae_dtype', type=str, default=None)
    parser.add_argument('--contrastive_dtype', type=str, default=None)
    parser.add_argument('--llm_dtype', type=str, default=None)

    # Audio discriminator (LSGAN) training knobs
    parser.add_argument('--use_audio_disc', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable audio discriminator (MPD/MSD/MSSTFTD) GAN training')
    parser.add_argument('--audio_disc_start_step', type=int, default=None,
                        help='Step at which disc update + generator adversarial loss kick in')
    parser.add_argument('--lambda_audio_adv', type=float, default=None,
                        help='Weight on generator adversarial loss (audio)')
    parser.add_argument('--lambda_audio_feature_matching', type=float, default=None,
                        help='Weight on generator feature matching loss (audio)')
    parser.add_argument('--lr_disc', type=float, default=None,
                        help='Learning rate for audio discriminator optimizer')
    parser.add_argument('--disc_max_grad_norm', type=float, default=None,
                        help='Grad clip for discriminator (null/unset = same as max_grad_norm)')
    parser.add_argument('--disc_dtype', type=str, default=None,
                        help='dtype for discriminator autocast (default fp32)')

    # Video discriminator (CausalVAE-style 3D PatchGAN, alternating G/D update)
    parser.add_argument('--use_video_disc', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable video discriminator with alternating G/D training')
    parser.add_argument('--video_disc_start_step', type=int, default=None,
                        help='Step at which video disc + generator adv loss kick in')
    parser.add_argument('--lambda_video_adv', type=float, default=None,
                        help='Static weight on generator adversarial loss (video); '
                             'ignored when --video_disc_adaptive_weight is set')
    parser.add_argument('--video_disc_loss_type', type=str, choices=['hinge', 'vanilla'], default=None,
                        help='Discriminator loss formulation for the video disc')
    parser.add_argument('--video_disc_adaptive_weight', action=argparse.BooleanOptionalAction, default=None,
                        help='Use VQGAN-style adaptive weight ||∇L_rec|| / ||∇L_g|| on '
                             'the decoder last layer instead of the static lambda')
    parser.add_argument('--video_disc_adaptive_weight_max', type=float, default=None,
                        help='Upper clamp for adaptive d_weight. Default 1.0. '
                             'Lower this (e.g. 0.1) when G is pretrained and D is fresh '
                             'to prevent adversarial runaway; raise (e.g. 1e4) for '
                             'legacy VQGAN behavior.')
    parser.add_argument('--video_disc_lazy_threshold', type=float, default=None,
                        help='If d_loss (averaged across ranks) falls below this threshold '
                             'at an accumulation boundary, skip the D optimizer update for '
                             'that boundary. Prevents D from becoming arbitrarily sharp when '
                             'G is pretrained. 0 disables. Typical 0.2~0.4 for hinge.')
    parser.add_argument('--distill_every_steps', action=argparse.BooleanOptionalAction, default=None,
                        help='Also run distill loss on D-only steps and accumulate its '
                             'grad into G params for the next G update (≈ 2x effective '
                             'batch size for the distill signal)')

    # Validation
    parser.add_argument('--val_segment_num_negatives', type=str, default=None)
    parser.add_argument('--val_segment_num_negative_videos', type=int, default=None)
    parser.add_argument('--val_global_num_negatives', type=str, default=None)
    parser.add_argument('--val_contrastive_max_samples', type=int, default=None,
                        help='Limit number of samples used for contrastive validation '
                             '(applies to both per-dataset and merged eval; per-rank count '
                             'is ceil(max_samples / world_size)).')

    # Experiment name suffix
    parser.add_argument('--exp_name_suffix', type=str, default=None)
    # Full experiment name override. When set, the experiment directory is
    # named exactly this (under output.exp_root); the auto-generated detail
    # suffix and exp_name_suffix are skipped entirely.
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Override the entire experiment tag. When set, '
                             'exp_dir = <output.exp_root>/<exp_name> and the '
                             'auto-detail suffix is bypassed.')

    # Semantic distillation
    parser.add_argument('--use_semantic_distill', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable semantic distillation loss (overrides config)')
    parser.add_argument('--semantic_model_path', type=str, default=None,
                        help='Path to Qwen3-Omni model for local semantic encoder')
    parser.add_argument('--semantic_api_url', type=str, default=None,
                        help='URL of the semantic feature extraction API (legacy)')
    parser.add_argument('--encoder_fps', type=float, default=None,
                        help='FPS for semantic encoder video sampling (default: 4)')
    parser.add_argument('--encoder_resolution', type=int, default=None,
                        help='Resolution for semantic encoder video input (default: 128)')
    parser.add_argument('--distill_vision_layer', type=int, default=None,
                        help='1-indexed vision encoder layer to distill from (Mode A only). '
                             'Vision has 27 layers; default (YAML) is 18. '
                             'Set to null/omit in YAML to use the final layer.')
    parser.add_argument('--distill_audio_layer', type=int, default=None,
                        help='1-indexed audio encoder layer to distill from (Mode A only). '
                             'Audio has 32 layers; default (YAML) is 24. '
                             'Set to null/omit in YAML to use the final layer.')
    parser.add_argument('--lambda_distill_image_cosine', type=float, default=None)
    parser.add_argument('--lambda_distill_image_distance', type=float, default=None)
    parser.add_argument('--lambda_distill_video_cosine', type=float, default=None)
    parser.add_argument('--lambda_distill_video_distance', type=float, default=None)
    parser.add_argument('--lambda_distill_audio_t_axis', type=float, default=None)
    parser.add_argument('--lambda_distill_audio_d_axis', type=float, default=None)
    parser.add_argument('--lambda_group_distill', type=float, default=None)
    parser.add_argument('--distill_margin_cosine', type=float, default=None)
    parser.add_argument('--distill_margin_distance', type=float, default=None)
    parser.add_argument('--distill_w_hyper', type=float, default=None)
    parser.add_argument('--distill_audio_type', type=str, default=None, choices=['d_axis', 't_axis'])
    # iREPA-style distillation options
    parser.add_argument('--distill_proj_type', type=str, default=None, choices=['conv', 'linear'],
                        help='Projector type: conv (iREPA) or linear (REPA)')
    parser.add_argument('--distill_proj_layers', type=int, default=None,
                        help='Number of conv layers in projector (1=single-layer legacy, 2+=multi-layer)')
    parser.add_argument('--distill_proj_hidden_dim', type=int, default=None,
                        help='Hidden dim for multi-layer projector (null=auto geometric mean)')
    parser.add_argument('--distill_use_conv3d', action=argparse.BooleanOptionalAction, default=None,
                        help='Use Conv3d instead of Conv2d in conv projector mode')
    parser.add_argument('--distill_proj_before_agg', action=argparse.BooleanOptionalAction, default=None,
                        help='Conv projection before temporal aggregation (True=new, False=legacy order)')
    parser.add_argument('--distill_dim_schedule', type=str, default=None,
                        choices=['fixed', 'doubling'],
                        help='Projector dimension schedule: fixed=constant hidden dim, doubling=double each layer (auto layers)')
    parser.add_argument('--distill_use_sampled', action=argparse.BooleanOptionalAction, default=None,
                        help='Use sampled latent for distillation (default: use mean)')
    parser.add_argument('--distill_spatial_norm', action=argparse.BooleanOptionalAction, default=None,
                        help='Apply iREPA spatial normalization on teacher features')
    parser.add_argument('--distill_spatial_norm_gamma', type=float, default=None,
                        help='Gamma for spatial normalization (default: 0.7)')
    parser.add_argument('--distill_use_dist_matrix', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable Marginal Distance Matrix Loss')
    parser.add_argument('--adaptive_distill_balance', action=argparse.BooleanOptionalAction, default=None,
                        help='Align distill losses with recon losses per modality')
    parser.add_argument('--adaptive_distill_use_gradient', action=argparse.BooleanOptionalAction, default=None,
                        help='Switch distill balance from loss-value ratio to gradient-norm ratio '
                             '(requires --adaptive_distill_balance; default False).')
    parser.add_argument('--adaptive_distill_video_ratio', type=float, default=None,
                        help='Target ratio of video distill to video recon loss')
    parser.add_argument('--adaptive_distill_audio_ratio', type=float, default=None,
                        help='Target ratio of audio distill to audio recon loss')

    # Mode C: remote upload distillation (cross-server, no shared storage)
    parser.add_argument('--distill_upload_mode', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable upload mode: send data to encoder service instead of file paths')
    parser.add_argument('--distill_video_gpu_map', type=str, default=None,
                        help='rank:gpu_id mapping, e.g. "0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3"')
    parser.add_argument('--distill_image_gpu_id', type=int, default=None,
                        help='GPU id on encoder server for image extraction')
    parser.add_argument('--distill_audio_gpu_id', type=int, default=None,
                        help='GPU id on encoder server for audio extraction')
    parser.add_argument('--distill_num_upload_workers', type=int, default=None,
                        help='Number of parallel upload threads per rank (default: 6)')
    parser.add_argument('--distill_processor_path', type=str, default=None,
                        help='Path to Qwen processor for client-side tokenization (enables tensor upload mode)')

    _hide_unsupported_public_options(parser)
    return parser


def merge_cli_to_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """将 CLI 参数合并到 config 字典，就地修改 cfg。"""
    loss_cfg = cfg.setdefault('loss', {})

    if getattr(args, 'use_llm_caption', None):
        raise NotImplementedError(
            "LLM caption training is not part of the public OmniVAE training boundary."
        )
    if getattr(args, 'eval_llm_caption', None):
        raise NotImplementedError(
            "LLM caption evaluation is not part of the public OmniVAE training boundary."
        )
    if getattr(args, 'use_image_video_alter', None):
        raise NotImplementedError(
            "Image+video alterstep training is not part of the public OmniVAE training boundary."
        )

    _bool_loss_overrides = [
        'use_video_recon', 'use_audio_recon', 'use_segment_contrastive',
        'use_global_contrastive', 'use_llm_caption', 'freeze_vae_encoders',
    ]
    for key in _bool_loss_overrides:
        val = getattr(args, key, None)
        if val is not None:
            loss_cfg[key] = val

    _float_loss_overrides = [
        'lambda_group_video', 'lambda_group_audio', 'lambda_group_contrastive',
        'lambda_segment_contrastive', 'lambda_global_contrastive',
        'lambda_video_kl', 'lambda_video_lpips', 'lambda_audio_kl',
        'adaptive_balance_audio_ratio', 'adaptive_balance_contrastive_ratio',
        'gradient_balance_video_ratio', 'gradient_balance_audio_ratio',
        'gradient_balance_clamp_max',
    ]
    for key in _float_loss_overrides:
        val = getattr(args, key, None)
        if val is not None:
            loss_cfg[key] = val

    _int_loss_overrides = [
        'global_contrastive_start_steps', 'gradient_balance_interval',
        'uncertainty_warmup_steps',
        'video_distill_start_step', 'audio_distill_start_step',
    ]
    for key in _int_loss_overrides:
        val = getattr(args, key, None)
        if val is not None:
            loss_cfg[key] = val

    if args.video_loss_reduction is not None:
        loss_cfg['video_loss_reduction'] = args.video_loss_reduction
    if args.video_learn_logvar is not None:
        loss_cfg['video_learn_logvar'] = args.video_learn_logvar
    if args.video_logvar_init is not None:
        loss_cfg['video_logvar_init'] = args.video_logvar_init

    # Video loss clamp
    if getattr(args, 'video_loss_clamp', None) is not None:
        loss_cfg['video_loss_clamp'] = args.video_loss_clamp
    for _clamp_key in ('video_recon_clamp_max', 'video_lpips_clamp_max', 'video_kl_clamp_max'):
        _clamp_val = getattr(args, _clamp_key, None)
        if _clamp_val is not None:
            loss_cfg[_clamp_key] = _clamp_val

    _bool_loss_balance = [
        'adaptive_loss_balance', 'adaptive_loss_balance_by_uncertainty',
        'adaptive_loss_balance_by_gradient', 'adaptive_loss_balance_v2',
    ]
    for key in _bool_loss_balance:
        val = getattr(args, key, None)
        if val is not None:
            loss_cfg[key] = val

    _adaptive_v2_passthrough = [
        ('adaptive_anchor_source', str),
        ('adaptive_anchor_ema_decay', float),
        ('adaptive_anchor_warmup_steps', int),
        ('adaptive_scale_clamp_min', float),
        ('adaptive_scale_clamp_max', float),
        ('adaptive_ratio_video', float),
        ('adaptive_ratio_audio', float),
        ('adaptive_ratio_contrastive', float),
        ('adaptive_anchor_source_stage1', str),
        ('adaptive_ratio_video_stage1', float),
        ('adaptive_ratio_audio_stage1', float),
        ('adaptive_ratio_contrastive_stage1', float),
        ('adaptive_v2_stage2_blend_steps', int),
        ('gradient_ratio_video_stage2', float),
        ('gradient_ratio_audio_stage2', float),
    ]
    for key, _caster in _adaptive_v2_passthrough:
        val = getattr(args, key, None)
        if val is not None:
            loss_cfg[key] = _caster(val)
    if getattr(args, 'adaptive_v2_stage2_use_gradient', None) is not None:
        loss_cfg['adaptive_v2_stage2_use_gradient'] = args.adaptive_v2_stage2_use_gradient

    # Phase freezing of video VAE
    if getattr(args, 'freeze_video_vae', None) is not None:
        loss_cfg['freeze_video_vae'] = args.freeze_video_vae
    if getattr(args, 'freeze_video_vae_until_step', None) is not None:
        loss_cfg['freeze_video_vae_until_step'] = int(args.freeze_video_vae_until_step)

    # Phase freezing of audio VAE (symmetric to video)
    if getattr(args, 'freeze_audio_vae', None) is not None:
        loss_cfg['freeze_audio_vae'] = args.freeze_audio_vae
    if getattr(args, 'freeze_audio_vae_until_step', None) is not None:
        loss_cfg['freeze_audio_vae_until_step'] = int(args.freeze_audio_vae_until_step)

    # Encoder-only freeze for the audio VAE
    if getattr(args, 'freeze_audio_encoder', None) is not None:
        loss_cfg['freeze_audio_encoder'] = args.freeze_audio_encoder

    # Encoder-only freeze for the video VAE (symmetric to audio)
    if getattr(args, 'freeze_video_encoder', None) is not None:
        loss_cfg['freeze_video_encoder'] = args.freeze_video_encoder

    if args.segment_avclip_start_steps is not None:
        loss_cfg['segment_avclip_start_steps'] = _parse_int_list(args.segment_avclip_start_steps)
    if args.segment_count_weights is not None:
        loss_cfg['segment_count_weights'] = _parse_float_list(args.segment_count_weights)

    # Video backbone overrides (model.video.*)
    video_cfg = cfg.setdefault('model', {}).setdefault('video', {})
    if getattr(args, 'video_model_name', None) is not None:
        video_cfg['model_name'] = args.video_model_name
        video_cfg['_cli_override'] = True
    if getattr(args, 'video_model_config', None) is not None:
        video_cfg['model_config'] = args.video_model_config
        video_cfg['_cli_override'] = True
    if getattr(args, 'pretrained_video_model_path', None) is not None:
        video_cfg['pretrained_model_name_or_path'] = args.pretrained_video_model_path
        video_cfg['_cli_override'] = True

    # Contrastive overrides
    contrastive_cfg = cfg.setdefault('model', {}).setdefault('contrastive', {})
    _contrastive_str = ['spatial_pool_mode', 'segment_temporal_pool_mode', 'global_temporal_pool_mode']
    _contrastive_int = [
        'spatial_merge_factor', 'transformer_nhead',
        'spatial_transformer_layers', 'segment_transformer_layers', 'global_transformer_layers',
        'cnn_num_blocks_per_stage', 'cnn_kernel_size',
    ]
    for key in _contrastive_str:
        val = getattr(args, key, None)
        if val is not None:
            contrastive_cfg[key] = val
    for key in _contrastive_int:
        val = getattr(args, key, None)
        if val is not None:
            contrastive_cfg[key] = val
    if args.contrastive_transformer_layers is not None:
        contrastive_cfg['transformer_layers'] = args.contrastive_transformer_layers
    if args.contrastive_transformer_dim is not None:
        contrastive_cfg['transformer_dim'] = args.contrastive_transformer_dim

    _global_size = args.contrastive_module_size
    for _key, _cli_val in [
        ('spatial_module_size', args.spatial_module_size),
        ('segment_module_size', args.segment_module_size),
        ('global_module_size', args.global_module_size),
    ]:
        val = _cli_val if _cli_val is not None else _global_size
        if val is not None:
            contrastive_cfg[_key] = val
    if args.use_sdpa is not None:
        contrastive_cfg['use_sdpa'] = args.use_sdpa
    if args.contrastive_use_mean is not None:
        contrastive_cfg['contrastive_use_mean'] = args.contrastive_use_mean

    # Contrastive head type + intra_seg_xattn overrides
    if getattr(args, 'contrastive_type', None) is not None:
        contrastive_cfg['contrastive_type'] = args.contrastive_type
    if getattr(args, 'contrastive_embed_dim', None) is not None:
        contrastive_cfg['embed_dim'] = int(args.contrastive_embed_dim)
    _intra_int_map = {
        'contrastive_nhead': 'nhead',
        'self_attn_layers': 'self_attn_layers',
        'cross_attn_layers': 'cross_attn_layers',
        'max_audio_tokens_per_seg': 'max_audio_tokens_per_seg',
        'max_spatial_h': 'max_spatial_h',
        'max_spatial_w': 'max_spatial_w',
        'contrastive_dim_feedforward': 'dim_feedforward',
    }
    for _arg_name, _cfg_key in _intra_int_map.items():
        val = getattr(args, _arg_name, None)
        if val is not None:
            contrastive_cfg[_cfg_key] = int(val)
    if getattr(args, 'contrastive_dropout', None) is not None:
        contrastive_cfg['dropout'] = float(args.contrastive_dropout)
    # args.qk_norm (defined below as a generic switch) is already routed into
    # training_cfg['qk_norm'] and then into contrastive_cfg['qk_norm'] by the
    # trainer, so we do NOT write it here to avoid double-handling.
    if getattr(args, 'contrastive_grad_scale_video', None) is not None:
        contrastive_cfg['contrastive_grad_scale_video'] = float(args.contrastive_grad_scale_video)
    if getattr(args, 'contrastive_grad_scale_audio', None) is not None:
        contrastive_cfg['contrastive_grad_scale_audio'] = float(args.contrastive_grad_scale_audio)
    if args.segment_count is not None:
        contrastive_cfg['segment_count'] = _parse_segment_count(args.segment_count)
    if args.num_negatives is not None:
        contrastive_cfg['num_negatives'] = _parse_int_list(args.num_negatives)
    if args.num_negative_videos is not None:
        contrastive_cfg['num_negative_videos'] = _parse_int_list(args.num_negative_videos)
    if args.same_long_video_priority is not None:
        contrastive_cfg['same_long_video_priority'] = bool(args.same_long_video_priority)
    if args.same_long_video_num_negatives is not None:
        contrastive_cfg['same_long_video_num_negatives'] = _parse_int_list(
            args.same_long_video_num_negatives
        )
    if args.num_negatives_with_sibling is not None:
        contrastive_cfg['num_negatives_with_sibling'] = _parse_int_list(
            args.num_negatives_with_sibling
        )
    if getattr(args, 'num_negatives_no_sibling', None) is not None:
        contrastive_cfg['num_negatives_no_sibling'] = _parse_int_list(
            args.num_negatives_no_sibling
        )
    for _eval_key in ('eval_video_recon', 'eval_audio_recon', 'eval_contrastive',
                      'eval_contrastive_in_all', 'eval_llm_caption'):
        _eval_val = getattr(args, _eval_key, None)
        if _eval_val is not None:
            loss_cfg[_eval_key] = _eval_val

    # Sync loss switches → model enabled flags
    contrastive_cfg['use_segment_loss'] = loss_cfg.get('use_segment_contrastive', True)
    contrastive_cfg['use_global_loss'] = loss_cfg.get('use_global_contrastive', True)
    contrastive_cfg['enabled'] = (
        loss_cfg.get('use_segment_contrastive', True)
        or loss_cfg.get('use_global_contrastive', True)
    )

    if loss_cfg.get('use_llm_caption', False):
        llm_cfg = cfg.setdefault('model', {}).setdefault('llm', {})
        llm_cfg['enabled'] = True

    # Training overrides
    training_cfg = cfg.setdefault('training', {})
    if args.lr is not None:
        training_cfg['lr'] = args.lr
    for _lr_key in ('lr_video_vae', 'lr_audio_vae', 'lr_contrastive_head',
                    'lr_llm_caption_head', 'lr_distill_proj', 'lr_video_logvar'):
        _lr_val = getattr(args, _lr_key, None)
        if _lr_val is not None:
            training_cfg[_lr_key] = float(_lr_val)
    if args.batch_size is not None:
        training_cfg['batch_size'] = args.batch_size
    if args.use_ema is not None:
        training_cfg['use_ema'] = args.use_ema
    if args.grad_log_steps is not None:
        training_cfg['grad_log_steps'] = args.grad_log_steps
    if args.gradient_checkpointing is not None:
        training_cfg['gradient_checkpointing'] = args.gradient_checkpointing
    if args.max_grad_norm is not None:
        training_cfg['max_grad_norm'] = args.max_grad_norm
    if args.qk_norm is not None:
        training_cfg['qk_norm'] = args.qk_norm
    if args.warmup_steps is not None:
        training_cfg['warmup_steps'] = args.warmup_steps
    if getattr(args, 'max_steps', None) is not None:
        training_cfg['max_steps'] = args.max_steps
    if getattr(args, 'eval_steps', None) is not None:
        training_cfg['eval_steps'] = args.eval_steps
    if getattr(args, 'save_steps', None) is not None:
        training_cfg['save_steps'] = args.save_steps
    if getattr(args, 'gradient_accumulation_steps', None) is not None:
        training_cfg['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if getattr(args, 'reset_scheduler_on_resume', None) is not None:
        training_cfg['reset_scheduler_on_resume'] = bool(args.reset_scheduler_on_resume)
    for _vv_key in ('lr_video_vae_warmup_steps', 'lr_video_vae_total_steps',
                    'lr_video_vae_start_step', 'lr_video_vae_min_ratio'):
        _vv_val = getattr(args, _vv_key, None)
        if _vv_val is not None:
            training_cfg[_vv_key] = _vv_val
    for _aa_key in ('lr_audio_vae_warmup_steps', 'lr_audio_vae_total_steps',
                    'lr_audio_vae_start_step', 'lr_audio_vae_min_ratio'):
        _aa_val = getattr(args, _aa_key, None)
        if _aa_val is not None:
            training_cfg[_aa_key] = _aa_val

    # Dtype
    for dtype_key in ['dtype', 'video_vae_dtype', 'audio_vae_dtype', 'contrastive_dtype', 'llm_dtype']:
        val = getattr(args, dtype_key, None)
        if val is not None:
            training_cfg[dtype_key] = val

    # Audio discriminator (LSGAN) CLI pass-through.
    # `use_audio_disc / audio_disc_start_step / lambda_audio_adv /
    #  lambda_audio_feature_matching` live under `loss.*`; `lr_disc /
    #  disc_max_grad_norm / disc_dtype` live under `training.*`.
    if getattr(args, 'use_audio_disc', None) is not None:
        loss_cfg['use_audio_disc'] = bool(args.use_audio_disc)
    if getattr(args, 'audio_disc_start_step', None) is not None:
        loss_cfg['audio_disc_start_step'] = int(args.audio_disc_start_step)
    if getattr(args, 'lambda_audio_adv', None) is not None:
        loss_cfg['lambda_audio_adv'] = float(args.lambda_audio_adv)
    if getattr(args, 'lambda_audio_feature_matching', None) is not None:
        loss_cfg['lambda_audio_feature_matching'] = float(args.lambda_audio_feature_matching)
    if getattr(args, 'lr_disc', None) is not None:
        training_cfg['lr_disc'] = float(args.lr_disc)
    if getattr(args, 'disc_max_grad_norm', None) is not None:
        training_cfg['disc_max_grad_norm'] = float(args.disc_max_grad_norm)
    if getattr(args, 'disc_dtype', None) is not None:
        training_cfg['disc_dtype'] = args.disc_dtype

    # Video discriminator (CausalVAE-style 3D PatchGAN, alternating G/D) CLI pass-through
    if getattr(args, 'use_video_disc', None) is not None:
        loss_cfg['use_video_disc'] = bool(args.use_video_disc)
    if getattr(args, 'video_disc_start_step', None) is not None:
        loss_cfg['video_disc_start_step'] = int(args.video_disc_start_step)
    if getattr(args, 'lambda_video_adv', None) is not None:
        loss_cfg['lambda_video_adv'] = float(args.lambda_video_adv)
    if getattr(args, 'video_disc_loss_type', None) is not None:
        loss_cfg['video_disc_loss_type'] = str(args.video_disc_loss_type)
    if getattr(args, 'video_disc_adaptive_weight', None) is not None:
        loss_cfg['video_disc_adaptive_weight'] = bool(args.video_disc_adaptive_weight)
    if getattr(args, 'video_disc_adaptive_weight_max', None) is not None:
        loss_cfg['video_disc_adaptive_weight_max'] = float(args.video_disc_adaptive_weight_max)
    if getattr(args, 'video_disc_lazy_threshold', None) is not None:
        loss_cfg['video_disc_lazy_threshold'] = float(args.video_disc_lazy_threshold)
    if getattr(args, 'distill_every_steps', None) is not None:
        loss_cfg['distill_every_steps'] = bool(args.distill_every_steps)

    # Data overrides
    data_cfg = cfg.setdefault('data', {})
    train_data_cfg = data_cfg.setdefault('train', {})
    if args.num_frames is not None:
        cfg['num_frames'] = args.num_frames
        for _section in ('train', 'val_video', 'val_contrastive', 'val_caption'):
            _sec = data_cfg.get(_section)
            if _sec is not None and 'num_frames' in _sec:
                _sec['num_frames'] = args.num_frames
        train_data_cfg['num_frames'] = args.num_frames
    if args.spatial_transform_mode is not None:
        train_data_cfg['spatial_transform_mode'] = args.spatial_transform_mode
    if args.spatial_roundtrip_short_edge is not None:
        _rt = int(args.spatial_roundtrip_short_edge)
        train_data_cfg['spatial_roundtrip_short_edge'] = _rt if _rt > 0 else None
    if args.train_metadata_path is not None:
        train_data_cfg['metadata_paths'] = {'train_main': args.train_metadata_path}
        train_data_cfg['metadata_weights'] = {'train_main': 1.0}
        train_data_cfg.pop('data_mixture_yaml', None)

    # Image+Video alterstep mixed training overrides (sit under data.train.image_*).
    if args.use_image_video_alter is not None:
        train_data_cfg['use_image_video_alter'] = bool(args.use_image_video_alter)
    if args.image_data_mixture_yaml is not None:
        train_data_cfg['image_data_mixture_yaml'] = args.image_data_mixture_yaml
    if args.image_dataset_path is not None:
        train_data_cfg['image_dataset_path'] = args.image_dataset_path
    if args.image_batch_size is not None:
        train_data_cfg['image_batch_size'] = int(args.image_batch_size)
    if args.video_batch_size is not None:
        train_data_cfg['video_batch_size'] = int(args.video_batch_size)
    if args.image_video_weights is not None:
        # Parse "a,b" -> [float(a), float(b)]
        _parts = [p.strip() for p in str(args.image_video_weights).split(',')]
        if len(_parts) != 2:
            raise ValueError(
                f"--image_video_weights must be 'image,video' (2 numbers), got "
                f"{args.image_video_weights!r}"
            )
        train_data_cfg['image_video_weights'] = [float(_parts[0]), float(_parts[1])]

    # Image-loader dispatch + Relaion-specific overrides (data.train.image_loader / image_relaion).
    if args.image_loader is not None:
        train_data_cfg['image_loader'] = str(args.image_loader)
    _relaion_arg_map = {
        'root': args.image_relaion_root,
        'slave_path': args.image_relaion_slave_path,
        'base_image_path': args.image_relaion_base_image_path,
        'split': args.image_relaion_split,
        'image_size': args.image_relaion_image_size,
        'center_crop': args.image_relaion_center_crop,
        'random_flip': args.image_relaion_random_flip,
        'recaption_prob': args.image_relaion_recaption_prob,
        'cache_dir': args.image_relaion_cache_dir,
        'max_samples': args.image_relaion_max_samples,
        'repeat': args.image_relaion_repeat,
    }
    _relaion_overrides = {k: v for k, v in _relaion_arg_map.items() if v is not None}
    if _relaion_overrides:
        existing_relaion = train_data_cfg.get('image_relaion') or {}
        if not isinstance(existing_relaion, dict):
            existing_relaion = {}
        existing_relaion.update(_relaion_overrides)
        train_data_cfg['image_relaion'] = existing_relaion

    # Semantic distillation overrides
    if args.use_semantic_distill is not None:
        loss_cfg['use_semantic_distill'] = args.use_semantic_distill
    distill_cfg = cfg.setdefault('model', {}).setdefault('distill', {})
    if args.use_semantic_distill is not None:
        distill_cfg['enabled'] = args.use_semantic_distill

    if args.distill_audio_type is not None:
        distill_cfg['audio_distill_type'] = args.distill_audio_type
    if args.distill_proj_type is not None:
        distill_cfg['distill_proj_type'] = args.distill_proj_type
    if args.distill_proj_layers is not None:
        distill_cfg['distill_proj_layers'] = args.distill_proj_layers
    if args.distill_proj_hidden_dim is not None:
        distill_cfg['distill_proj_hidden_dim'] = args.distill_proj_hidden_dim
    if args.distill_use_conv3d is not None:
        distill_cfg['distill_use_conv3d'] = args.distill_use_conv3d
    if args.distill_proj_before_agg is not None:
        distill_cfg['distill_proj_before_agg'] = args.distill_proj_before_agg
    if args.distill_dim_schedule is not None:
        distill_cfg['distill_dim_schedule'] = args.distill_dim_schedule
    if args.distill_use_sampled is not None:
        distill_cfg['distill_use_sampled'] = args.distill_use_sampled

    _distill_overrides = [
        ('semantic_model_path', loss_cfg),
        ('semantic_api_url', loss_cfg),
        ('encoder_fps', loss_cfg),
        ('encoder_resolution', loss_cfg),
        ('distill_vision_layer', loss_cfg),
        ('distill_audio_layer', loss_cfg),
        ('lambda_distill_image_cosine', loss_cfg),
        ('lambda_distill_image_distance', loss_cfg),
        ('lambda_distill_video_cosine', loss_cfg),
        ('lambda_distill_video_distance', loss_cfg),
        ('lambda_distill_audio_t_axis', loss_cfg),
        ('lambda_distill_audio_d_axis', loss_cfg),
        ('lambda_group_distill', loss_cfg),
        ('distill_margin_cosine', loss_cfg),
        ('distill_margin_distance', loss_cfg),
        ('distill_w_hyper', loss_cfg),
        ('distill_audio_type', loss_cfg),
        ('distill_spatial_norm', loss_cfg),
        ('distill_spatial_norm_gamma', loss_cfg),
        ('distill_use_dist_matrix', loss_cfg),
        ('adaptive_distill_video_ratio', loss_cfg),
        ('adaptive_distill_audio_ratio', loss_cfg),
    ]
    for attr_name, target_dict in _distill_overrides:
        val = getattr(args, attr_name, None)
        if val is not None:
            target_dict[attr_name] = val
    if args.adaptive_distill_balance is not None:
        loss_cfg['adaptive_distill_balance'] = args.adaptive_distill_balance
    if getattr(args, 'adaptive_distill_use_gradient', None) is not None:
        loss_cfg['adaptive_distill_use_gradient'] = args.adaptive_distill_use_gradient

    # Mode C: remote upload distillation
    if args.distill_upload_mode is not None:
        loss_cfg['distill_upload_mode'] = args.distill_upload_mode
    if args.distill_video_gpu_map is not None:
        loss_cfg['distill_video_gpu_map'] = args.distill_video_gpu_map
    if args.distill_image_gpu_id is not None:
        loss_cfg['distill_image_gpu_id'] = args.distill_image_gpu_id
    if args.distill_audio_gpu_id is not None:
        loss_cfg['distill_audio_gpu_id'] = args.distill_audio_gpu_id
    if args.distill_num_upload_workers is not None:
        loss_cfg['distill_num_upload_workers'] = args.distill_num_upload_workers
    if args.distill_processor_path is not None:
        loss_cfg['distill_processor_path'] = args.distill_processor_path

    # Contrastive validation negatives
    val_contrastive_cfg = cfg.setdefault('data', {}).setdefault('val_contrastive', {})
    if args.val_segment_num_negatives is not None:
        val_contrastive_cfg['val_segment_num_negatives'] = _parse_positive_int_list(
            args.val_segment_num_negatives,
            default_value=64,
            field_name='val_segment_num_negatives',
            cfg=cfg,
        )
    if args.val_segment_num_negative_videos is not None:
        val_contrastive_cfg['val_segment_num_negative_videos'] = args.val_segment_num_negative_videos
    if args.val_global_num_negatives is not None:
        val_contrastive_cfg['val_global_num_negatives'] = _parse_positive_int_list(
            args.val_global_num_negatives,
            default_value=32,
            field_name='val_global_num_negatives',
            cfg=cfg,
        )
    if args.val_contrastive_max_samples is not None:
        val_contrastive_cfg['max_samples'] = args.val_contrastive_max_samples


def _build_detail_suffix(cfg: Dict[str, Any]) -> str:
    """Generate a detail suffix from the merged config describing active losses."""
    loss_cfg = cfg.get('loss', {})
    training_cfg = cfg.get('training', {})
    model_cfg = cfg.get('model', {})

    parts = []

    if loss_cfg.get('use_video_recon', True):
        parts.append("Vrec")
    if loss_cfg.get('use_audio_recon', True):
        parts.append("Arec")
    if loss_cfg.get('use_segment_contrastive', True):
        parts.append("Seg")
    if loss_cfg.get('use_global_contrastive', True):
        parts.append("Glob")
    if loss_cfg.get('use_llm_caption', False):
        parts.append("LLM")

    suffix = "_".join(parts) if parts else "train"

    bs = training_cfg.get('batch_size', 1)
    suffix += f"_bs{bs}"

    _mgn = training_cfg.get('max_grad_norm', 1.0)
    if _mgn != 1.0:
        suffix += f"_mgn{_mgn:g}"

    # Group weights
    _gv = loss_cfg.get('lambda_group_video', 1.0)
    _ga = loss_cfg.get('lambda_group_audio', 1.0)
    _gc = loss_cfg.get('lambda_group_contrastive', 1.0)
    suffix += f"_gv{_gv:g}_ga{_ga:g}_gc{_gc:g}"

    _vkl = loss_cfg.get('lambda_video_kl', 1e-6)
    if _vkl != 1e-6:
        suffix += f"_vkl{_vkl:g}"

    # Dtype
    _dtype = training_cfg.get('dtype', 'bfloat16')
    if _dtype not in ('bfloat16', 'bf16'):
        suffix += f"_{_dtype}"

    _mod_abbrev = [
        ('video_vae_dtype', 'vvd'),
        ('audio_vae_dtype', 'avd'),
        ('contrastive_dtype', 'ctd'),
        ('llm_dtype', 'lld'),
    ]
    _mod_parts = []
    for _mkey, _mab in _mod_abbrev:
        _mval = training_cfg.get(_mkey)
        if _mval is not None:
            _short = _mval.replace('bfloat16', 'bf16').replace('float16', 'fp16').replace('float32', 'fp32')
            _mod_parts.append(f"{_mab}-{_short}")
    if _mod_parts:
        suffix += "_" + "_".join(_mod_parts)

    # Adaptive balance
    if loss_cfg.get('adaptive_loss_balance_v2', False):
        _src = str(loss_cfg.get('adaptive_anchor_source', 'video_vae'))
        _src_abbrev = {'video_vae': 'v', 'audio_vae': 'a', 'contrastive': 'c'}.get(_src, _src)
        _rv = loss_cfg.get('adaptive_ratio_video', 1.0)
        _ra = loss_cfg.get('adaptive_ratio_audio', 1.0)
        _rc = loss_cfg.get('adaptive_ratio_contrastive', 1.0)
        suffix += f"_adav2-{_src_abbrev}-v{_rv:g}a{_ra:g}c{_rc:g}"
        # Stage1 overrides (only when freeze is enabled and user customised).
        if (loss_cfg.get('freeze_video_vae', False)
                or loss_cfg.get('freeze_audio_vae', False)):
            _s1_src = loss_cfg.get('adaptive_anchor_source_stage1')
            _s1_rv = loss_cfg.get('adaptive_ratio_video_stage1')
            _s1_ra = loss_cfg.get('adaptive_ratio_audio_stage1')
            _s1_rc = loss_cfg.get('adaptive_ratio_contrastive_stage1')
            if _s1_src or _s1_rv is not None or _s1_ra is not None or _s1_rc is not None:
                _s1_abbrev = {'video_vae': 'v', 'audio_vae': 'a', 'contrastive': 'c'}.get(
                    _s1_src, _src_abbrev
                ) if _s1_src else _src_abbrev
                _s1_rv_eff = _s1_rv if _s1_rv is not None else _rv
                _s1_ra_eff = _s1_ra if _s1_ra is not None else _ra
                _s1_rc_eff = _s1_rc if _s1_rc is not None else _rc
                suffix += f"_s1-{_s1_abbrev}-v{_s1_rv_eff:g}a{_s1_ra_eff:g}c{_s1_rc_eff:g}"
        # Stage2 gradient-balance hybrid marker.
        if loss_cfg.get('adaptive_v2_stage2_use_gradient', False):
            _bs = int(loss_cfg.get('adaptive_v2_stage2_blend_steps', 0))
            _s2_vr = loss_cfg.get('gradient_ratio_video_stage2')
            _s2_ar = loss_cfg.get('gradient_ratio_audio_stage2')
            if _s2_vr is None:
                _s2_vr = loss_cfg.get('gradient_balance_video_ratio', 0.5)
            if _s2_ar is None:
                _s2_ar = loss_cfg.get('gradient_balance_audio_ratio', 0.5)
            suffix += f"_s2grad-v{_s2_vr:g}a{_s2_ar:g}"
            if _bs > 0:
                suffix += f"-blend{_bs}"
    elif loss_cfg.get('adaptive_loss_balance', False):
        _ar = loss_cfg.get('adaptive_balance_audio_ratio', 0.5)
        _cr = loss_cfg.get('adaptive_balance_contrastive_ratio', 0.5)
        suffix += f"_adaloss-a{_ar:g}-c{_cr:g}"
    elif loss_cfg.get('adaptive_loss_balance_by_uncertainty', False):
        suffix += "_uncloss"
    elif loss_cfg.get('adaptive_loss_balance_by_gradient', False):
        suffix += "_gradloss"

    # Contrastive gradient scaling
    _contrastive_cfg = model_cfg.get('contrastive', {})
    _gsv = _contrastive_cfg.get('contrastive_grad_scale_video', 1.0)
    _gsa = _contrastive_cfg.get('contrastive_grad_scale_audio', 1.0)
    if _gsv != 1.0 or _gsa != 1.0:
        suffix += f"_cgs-v{_gsv:g}a{_gsa:g}"

    # Phase freezing of video VAE
    if loss_cfg.get('freeze_video_vae', False):
        _unfreeze_at = int(loss_cfg.get('freeze_video_vae_until_step', 0))
        suffix += f"_frzV{_unfreeze_at}"

    # Phase freezing of audio VAE (symmetric to video)
    if loss_cfg.get('freeze_audio_vae', False):
        _unfreeze_at = int(loss_cfg.get('freeze_audio_vae_until_step', 0))
        suffix += f"_frzA{_unfreeze_at}"

    # Encoder-only freeze markers (decoder/GAN-only finetune)
    if loss_cfg.get('freeze_video_encoder', False) and not loss_cfg.get('freeze_video_vae', False):
        suffix += "_frzVenc"
    if loss_cfg.get('freeze_audio_encoder', False) and not loss_cfg.get('freeze_audio_vae', False):
        suffix += "_frzAenc"

    # Video backbone override marker (only emit when CLI actually overrode it)
    _video_cfg_here = model_cfg.get('video', {})
    if _video_cfg_here.get('_cli_override', False):
        _video_model_name = _video_cfg_here.get('model_name')
        if _video_model_name:
            _vm_abbrev = {'WanVAE': 'wanvae', 'WanVAE22': 'wanvae22'}.get(
                _video_model_name, str(_video_model_name).lower()
            )
            suffix += f"_vmodel-{_vm_abbrev}"

    if loss_cfg.get('use_semantic_distill', False):
        dp = []
        distill_model_cfg = model_cfg.get('distill', {})

        proj = distill_model_cfg.get('distill_proj_type', loss_cfg.get('distill_proj_type', 'conv'))
        dp.append(f"proj{proj}")
        if proj == 'conv' and distill_model_cfg.get('distill_use_conv3d', loss_cfg.get('distill_use_conv3d', False)):
            dp.append("3d")

        n_layers = distill_model_cfg.get('distill_proj_layers', 1)
        if n_layers != 1:
            dp.append(f"L{n_layers}")

        _dim_sched = distill_model_cfg.get('distill_dim_schedule', 'fixed')
        if _dim_sched != 'fixed':
            dp.append(f"ds{_dim_sched}")

        if distill_model_cfg.get('distill_use_sampled', False):
            dp.append("sampled")

        proj_before = distill_model_cfg.get('distill_proj_before_agg', True)
        if not proj_before:
            dp.append("aggfirst")

        if loss_cfg.get('distill_spatial_norm', True):
            gamma = loss_cfg.get('distill_spatial_norm_gamma', 0.7)
            dp.append(f"snorm{gamma}")
        else:
            dp.append("nosnorm")

        margin_cos = loss_cfg.get('distill_margin_cosine', 0.0)
        dp.append(f"mc{margin_cos}")

        if loss_cfg.get('distill_use_dist_matrix', False):
            margin_dist = loss_cfg.get('distill_margin_distance', 0.25)
            dp.append(f"dmat{margin_dist}")

        audio_type = distill_model_cfg.get('audio_distill_type', loss_cfg.get('distill_audio_type', 'd_axis'))
        dp.append(f"aud{audio_type}")

        w_hyper = loss_cfg.get('distill_w_hyper', 0.1)
        dp.append(f"wh{w_hyper}")

        lg = loss_cfg.get('lambda_group_distill', 1.0)
        dp.append(f"lg{lg}")

        lic = loss_cfg.get('lambda_distill_image_cosine', 1.0)
        lid = loss_cfg.get('lambda_distill_image_distance', 1.0)
        lvc = loss_cfg.get('lambda_distill_video_cosine', 1.0)
        lvd = loss_cfg.get('lambda_distill_video_distance', 1.0)
        dp.append(f"ic{lic}_id{lid}_vc{lvc}_vd{lvd}")

        if audio_type == 'd_axis':
            lad = loss_cfg.get('lambda_distill_audio_d_axis', 120.0)
            dp.append(f"ad{lad}")
        else:
            lat = loss_cfg.get('lambda_distill_audio_t_axis', 1.0)
            dp.append(f"at{lat}")

        if loss_cfg.get('adaptive_distill_balance', False):
            vr = loss_cfg.get('adaptive_distill_video_ratio', 0.1)
            ar = loss_cfg.get('adaptive_distill_audio_ratio', 0.1)
            dp.append(f"adabal_v{vr}_a{ar}")

        efps = loss_cfg.get('encoder_fps', 4)
        eres = loss_cfg.get('encoder_resolution', 128)
        dp.append(f"fps{efps}_res{eres}")

        suffix += "_distill_" + "_".join(dp)

    if loss_cfg.get('video_learn_logvar', False):
        _vli = loss_cfg.get('video_logvar_init', 0.0)
        suffix += f"_llogvar{_vli:g}"

    if loss_cfg.get('video_loss_clamp', False):
        suffix += "_vlclamp"

    _ga = int(training_cfg.get('gradient_accumulation_steps', 1))
    if _ga > 1:
        suffix += f"_ga{_ga}"

    _vv_w = training_cfg.get('lr_video_vae_warmup_steps')
    _vv_t = training_cfg.get('lr_video_vae_total_steps')
    _vv_s = training_cfg.get('lr_video_vae_start_step')
    _vv_m = training_cfg.get('lr_video_vae_min_ratio')
    if any(v is not None for v in (_vv_w, _vv_t, _vv_s, _vv_m)):
        _parts = []
        if _vv_w is not None:
            _parts.append(f"w{int(_vv_w)}")
        if _vv_t is not None:
            _parts.append(f"t{int(_vv_t)}")
        if _vv_s is not None:
            _parts.append(f"s{int(_vv_s)}")
        if _vv_m is not None and float(_vv_m) != 0.0:
            _parts.append(f"m{float(_vv_m):g}")
        if _parts:
            suffix += "_vvsched-" + "".join(_parts)

    _aa_w = training_cfg.get('lr_audio_vae_warmup_steps')
    _aa_t = training_cfg.get('lr_audio_vae_total_steps')
    _aa_s = training_cfg.get('lr_audio_vae_start_step')
    _aa_m = training_cfg.get('lr_audio_vae_min_ratio')
    if any(v is not None for v in (_aa_w, _aa_t, _aa_s, _aa_m)):
        _parts = []
        if _aa_w is not None:
            _parts.append(f"w{int(_aa_w)}")
        if _aa_t is not None:
            _parts.append(f"t{int(_aa_t)}")
        if _aa_s is not None:
            _parts.append(f"s{int(_aa_s)}")
        if _aa_m is not None and float(_aa_m) != 0.0:
            _parts.append(f"m{float(_aa_m):g}")
        if _parts:
            suffix += "_avsched-" + "".join(_parts)

    return suffix


def _truncate_path_components(tag: str, max_component: int = 240) -> str:
    """Ensure no single path component exceeds *max_component* bytes.

    ext4 limits a single directory / file name to 255 bytes.  We use 240
    as the default ceiling to leave room for sub-entries like ``/log``.
    When truncation is needed the component is shortened and a short
    hash of the original is appended so that uniqueness is preserved.
    """
    import hashlib
    parts = tag.split('/')
    out = []
    for p in parts:
        if len(p.encode('utf-8')) > max_component:
            h = hashlib.md5(p.encode('utf-8')).hexdigest()[:8]
            truncated = p[:max_component - 10]
            while len(truncated.encode('utf-8')) > max_component - 10:
                truncated = truncated[:-1]
            p = f"{truncated}_{h}"
        out.append(p)
    return '/'.join(out)


def build_experiment_tag(args: argparse.Namespace, cfg: Dict[str, Any], rank: int) -> None:
    """Build experiment tag from config and CLI args, store in args.tag.

    If ``args.tag`` is already set (e.g. from the shell launcher), the
    auto-generated detail suffix is appended so the directory name still
    reflects the active losses and hyper-parameters.

    When ``args.exp_name`` is provided, it replaces only the leaf
    component: the config-derived prefix from ``args.tag`` is kept so old
    and new experiments still live side by side under the same group
    directory, but the auto-detail suffix and ``exp_name_suffix`` are
    skipped. Final tag is ``<args.tag>/<exp_name>`` (or just ``<exp_name>``
    when ``args.tag`` is None). This is the escape hatch for users who
    want a clean, human-readable leaf name.
    """
    if getattr(args, 'exp_name', None):
        if args.tag:
            tag = f"{args.tag}/{args.exp_name}"
        else:
            tag = args.exp_name
        tag = _truncate_path_components(tag)
        args.tag = tag
        if rank == 0:
            logging.info(f"Experiment tag (--exp_name override): {tag}")
        return

    detail = _build_detail_suffix(cfg)

    if args.tag is not None:
        tag = f"{args.tag}/{detail}"
    else:
        tag = detail

    if getattr(args, 'train_metadata_path', None):
        import os as _os
        _ds_name = _os.path.basename(_os.path.normpath(args.train_metadata_path))
        if _ds_name:
            tag += f"_data-{_ds_name}"

    if args.exp_name_suffix:
        tag += f"_{args.exp_name_suffix}"

    tag = _truncate_path_components(tag)
    args.tag = tag

    if rank == 0:
        logging.info(f"Experiment tag: {tag}")
