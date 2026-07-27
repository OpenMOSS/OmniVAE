#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${VIDEO_CKPT:?Set VIDEO_CKPT to the stage2 video VAE checkpoint directory or file.}"
: "${AUDIO_CKPT:?Set AUDIO_CKPT to the stage2 audio VAE checkpoint directory or file.}"

FREEZE_STEPS="${FREEZE_STEPS:-20000}"

run_train "${CFG_ALIGN}" \
  --pretrained_video_checkpoint "${VIDEO_CKPT}" \
  --pretrained_audio_checkpoint "${AUDIO_CKPT}" \
  --use_video_recon \
  --use_audio_recon \
  --use_segment_contrastive \
  --no_global_contrastive \
  --no_audio_disc \
  --no_video_disc \
  --batch_size 1 \
  --num_frames 193 \
  --dtype fp32 \
  --video_vae_dtype bf16 \
  --gradient_checkpointing \
  --lambda_video_kl 1e-4 \
  --contrastive_use_mean \
  --spatial_pool_mode transformer \
  --segment_temporal_pool_mode conv \
  --global_temporal_pool_mode transformer \
  --contrastive_module_size large \
  --spatial_merge_factor 2 \
  --num_negatives 96 \
  --same_long_video_priority \
  --same_long_video_num_negatives 48 \
  --num_negatives_no_sibling 24 \
  --use_semantic_distill \
  --semantic_model_path "${SEMANTIC_MODEL_PATH}" \
  --encoder_fps 4 \
  --encoder_resolution 256 \
  --distill_vision_layer 18 \
  --distill_audio_type t_axis \
  --distill_dim_schedule doubling \
  --distill_proj_before_agg \
  --no_distill_use_sampled \
  --distill_spatial_norm \
  --distill_spatial_norm_gamma 0.7 \
  --adaptive_distill_balance \
  --adaptive_distill_video_ratio 0.2 \
  --adaptive_distill_audio_ratio 0.2 \
  --freeze_video_vae \
  --freeze_video_vae_until_step "${FREEZE_STEPS}" \
  --video_distill_start_step "${FREEZE_STEPS}" \
  --audio_distill_start_step 0 \
  --no_adaptive_loss_balance \
  --adaptive_loss_balance_v2 \
  --adaptive_anchor_source contrastive \
  --adaptive_ratio_video 1.0 \
  --adaptive_ratio_audio 1.0 \
  --adaptive_ratio_contrastive 1.5 \
  --adaptive_anchor_source_stage1 contrastive \
  --adaptive_ratio_video_stage1 0.0 \
  --adaptive_ratio_audio_stage1 1.0 \
  --adaptive_ratio_contrastive_stage1 2.0 \
  --lr 1e-4 \
  --lr_audio_vae 5e-5 \
  --lr_video_vae 2e-5 \
  --lr_video_vae_warmup_steps 6000 \
  --lr_video_vae_min_ratio 0.1 \
  --exp_name_suffix "loss_2_1_3_2stage_frz${FREEZE_STEPS}_distill_audt_videw_split_ckpt" \
  --reset_scheduler_on_resume
