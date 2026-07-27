#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_train "${CFG_RECON}" \
  --use_video_recon \
  --no_audio_recon \
  --no_segment_contrastive \
  --no_global_contrastive \
  --batch_size 1 \
  --lambda_video_kl 1e-4 \
  --video_vae_dtype bf16 \
  --distill_vision_layer 18 \
  --gradient_checkpointing \
  --num_frames 121 \
  --use_semantic_distill \
  --semantic_model_path "${SEMANTIC_MODEL_PATH}" \
  --encoder_fps 4 \
  --encoder_resolution 256 \
  --distill_proj_before_agg \
  --distill_spatial_norm \
  --distill_spatial_norm_gamma 0.7 \
  --adaptive_distill_balance \
  --adaptive_distill_video_ratio 0.5 \
  --distill_dim_schedule doubling \
  --no_distill_use_sampled \
  --use_video_disc \
  --video_disc_adaptive_weight \
  --video_disc_loss_type hinge \
  --lr_disc 5e-6 \
  --video_disc_start_step 3000 \
  --lambda_video_adv 0.1 \
  --distill_every_steps \
  --exp_name_suffix wan2_2_fwbf16_193_layers18_disc_smalllr_64
