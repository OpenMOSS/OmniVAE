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
  --no_semantic_distill \
  --gradient_checkpointing \
  --num_frames 121 \
  --use_video_disc \
  --video_disc_adaptive_weight \
  --video_disc_loss_type hinge \
  --lr_disc 5e-6 \
  --video_disc_start_step 3000 \
  --lambda_video_adv 0.1 \
  --exp_name_suffix wan2_2_no_distill
