#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_train "${CFG_RECON}" \
  --no_video_recon \
  --use_audio_recon \
  --no_segment_contrastive \
  --no_global_contrastive \
  --batch_size 4 \
  --dtype fp32 \
  --num_frames 121 \
  --no_semantic_distill \
  --exp_name_suffix audio_moredata_ct \
  --use_audio_disc \
  --lr 1.0e-5 \
  --lr_disc 1.0e-4 \
  --lambda_audio_adv 1.0 \
  --lambda_audio_feature_matching 2.0
