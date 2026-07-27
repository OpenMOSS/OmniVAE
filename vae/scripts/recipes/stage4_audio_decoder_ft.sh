#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${PRETRAINED_AUDIO_CKPT:?Set PRETRAINED_AUDIO_CKPT to the stage3 checkpoint.}"
: "${PRETRAINED_DISC_CKPT:?Set PRETRAINED_DISC_CKPT to an audio discriminator checkpoint.}"

run_train "${CFG_RECON}" \
  --no_video_recon \
  --use_audio_recon \
  --no_segment_contrastive \
  --no_global_contrastive \
  --no_semantic_distill \
  --freeze_audio_encoder \
  --batch_size 4 \
  --dtype fp32 \
  --num_frames 121 \
  --use_audio_disc \
  --lr 1.0e-5 \
  --lr_disc 5e-4 \
  --lambda_audio_adv 1.0 \
  --lambda_audio_feature_matching 2.0 \
  --lambda_audio_kl 0 \
  --pretrained_audio_checkpoint "${PRETRAINED_AUDIO_CKPT}" \
  --no_continue \
  --pretrained_disc_checkpoint "${PRETRAINED_DISC_CKPT}" \
  --no-pretrained_disc_load_optim \
  --exp_name_suffix audio_decoder_only_distill_avclip
