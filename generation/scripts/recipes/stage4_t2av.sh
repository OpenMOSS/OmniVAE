#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${OMNIVAE_VIDEO_CKPT:=${OMNIVAE_CKPT:-}}"
: "${OMNIVAE_AUDIO_CKPT:=${OMNIVAE_CKPT:-}}"
: "${OMNIVAE_VIDEO_CKPT:?Set OMNIVAE_VIDEO_CKPT or OMNIVAE_CKPT to an OmniVAE checkpoint.}"
: "${OMNIVAE_AUDIO_CKPT:?Set OMNIVAE_AUDIO_CKPT or OMNIVAE_CKPT to an OmniVAE checkpoint.}"
: "${T2V_TRANSFORMER_CKPT:?Set T2V_TRANSFORMER_CKPT to the stage2 transformer directory.}"
: "${T2A_TRANSFORMER_CKPT:?Set T2A_TRANSFORMER_CKPT to the stage3 transformer directory.}"

name="${OMNIGEN_RUN_NAME:-t2av_omnivae}"
batch_size="${OMNIGEN_BATCH_SIZE:-2}"
grad_accum="${OMNIGEN_GRAD_ACCUM:-1}"
backbone_lr="${OMNIGEN_BACKBONE_LR:-3.0e-5}"
bridge_lr="${OMNIGEN_BRIDGE_LR:-6.0e-5}"
validate_at="${OMNIGEN_VALIDATE_AT:-0}"

bash scripts/av/train.sh configs/av/t2av.yaml \
  --name "${name}" \
  --bs "${batch_size}" \
  --grad_accum "${grad_accum}" \
  --backbone_lr "${backbone_lr}" \
  --bridge_lr "${bridge_lr}" \
  --pretrained_t2v "${T2V_TRANSFORMER_CKPT}" \
  --pretrained_t2a "${T2A_TRANSFORMER_CKPT}" \
  --video_vae_path "${OMNIVAE_VIDEO_CKPT}" \
  --audio_vae_path "${OMNIVAE_AUDIO_CKPT}" \
  --video_vae_type omnivae \
  --audio_vae_type omnivae \
  --vae_use_ema "${OMNIVAE_USE_EMA:-false}" \
  --validate_at "${validate_at}" \
  "$@"
