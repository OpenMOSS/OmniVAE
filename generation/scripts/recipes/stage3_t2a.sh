#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${OMNIVAE_AUDIO_CKPT:=${OMNIVAE_CKPT:-}}"
: "${OMNIVAE_AUDIO_CKPT:?Set OMNIVAE_AUDIO_CKPT or OMNIVAE_CKPT to an OmniVAE checkpoint.}"

name="${OMNIGEN_RUN_NAME:-t2a_omnivae}"
batch_size="${OMNIGEN_BATCH_SIZE:-8}"
learning_rate="${OMNIGEN_LR:-1.0e-4}"
grad_accum="${OMNIGEN_GRAD_ACCUM:-1}"
validation_steps="${OMNIGEN_VALIDATION_STEPS:-5000}"

bash scripts/audio/train.sh configs/audio/t2a.yaml \
  --name "${name}" \
  --bs "${batch_size}" \
  --lr "${learning_rate}" \
  --grad_accum "${grad_accum}" \
  --validation_steps "${validation_steps}" \
  --audio_vae_path "${OMNIVAE_AUDIO_CKPT}" \
  --audio_vae_type omnivae \
  --vae_branch audio \
  --vae_use_ema "${OMNIVAE_USE_EMA:-false}" \
  "$@"
