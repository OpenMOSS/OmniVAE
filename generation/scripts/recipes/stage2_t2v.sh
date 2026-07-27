#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${OMNIVAE_CKPT:?Set OMNIVAE_CKPT to an OmniVAE Trainer_xxxxxx/state_dict.pt checkpoint.}"
: "${T2I_TRANSFORMER_CKPT:?Set T2I_TRANSFORMER_CKPT to the stage1 transformer directory.}"

name="${OMNIGEN_RUN_NAME:-t2v_omnivae}"
batch_size="${OMNIGEN_BATCH_SIZE:-2}"
learning_rate="${OMNIGEN_LR:-1.0e-4}"
grad_accum="${OMNIGEN_GRAD_ACCUM:-1}"

bash scripts/audio/train.sh configs/visual/t2v.yaml \
  --name "${name}" \
  --bs "${batch_size}" \
  --lr "${learning_rate}" \
  --grad_accum "${grad_accum}" \
  --vae_path "${OMNIVAE_CKPT}" \
  --vae_type omnivae \
  --vae_branch video \
  --vae_use_ema "${OMNIVAE_USE_EMA:-false}" \
  --init_from_transformer "${T2I_TRANSFORMER_CKPT}" \
  "$@"
