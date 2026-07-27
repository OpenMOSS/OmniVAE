#!/usr/bin/env bash

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RECIPE_DIR}/../.." && pwd)"

export OMNIVAE_REPO_ROOT="${OMNIVAE_REPO_ROOT:-${REPO_ROOT}}"
export OMNIVAE_CKPT_ROOT="${OMNIVAE_CKPT_ROOT:-${REPO_ROOT}/ckpts}"
export OMNIVAE_DATA_ROOT="${OMNIVAE_DATA_ROOT:-${REPO_ROOT}/data}"
export OMNIVAE_EXP_ROOT="${OMNIVAE_EXP_ROOT:-${REPO_ROOT}/exp}"
export OMNIVAE_SEMANTIC_MODEL="${OMNIVAE_SEMANTIC_MODEL:-${OMNIVAE_CKPT_ROOT}/qwen3_avencoder_service}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/scripts/train_local.sh}"
CFG_RECON="${CFG_RECON:-configs/audio_video_vae/omnivae_recon_distill_wan22.yaml}"
CFG_ALIGN="${CFG_ALIGN:-configs/audio_video_vae/omnivae_av_align_wan22_24fps.yaml}"
SEMANTIC_MODEL_PATH="${SEMANTIC_MODEL_PATH:-${OMNIVAE_SEMANTIC_MODEL}}"

run_train() {
    bash "${TRAIN_SCRIPT}" "$@"
}
