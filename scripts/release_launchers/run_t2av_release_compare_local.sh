#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher for an already allocated local/PET environment.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/../test_output/t2av_release_compare_set3_large}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CFG="${CFG:-4}"
STEP="${STEP:-200000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
VAE_MODE="${VAE_MODE:-release}"

cmd=(
    bash "${REPO_ROOT}/scripts/release_eval/t2av/run_release_t2av_eval_compare.sh"
    --mode run
    --gpus "${GPUS}"
    --cfg "${CFG}"
    --step "${STEP}"
    --max-examples "${MAX_EXAMPLES}"
    --vae-mode "${VAE_MODE}"
    --output-root "${OUTPUT_ROOT}"
)

cd "${REPO_ROOT}"
echo "Executing: ${cmd[*]}"
"${cmd[@]}"
