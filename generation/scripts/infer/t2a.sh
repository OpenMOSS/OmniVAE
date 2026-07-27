#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${T2A_CHECKPOINT:?Set T2A_CHECKPOINT to a checkpoint directory.}"

config="${T2A_CONFIG:-configs/audio/t2a.yaml}"
output_dir="${T2A_OUTPUT_DIR:-outputs/t2a}"

cmd=(python infer/audio/run_eval.py
  --config "${config}"
  --checkpoint "${T2A_CHECKPOINT}"
  --output-dir "${output_dir}"
  --num-prompts "${T2A_NUM_PROMPTS:-8}"
)

if [ -n "${OMNIVAE_AUDIO_CKPT:-${OMNIVAE_CKPT:-}}" ] && [ -n "${T2A_AUDIO_VAE_OVERRIDE_NAME:-}" ]; then
  cmd+=(--audio-vae-override "${T2A_AUDIO_VAE_OVERRIDE_NAME}=omnivae:${OMNIVAE_AUDIO_CKPT:-${OMNIVAE_CKPT}}")
fi
cmd+=("$@")

printf "Executing:"
printf " %q" "${cmd[@]}"
printf "\n"
if [ "${OMNIGEN_DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
"${cmd[@]}"
