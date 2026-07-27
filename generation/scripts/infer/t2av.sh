#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${T2AV_CHECKPOINT:?Set T2AV_CHECKPOINT to a checkpoint directory.}"

output_dir="${T2AV_OUTPUT_DIR:-outputs/t2av}"
prompt_manifest="${T2AV_PROMPT_MANIFEST:-examples/prompts/t2av_valid.jsonl}"

cmd=(python infer/t2av/infer_t2av.py
  --ckpt "${T2AV_CHECKPOINT}"
  --output-dir "${output_dir}"
  --prompt-manifest "${prompt_manifest}"
  --limit "${T2AV_LIMIT:-4}"
)

if [ -n "${OMNIVAE_VIDEO_CKPT:-${OMNIVAE_CKPT:-}}" ]; then
  cmd+=(--vae-type omnivae --vae-path "${OMNIVAE_VIDEO_CKPT:-${OMNIVAE_CKPT}}")
fi
if [ -n "${OMNIVAE_AUDIO_CKPT:-${OMNIVAE_CKPT:-}}" ]; then
  cmd+=(--audio-vae-type omnivae --audio-vae-path "${OMNIVAE_AUDIO_CKPT:-${OMNIVAE_CKPT}}")
fi
cmd+=("$@")

printf "Executing:"
printf " %q" "${cmd[@]}"
printf "\n"
if [ "${OMNIGEN_DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
"${cmd[@]}"
