#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${T2V_CHECKPOINT:?Set T2V_CHECKPOINT to a checkpoint directory.}"
: "${T2V_PROMPT_MANIFEST:=examples/prompts/t2v_prompts.jsonl}"

output_dir="${T2V_OUTPUT_DIR:-outputs/t2v}"

cmd=(python infer/t2v/infer_t2v.py
  --checkpoint-dir "${T2V_CHECKPOINT}"
  --prompt-manifest "${T2V_PROMPT_MANIFEST}"
  --output-dir "${output_dir}"
  --limit "${T2V_LIMIT:-8}"
)

if [ -n "${OMNIVAE_CKPT:-}" ]; then
  cmd+=(--vae-type omnivae --vae-path "${OMNIVAE_CKPT}")
fi
cmd+=("$@")

printf "Executing:"
printf " %q" "${cmd[@]}"
printf "\n"
if [ "${OMNIGEN_DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
"${cmd[@]}"
