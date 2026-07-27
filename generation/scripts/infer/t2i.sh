#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${T2I_CHECKPOINT:?Set T2I_CHECKPOINT to a checkpoint directory.}"

config="${T2I_CONFIG:-configs/visual/t2i.yaml}"
output_dir="${T2I_OUTPUT_DIR:-outputs/t2i}"

cmd=(python infer/image/run_eval.py
  --config "${config}"
  --checkpoint "${T2I_CHECKPOINT}"
  --output-dir "${output_dir}"
  --num-samples "${T2I_NUM_SAMPLES:-64}"
  --batch-size "${T2I_BATCH_SIZE:-4}"
)

if [ -n "${T2I_ANNOTATIONS_JSON:-}" ]; then
  cmd+=(--annotations-json "${T2I_ANNOTATIONS_JSON}")
fi
if [ -n "${T2I_IMAGES_DIR:-}" ]; then
  cmd+=(--images-dir "${T2I_IMAGES_DIR}")
fi
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
