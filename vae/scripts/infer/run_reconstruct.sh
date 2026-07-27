#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export OMNIVAE_REPO_ROOT="${OMNIVAE_REPO_ROOT:-${REPO_ROOT}}"
export OMNIVAE_CKPT_ROOT="${OMNIVAE_CKPT_ROOT:-${REPO_ROOT}/ckpts}"
export OMNIVAE_DATA_ROOT="${OMNIVAE_DATA_ROOT:-${REPO_ROOT}/data}"
export OMNIVAE_EXP_ROOT="${OMNIVAE_EXP_ROOT:-${REPO_ROOT}/exp}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-video}"
CONFIG="${INFER_CONFIG:-${REPO_ROOT}/configs/audio_video_vae/omnivae_recon_distill_wan22.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${OMNIVAE_EXP_ROOT}/infer/reconstruct/${MODE}}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT_LIST="${CHECKPOINT_LIST:-${REPO_ROOT}/examples/eval/checkpoints.txt}"
  CHECKPOINT="$(grep -vE '^[[:space:]]*(#|$)' "${CHECKPOINT_LIST}" | head -n 1 || true)"
fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] Set CHECKPOINT=/path/to/state_dict.pt or provide CHECKPOINT_LIST." >&2
  exit 1
fi

args=(
  -m omnivae.infer.reconstruct
  --mode "${MODE}"
  --checkpoint "${CHECKPOINT}"
  --config "${CONFIG}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${INPUT_FILE:-}" ]]; then
  args+=(--input_file "${INPUT_FILE}")
elif [[ -n "${INPUT_DIR:-}" ]]; then
  args+=(--input_dir "${INPUT_DIR}")
elif [[ -n "${INPUT_JSONL:-}" ]]; then
  args+=(--input_jsonl "${INPUT_JSONL}")
else
  if [[ "${MODE}" == "audio" ]]; then
    args+=(--input_jsonl "${REPO_ROOT}/examples/eval/audio_recon.jsonl")
  else
    args+=(--input_jsonl "${REPO_ROOT}/examples/eval/video_recon.jsonl")
  fi
fi

if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  args+=(--max_examples "${MAX_EXAMPLES}")
fi
if [[ -n "${AUDIO_SAMPLE_RATE:-}" ]]; then
  args+=(--sample_rate "${AUDIO_SAMPLE_RATE}")
fi
if [[ -n "${MAX_DURATION:-}" ]]; then
  args+=(--max_duration "${MAX_DURATION}")
fi
if [[ -n "${NUM_FRAMES:-}" ]]; then
  args+=(--num_frames "${NUM_FRAMES}")
fi
if [[ -n "${TARGET_FPS:-}" ]]; then
  args+=(--target_fps "${TARGET_FPS}")
fi
if [[ -n "${RESOLUTION:-}" ]]; then
  args+=(--resolution "${RESOLUTION}")
fi
if [[ "${SAVE_INPUTS:-0}" == "1" ]]; then
  args+=(--save_inputs)
fi
if [[ "${NO_TORCHCODEC:-0}" == "1" ]]; then
  args+=(--no_torchcodec)
fi
if [[ "${OMNIVAE_DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run)
fi

exec "${PYTHON_BIN}" "${args[@]}" "$@"
