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
CHECKPOINT_LIST="${CHECKPOINT_LIST:-${REPO_ROOT}/examples/eval/checkpoints.txt}"
INPUT_JSONL="${AUDIO_EVAL_JSONL:-${REPO_ROOT}/examples/eval/audio_recon.jsonl}"
CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/audio_video_vae/omnivae_recon_distill_wan22.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${OMNIVAE_EXP_ROOT}/eval/audio_recon}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVALUATION_DOMAIN="${EVALUATION_DOMAIN:-audio}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-float32}"

args=(
  -m omnivae.eval.reconstruction.audio_recon
  --checkpoint_list "${CHECKPOINT_LIST}"
  --config "${CONFIG}"
  --input_jsonl "${INPUT_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --evaluation_domain "${EVALUATION_DOMAIN}"
  --inference_dtype "${INFERENCE_DTYPE}"
)

if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  args+=(--max_examples "${MAX_EXAMPLES}")
fi
if [[ -n "${AUDIO_SAMPLE_RATE:-}" ]]; then
  args+=(--sample_rate "${AUDIO_SAMPLE_RATE}")
fi
if [[ -n "${MAX_DURATION:-}" ]]; then
  args+=(--max_duration "${MAX_DURATION}")
fi
if [[ "${COMPUTE_STOI:-0}" == "1" ]]; then
  args+=(--compute_stoi)
fi
if [[ "${COMPUTE_PESQ:-0}" == "1" ]]; then
  args+=(--compute_pesq)
fi
if [[ -n "${COMPUTE_DAC_METRICS:-}" ]]; then
  if [[ "${COMPUTE_DAC_METRICS}" == "1" ]]; then
    args+=(--compute_dac_metrics)
  else
    args+=(--no-compute_dac_metrics)
  fi
fi
if [[ -n "${COMPUTE_VISQOL:-}" ]]; then
  if [[ "${COMPUTE_VISQOL}" == "1" ]]; then
    args+=(--compute_visqol)
  else
    args+=(--no-compute_visqol)
  fi
fi
if [[ -n "${COMPUTE_SPEAKER_SIMILARITY:-}" ]]; then
  if [[ "${COMPUTE_SPEAKER_SIMILARITY}" == "1" ]]; then
    args+=(--compute_speaker_similarity)
  else
    args+=(--no-compute_speaker_similarity)
  fi
fi
if [[ -n "${SPEAKER_SIMILARITY_MODEL:-}" ]]; then
  args+=(--speaker_similarity_model "${SPEAKER_SIMILARITY_MODEL}")
fi
if [[ "${OMNIVAE_DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run)
fi

exec "${PYTHON_BIN}" "${args[@]}" "$@"
