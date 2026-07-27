#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

INPUT_DIR="${1:-${ROOT_DIR}/mini_testset}"
OUTPUT_DIR="${2:-${ROOT_DIR}/eval_outputs/$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${3:-${VERSE_BENCH_DATA_DIR}}"
MODELS_DIR="${4:-${VERSE_MODELS_DIR}}"

INPUT_DIR="$(abs_path "${INPUT_DIR}")"
OUTPUT_DIR="$(abs_path "${OUTPUT_DIR}")"
DATA_DIR="$(abs_path "${DATA_DIR}")"
MODELS_DIR="$(abs_path "${MODELS_DIR}")"

if [[ ! -x "${VERSE_ENV_PREFIX}/bin/python" ]]; then
  echo "ERROR: Verse-Bench env is missing. Run: bash ${ROOT_DIR}/setup_verse_bench.sh" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: Verse-Bench data dir not found: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "ERROR: models dir not found: ${MODELS_DIR}" >&2
  exit 1
fi

require_verse_model_files "${MODELS_DIR}"

mkdir -p "${OUTPUT_DIR}"
export MODELS_PATH="${MODELS_DIR}"

cd "${ROOT_DIR}"
conda_run python calculate_metrics.py \
  --input_dir "${INPUT_DIR}" \
  --verse_bench_dir "${DATA_DIR}" \
  --models_path "${MODELS_DIR}" \
  | tee "${OUTPUT_DIR}/metrics.log"

echo "Saved log to ${OUTPUT_DIR}/metrics.log"
