#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"

echo "[check] root:   ${ROOT_DIR}"
echo "[check] env:    ${VERSE_ENV_PREFIX}"
echo "[check] models: ${VERSE_MODELS_DIR}"
echo "[check] data:   ${VERSE_BENCH_DATA_DIR}"
echo "[check] cache:  ${VERSE_CACHE_DIR}"

if [[ ! -x "${VERSE_ENV_PREFIX}/bin/python" ]]; then
  echo "ERROR: Verse-Bench env is missing. Run: bash ${ROOT_DIR}/setup_verse_bench.sh" >&2
  exit 1
fi

require_verse_model_files "${VERSE_MODELS_DIR}"

for path in \
  "${VERSE_BENCH_DATA_DIR}/set1" \
  "${VERSE_BENCH_DATA_DIR}/set2/data" \
  "${VERSE_BENCH_DATA_DIR}/set3/data"; do
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: Verse-Bench dataset path is missing: ${path}" >&2
    exit 1
  fi
done

conda_run python -m py_compile \
  "${ROOT_DIR}/calculate_metrics.py" \
  "${ROOT_DIR}/scripts/prefetch_auxiliary.py" \
  "${ROOT_DIR}/syncnet/syncnet_inferencer.py" \
  "${ROOT_DIR}/fd/training/data.py"

conda_run python - <<'PY'
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
import torchvision
import torchaudio
import transformers
import numpy as np
from fd.training.data import tokenizer

assert os.environ["MODELS_PATH"]
assert torch.cuda.is_available(), "CUDA is not available"
tokens = tokenizer("quick Verse-Bench check", tmodel="roberta")
assert tokens["input_ids"].shape[0] == 77

print("[check] torch", torch.__version__, "cuda", torch.version.cuda)
print("[check] torchvision", torchvision.__version__)
print("[check] torchaudio", torchaudio.__version__)
print("[check] transformers", transformers.__version__)
print("[check] numpy", np.__version__)
print("[check] cv2", cv2.__version__)
print("[check] roberta tokenizer OK")
PY

echo "[check] Verse-Bench setup looks ready."
