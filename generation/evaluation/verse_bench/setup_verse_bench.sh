#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/common.sh"

bootstrap_verse_runtime

echo "Verse-Bench setup complete."
echo "  env:     ${VERSE_ENV_PREFIX}"
echo "  models:  ${VERSE_MODELS_DIR}"
echo "  data:    ${VERSE_BENCH_DATA_DIR}"
echo "  cache:   ${VERSE_CACHE_DIR}"
