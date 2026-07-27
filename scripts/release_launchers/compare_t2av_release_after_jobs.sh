#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/../test_output/t2av_release_compare_set3_large}"

bash "${OUTPUT_ROOT}/compare_after_jobs.sh"
