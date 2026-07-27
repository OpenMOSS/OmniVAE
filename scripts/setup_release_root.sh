#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Usage: source scripts/setup_release_root.sh [/path/to/omnivae_release]" >&2
    exit 2
fi

if [[ $# -gt 0 ]]; then
    OMNIVAE_RELEASE_ROOT="$1"
elif [[ -z "${OMNIVAE_RELEASE_ROOT:-}" ]]; then
    for candidate in \
        "${REPO_ROOT}/open_source" \
        "${REPO_ROOT}/../open_source" \
        "${REPO_ROOT}/open_source/open_source" \
        "${REPO_ROOT}/../open_source/open_source"; do
        if [[ -d "${candidate}/models" && -d "${candidate}/eval" ]]; then
            OMNIVAE_RELEASE_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi

if [[ -z "${OMNIVAE_RELEASE_ROOT:-}" ]]; then
    echo "Could not infer OMNIVAE_RELEASE_ROOT; pass it explicitly." >&2
    return 2
fi

export OMNIVAE_RELEASE_ROOT
export OPEN_SOURCE_ROOT="${OMNIVAE_RELEASE_ROOT}"
echo "OMNIVAE_RELEASE_ROOT=${OMNIVAE_RELEASE_ROOT}"
