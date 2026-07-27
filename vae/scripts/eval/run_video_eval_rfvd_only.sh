#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export OMNIVAE_REPO_ROOT="${OMNIVAE_REPO_ROOT:-${REPO_ROOT}}"
export OMNIVAE_EXP_ROOT="${OMNIVAE_EXP_ROOT:-${REPO_ROOT}/exp}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RELEASE_ROOT="${OMNIVAE_RELEASE_ROOT:-${OPEN_SOURCE_ROOT:-}}"
if [[ -z "${RELEASE_ROOT}" ]]; then
  for candidate in \
    "${REPO_ROOT}/../open_source" \
    "${REPO_ROOT}/../../open_source" \
    "${REPO_ROOT}/../open_source/open_source" \
    "${REPO_ROOT}/../../open_source/open_source"; do
    if [[ -d "${candidate}/models" && -d "${candidate}/eval" ]]; then
      RELEASE_ROOT="$(cd "${candidate}" && pwd)"
      break
    fi
  done
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RFVD_OUTPUT_ROOTS="${RFVD_OUTPUT_ROOTS:-${OMNIVAE_EXP_ROOT}/eval/video_recon}"
RFVD_BATCH_SIZE="${RFVD_BATCH_SIZE:-2}"
RFVD_GROUP_JSONLS="${RFVD_GROUP_JSONLS:-}"

args=(-m omnivae.eval.reconstruction.video_rfvd_only --batch_size "${RFVD_BATCH_SIZE}")

IFS=':' read -r -a roots <<< "${RFVD_OUTPUT_ROOTS}"
for root in "${roots[@]}"; do
  [[ -n "${root}" ]] && args+=(--output_root "${root}")
done

if [[ -n "${RFVD_GROUP_JSONLS}" ]]; then
  IFS=':' read -r -a group_jsonls <<< "${RFVD_GROUP_JSONLS}"
  for jsonl in "${group_jsonls[@]}"; do
    [[ -n "${jsonl}" ]] && args+=(--group_jsonl "${jsonl}")
  done
fi

if [[ -n "${MAX_TASKS:-}" ]]; then
  args+=(--max_tasks "${MAX_TASKS}")
fi
if [[ -n "${MAX_VIDEOS:-}" ]]; then
  args+=(--max_videos "${MAX_VIDEOS}")
fi
if [[ -z "${I3D_TORCHSCRIPT_PT:-}" && -n "${RELEASE_ROOT}" && -f "${RELEASE_ROOT}/eval/models/vae/fvd/i3d_torchscript.pt" ]]; then
  I3D_TORCHSCRIPT_PT="${RELEASE_ROOT}/eval/models/vae/fvd/i3d_torchscript.pt"
fi
if [[ -n "${I3D_TORCHSCRIPT_PT:-}" ]]; then
  args+=(--i3d_torchscript_pt "${I3D_TORCHSCRIPT_PT}")
fi
if [[ "${OMNIVAE_DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run)
fi

exec "${PYTHON_BIN}" "${args[@]}" "$@"
