#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/../OmniVAE:${PYTHONPATH:-}"

CKPT_PATH="${CKPT_PATH:-}"
CFG_PATH="${CFG_PATH:-}"
VID_PATH="${VID_PATH:-}"
OFFSET_SEC="${OFFSET_SEC:-0.0}"
V_START_I_SEC="${V_START_I_SEC:-0.0}"
DEVICE="${DEVICE:-cuda:0}"
TOPK="${TOPK:-5}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_path=*) CKPT_PATH="${1#*=}"; shift ;;
    --ckpt_path) CKPT_PATH="$2"; shift 2 ;;
    --cfg_path=*) CFG_PATH="${1#*=}"; shift ;;
    --cfg_path) CFG_PATH="$2"; shift 2 ;;
    --vid_path=*) VID_PATH="${1#*=}"; shift ;;
    --vid_path) VID_PATH="$2"; shift 2 ;;
    --offset_sec=*) OFFSET_SEC="${1#*=}"; shift ;;
    --offset_sec) OFFSET_SEC="$2"; shift 2 ;;
    --v_start_i_sec=*) V_START_I_SEC="${1#*=}"; shift ;;
    --v_start_i_sec) V_START_I_SEC="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --topk=*) TOPK="${1#*=}"; shift ;;
    --topk) TOPK="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${CKPT_PATH}" || -z "${VID_PATH}" ]]; then
  echo "Usage: CKPT_PATH=/path/model.pt VID_PATH=/path/video.mp4 bash scripts/infer/predict_offset.sh [CFG_PATH=/path/cfg.yaml]" >&2
  exit 1
fi

CMD=(python -m omnivae_sync.infer.predict_offset
  --ckpt_path "${CKPT_PATH}"
  --vid_path "${VID_PATH}"
  --offset_sec "${OFFSET_SEC}"
  --v_start_i_sec "${V_START_I_SEC}"
  --device "${DEVICE}"
  --topk "${TOPK}"
)
if [[ -n "${CFG_PATH}" ]]; then
  CMD+=(--cfg_path "${CFG_PATH}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "${OMNIVAE_SYNC_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}"
fi
