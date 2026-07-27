#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/sync_24fps_nonspeech_vae.yaml}"
EXP_ROOT="${OMNIVAE_SYNC_EXP_ROOT:-${ROOT_DIR}/outputs}"
LOGDIR="${LOGDIR:-${EXP_ROOT}/sync_models}"

AV_VAE_CONFIG="${AV_VAE_CONFIG:-~}"
VIDEO_AV_VAE_CONFIG="${VIDEO_AV_VAE_CONFIG:-~}"
AUDIO_AV_VAE_CONFIG="${AUDIO_AV_VAE_CONFIG:-~}"
CONTRASTIVE_HEAD_CONFIG="${CONTRASTIVE_HEAD_CONFIG:-~}"
VAE_PRETRAINED="${VAE_PRETRAINED:-~}"
VIDEO_VAE_PRETRAINED="${VIDEO_VAE_PRETRAINED:-~}"
AUDIO_VAE_PRETRAINED="${AUDIO_VAE_PRETRAINED:-~}"
CKPT_PATH="${CKPT_PATH:-~}"

FPS="${FPS:-24}"
CROP_LEN_SEC="${CROP_LEN_SEC:-6}"
SKIP_TEMPORAL_POOL="${SKIP_TEMPORAL_POOL:-False}"
AUDIO_MERGE="${AUDIO_MERGE:-1}"
LR="${LR:-2e-6}"
MAX_OFF_SEC="${MAX_OFF_SEC:-2}"
SUFFIX="${SUFFIX:-}"
NPROC="${NPROC:-}"
AUTO_RESUME="${AUTO_RESUME:-True}"
FREEZE_CONTRASTIVE_HEAD="${FREEZE_CONTRASTIVE_HEAD:-True}"
INIT_CONTRASTIVE_HEAD_FROM_CKPT="${INIT_CONTRASTIVE_HEAD_FROM_CKPT:-True}"
LOAD_CONTRASTIVE_HEAD_FROM_EXTERNAL="${LOAD_CONTRASTIVE_HEAD_FROM_EXTERNAL:-True}"

CKPT_PATH_USER_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config=*) CONFIG="${1#*=}"; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --av_vae_config=*) AV_VAE_CONFIG="${1#*=}"; shift ;;
    --av_vae_config) AV_VAE_CONFIG="$2"; shift 2 ;;
    --video_av_vae_config=*) VIDEO_AV_VAE_CONFIG="${1#*=}"; shift ;;
    --video_av_vae_config) VIDEO_AV_VAE_CONFIG="$2"; shift 2 ;;
    --audio_av_vae_config=*) AUDIO_AV_VAE_CONFIG="${1#*=}"; shift ;;
    --audio_av_vae_config) AUDIO_AV_VAE_CONFIG="$2"; shift 2 ;;
    --contrastive_head_config=*) CONTRASTIVE_HEAD_CONFIG="${1#*=}"; shift ;;
    --contrastive_head_config) CONTRASTIVE_HEAD_CONFIG="$2"; shift 2 ;;
    --vae_pretrained=*) VAE_PRETRAINED="${1#*=}"; shift ;;
    --vae_pretrained) VAE_PRETRAINED="$2"; shift 2 ;;
    --video_vae_pretrained=*) VIDEO_VAE_PRETRAINED="${1#*=}"; shift ;;
    --video_vae_pretrained) VIDEO_VAE_PRETRAINED="$2"; shift 2 ;;
    --audio_vae_pretrained=*) AUDIO_VAE_PRETRAINED="${1#*=}"; shift ;;
    --audio_vae_pretrained) AUDIO_VAE_PRETRAINED="$2"; shift 2 ;;
    --ckpt_path=*) CKPT_PATH="${1#*=}"; CKPT_PATH_USER_SET=1; shift ;;
    --ckpt_path) CKPT_PATH="$2"; CKPT_PATH_USER_SET=1; shift 2 ;;
    --fps=*) FPS="${1#*=}"; shift ;;
    --fps) FPS="$2"; shift 2 ;;
    --crop_len_sec=*) CROP_LEN_SEC="${1#*=}"; shift ;;
    --crop_len_sec) CROP_LEN_SEC="$2"; shift 2 ;;
    --skip_temporal_pool=*) SKIP_TEMPORAL_POOL="${1#*=}"; shift ;;
    --skip_temporal_pool) SKIP_TEMPORAL_POOL="$2"; shift 2 ;;
    --audio_merge=*) AUDIO_MERGE="${1#*=}"; shift ;;
    --audio_merge) AUDIO_MERGE="$2"; shift 2 ;;
    --lr=*) LR="${1#*=}"; shift ;;
    --lr) LR="$2"; shift 2 ;;
    --max_off_sec=*) MAX_OFF_SEC="${1#*=}"; shift ;;
    --max_off_sec) MAX_OFF_SEC="$2"; shift 2 ;;
    --suffix=*) SUFFIX="${1#*=}"; shift ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    --tail_more=*) echo "Ignoring deprecated $1; OmniVAE-Sync now trains on crop_len_sec only."; shift ;;
    --tail_more) echo "Ignoring deprecated --tail_more $2; OmniVAE-Sync now trains on crop_len_sec only."; shift 2 ;;
    --tail_time=*) echo "Ignoring deprecated $1; OmniVAE-Sync now trains on crop_len_sec only."; shift ;;
    --tail_time) echo "Ignoring deprecated --tail_time $2; OmniVAE-Sync now trains on crop_len_sec only."; shift 2 ;;
    --nproc=*) NPROC="${1#*=}"; shift ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --auto_resume=*) AUTO_RESUME="${1#*=}"; shift ;;
    --auto_resume) AUTO_RESUME="$2"; shift 2 ;;
    --freeze_contrastive_head=*) FREEZE_CONTRASTIVE_HEAD="${1#*=}"; shift ;;
    --freeze_contrastive_head) FREEZE_CONTRASTIVE_HEAD="$2"; shift 2 ;;
    --init_contrastive_head_from_ckpt=*) INIT_CONTRASTIVE_HEAD_FROM_CKPT="${1#*=}"; shift ;;
    --init_contrastive_head_from_ckpt) INIT_CONTRASTIVE_HEAD_FROM_CKPT="$2"; shift 2 ;;
    --load_contrastive_head_from_external=*) LOAD_CONTRASTIVE_HEAD_FROM_EXTERNAL="${1#*=}"; shift ;;
    --load_contrastive_head_from_external) LOAD_CONTRASTIVE_HEAD_FROM_EXTERNAL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${NPROC}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L 2>/dev/null | wc -l)"
    NPROC="${NPROC:-1}"
  else
    NPROC=1
  fi
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((RANDOM % 10000 + 20000))}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-30}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/../OmniVAE:${PYTHONPATH:-}"

RUN_NAME="${FPS}fps"
if [[ "${SKIP_TEMPORAL_POOL}" == "True" || "${SKIP_TEMPORAL_POOL}" == "true" ]]; then
  RUN_NAME="${RUN_NAME}_npool"
else
  RUN_NAME="${RUN_NAME}_pool"
fi
CROP_TAG="${CROP_LEN_SEC//./p}"
RUN_NAME="${RUN_NAME}_crop${CROP_TAG}s"
if [[ "${AUDIO_MERGE}" -gt 1 ]]; then
  RUN_NAME="${RUN_NAME}_amerge${AUDIO_MERGE}"
fi
RUN_NAME="${RUN_NAME}_lr${LR//-/m}"
if [[ "${MAX_OFF_SEC}" != "2" ]]; then
  RUN_NAME="${RUN_NAME}_off${MAX_OFF_SEC//./p}s"
fi
TAIL_EXTRA_SEC=0
if [[ "${VIDEO_VAE_PRETRAINED}" != "~" ]]; then RUN_NAME="${RUN_NAME}_vext"; fi
if [[ "${AUDIO_VAE_PRETRAINED}" != "~" ]]; then RUN_NAME="${RUN_NAME}_aext"; fi
if [[ "${FREEZE_CONTRASTIVE_HEAD}" == "False" || "${FREEZE_CONTRASTIVE_HEAD}" == "false" ]]; then
  RUN_NAME="${RUN_NAME}_ctun"
fi
if [[ "${INIT_CONTRASTIVE_HEAD_FROM_CKPT}" == "False" || "${INIT_CONTRASTIVE_HEAD_FROM_CKPT}" == "false" ]]; then
  RUN_NAME="${RUN_NAME}_ctrand"
fi
if [[ "${CONTRASTIVE_HEAD_CONFIG}" != "~" ]]; then RUN_NAME="${RUN_NAME}_ctcfg"; fi
if [[ -n "${SUFFIX}" ]]; then RUN_NAME="${RUN_NAME}_${SUFFIX}"; fi

RESUME_FLAG=False
if [[ "${CKPT_PATH_USER_SET}" == "1" && "${CKPT_PATH}" != "~" ]]; then
  RESUME_FLAG=True
elif [[ "${AUTO_RESUME}" == "True" || "${AUTO_RESUME}" == "true" ]]; then
  shopt -s nullglob
  CANDIDATE_DIRS=("${LOGDIR}/${RUN_NAME}"*/)
  shopt -u nullglob
  if [[ ${#CANDIDATE_DIRS[@]} -gt 0 ]]; then
    TARGET_DIR="$(ls -dt "${CANDIDATE_DIRS[@]}" | head -n1)"
    TARGET_DIR="${TARGET_DIR%/}"
    REAL_RUN_NAME="$(basename "${TARGET_DIR}")"
    if [[ -f "${TARGET_DIR}/${REAL_RUN_NAME}_latest.pt" ]]; then
      CKPT_PATH="${TARGET_DIR}/${REAL_RUN_NAME}_latest.pt"
      RESUME_FLAG=True
    elif [[ -f "${TARGET_DIR}/${REAL_RUN_NAME}.pt" ]]; then
      CKPT_PATH="${TARGET_DIR}/${REAL_RUN_NAME}.pt"
      RESUME_FLAG=True
    fi
  fi
fi

VIDEO_LATENT_LEN="$(awk -v f="${FPS}" -v c="${CROP_LEN_SEC}" 'BEGIN { print int(f * c / 4) }')"
if [[ "${SKIP_TEMPORAL_POOL}" == "True" || "${SKIP_TEMPORAL_POOL}" == "true" ]]; then
  AUDIO_TOKENS="$(awk -v c="${CROP_LEN_SEC}" -v m="${AUDIO_MERGE}" 'BEGIN { print int(250 * c / 5 / m + 0.5) }')"
else
  AUDIO_TOKENS="${VIDEO_LATENT_LEN}"
fi
POS_EMB_SIZE="$((2 + VIDEO_LATENT_LEN + AUDIO_TOKENS + 100))"
NOW="$(date +"%Y-%m-%dT%H-%M-%S")"

CMD=(
  torchrun
  --nnodes "${NNODES:-1}"
  --nproc-per-node "${NPROC}"
  --node_rank "${NODE_RANK:-0}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  -m omnivae_sync.train
  start_time="${NOW}"
  run_name="${RUN_NAME}"
  config="${CONFIG}"
  logging.logdir="${LOGDIR}"
  logging.use_wandb=False
  training.patience=100
  training.base_learning_rate="${LR}"
  av_vae_config="${AV_VAE_CONFIG}"
  video_av_vae_config="${VIDEO_AV_VAE_CONFIG}"
  audio_av_vae_config="${AUDIO_AV_VAE_CONFIG}"
  contrastive_head_config="${CONTRASTIVE_HEAD_CONFIG}"
  vae_pretrained="${VAE_PRETRAINED}"
  video_vae_pretrained="${VIDEO_VAE_PRETRAINED}"
  audio_vae_pretrained="${AUDIO_VAE_PRETRAINED}"
  freeze_contrastive_head="${FREEZE_CONTRASTIVE_HEAD}"
  init_contrastive_head_from_ckpt="${INIT_CONTRASTIVE_HEAD_FROM_CKPT}"
  load_contrastive_head_from_external="${LOAD_CONTRASTIVE_HEAD_FROM_EXTERNAL}"
  ckpt_path="${CKPT_PATH}"
  training.resume="${RESUME_FLAG}"
  data.crop_len_sec="${CROP_LEN_SEC}"
  data.target_fps="${FPS}"
  model.params.source_vfps="${FPS}"
  model.params.model_target_vfps="${FPS}"
  model.params.skip_temporal_pool="${SKIP_TEMPORAL_POOL}"
  model.params.audio_merge_factor="${AUDIO_MERGE}"
  data.max_off_sec="${MAX_OFF_SEC}"
  data.tail_extra_sec="${TAIL_EXTRA_SEC}"
  model.params.tail_extra_sec="${TAIL_EXTRA_SEC}"
  "model.params.transformer.params.pos_emb_cfg.params.block_shape=[${POS_EMB_SIZE}]"
)

echo "Run name: ${RUN_NAME}"
echo "Log dir : ${LOGDIR}"
echo "NPROC   : ${NPROC}"
echo "Resume  : ${RESUME_FLAG}"
echo "CKPT    : ${CKPT_PATH}"

if [[ "${OMNIVAE_SYNC_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}"
fi
