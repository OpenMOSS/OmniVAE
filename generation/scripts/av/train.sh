#!/usr/bin/env bash
# ============================================================================
# Z-Image joint Text-to-Audio-Video (T2AV) trainer launcher.
#
# Composes a video branch (t2v) + audio branch (t2a) under the bridge
# cross-attention stack, warm-starts each branch from its single-modality
# checkpoint, and trains with dual sigma shift + heterogeneous LR.
#
# Usage:
#   bash scripts/av/train.sh <config.yaml> [--gpus 0,1,2,3] [--debug] [--no_compile]
#                                          [--name <run-name>]
#                                          [--bs <N>] [--grad_accum <N>]
#                                          [--backbone_lr <F>] [--bridge_lr <F>]
#                                          [--bridge_interval <N>]
#                                          [--max_train_steps <N>]
#                                          [--validation_steps <N>]
#                                          [--validate_at <N|N1,N2,...>]   # one-shot validation triggers
#                                          [--pretrained_t2v <path>] [--pretrained_t2a <path>]
#                                          [--video_vae_path <path>] [--audio_vae_path <path>]
#                                          [--muon_shard_across_ranks <N>]   # default: 8 (1=disable, 0=all-ranks)
#                                          [--shift_v <F>] [--shift_a <F>]
#                                          [--bridge_dropout_prob <F>]   # default: yaml (0.1)
#                                          [--resume_from_checkpoint <path|latest|latest_persistent>]
#                                          [-- <extra args forwarded to train_zimage_t2av.py>]
#
# Examples:
#   bash scripts/av/train.sh configs/av/t2av.yaml
#   bash scripts/av/train.sh configs/av/t2av.yaml --gpus 0,1 --debug
#   bash scripts/av/train.sh configs/av/t2av.yaml --no_compile --validation_steps 1   # smoke
#   bash scripts/av/train.sh configs/av/t2av.yaml --bs 1 --grad_accum 4 --bridge_lr 5e-5
#   bash scripts/av/train.sh configs/av/t2av.yaml --pretrained_t2v /path/.../transformer \
#                                                 --pretrained_t2a /path/.../transformer
#   bash scripts/av/train.sh configs/av/t2av.yaml --name t2av_recon \
#                                                 --video_vae_path /path/to/wan22_vae \
#                                                 --audio_vae_path /path/to/dac.pth
#
# Multi-node contract (matches the in-house torchelastic-style launcher):
#   When PET_NNODES>1 the script picks up
#     PET_NNODES / PET_NPROC_PER_NODE / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT
#   automatically. When unset, falls back to single-node multi-GPU.
#
# Optional environment overrides:
#   OMNIGEN_ENV_SH      : optional shell file to source before launching
#   OMNIGEN_CONDA_ENV   : conda env to activate (skipped if empty)
#   OMNIGEN_PYTHON      : explicit python interpreter to use
#   OMNIGEN_DRY_RUN=1   : print the torch distributed command and exit
# ============================================================================

set -euo pipefail
echo "===================== start train (T2AV) ====================="

# ============ Optional environment / conda activation ========================
export HF_HOME="${HF_HOME:-${OMNIGEN_HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}}"

env_sh="${OMNIGEN_ENV_SH:-}"
if [ -n "${env_sh}" ]; then
    if [ -f "${env_sh}" ]; then
        # shellcheck disable=SC1090
        source "${env_sh}"
    else
        echo "Warning: OMNIGEN_ENV_SH=${env_sh} not found, skipping source"
    fi
fi
conda_env="${OMNIGEN_CONDA_ENV:-}"
if [ -n "${conda_env}" ]; then
    if command -v conda >/dev/null 2>&1; then
        conda activate "${conda_env}"
    else
        echo "Warning: conda not on PATH, ignoring OMNIGEN_CONDA_ENV=${conda_env}"
    fi
fi
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
which python || true

# ============ work_dir = repo root (this script lives in <repo>/scripts/av) =
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
work_dir="$(cd "$(dirname "${script_path}")/../.." && pwd)"
cd "${work_dir}"

# ============ Argument parsing ==============================================
config=""
gpu_ids=""
debug_mode=0
no_compile=0
validation_steps=""
validate_at=""
resume_arg=""
run_name=""
batch_size=""
grad_accum=""
backbone_lr=""
bridge_lr=""
bridge_interval=""
max_train_steps=""
pretrained_t2v=""
pretrained_t2a=""
video_vae_path=""
audio_vae_path=""
video_vae_type=""
audio_vae_type=""
vae_branch=""
vae_use_ema=""
muon_shard_across_ranks="8"             # default: shard Muon NS across 8 ranks
shift_v=""
shift_a=""
bridge_dropout_prob=""
passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            gpu_ids="$2"; shift 2 ;;
        --debug)
            debug_mode=1; shift ;;
        --no_compile)
            no_compile=1; shift ;;
        --validation_steps)
            validation_steps="$2"; shift 2 ;;
        --validate_at|--validation_force_steps)
            # Comma-separated list of one-shot validation triggers, e.g.
            # `--validate_at 0` (pre-train-loop sanity check on warm-start
            # weights), `--validate_at 1` (after the first optimizer step),
            # or `--validate_at 0,500,1000`. Composes with the periodic
            # `--validation_steps` cadence.
            validate_at="$2"; shift 2 ;;
        --resume_from_checkpoint)
            resume_arg="--resume_from_checkpoint $2"; shift 2 ;;
        --name)
            run_name="$2"; shift 2 ;;
        --bs|--batch_size|--per_device_batch_size)
            batch_size="$2"; shift 2 ;;
        --grad_accum|--gradient_accumulation_steps)
            grad_accum="$2"; shift 2 ;;
        --backbone_lr)
            backbone_lr="$2"; shift 2 ;;
        --bridge_lr)
            bridge_lr="$2"; shift 2 ;;
        --bridge_interval)
            bridge_interval="$2"; shift 2 ;;
        --max_train_steps)
            max_train_steps="$2"; shift 2 ;;
        --pretrained_t2v)
            pretrained_t2v="$2"; shift 2 ;;
        --pretrained_t2a)
            pretrained_t2a="$2"; shift 2 ;;
        --video_vae_path|--vae_path)
            # Mirror the t2v launcher's `--vae_path` for muscle memory; in
            # joint AV it always means the *video* VAE. Use
            # --audio_vae_path explicitly for the DAC checkpoint.
            video_vae_path="$2"; shift 2 ;;
        --audio_vae_path)
            audio_vae_path="$2"; shift 2 ;;
        --video_vae_type|--vae_type)
            # Mirror t2v `--vae_type`; default joint AV recipe is wan2_2_vae,
            # but switch here to e.g. `omnivae` if you point --video_vae_path
            # at a OmniVAE Trainer_xxxxx/state_dict.pt checkpoint.
            video_vae_type="$2"; shift 2 ;;
        --audio_vae_type)
            audio_vae_type="$2"; shift 2 ;;
        --vae_branch)
            vae_branch="$2"; shift 2 ;;
        --vae_use_ema)
            vae_use_ema="$2"; shift 2 ;;
        --muon_shard_across_ranks|--muon_shard)
            muon_shard_across_ranks="$2"; shift 2 ;;
        --shift_v)
            shift_v="$2"; shift 2 ;;
        --shift_a)
            shift_a="$2"; shift 2 ;;
        --bridge_dropout_prob|--bridge_dropout)
            bridge_dropout_prob="$2"; shift 2 ;;
        --)
            shift
            passthrough+=("$@")
            break ;;
        -h|--help)
            sed -n '2,40p' "${script_path}"
            exit 0 ;;
        *)
            if [ -z "${config}" ]; then
                config="$1"; shift
            else
                passthrough+=("$1"); shift
            fi
            ;;
    esac
done

if [ -z "${config}" ]; then
    echo "Error: missing <config.yaml>"
    echo "Usage: bash $0 <config.yaml> [--gpus 0,1,2,3] [--debug] [--resume_from_checkpoint <path>]"
    exit 1
fi
if [ ! -f "${config}" ]; then
    echo "Error: config file not found: ${config}"
    exit 1
fi

# ============ Tag / launch log dir ==========================================
tag="$(basename "${config}" .yaml)"
tag_name="${tag//\//_}"

yaml_field() {
    awk -v key="$1" '
        /^experiment:/ { in_exp=1; next }
        in_exp && /^[^[:space:]]/ { in_exp=0 }
        in_exp && $1==key":" { sub("^[[:space:]]*"key":[[:space:]]*", "", $0); print; exit }
    ' "${config}"
}

output_dir="$(yaml_field output_dir)"
output_dir="${output_dir%\"}"; output_dir="${output_dir#\"}"
output_dir="${output_dir%\'}"; output_dir="${output_dir#\'}"

yaml_name="$(yaml_field name)"
yaml_name="${yaml_name%\"}"; yaml_name="${yaml_name#\"}"
yaml_name="${yaml_name%\'}"; yaml_name="${yaml_name#\'}"

resolved_name="${run_name:-${yaml_name}}"

if [ -z "${output_dir}" ]; then
    output_dir="exp/${resolved_name:-${tag}}"
elif [ -n "${resolved_name}" ]; then
    leaf="$(basename "${output_dir}")"
    if [ -n "${yaml_name}" ] && [ "${leaf}" = "${yaml_name}" ]; then
        output_dir="$(dirname "${output_dir}")/${resolved_name}"
    else
        output_dir="${output_dir%/}/${resolved_name}"
    fi
fi
launch_log_dir="${output_dir}/launch_logs"
mkdir -p "${launch_log_dir}"

# ============ Distributed env (PET_* injected by the cluster scheduler) =====
if [ -n "${PET_NNODES:-}" ] && [ "${PET_NNODES}" -gt 1 ] 2>/dev/null; then
    dist_mode="distributed"

    MASTER_IP=""
    for i in $(seq 1 120); do
        MASTER_IP=$(getent hosts "${PET_MASTER_ADDR}" 2>/dev/null | awk '{print $1}' | head -n 1 || true)
        [ -n "${MASTER_IP}" ] && break
        echo "Waiting for DNS resolution of ${PET_MASTER_ADDR}... (${i}/120)"
        sleep 2
    done
    if [ -z "${MASTER_IP}" ]; then
        echo "Warning: getent failed, trying nslookup..."
        MASTER_IP=$(nslookup "${PET_MASTER_ADDR}" 2>/dev/null | awk '/^Address: /{print $2; exit}')
    fi
    if [ -z "${MASTER_IP}" ]; then
        echo "Error: cannot resolve PET_MASTER_ADDR=${PET_MASTER_ADDR}"
        exit 1
    fi
    echo "Resolved MASTER_ADDR: ${PET_MASTER_ADDR} -> ${MASTER_IP}"

    export MASTER_ADDR="${MASTER_IP}"
    export MASTER_PORT="${PET_MASTER_PORT}"

    NNODES="${PET_NNODES}"
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
    NODE_RANK="${PET_NODE_RANK}"

    log_stdout="${launch_log_dir}/${tag_name}_node${NODE_RANK}_stdout.log"
    log_stderr="${launch_log_dir}/${tag_name}_node${NODE_RANK}_stderr.log"
else
    dist_mode="standalone"
    NNODES=1
    NODE_RANK=0
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"

    log_stdout="${launch_log_dir}/${tag_name}_stdout.log"
    log_stderr="${launch_log_dir}/${tag_name}_stderr.log"
fi

# ============ GPU / debug overrides =========================================
if [ "${debug_mode}" -eq 1 ]; then
    dist_mode="standalone"
    NNODES=1
    NODE_RANK=0
    if [ -z "${gpu_ids}" ]; then
        gpu_ids="0"
    else
        gpu_ids="$(echo "${gpu_ids}" | cut -d',' -f1)"
    fi
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    NPROC_PER_NODE=1
elif [ "${dist_mode}" = "standalone" ]; then
    if [ -n "${gpu_ids}" ]; then
        export CUDA_VISIBLE_DEVICES="${gpu_ids}"
        NPROC_PER_NODE=$(echo "${gpu_ids}" | tr ',' '\n' | wc -l)
    else
        NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    fi
    if [ "${NPROC_PER_NODE}" -le 0 ]; then
        if [ "${OMNIGEN_DRY_RUN:-0}" = "1" ]; then
            NPROC_PER_NODE=1
        else
            echo "Error: no GPU detected"
            exit 1
        fi
    fi
fi
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ============ Runtime env vars (training defaults) ==========================
export TORCHINDUCTOR_MIX_ORDER_REDUCTION="${TORCHINDUCTOR_MIX_ORDER_REDUCTION:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-default}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ----- Host-RAM hygiene for AV dataloader workers -----
# torchcodec (ffmpeg) inside dataloader workers spawns several decode
# threads, which under glibc's default behaviour gives each worker up
# to ``8 * num_cpu`` malloc arenas. Each arena holds onto pages it
# never returns to the OS, so a long-lived ``persistent_workers=True``
# worker accumulates significant RSS even with no real Python-side
# leak. ``MALLOC_ARENA_MAX=2`` caps that to a tight ceiling and is the
# canonical fix for ML workloads of this shape; combined with the
# ``num_workers=6 / prefetch_factor=2`` knobs in t2av.yaml it brought
# the OOM-killer event rate to zero in our internal runs. Override by
# exporting MALLOC_ARENA_MAX before invoking this script.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# Aggressive ``malloc_trim`` threshold: when free space inside an
# arena exceeds 128 KiB, return it to the OS instead of caching it
# forever. Without this the arenas grow unbounded even with
# ARENA_MAX=2. 128 KiB is conservative; raise toward 1 MiB if you
# observe trimming overhead in profiling (rare in practice).
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

# NCCL: long all-reduce backward needs generous timeouts for first-step
# compile + DAC encode + Wan2.2 VAE encode.
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-30}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600000}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

python_bin="${OMNIGEN_PYTHON:-$(command -v python)}"
python_scripts="train_zimage_t2av.py"

# ============ Banner ========================================================
echo "============================================"
if [ "${dist_mode}" = "distributed" ]; then
    echo "  Z-Image T2AV: multi-node multi-GPU"
else
    echo "  Z-Image T2AV: single-node multi-GPU"
fi
echo "============================================"
echo "Mode             : ${dist_mode}"
echo "Config           : ${config}"
echo "Tag              : ${tag}"
echo "Run name         : ${run_name:-<from yaml>}"
echo "Output dir       : ${output_dir}"
echo "NNODES           : ${NNODES}"
echo "NPROC/NODE       : ${NPROC_PER_NODE}"
echo "NODE_RANK        : ${NODE_RANK}"
echo "WORLD_SIZE       : ${WORLD_SIZE}"
echo "MASTER_ADDR      : ${MASTER_ADDR}"
echo "MASTER_PORT      : ${MASTER_PORT}"
echo "CUDA_VISIBLE     : ${CUDA_VISIBLE_DEVICES:-all}"
echo "Debug            : ${debug_mode}"
echo "No-compile       : ${no_compile}"
echo "Val-steps        : ${validation_steps:-yaml default}"
echo "Validate-at      : ${validate_at:-none (yaml controls)}"
echo "Batch size       : ${batch_size:-yaml default}"
echo "Grad accum       : ${grad_accum:-yaml default}"
echo "Backbone LR      : ${backbone_lr:-yaml default}"
echo "Bridge LR        : ${bridge_lr:-yaml default}"
echo "Bridge interval  : ${bridge_interval:-yaml default}"
echo "Max train steps  : ${max_train_steps:-yaml default}"
echo "Pretrained T2V   : ${pretrained_t2v:-yaml default}"
echo "Pretrained T2A   : ${pretrained_t2a:-yaml default}"
echo "Video VAE path   : ${video_vae_path:-yaml default}"
echo "Audio VAE path   : ${audio_vae_path:-yaml default}"
echo "Video VAE type   : ${video_vae_type:-yaml default}"
echo "Audio VAE type   : ${audio_vae_type:-yaml default}"
echo "VAE branch       : ${vae_branch:-yaml default}"
echo "VAE use_ema      : ${vae_use_ema:-yaml default}"
echo "Muon shard ranks : ${muon_shard_across_ranks:-yaml default}"
echo "Sigma shift V    : ${shift_v:-yaml default}"
echo "Sigma shift A    : ${shift_a:-yaml default}"
echo "Bridge dropout   : ${bridge_dropout_prob:-yaml default}"
echo "Resume           : ${resume_arg:-none (yaml controls)}"
echo "Passthrough      : ${passthrough[*]:-(none)}"
echo "Python           : ${python_bin}"
echo "============================================"

# ============ Launch ========================================================
cmd=("${python_bin}" -m torch.distributed.run
    --nnodes="${NNODES}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    "${python_scripts}"
    --config "${config}"
)
if [ -n "${resume_arg}" ]; then
    cmd+=(${resume_arg})
fi
if [ "${no_compile}" -eq 1 ]; then
    cmd+=(--no_compile)
fi
if [ -n "${validation_steps}" ]; then
    cmd+=(--validation_steps "${validation_steps}")
fi
if [ -n "${validate_at}" ]; then
    cmd+=(--validate_at "${validate_at}")
fi
if [ -n "${run_name}" ]; then
    cmd+=(--name "${run_name}")
fi
if [ -n "${batch_size}" ]; then
    cmd+=(--per_device_batch_size "${batch_size}")
fi
if [ -n "${grad_accum}" ]; then
    cmd+=(--gradient_accumulation_steps "${grad_accum}")
fi
if [ -n "${backbone_lr}" ]; then
    cmd+=(--backbone_lr "${backbone_lr}")
fi
if [ -n "${bridge_lr}" ]; then
    cmd+=(--bridge_lr "${bridge_lr}")
fi
if [ -n "${bridge_interval}" ]; then
    cmd+=(--bridge_interval "${bridge_interval}")
fi
if [ -n "${max_train_steps}" ]; then
    cmd+=(--max_train_steps "${max_train_steps}")
fi
if [ -n "${pretrained_t2v}" ]; then
    cmd+=(--pretrained_t2v "${pretrained_t2v}")
fi
if [ -n "${pretrained_t2a}" ]; then
    cmd+=(--pretrained_t2a "${pretrained_t2a}")
fi
if [ -n "${video_vae_path}" ]; then
    cmd+=(--video_vae_path "${video_vae_path}")
fi
if [ -n "${audio_vae_path}" ]; then
    cmd+=(--audio_vae_path "${audio_vae_path}")
fi
if [ -n "${video_vae_type}" ]; then
    cmd+=(--video_vae_type "${video_vae_type}")
fi
if [ -n "${audio_vae_type}" ]; then
    cmd+=(--audio_vae_type "${audio_vae_type}")
fi
if [ -n "${vae_branch}" ]; then
    cmd+=(--vae_branch "${vae_branch}")
fi
if [ -n "${vae_use_ema}" ]; then
    cmd+=(--vae_use_ema "${vae_use_ema}")
fi
if [ -n "${muon_shard_across_ranks}" ]; then
    cmd+=(--muon_shard_across_ranks "${muon_shard_across_ranks}")
fi
if [ -n "${shift_v}" ]; then
    cmd+=(--shift_v "${shift_v}")
fi
if [ -n "${shift_a}" ]; then
    cmd+=(--shift_a "${shift_a}")
fi
if [ -n "${bridge_dropout_prob}" ]; then
    cmd+=(--bridge_dropout_prob "${bridge_dropout_prob}")
fi
if [ "${#passthrough[@]}" -gt 0 ]; then
    cmd+=("${passthrough[@]}")
fi

printf "Executing:"
printf " %q" "${cmd[@]}"
printf "\n"
if [ "${OMNIGEN_DRY_RUN:-0}" = "1" ]; then
    exit 0
fi
"${cmd[@]}" 2> >(tee -a "${log_stderr}" >&2) | tee -a "${log_stdout}"

echo "Training completed!"
