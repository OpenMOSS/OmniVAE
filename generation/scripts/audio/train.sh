#!/usr/bin/env bash
# ============================================================================
# Z-Image audio trainer launcher (single-node and multi-node via torchrun).
#
# Usage:
#   bash scripts/audio/train_zimage.sh <config.yaml> [--gpus 0,1,2,3] [--debug] [--no_compile]
#                                                   [--name <run-name>]
#                                                   [--size <1b|2.5b|5b>]
#                                                   [--bs <N>]
#                                                   [--lr <F>]
#                                                   [--grad_accum <N>]
#                                                   [--validation_steps N]
#                                                   [--resume_from_checkpoint <path|latest|latest_persistent>]
#                                                   [--vae_path <path>] [--audio_vae_path <path>]
#                                                   [--vae_type <name>] [--audio_vae_type <name>]
#                                                   [--vae_branch <video|audio|both>]
#                                                   [--vae_use_ema <true|false>]
#                                                   [-- <extra args forwarded to train_zimage.py>]
# Examples:
#   bash scripts/audio/train_zimage.sh configs/audio/t2a_zimage_dac_fast.yaml
#   bash scripts/audio/train_zimage.sh configs/audio/t2a_zimage_dac_fast.yaml --gpus 0,1
#   bash scripts/audio/train_zimage.sh configs/audio/t2a_zimage_dac_fast.yaml --debug
#   bash scripts/audio/train_zimage.sh configs/audio/t2a_zimage_dac_fast.yaml --no_compile         # skip torch.compile, fast first-step val
#   bash scripts/audio/train_zimage.sh configs/audio/t2a_zimage_dac_fast.yaml --validation_steps 1 # trigger val every step (debug)
#   bash scripts/audio/train_zimage.sh configs/audio/t2a.yaml --name my-tta-run-v3                 # rewrite experiment.name + output_dir leaf + wandb.run_name
#   bash scripts/audio/train_zimage.sh configs/visual/t2i.yaml --size 5b --bs 8 --lr 5e-5 --grad_accum 4
#                                                                                                    # scale up to 5B on the fly (overrides yaml dim/layers/lr/bs)
#   bash scripts/audio/train_zimage.sh configs/audio/t2a_only.yaml \
#       --audio_vae_path /path/to/OmniVAE/Trainer_00010000/state_dict.pt \
#       --audio_vae_type omnivae --vae_branch both                                                    # swap audio VAE to a OmniVAE ckpt without editing yaml
#   bash scripts/audio/train_zimage.sh configs/visual/t2i.yaml \
#       --vae_path /path/to/OmniVAE/Trainer_00010000/state_dict.pt \
#       --vae_type omnivae --vae_use_ema false                                                        # swap visual VAE to a OmniVAE ckpt
#
# Multi-node contract (matches the in-house torchelastic-style launcher):
#   When PET_NNODES>1 the script picks up
#     PET_NNODES / PET_NPROC_PER_NODE / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT
#   automatically. When unset, the script falls back to single-node multi-GPU.
#
# Optional environment overrides:
#   OMNIGEN_ENV_SH      : optional shell file to source before launching
#   OMNIGEN_CONDA_ENV   : conda env to activate (skipped if empty)
#   OMNIGEN_PYTHON      : explicit python interpreter to use
#   OMNIGEN_DRY_RUN=1   : print the torch distributed command and exit
# ============================================================================

set -euo pipefail
echo "===================== start train ====================="

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
# ============ work_dir = repo root (this script lives in <repo>/scripts/audio) =
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
work_dir="$(cd "$(dirname "${script_path}")/../.." && pwd)"
cd "${work_dir}"

# ============ Argument parsing ==============================================
config=""
gpu_ids=""
debug_mode=0
no_compile=0
validation_steps=""
resume_arg=""
run_name=""
size=""
batch_size=""
learning_rate=""
grad_accum=""
vae_path=""
audio_vae_path=""
vae_type=""
audio_vae_type=""
vae_branch=""
vae_use_ema=""
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
        --resume_from_checkpoint)
            resume_arg="--resume_from_checkpoint $2"; shift 2 ;;
        --name)
            run_name="$2"; shift 2 ;;
        --size)
            size="$2"; shift 2 ;;
        --bs|--batch_size|--per_device_batch_size)
            batch_size="$2"; shift 2 ;;
        --lr|--learning_rate)
            learning_rate="$2"; shift 2 ;;
        --grad_accum|--gradient_accumulation_steps)
            grad_accum="$2"; shift 2 ;;
        --vae_path|--vae_model_name_or_path)
            vae_path="$2"; shift 2 ;;
        --audio_vae_path|--audio_vae_model_path)
            audio_vae_path="$2"; shift 2 ;;
        --vae_type)
            vae_type="$2"; shift 2 ;;
        --audio_vae_type)
            audio_vae_type="$2"; shift 2 ;;
        --vae_branch)
            vae_branch="$2"; shift 2 ;;
        --vae_use_ema)
            vae_use_ema="$2"; shift 2 ;;
        --)
            shift
            passthrough+=("$@")
            break ;;
        -h|--help)
            sed -n '2,32p' "${script_path}"
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

# Read experiment.output_dir and experiment.name from yaml. We mirror the
# python-side _sync_run_identity here so launcher logs land in the same dir
# the trainer will eventually use, regardless of which yaml convention the
# user picked (full-path vs base-only).
yaml_field() {
    # $1 = field name under experiment:
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

# Resolved name precedence: --name CLI > experiment.name in yaml.
resolved_name="${run_name:-${yaml_name}}"

if [ -z "${output_dir}" ]; then
    output_dir="exp/${resolved_name:-${tag}}"
elif [ -n "${resolved_name}" ]; then
    # Detect the yaml convention via the yaml's declared experiment.name:
    #   leaf == yaml_name -> legacy full-path style; replace leaf
    #   else              -> base-only style; append leaf
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
# Inductor: silence torch._inductor reduction reordering that interferes with
# z-image's reduce-overhead compilation path.
export TORCHINDUCTOR_MIX_ORDER_REDUCTION="${TORCHINDUCTOR_MIX_ORDER_REDUCTION:-0}"

export PYTHONWARNINGS="${PYTHONWARNINGS:-default}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NCCL: same defaults as the in-house template; long all-reduce backward needs
# generous timeouts for first-step compile + DAC encode.
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-30}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600000}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

python_bin="${OMNIGEN_PYTHON:-$(command -v python)}"
python_scripts="train_zimage.py"

# ============ Banner ========================================================
echo "============================================"
if [ "${dist_mode}" = "distributed" ]; then
    echo "  Z-Image audio: multi-node multi-GPU"
else
    echo "  Z-Image audio: single-node multi-GPU"
fi
echo "============================================"
echo "Mode         : ${dist_mode}"
echo "Config       : ${config}"
echo "Tag          : ${tag}"
echo "Run name     : ${run_name:-<from yaml>}"
echo "Output dir   : ${output_dir}"
echo "NNODES       : ${NNODES}"
echo "NPROC/NODE   : ${NPROC_PER_NODE}"
echo "NODE_RANK    : ${NODE_RANK}"
echo "WORLD_SIZE   : ${WORLD_SIZE}"
echo "MASTER_ADDR  : ${MASTER_ADDR}"
echo "MASTER_PORT  : ${MASTER_PORT}"
echo "CUDA_VISIBLE : ${CUDA_VISIBLE_DEVICES:-all}"
echo "Debug        : ${debug_mode}"
echo "No-compile   : ${no_compile}"
echo "Val-steps    : ${validation_steps:-yaml default}"
echo "Model size   : ${size:-yaml default}"
echo "Batch size   : ${batch_size:-yaml default}"
echo "Learning rate: ${learning_rate:-yaml default}"
echo "Grad accum   : ${grad_accum:-yaml default}"
echo "VAE path     : ${vae_path:-yaml default}"
echo "Audio VAE    : ${audio_vae_path:-yaml default}"
echo "VAE type     : ${vae_type:-yaml default}"
echo "Audio VAE type: ${audio_vae_type:-yaml default}"
echo "VAE branch   : ${vae_branch:-yaml default}"
echo "VAE use_ema  : ${vae_use_ema:-yaml default}"
echo "Resume       : ${resume_arg:-none (yaml controls)}"
echo "Passthrough  : ${passthrough[*]:-(none)}"
echo "Python       : ${python_bin}"
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
if [ -n "${run_name}" ]; then
    cmd+=(--name "${run_name}")
fi
if [ -n "${size}" ]; then
    cmd+=(--size "${size}")
fi
if [ -n "${batch_size}" ]; then
    cmd+=(--per_device_batch_size "${batch_size}")
fi
if [ -n "${learning_rate}" ]; then
    cmd+=(--learning_rate "${learning_rate}")
fi
if [ -n "${grad_accum}" ]; then
    cmd+=(--gradient_accumulation_steps "${grad_accum}")
fi
if [ -n "${vae_path}" ]; then
    cmd+=(--vae_path "${vae_path}")
fi
if [ -n "${audio_vae_path}" ]; then
    cmd+=(--audio_vae_path "${audio_vae_path}")
fi
if [ -n "${vae_type}" ]; then
    cmd+=(--vae_type "${vae_type}")
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
