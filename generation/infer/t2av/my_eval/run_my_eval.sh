#!/usr/bin/env bash
# Multi-node / multi-GPU launcher for the distributed T2AV evaluation toolkit.
#
# By default we run inside the verse-bench conda env produced by
#   generation/evaluation/verse_bench/setup_verse_bench.sh
# That env already ships all the heavy dependencies we need -- Synchformer,
# ImageBind, audiobox-aesthetics,
# aesthetic-predictor-v2-5, MANIQA, LAION-CLAP, pyiqa, onnxruntime-gpu,
# audiotools / pyloudnorm, etc. -- plus its MODELS_PATH already points at the
# weight tree we depend on.
#
# Behaviour:
#
#   1. Source Verse-Bench/scripts/common.sh to import VERSE_ENV_PREFIX,
#      MODELS_PATH, HF_HOME, TORCH_HOME and friends.
#   2. Verify ${VERSE_ENV_PREFIX}/bin/python exists; prepend it to PATH.
#   3. Resolve PET_MASTER_ADDR to an IP (with retries), expose MASTER_ADDR/PORT.
#   4. NNODES / NPROC_PER_NODE / NODE_RANK fall back to local GPU count when
#      PET_NNODES is unset (standalone single-node mode).
#   5. Invoke ``${VERSE_ENV_PREFIX}/bin/python -m torch.distributed.run`` so we
#      never accidentally pick up a different system python.
#
# Override knobs (env vars):
#
#   MY_EVAL_VERSE_BENCH_ROOT   absolute path to Verse-Bench/ (default: relative
#                              to this script).
#   MY_EVAL_PYTHON             absolute path to a python binary; takes
#                              precedence over the verse-bench env.
#   SKIP_VERSE_COMMON=1        do not source common.sh (assume caller already
#                              activated an env).
set -euo pipefail

echo "===================== start my_eval (T2AV) ====================="

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
my_eval_dir="$(cd "$(dirname "${script_path}")" && pwd)"
# my_eval/ -> t2av/ -> infer/ -> generation/ -> OmniVAE/  (4 levels up)
project_root="$(cd "${my_eval_dir}/../../../.." && pwd)"
cd "${project_root}"

MY_EVAL_VERSE_BENCH_ROOT="${MY_EVAL_VERSE_BENCH_ROOT:-${project_root}/generation/evaluation/verse_bench}"
_USER_MODELS_PATH_SET="${MODELS_PATH+x}"
_USER_TORCH_HOME_SET="${TORCH_HOME+x}"
_USER_MY_EVAL_VERSE_MODELS_SET="${MY_EVAL_VERSE_MODELS+x}"

# ----------------------------------------------------------------------
# Argument forwarding -- run_my_eval.py gets all "real" flags.
# ----------------------------------------------------------------------
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-verse-common) export SKIP_VERSE_COMMON=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: bash run_my_eval.sh [options]

Options forwarded to run_my_eval.py:
  --sample-root DIR              [required]
  --eval-output-root DIR         [required]
  --experiments NAME [NAME ...]
  --steps "S1 S2"
  --cfg {dual,simple,both}       default: dual
  --kinds av_sync_imagebind,...  default: core metrics (excludes audio_amplitude)
  --extra-kinds audio_amplitude append optional kind(s) to selected/default set
  --optional-metrics AV-Align,LSE-D
                                 optional submetrics disabled by default
  --limit N                      default: 0 (all samples)
  --skip-completed
  --scan-workers N               parallel target-completion scan workers
  --build-manifest-script PATH
  --valid-jsonl PATH
  --input-jsonl PATH            alias for --valid-jsonl
  --max-ckpt-per-experiment N
  --dispatch-mode {kind-major-reuse,subtask,data-parallel,sample-major}
  --sample-major-chunk-size N  records per microbatch for sample-major mode
  --watch                        Poll --sample-root continuously for new targets
  --watch-interval SECONDS       default: 180
  --watch-min-age SECONDS        wait until newest sample file is this old
  --watch-skip-existing          only evaluate targets discovered after startup
  --watch-max-passes N           debug/testing; 0 means never stop

Local options:
  --skip-verse-common            Do not source Verse-Bench/scripts/common.sh
                                 (assume the caller already activated a python
                                 env with all dependencies installed).

Environment overrides:
  MY_EVAL_VERSE_BENCH_ROOT       optional absolute path to a Verse-Bench tree.
  MY_EVAL_PYTHON                 absolute path to a python binary; takes
                                 precedence over the verse-bench env
  -h, --help                     Show this help
EOF
            exit 0 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

# ----------------------------------------------------------------------
# Python environment: by default reuse the Verse-Bench conda env so this
# script lines up with the bundled Verse-Bench metric dependencies.
# ----------------------------------------------------------------------
PYTHON_BIN_RESOLVED=""

if [[ -n "${MY_EVAL_PYTHON:-}" ]]; then
    if [[ ! -x "${MY_EVAL_PYTHON}" ]]; then
        echo "ERROR: MY_EVAL_PYTHON=${MY_EVAL_PYTHON} is not executable" >&2
        exit 1
    fi
    echo "[my_eval] using explicit MY_EVAL_PYTHON=${MY_EVAL_PYTHON}"
    PYTHON_BIN_RESOLVED="${MY_EVAL_PYTHON}"
    export PATH="$(dirname "${PYTHON_BIN_RESOLVED}"):${PATH}"
elif [[ "${SKIP_VERSE_COMMON:-0}" == "1" ]]; then
    echo "[my_eval] SKIP_VERSE_COMMON=1: not sourcing common.sh; relying on current shell"
    PYTHON_BIN_RESOLVED="$(command -v python || true)"
    if [[ -z "${PYTHON_BIN_RESOLVED}" ]]; then
        echo "ERROR: no python on PATH; either drop SKIP_VERSE_COMMON or set MY_EVAL_PYTHON" >&2
        exit 1
    fi
else
    if [[ ! -f "${MY_EVAL_VERSE_BENCH_ROOT}/scripts/common.sh" ]]; then
        echo "[my_eval] Verse-Bench common.sh not found; using current python."
        echo "[my_eval] Set MY_EVAL_VERSE_BENCH_ROOT to reuse a Verse-Bench env."
        PYTHON_BIN_RESOLVED="$(command -v python || true)"
        if [[ -z "${PYTHON_BIN_RESOLVED}" ]]; then
            echo "ERROR: no python on PATH; set MY_EVAL_PYTHON" >&2
            exit 1
        fi
    else
        echo "[my_eval] sourcing ${MY_EVAL_VERSE_BENCH_ROOT}/scripts/common.sh"
        # shellcheck disable=SC1090,SC1091
        source "${MY_EVAL_VERSE_BENCH_ROOT}/scripts/common.sh"
        if [[ ! -x "${VERSE_ENV_PREFIX}/bin/python" ]]; then
            echo "ERROR: verse-bench env missing at ${VERSE_ENV_PREFIX}" >&2
            echo "       Run: bash ${MY_EVAL_VERSE_BENCH_ROOT}/setup_verse_bench.sh" >&2
            echo "       Or pass --skip-verse-common after activating your own env." >&2
            exit 1
        fi
        PYTHON_BIN_RESOLVED="${VERSE_ENV_PREFIX}/bin/python"
        export PATH="${VERSE_ENV_PREFIX}/bin:${PATH}"
        # common.sh prepends ROOT_DIR to PYTHONPATH; that exposes optional
        # Verse-Bench helper packages when users choose to provide them.
    fi
fi

OPEN_SOURCE_ROOT="${OMNIVAE_RELEASE_ROOT:-${OPEN_SOURCE_ROOT:-}}"
if [[ -z "${OPEN_SOURCE_ROOT}" ]]; then
    for candidate in \
        "${project_root}/open_source" \
        "${project_root}/../open_source" \
        "${project_root}/open_source/open_source" \
        "${project_root}/../open_source/open_source"; do
        if [[ -d "${candidate}/models" && -d "${candidate}/eval" ]]; then
            OPEN_SOURCE_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
if [[ -z "${OPEN_SOURCE_ROOT}" ]]; then
    OPEN_SOURCE_ROOT="${project_root}/open_source"
fi
OPEN_SOURCE_T2AV_EVAL_MODELS="${MY_EVAL_OPEN_SOURCE_T2AV_MODELS:-${OPEN_SOURCE_ROOT}/eval/models/t2av}"
if [[ ! -d "${OPEN_SOURCE_T2AV_EVAL_MODELS}" && -d "${project_root}/open_source/eval_models/t2av" ]]; then
    OPEN_SOURCE_T2AV_EVAL_MODELS="${project_root}/open_source/eval_models/t2av"
fi
if [[ -d "${OPEN_SOURCE_T2AV_EVAL_MODELS}" ]]; then
    echo "[my_eval] using open-source T2AV eval models: ${OPEN_SOURCE_T2AV_EVAL_MODELS}"

    if [[ -z "${_USER_MODELS_PATH_SET}" ]]; then
        if [[ -n "${_USER_MY_EVAL_VERSE_MODELS_SET}" ]]; then
            export MODELS_PATH="${MY_EVAL_VERSE_MODELS}"
        else
            export MODELS_PATH="${OPEN_SOURCE_T2AV_EVAL_MODELS}/verse_models"
        fi
    fi
    if [[ -z "${_USER_MY_EVAL_VERSE_MODELS_SET}" ]]; then
        export MY_EVAL_VERSE_MODELS="${MODELS_PATH:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/verse_models}"
    fi
    if [[ -z "${_USER_TORCH_HOME_SET}" ]]; then
        export TORCH_HOME="${OPEN_SOURCE_T2AV_EVAL_MODELS}/torch_cache"
    fi

    export MY_EVAL_PE_AV_MODEL_DIR="${MY_EVAL_PE_AV_MODEL_DIR:-${PE_AV_MODEL_DIR:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/pe_av_alignment/facebook_pe_av_large}}"
    export MY_EVAL_SYNCHFORMER_CKPT="${MY_EVAL_SYNCHFORMER_CKPT:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/verse_models/24-01-04T16-39-21.pt}"
    export MY_EVAL_AUDIOBOX_CKPT="${MY_EVAL_AUDIOBOX_CKPT:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/verse_models/audiobox-aesthetics/checkpoint.pt}"
    export MY_EVAL_AUDIO_IS_CLAP_DIR="${MY_EVAL_AUDIO_IS_CLAP_DIR:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/audio_quality_metrics/audio_is_clap}"
    export MY_EVAL_PANNS_HOME="${MY_EVAL_PANNS_HOME:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/audio_quality_metrics/audio_is_clap/pann_home}"
    export MY_EVAL_DNSMOS_DIR="${MY_EVAL_DNSMOS_DIR:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/audio_quality_metrics/dnsmos}"
    export MY_EVAL_SYNCNET_MODEL="${MY_EVAL_SYNCNET_MODEL:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/lip_sync/syncnet_v2.model}"
    export MY_EVAL_S3FD_WEIGHT="${MY_EVAL_S3FD_WEIGHT:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/lip_sync/s3fd_face_detector/sfd_face.pth}"
fi

echo "[my_eval] python: ${PYTHON_BIN_RESOLVED}"
echo "[my_eval] HF_HOME=${HF_HOME:-<unset>}"
echo "[my_eval] TORCH_HOME=${TORCH_HOME:-<unset>}"
echo "[my_eval] MODELS_PATH=${MODELS_PATH:-<unset>}"
echo "[my_eval] MY_EVAL_VERSE_MODELS=${MY_EVAL_VERSE_MODELS:-<unset>}"
echo "[my_eval] MY_EVAL_PE_AV_MODEL_DIR=${MY_EVAL_PE_AV_MODEL_DIR:-<unset>}"
echo "[my_eval] MY_EVAL_AUDIO_IS_CLAP_DIR=${MY_EVAL_AUDIO_IS_CLAP_DIR:-<unset>}"
echo "[my_eval] MY_EVAL_DNSMOS_DIR=${MY_EVAL_DNSMOS_DIR:-<unset>}"

_arg_value() {
    local flag="$1"
    local i
    for ((i = 0; i < ${#PASS_ARGS[@]}; i++)); do
        case "${PASS_ARGS[$i]}" in
            "${flag}")
                if (( i + 1 < ${#PASS_ARGS[@]} )); then
                    echo "${PASS_ARGS[$((i + 1))]}"
                fi
                return 0 ;;
            "${flag}="*)
                echo "${PASS_ARGS[$i]#${flag}=}"
                return 0 ;;
        esac
    done
}

_contains_token() {
    local raw="$1"
    local wanted="$2"
    local token
    raw="${raw//,/ }"
    for token in ${raw}; do
        if [[ "${token}" == "${wanted}" ]]; then
            return 0
        fi
    done
    return 1
}

_will_run_pe_av() {
    local kinds extra skip
    kinds="$(_arg_value --kinds)"
    extra="$(_arg_value --extra-kinds)"
    skip="$(_arg_value --skip-kinds)"
    if _contains_token "${skip}" "pe_av"; then
        return 1
    fi
    if [[ -z "${kinds}" ]]; then
        # Default kind set includes pe_av.
        return 0
    fi
    if _contains_token "${kinds}" "pe_av"; then
        return 0
    fi
    if _contains_token "${extra}" "pe_av"; then
        return 0
    fi
    return 1
}

if _will_run_pe_av && [[ "${MY_EVAL_SKIP_PE_AV_PREFLIGHT:-0}" != "1" ]]; then
    echo "[my_eval] PE-AV preflight: checking python deps and local model"
    "${PYTHON_BIN_RESOLVED}" - <<'PY'
import importlib
import os
import sys
from pathlib import Path

model_dir = Path(
    os.environ.get(
        "MY_EVAL_PE_AV_MODEL_DIR",
        os.environ.get(
            "PE_AV_MODEL_DIR",
            "",
        ),
    )
).expanduser()

errors = []
if not model_dir.is_dir():
    errors.append(f"PE-AV model dir not found: {model_dir}")
else:
    required = [
        "config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    has_weights = any(
        (model_dir / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if missing:
        errors.append(f"PE-AV model dir missing files: {', '.join(missing)}")
    if not has_weights:
        errors.append(f"PE-AV model dir has no model weights: {model_dir}")

try:
    from transformers.models.pe_audio_video import PeAudioVideoModel, PeAudioVideoProcessor  # noqa: F401
except Exception as exc:
    errors.append(
        "cannot import transformers.models.pe_audio_video "
        f"({exc.__class__.__name__}: {exc})"
    )

try:
    importlib.import_module("timm")
except Exception as exc:
    errors.append(f"cannot import timm ({exc.__class__.__name__}: {exc})")

if errors:
    print("\n[my_eval] PE-AV preflight FAILED", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    print(
        "\nFix before launching the full evaluation:\n"
        "  bash generation/infer/t2av/my_eval/setup_my_eval_deps.sh\n"
        "  bash generation/infer/t2av/my_eval/check_weights.sh\n"
        "or set MY_EVAL_PYTHON to an env with PE-AV-capable transformers+timm.\n"
        "Set MY_EVAL_SKIP_PE_AV_PREFLIGHT=1 only if you intentionally want to bypass this check.",
        file=sys.stderr,
    )
    sys.exit(2)

print(f"[my_eval] PE-AV preflight OK: {model_dir}")
PY
fi

# ----------------------------------------------------------------------
# Distributed env (mirrors eval_versebench_t2av.sh L220-L262).
# ----------------------------------------------------------------------
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
        MASTER_IP=$(nslookup "${PET_MASTER_ADDR}" 2>/dev/null | awk '/^Address: /{print $2; exit}' || true)
    fi
    if [ -z "${MASTER_IP}" ]; then
        echo "Error: cannot resolve PET_MASTER_ADDR=${PET_MASTER_ADDR}" >&2
        exit 1
    fi
    echo "Resolved MASTER_ADDR: ${PET_MASTER_ADDR} -> ${MASTER_IP}"

    export MASTER_ADDR="${MASTER_IP}"
    export MASTER_PORT="${PET_MASTER_PORT}"

    NNODES="${PET_NNODES}"
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
    NODE_RANK="${PET_NODE_RANK}"
else
    dist_mode="standalone"
    NNODES=1
    NODE_RANK=0
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    if [ -n "${PET_NPROC_PER_NODE:-}" ]; then
        NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
    elif [ -n "${NPROC_PER_NODE:-}" ]; then
        :
    else
        NPROC_PER_NODE="$("${PYTHON_BIN_RESOLVED}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
        if [ "${NPROC_PER_NODE}" -le 0 ]; then
            NPROC_PER_NODE=1
        fi
    fi
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

cuda_diag="$("${PYTHON_BIN_RESOLVED}" - <<'PY'
import os
import sys
import torch

print(
    f"python={sys.executable}; "
    f"torch={torch.__version__}; "
    f"torch_cuda={torch.version.cuda}; "
    f"cuda_available={torch.cuda.is_available()}; "
    f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}; "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
)
if torch.cuda.is_available():
    print("cuda_devices=" + ",".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
PY
)"

echo "============================================"
echo "  my_eval Dispatcher"
echo "============================================"
echo "Mode                       : ${dist_mode}"
echo "NNODES                     : ${NNODES}"
echo "NPROC/NODE                 : ${NPROC_PER_NODE}"
echo "NODE_RANK                  : ${NODE_RANK}"
echo "WORLD_SIZE                 : ${WORLD_SIZE}"
echo "MASTER_ADDR                : ${MASTER_ADDR}"
echo "MASTER_PORT                : ${MASTER_PORT}"
echo "CUDA_VISIBLE               : ${CUDA_VISIBLE_DEVICES:-all}"
echo "Forwarded args             : ${PASS_ARGS[*]:-<none>}"
echo "Python interpreter         : ${PYTHON_BIN_RESOLVED}"
echo "CUDA diagnostic            : ${cuda_diag}"
echo "============================================"

export PYTHONUNBUFFERED=1
# Subprocess workers inherit these and skip the dozens of FutureWarning /
# UserWarning / DeprecationWarning emitted by torch / transformers / timm /
# pyiqa / pytorch-lightning at import time.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning}"
# HF_HOME / TRANSFORMERS_CACHE are now redundant; the latter is deprecated and
# spams a warning per worker. ``common.sh`` already sets HF_HOME for us.
unset TRANSFORMERS_CACHE

# Use ``python -m torch.distributed.run`` rather than the ``torchrun`` script.
# This guarantees we always invoke the verse-bench env's python no matter what
# shebang the ``torchrun`` console-script picks up.
"${PYTHON_BIN_RESOLVED}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${my_eval_dir}/run_my_eval.py" \
    "${PASS_ARGS[@]}"

echo "my_eval evaluation completed!"
