#!/usr/bin/env bash
set -euo pipefail

# Run release validation for T2AV recon and recon_distill_avclip, then compare
# against the previous eval_final_522 run.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OPEN_SOURCE_ROOT="${OMNIVAE_RELEASE_ROOT:-${OPEN_SOURCE_ROOT:-}}"
if [[ -z "${OPEN_SOURCE_ROOT}" ]]; then
    for candidate in \
        "${REPO_ROOT}/open_source" \
        "${REPO_ROOT}/../open_source" \
        "${REPO_ROOT}/open_source/open_source" \
        "${REPO_ROOT}/../open_source/open_source"; do
        if [[ -d "${candidate}/models" && -d "${candidate}/eval" ]]; then
            OPEN_SOURCE_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
if [[ -z "${OPEN_SOURCE_ROOT}" ]]; then
    OPEN_SOURCE_ROOT="${REPO_ROOT}/open_source"
fi
export OPEN_SOURCE_ROOT

MODE="run"
GPUS="0,1,2,3,4,5,6,7"
CFG="4"
STEP="200000"
MAX_EXAMPLES="0"
TYPES="set3-large"
VALIDATION_JSONL="${OPEN_SOURCE_ROOT}/eval/data/t2av/versebench_minimal/versebench_t2av_infer_minimal.jsonl"
CKPT_ROOT="${OPEN_SOURCE_ROOT}/models/dit/t2av"
REFERENCE_EVAL_ROOT="${REPO_ROOT}/eval/t2av/eval_final_522/cfg_4/shard_00"
OUTPUT_ROOT="${REPO_ROOT}/../test_output/t2av_release_compare_set3_large"
EXPERIMENTS=("t2av_recon" "t2av_recon_distill_avclip")
VAE_MODE="release"
RESUME_INFERENCE=1
RESUME_EVAL=1
RUN_COMPARE=1
EXTRA_VALIDATE_ARGS=()
TEE_LOG=1

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/release_eval/t2av/run_release_t2av_eval_compare.sh [options]

Modes:
  --mode run          Run inference + my_eval now. Works on local multi-GPU and
                      PET multi-node jobs because validate_checkpoints.sh reads
                      PET_NNODES/PET_NPROC_PER_NODE/PET_NODE_RANK.
  --mode compare-only Only compare an existing eval output root to the reference.

Core options:
  --gpus IDS                 Local GPU IDs, default 0,1,2,3,4,5,6,7.
  --cfg VALUE                CFG value, default 4.
  --step STEP                Checkpoint step, default 200000.
  --max-examples N           Number of filtered jsonl rows to run, default 0
                             means all.
  --types TYPES              Validation type filter, default set3-large.
  --output-root DIR          Shared storage output root.
  --reference-eval-root DIR  Previous eval root to compare with.
  --validation-jsonl PATH    Aligned VerseBench jsonl.
  --vae-mode release          Public release mode. Other modes are rejected.

Pass-through:
  --validate-arg ARG         Append one raw arg to validate_checkpoints.sh;
                             repeat for flag/value pairs.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --cfg)
            CFG="$2"; shift 2 ;;
        --step|--steps)
            STEP="$2"; shift 2 ;;
        --max-examples)
            MAX_EXAMPLES="$2"; shift 2 ;;
        --types)
            TYPES="$2"; shift 2 ;;
        --output-root)
            OUTPUT_ROOT="$(readlink -f "$2" 2>/dev/null || python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"; shift 2 ;;
        --reference-eval-root)
            REFERENCE_EVAL_ROOT="$(readlink -f "$2" 2>/dev/null || python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"; shift 2 ;;
        --validation-jsonl)
            VALIDATION_JSONL="$(readlink -f "$2" 2>/dev/null || python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"; shift 2 ;;
        --ckpt-root)
            CKPT_ROOT="$(readlink -f "$2" 2>/dev/null || python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")"; shift 2 ;;
        --vae-mode)
            VAE_MODE="$2"; shift 2 ;;
        --resume-inference)
            RESUME_INFERENCE=1; shift ;;
        --no-resume-inference)
            RESUME_INFERENCE=0; shift ;;
        --resume-eval)
            RESUME_EVAL=1; shift ;;
        --no-resume-eval)
            RESUME_EVAL=0; shift ;;
        --run-compare)
            RUN_COMPARE=1; shift ;;
        --no-run-compare)
            RUN_COMPARE=0; shift ;;
        --qz-gpus-per-task|--qz-instances|--qz-cfg-splits|--qz-job-prefix|--qz-arg)
            echo "ERROR: QZ submit options are not part of the public release script." >&2
            echo "       Run --mode run inside your allocated local/PET job." >&2
            exit 2 ;;
        --qz-dry-run)
            echo "ERROR: --qz-dry-run is not part of the public release script." >&2
            exit 2 ;;
        --validate-arg)
            EXTRA_VALIDATE_ARGS+=("$2"); shift 2 ;;
        --no-tee-log)
            TEE_LOG=0; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

case "${MODE}" in
    run|compare-only) ;;
    qz-submit)
        echo "ERROR: --mode qz-submit is not part of the public release script." >&2
        echo "       Run --mode run inside your allocated local/PET job." >&2
        exit 2 ;;
    *)
        echo "ERROR: --mode must be run or compare-only" >&2
        exit 2 ;;
esac
case "${VAE_MODE}" in
    release|reference) ;;
    *)
        echo "ERROR: --vae-mode must be release or reference" >&2
        exit 2 ;;
esac

INFER_OUTPUT_ROOT="${OUTPUT_ROOT}/inference"
EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}/eval"
COMPARE_OUTPUT_ROOT="${OUTPUT_ROOT}/compare"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${INFER_OUTPUT_ROOT}" "${EVAL_OUTPUT_ROOT}" "${COMPARE_OUTPUT_ROOT}" "${LOG_ROOT}"

NODE_RANK_FOR_LOG="${PET_NODE_RANK:-0}"
LOG_FILE="${LOG_ROOT}/${MODE}_node${NODE_RANK_FOR_LOG}_$(date +%Y%m%d_%H%M%S).log"
if [[ "${TEE_LOG}" == "1" ]]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

is_primary_node() {
    [[ "${PET_NODE_RANK:-0}" == "0" ]]
}

COMMON_VALIDATE_ARGS=(
    --validation-name versebench_expanded_new
    --text-field av_caption
    --index-field index
    --type-field type
)
if [[ -n "${TYPES}" ]]; then
    COMMON_VALIDATE_ARGS+=(--types "${TYPES}")
fi

VAE_VALIDATE_ARGS=()
if [[ "${VAE_MODE}" == "reference" ]]; then
    echo "ERROR: --vae-mode reference is not part of the public release package." >&2
    echo "       Use --vae-mode release, or pass explicit --validate-arg VAE overrides." >&2
    exit 2
fi

COMPARE_CMD=(
    python "${SCRIPT_DIR}/compare_t2av_eval_to_reference.py"
    --new-eval-root "${EVAL_OUTPUT_ROOT}"
    --reference-eval-root "${REFERENCE_EVAL_ROOT}"
    --output-dir "${COMPARE_OUTPUT_ROOT}"
    --experiments "${EXPERIMENTS[@]}"
    --step "${STEP}"
    --cfg-dir "cfg_dual_g${CFG}"
)

write_compare_helper() {
    local helper="${OUTPUT_ROOT}/compare_after_jobs.sh"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'set -euo pipefail\n'
        printf 'cd %q\n' "${REPO_ROOT}"
        printf '%q ' "${COMPARE_CMD[@]}"
        printf '\n'
    } > "${helper}"
    chmod +x "${helper}"
    echo "Compare helper: ${helper}"
}

print_config() {
    echo "================ T2AV release eval compare ================"
    echo "mode                : ${MODE}"
    echo "repo root           : ${REPO_ROOT}"
    echo "open source root    : ${OPEN_SOURCE_ROOT}"
    echo "ckpt root           : ${CKPT_ROOT}"
    echo "validation jsonl    : ${VALIDATION_JSONL}"
    echo "experiments         : ${EXPERIMENTS[*]}"
    echo "step/cfg/max/types  : ${STEP} / ${CFG} / ${MAX_EXAMPLES} / ${TYPES:-all}"
    echo "vae mode            : ${VAE_MODE}"
    echo "output root         : ${OUTPUT_ROOT}"
    echo "inference output    : ${INFER_OUTPUT_ROOT}"
    echo "eval output         : ${EVAL_OUTPUT_ROOT}"
    echo "compare output      : ${COMPARE_OUTPUT_ROOT}"
    echo "reference eval      : ${REFERENCE_EVAL_ROOT}"
    echo "PET_NNODES          : ${PET_NNODES:-unset}"
    echo "PET_NODE_RANK       : ${PET_NODE_RANK:-unset}"
    echo "PET_NPROC_PER_NODE  : ${PET_NPROC_PER_NODE:-unset}"
    echo "local gpus          : ${GPUS}"
    echo "log file            : ${LOG_FILE}"
    echo "==========================================================="
}

run_compare() {
    if ! is_primary_node; then
        echo "[compare] skip on PET_NODE_RANK=${PET_NODE_RANK:-0}"
        return 0
    fi
    if [[ ! -d "${REFERENCE_EVAL_ROOT}" ]]; then
        echo "[compare] reference eval root not found; skipping comparison."
        echo "          reference eval: ${REFERENCE_EVAL_ROOT}"
        echo "          pass --reference-eval-root to compare."
        return 0
    fi
    echo "[compare] executing: ${COMPARE_CMD[*]}"
    "${COMPARE_CMD[@]}"
}

run_validate() {
    local validate_cmd=(
        bash "${REPO_ROOT}/generation/scripts/av/validate_checkpoints.sh"
        --gpus "${GPUS}"
        --ckpt-root "${CKPT_ROOT}"
        --validation-jsonl "${VALIDATION_JSONL}"
        --steps "${STEP}"
        --order desc
        --cfg "${CFG}"
        --max-examples "${MAX_EXAMPLES}"
        --output-root "${INFER_OUTPUT_ROOT}"
        --experiments "${EXPERIMENTS[@]}"
        --run-my-eval
        --eval-output-root "${EVAL_OUTPUT_ROOT}"
        --my-eval-cfg dual
    )
    if [[ "${RESUME_INFERENCE}" == "1" ]]; then
        validate_cmd+=(--resume-inference)
    else
        validate_cmd+=(--no-resume-inference)
    fi
    if [[ "${RESUME_EVAL}" == "1" ]]; then
        validate_cmd+=(--resume-eval)
    else
        validate_cmd+=(--no-resume-eval)
    fi
    validate_cmd+=("${COMMON_VALIDATE_ARGS[@]}")
    validate_cmd+=("${VAE_VALIDATE_ARGS[@]}")
    validate_cmd+=("${EXTRA_VALIDATE_ARGS[@]}")

    cd "${REPO_ROOT}"
    echo "[run] executing: ${validate_cmd[*]}"
    "${validate_cmd[@]}"
}

print_config
write_compare_helper

case "${MODE}" in
    run)
        run_validate
        if [[ "${RUN_COMPARE}" == "1" ]]; then
            run_compare
        fi
        ;;
    compare-only)
        run_compare
        ;;
esac
