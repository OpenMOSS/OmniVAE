#!/usr/bin/env bash
# Inference-only T2AV validation over historical checkpoints.
#
# Usage:
#   bash scripts/av/validate_checkpoints.sh \
#       --ckpt-root /path/to/run-or-sweep-root \
#       --validation-jsonl /path/to/valid.jsonl \
#       --types set3-large,set4-large \
#       --min-step 5000 --max-step 35000 --step-multiple 5000 \
#       --cfg 4 \
#       --output-root /path/to/generated/samples \
#       --run-my-eval --eval-output-root /path/to/my_eval/results
#
# Eval-only / profiling examples:
#   bash scripts/av/validate_checkpoints.sh --eval-only --dino-only \
#       --output-root /path/to/existing/samples \
#       --eval-output-root /path/to/my_eval/results \
#       --gpu-monitor --no-my-eval-skip-completed

set -euo pipefail
echo "===================== start checkpoint validation (T2AV) ====================="

OMNIGEN_BASHRC="${OMNIGEN_BASHRC:-${LAION_BASHRC:-}}"
INFER_CONDA_ENV="${INFER_CONDA_ENV-dit}"
TS="$(date +%Y%m%d_%H%M%S)"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
work_dir="$(cd "$(dirname "${script_path}")/../.." && pwd)"
project_root="$(cd "${work_dir}/.." && pwd)"
OPEN_SOURCE_ROOT="${OMNIVAE_RELEASE_ROOT:-${OPEN_SOURCE_ROOT:-}}"
if [[ -z "${OPEN_SOURCE_ROOT}" ]]; then
    for candidate in \
        "${project_root}/open_source" \
        "${project_root}/../open_source" \
        "${project_root}/../../open_source" \
        "${project_root}/open_source/open_source" \
        "${project_root}/../open_source/open_source" \
        "${project_root}/../../open_source/open_source"; do
        if [[ -d "${candidate}/models" && -d "${candidate}/eval" ]]; then
            OPEN_SOURCE_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
if [[ -z "${OPEN_SOURCE_ROOT}" ]]; then
    OPEN_SOURCE_ROOT="${project_root}/open_source"
fi
export OPEN_SOURCE_ROOT
OPEN_SOURCE_VAE_ROOT="${OPEN_SOURCE_VAE_ROOT:-${OPEN_SOURCE_ROOT}/models/vae}"
if [[ ! -d "${OPEN_SOURCE_VAE_ROOT}" && -d "${project_root}/open_source/vae_release" ]]; then
    OPEN_SOURCE_VAE_ROOT="${project_root}/open_source/vae_release"
fi
script_start_epoch="$(date +%s)"
gpu_ids=""
debug_mode=0
skip_bashrc=0
test_mode=0
use_default_vae_overrides=1
run_my_eval=0
my_eval_output_root=""
my_eval_cfg="both"
my_eval_dispatch_mode="kind-major-reuse"
my_eval_scan_workers="${MY_EVAL_SCAN_WORKERS:-32}"
my_eval_skip_completed=1
resume_inference_arg=1
eval_only_arg=0
dino_only_arg=0
gpu_monitor_arg=0
gpu_monitor_interval="${GPU_MONITOR_INTERVAL:-1}"
gpu_monitor_dir_arg=""
ckpt_root_arg=""
output_root_arg=""
validation_jsonl_arg=""
steps_arg=""
cfg_arg=""
max_examples_arg=""
experiments_arg=()
latest_arg=0
dry_run_arg=0
passthrough=()
my_eval_extra=()

DEFAULT_VIDEO_VAE_OVERRIDES=(
    "t2av_recon=univae:${OPEN_SOURCE_VAE_ROOT}/video_only/recon"
    "t2av_recon_distill=univae:${OPEN_SOURCE_VAE_ROOT}/video_only/recon_distill"
    "t2av_recon_avclip=univae:${OPEN_SOURCE_VAE_ROOT}/audio_video/recon_avclip"
    "t2av_recon_distill_avclip=univae:${OPEN_SOURCE_VAE_ROOT}/audio_video/recon_distill_avclip"
    "2_t2av_recon_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/video_only/recon"
    "2_t2av_recon_distill_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/video_only/recon_distill"
    "2_t2av_recon_avclip_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/audio_video/recon_avclip"
    "2_t2av_recon_distill_avclip_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/audio_video/recon_distill_avclip"
)
DEFAULT_AUDIO_VAE_OVERRIDES=(
    "t2av_recon=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon"
    "t2av_recon_distill=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon_distill"
    "t2av_recon_avclip=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon_avclip_ft_decoder"
    "t2av_recon_distill_avclip=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon_distill_avclip_ft_decoder"
    "2_t2av_recon_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon"
    "2_t2av_recon_distill_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon_distill"
    "2_t2av_recon_avclip_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon_avclip_ft_decoder"
    "2_t2av_recon_distill_avclip_lr2=univae:${OPEN_SOURCE_VAE_ROOT}/audio_only/recon_distill_avclip_ft_decoder"
)

resolve_arg_path() {
    local path="$1"
    if [ -z "${path}" ]; then
        return
    fi
    readlink -f "${path}" 2>/dev/null || python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${path}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            gpu_ids="$2"; shift 2 ;;
        --debug)
            debug_mode=1; shift ;;
        --test)
            test_mode=1; run_my_eval=1; shift ;;
        --eval-only|--skip-inference)
            eval_only_arg=1; run_my_eval=1; shift ;;
        --resume-inference)
            resume_inference_arg=1; passthrough+=("$1"); shift ;;
        --no-resume-inference)
            resume_inference_arg=0; passthrough+=("$1"); shift ;;
        --dino-only|--identity-dino-only)
            dino_only_arg=1; run_my_eval=1; shift ;;
        --gpu-monitor)
            gpu_monitor_arg=1; shift ;;
        --no-gpu-monitor)
            gpu_monitor_arg=0; shift ;;
        --gpu-monitor-interval)
            gpu_monitor_interval="$2"; shift 2 ;;
        --gpu-monitor-dir)
            gpu_monitor_dir_arg="$2"; shift 2 ;;
        --skip-bashrc)
            skip_bashrc=1; shift ;;
        --no-default-vae-overrides|--no-default-t2av-vae-overrides)
            use_default_vae_overrides=0; shift ;;
        --run-my-eval|--my-eval)
            run_my_eval=1; shift ;;
        --no-my-eval)
            run_my_eval=0; shift ;;
        --eval-output-root|--my-eval-output-root)
            my_eval_output_root="$(resolve_arg_path "$2")"; shift 2 ;;
        --my-eval-cfg)
            my_eval_cfg="$2"; shift 2 ;;
        --my-eval-dispatch-mode)
            my_eval_dispatch_mode="$2"; shift 2 ;;
        --my-eval-scan-workers)
            my_eval_scan_workers="$2"; shift 2 ;;
        --resume-eval|--my-eval-skip-completed)
            my_eval_skip_completed=1; shift ;;
        --no-resume-eval|--no-my-eval-skip-completed)
            my_eval_skip_completed=0; shift ;;
        --my-eval-arg)
            my_eval_extra+=("$2"); shift 2 ;;
        --ckpt-root|--ckpt_root|--ckpt)
            ckpt_root_arg="$(resolve_arg_path "$2")"; passthrough+=("$1" "${ckpt_root_arg}"); shift 2 ;;
        --output-root|--output_root)
            output_root_arg="$(resolve_arg_path "$2")"; passthrough+=("$1" "${output_root_arg}"); shift 2 ;;
        --validation-jsonl|--valid-jsonl|--validation_jsonl|--valid_jsonl)
            validation_jsonl_arg="$(resolve_arg_path "$2")"; passthrough+=("$1" "${validation_jsonl_arg}"); shift 2 ;;
        --steps)
            steps_arg="$2"; passthrough+=("$1" "$2"); shift 2 ;;
        --cfg)
            cfg_arg="$2"; passthrough+=("$1" "$2"); shift 2 ;;
        --max-examples|--max_examples)
            max_examples_arg="$2"; passthrough+=("$1" "$2"); shift 2 ;;
        --experiments|--experiment|--experiment-name|--experiment-names)
            passthrough+=("$1"); shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                experiments_arg+=("$1")
                passthrough+=("$1")
                shift
            done ;;
        --latest|--only-latest)
            latest_arg=1; passthrough+=("$1"); shift ;;
        --dry-run|--dry_run)
            dry_run_arg=1; passthrough+=("$1"); shift ;;
        --)
            shift
            passthrough+=("$@")
            break ;;
        *)
            passthrough+=("$1"); shift ;;
    esac
done

if [ "${skip_bashrc}" -eq 0 ] && [ -n "${OMNIGEN_BASHRC}" ]; then
    if [ -f "${OMNIGEN_BASHRC}" ]; then
        # shellcheck disable=SC1090
        source "${OMNIGEN_BASHRC}"
    else
        echo "Warning: OMNIGEN_BASHRC=${OMNIGEN_BASHRC} not found, skipping source"
    fi
fi
if [ -n "${INFER_CONDA_ENV}" ]; then
    if command -v conda >/dev/null 2>&1; then
        if ! conda activate "${INFER_CONDA_ENV}" 2>/dev/null; then
            CONDA_BASE="$(conda info --base 2>/dev/null || true)"
            if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
                # shellcheck disable=SC1091
                source "${CONDA_BASE}/etc/profile.d/conda.sh"
                conda activate "${INFER_CONDA_ENV}"
            else
                echo "Warning: could not initialize conda, continuing in current environment"
            fi
        fi
    else
        echo "Warning: conda not on PATH, ignoring INFER_CONDA_ENV=${INFER_CONDA_ENV}"
    fi
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-30}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600000}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# Post-validation my_eval batch defaults for H200 jobs. These remain
# overrideable from the job environment; set MY_EVAL_H200_BATCH_DEFAULTS=0 to
# keep the lower per-metric code defaults.
if [[ "${MY_EVAL_H200_BATCH_DEFAULTS:-1}" != "0" ]]; then
    export MY_EVAL_PE_AV_BATCH_SIZE="${MY_EVAL_PE_AV_BATCH_SIZE:-32}"
    export MY_EVAL_IMAGEBIND_BATCH_SIZE="${MY_EVAL_IMAGEBIND_BATCH_SIZE:-32}"
    export MY_EVAL_CLAP_BATCH_SIZE="${MY_EVAL_CLAP_BATCH_SIZE:-64}"
    export MY_EVAL_DINO_BATCH_SIZE="${MY_EVAL_DINO_BATCH_SIZE:-32}"
    export MY_EVAL_AESTHETIC_BATCH_SIZE="${MY_EVAL_AESTHETIC_BATCH_SIZE:-32}"
    export MY_EVAL_MANIQA_PATCH_BATCH_SIZE="${MY_EVAL_MANIQA_PATCH_BATCH_SIZE:-128}"
    export MY_EVAL_AUDIOBOX_BATCH_SIZE="${MY_EVAL_AUDIOBOX_BATCH_SIZE:-32}"
    export MY_EVAL_WER_BATCH_SIZE="${MY_EVAL_WER_BATCH_SIZE:-16}"
    export MY_EVAL_LIPSYNC_BATCH_SIZE="${MY_EVAL_LIPSYNC_BATCH_SIZE:-32}"
    export MY_EVAL_RAFT_BATCH_SIZE="${MY_EVAL_RAFT_BATCH_SIZE:-8}"
fi

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
work_dir="$(cd "$(dirname "${script_path}")/../.." && pwd)"
project_root="$(cd "${work_dir}/.." && pwd)"
cd "${work_dir}"

resolve_path_best_effort() {
    local path="$1"
    if [ -z "${path}" ]; then
        return
    fi
    readlink -f "${path}" 2>/dev/null || python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${path}"
}

dirname_n() {
    local path="$1"
    local count="$2"
    local i
    for i in $(seq 1 "${count}"); do
        path="$(dirname "${path}")"
    done
    echo "${path}"
}

infer_sample_root() {
    local root="$1"
    local resolved
    resolved="$(resolve_path_best_effort "${root}")"
    if [ -z "${resolved}" ]; then
        return
    fi
    if [ -d "${resolved}/checkpoints/snapshots" ]; then
        dirname "${resolved}"
    elif [[ "$(basename "${resolved}")" == checkpoint-* ]]; then
        dirname_n "${resolved}" 4
    else
        echo "${resolved}"
    fi
}

bump_port() {
    local port="$1"
    if [[ "${port}" =~ ^[0-9]+$ ]]; then
        echo "$((port + 1))"
    else
        echo "${port}"
    fi
}

write_pipeline_timing() {
    local output_path="$1"
    local inference_started="$2"
    local inference_completed="$3"
    local eval_started="$4"
    local eval_completed="$5"
    mkdir -p "$(dirname "${output_path}")"
python - "$output_path" "$script_start_epoch" "$inference_started" "$inference_completed" "$eval_started" "$eval_completed" "$test_mode" "$ckpt_root_arg" "$output_root_arg" "$my_eval_output_root" <<'PY'
import json
import sys
import time
from pathlib import Path

(
    output_path,
    script_start,
    inference_started,
    inference_completed,
    eval_started,
    eval_completed,
    test_mode,
    ckpt_root,
    output_root,
    eval_output_root,
) = sys.argv[1:]

def as_int(value):
    try:
        return int(value)
    except Exception:
        return 0

def _float_stat(stats, field, key):
    try:
        return float(stats.get(field, {}).get(key, 0.0))
    except Exception:
        return 0.0

def collect_metric_timings(eval_output_root):
    root = Path(eval_output_root).expanduser()
    if not eval_output_root or not root.is_dir():
        return {
            "files": [],
            "targets": [],
            "by_metric_kind": {},
        }

    files = sorted(root.rglob("eval_timing_summary.json"))
    targets = []
    by_metric = {}
    for summary_path in files:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        target_rel = str(summary_path.parent.relative_to(root))
        metrics = {}
        for kind, item in sorted((data.get("by_kind") or {}).items()):
            fields = item.get("timing_fields") or {}
            metric_payload = {
                "count": int(item.get("count", 0)),
                "wall_elapsed_sec": _float_stat(fields, "elapsed_sec", "max"),
                "rank_elapsed_sec_sum": _float_stat(fields, "elapsed_sec", "sum"),
                "module_import_wall_sec": _float_stat(fields, "module_import_elapsed_sec", "max"),
                "model_load_wall_sec": _float_stat(fields, "model_load_elapsed_sec", "max"),
                "model_load_rank_sum_sec": _float_stat(fields, "model_load_elapsed_sec", "sum"),
                "metric_compute_wall_sec": _float_stat(fields, "metric_compute_elapsed_sec", "max"),
                "metric_compute_rank_sum_sec": _float_stat(fields, "metric_compute_elapsed_sec", "sum"),
                "summary_wall_sec": _float_stat(fields, "summary_elapsed_sec", "max"),
                "barrier_wall_sec": _float_stat(fields, "barrier_elapsed_sec", "max"),
                "statuses": item.get("statuses", {}),
            }
            metrics[kind] = metric_payload
            agg = by_metric.setdefault(kind, {
                "num_targets": 0,
                "wall_elapsed_sec_sum": 0.0,
                "rank_elapsed_sec_sum": 0.0,
                "model_load_wall_sec_sum": 0.0,
                "model_load_rank_sum_sec": 0.0,
                "metric_compute_wall_sec_sum": 0.0,
                "metric_compute_rank_sum_sec": 0.0,
                "summary_wall_sec_sum": 0.0,
                "barrier_wall_sec_sum": 0.0,
            })
            agg["num_targets"] += 1
            agg["wall_elapsed_sec_sum"] += metric_payload["wall_elapsed_sec"]
            agg["rank_elapsed_sec_sum"] += metric_payload["rank_elapsed_sec_sum"]
            agg["model_load_wall_sec_sum"] += metric_payload["model_load_wall_sec"]
            agg["model_load_rank_sum_sec"] += metric_payload["model_load_rank_sum_sec"]
            agg["metric_compute_wall_sec_sum"] += metric_payload["metric_compute_wall_sec"]
            agg["metric_compute_rank_sum_sec"] += metric_payload["metric_compute_rank_sum_sec"]
            agg["summary_wall_sec_sum"] += metric_payload["summary_wall_sec"]
            agg["barrier_wall_sec_sum"] += metric_payload["barrier_wall_sec"]
        targets.append({
            "target": target_rel,
            "timing_file": str(summary_path),
            "metrics": metrics,
        })
    return {
        "files": [str(path) for path in files],
        "targets": targets,
        "by_metric_kind": by_metric,
    }

script_start_i = as_int(script_start)
inference_started_i = as_int(inference_started)
inference_completed_i = as_int(inference_completed)
eval_started_i = as_int(eval_started)
eval_completed_i = as_int(eval_completed)
completed_i = eval_completed_i or inference_completed_i or int(time.time())
payload = {
    "test_mode": bool(as_int(test_mode)),
    "ckpt_root": ckpt_root or None,
    "sample_output_root": output_root or None,
    "eval_output_root": eval_output_root or None,
    "started_at_unix": script_start_i,
    "completed_at_unix": completed_i,
    "total_elapsed_sec": max(0, completed_i - script_start_i),
    "inference": {
        "started_at_unix": inference_started_i or None,
        "completed_at_unix": inference_completed_i or None,
        "elapsed_sec": (
            max(0, inference_completed_i - inference_started_i)
            if inference_started_i and inference_completed_i else None
        ),
    },
    "my_eval": {
        "started_at_unix": eval_started_i or None,
        "completed_at_unix": eval_completed_i or None,
        "elapsed_sec": (
            max(0, eval_completed_i - eval_started_i)
            if eval_started_i and eval_completed_i else None
        ),
    },
    "metric_timing_files": "Per target: <eval-output-root>/<experiment>/<step>/<cfg>/eval_timing_summary.json",
    "metric_timings": collect_metric_timings(eval_output_root),
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
print(f"[timing] wrote {output_path}", flush=True)
PY
}

gpu_monitor_last_pid=""
gpu_monitor_last_file=""
gpu_monitor_pids=()

stop_gpu_monitor() {
    local pid="${1:-}"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

stop_all_gpu_monitors() {
    local pid
    for pid in "${gpu_monitor_pids[@]:-}"; do
        stop_gpu_monitor "${pid}"
    done
}
trap stop_all_gpu_monitors EXIT

start_gpu_monitor() {
    local phase="$1"
    local output_dir="$2"
    gpu_monitor_last_pid=""
    gpu_monitor_last_file=""
    if [ "${gpu_monitor_arg}" -ne 1 ]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "Warning: --gpu-monitor requested but nvidia-smi is not available" >&2
        return
    fi
    mkdir -p "${output_dir}"
    gpu_monitor_last_file="${output_dir}/gpu_monitor_${phase}.csv"
    printf "epoch,phase,timestamp,index,uuid,name,utilization_gpu_pct,utilization_memory_pct,memory_used_mb,memory_total_mb,power_draw_w\n" > "${gpu_monitor_last_file}"
    (
        while true; do
            epoch="$(date +%s.%N)"
            nvidia-smi \
                --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits 2>/dev/null \
                | awk -v epoch="${epoch}" -v phase="${phase}" 'BEGIN { FS=", "; OFS="," } { print epoch, phase, $1, $2, $3, $4, $5, $6, $7, $8, $9 }'
            sleep "${gpu_monitor_interval}"
        done
    ) >> "${gpu_monitor_last_file}" &
    gpu_monitor_last_pid="$!"
    gpu_monitor_pids+=("${gpu_monitor_last_pid}")
    echo "[gpu-monitor] ${phase}: pid=${gpu_monitor_last_pid} -> ${gpu_monitor_last_file}"
}

run_timing_postprocess() {
    local pipeline_timing_path="$1"
    local timing_dir="$2"
    shift 2
    local monitor_files=("$@")
    local analyze_cmd

    if [ ! -f "${pipeline_timing_path}" ]; then
        return
    fi
    if [ -f "infer/t2av/plot_timing_report.py" ]; then
        python infer/t2av/plot_timing_report.py \
            --pipeline-timing "${pipeline_timing_path}" \
            --output-dir "${timing_dir}" \
            || echo "Warning: timing HTML generation failed" >&2
    fi
    if [ -f "infer/t2av/analyze_timing_bottlenecks.py" ]; then
        analyze_cmd=(python infer/t2av/analyze_timing_bottlenecks.py
            --pipeline-timing "${pipeline_timing_path}"
            --output-dir "${timing_dir}"
        )
        local f
        for f in "${monitor_files[@]:-}"; do
            if [ -n "${f}" ] && [ -f "${f}" ]; then
                analyze_cmd+=(--gpu-monitor "${f}")
            fi
        done
        "${analyze_cmd[@]}" || echo "Warning: bottleneck analysis failed" >&2
    fi
}

if [ "${test_mode}" -eq 1 ]; then
    echo "[test] enabled: latest checkpoint per experiment, CFG=4, max_examples=8, then my_eval"
    if [ -z "${cfg_arg}" ]; then
        cfg_arg="4"
        passthrough+=(--cfg "${cfg_arg}")
    fi
    if [ -z "${max_examples_arg}" ]; then
        max_examples_arg="8"
        passthrough+=(--max-examples "${max_examples_arg}")
    fi
    if [ "${latest_arg}" -eq 0 ]; then
        latest_arg=1
        passthrough+=(--latest)
    fi
    if [ -z "${output_root_arg}" ]; then
        output_root_arg="${project_root}/runs/t2av/test_${TS}"
        passthrough+=(--output-root "${output_root_arg}")
    fi
    if [ -z "${my_eval_output_root}" ]; then
        my_eval_output_root="${project_root}/eval/t2av/test_${TS}"
    fi
    my_eval_extra+=(--max-ckpt-per-experiment 1)
fi

if [ -z "${cfg_arg}" ]; then
    cfg_arg="4"
    passthrough+=(--cfg "${cfg_arg}")
fi

if [ "${dino_only_arg}" -eq 1 ]; then
    echo "[dino-only] enabled: my_eval will run only identity_dino"
    my_eval_extra+=(--kinds identity_dino)
fi

if [ "${eval_only_arg}" -eq 1 ]; then
    if [ -z "${output_root_arg}" ]; then
        echo "ERROR: --eval-only requires --output-root pointing at an existing sample root" >&2
        exit 2
    fi
    if [ ! -d "${output_root_arg}" ]; then
        echo "ERROR: --eval-only sample root not found: ${output_root_arg}" >&2
        exit 2
    fi
fi

if [ -n "${PET_NNODES:-}" ] && [ "${PET_NNODES}" -gt 1 ] 2>/dev/null; then
    MASTER_IP=""
    for i in $(seq 1 120); do
        MASTER_IP=$(getent hosts "${PET_MASTER_ADDR}" 2>/dev/null | awk '{print $1}' | head -n 1 || true)
        [ -n "${MASTER_IP}" ] && break
        echo "Waiting for DNS resolution of ${PET_MASTER_ADDR}... (${i}/120)"
        sleep 2
    done
    if [ -z "${MASTER_IP}" ]; then
        MASTER_IP=$(nslookup "${PET_MASTER_ADDR}" 2>/dev/null | awk '/^Address: /{print $2; exit}')
    fi
    if [ -z "${MASTER_IP}" ]; then
        echo "Error: cannot resolve PET_MASTER_ADDR=${PET_MASTER_ADDR}"
        exit 1
    fi
    export MASTER_ADDR="${MASTER_IP}"
    export MASTER_PORT="${PET_MASTER_PORT}"
    NNODES="${PET_NNODES}"
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
    NODE_RANK="${PET_NODE_RANK}"
else
    NNODES=1
    NODE_RANK=0
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    if [ "${debug_mode}" -eq 1 ]; then
        if [ -z "${gpu_ids}" ]; then
            gpu_ids="0"
        else
            gpu_ids="$(echo "${gpu_ids}" | cut -d',' -f1)"
        fi
        export CUDA_VISIBLE_DEVICES="${gpu_ids}"
        NPROC_PER_NODE=1
    elif [ -n "${gpu_ids}" ]; then
        export CUDA_VISIBLE_DEVICES="${gpu_ids}"
        NPROC_PER_NODE=$(echo "${gpu_ids}" | tr ',' '\n' | wc -l)
    else
        NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    fi
    if [ "${NPROC_PER_NODE}" -le 0 ]; then
        echo "Error: no GPU detected"
        exit 1
    fi
fi

echo "Mode             : T2AV checkpoint validation"
echo "Work dir         : ${work_dir}"
echo "NNODES           : ${NNODES}"
echo "NPROC/NODE       : ${NPROC_PER_NODE}"
echo "NODE_RANK        : ${NODE_RANK}"
echo "MASTER_ADDR      : ${MASTER_ADDR}"
echo "MASTER_PORT      : ${MASTER_PORT}"
echo "CUDA_VISIBLE     : ${CUDA_VISIBLE_DEVICES:-all}"
echo "Python           : $(command -v python)"
echo "Test mode        : ${test_mode}"
echo "Eval only        : ${eval_only_arg}"
echo "Resume inference : ${resume_inference_arg}"
echo "DINO only        : ${dino_only_arg}"
echo "Default VAE ovrd : ${use_default_vae_overrides}"
echo "Run my_eval      : ${run_my_eval}"
echo "my_eval cfg      : ${my_eval_cfg}"
echo "my_eval dispatch : ${my_eval_dispatch_mode}"
echo "my_eval batches  : PE_AV=${MY_EVAL_PE_AV_BATCH_SIZE:-2} ImageBind=${MY_EVAL_IMAGEBIND_BATCH_SIZE:-4} CLAP=${MY_EVAL_CLAP_BATCH_SIZE:-16} DINO=${MY_EVAL_DINO_BATCH_SIZE:-16} Aesthetic=${MY_EVAL_AESTHETIC_BATCH_SIZE:-16} AudioBox=${MY_EVAL_AUDIOBOX_BATCH_SIZE:-8} WER=${MY_EVAL_WER_BATCH_SIZE:-8}"
echo "Resume my_eval   : ${my_eval_skip_completed}"
echo "GPU monitor      : ${gpu_monitor_arg} (interval=${gpu_monitor_interval}s)"
echo "Args             : ${passthrough[*]}"
echo "============================================"

cmd=(torchrun
    --nnodes="${NNODES}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    infer/t2av/validate_t2av_checkpoints.py
)
if [ "${use_default_vae_overrides}" -eq 1 ]; then
    for item in "${DEFAULT_VIDEO_VAE_OVERRIDES[@]}"; do
        cmd+=(--video-vae-override "${item}")
    done
    for item in "${DEFAULT_AUDIO_VAE_OVERRIDES[@]}"; do
        cmd+=(--audio-vae-override "${item}")
    done
fi
cmd+=("${passthrough[@]}")

if [ -n "${gpu_monitor_dir_arg}" ]; then
    timing_output_dir="$(resolve_path_best_effort "${gpu_monitor_dir_arg}")"
elif [ -n "${my_eval_output_root}" ]; then
    timing_output_dir="$(resolve_path_best_effort "${my_eval_output_root}")/_timing"
elif [ -n "${output_root_arg}" ]; then
    timing_output_dir="$(resolve_path_best_effort "${output_root_arg}")/_timing"
else
    timing_output_dir="${project_root}/runs/t2av/timing_${TS}"
fi

inference_start_epoch=0
inference_end_epoch=0
inference_gpu_monitor_csv=""
if [ "${eval_only_arg}" -eq 1 ]; then
    echo "[eval-only] skipping checkpoint validation/inference; using existing samples under ${output_root_arg}"
else
    echo "Executing: ${cmd[*]}"
    start_gpu_monitor "inference" "${timing_output_dir}"
    inference_gpu_monitor_pid="${gpu_monitor_last_pid}"
    inference_gpu_monitor_csv="${gpu_monitor_last_file}"
    inference_start_epoch="$(date +%s)"
    "${cmd[@]}"
    inference_end_epoch="$(date +%s)"
    stop_gpu_monitor "${inference_gpu_monitor_pid}"
fi

echo "Checkpoint validation stage completed!"

eval_start_epoch=0
eval_end_epoch=0
if [ "${run_my_eval}" -eq 1 ] && [ "${dry_run_arg}" -eq 0 ]; then
    if [ -n "${output_root_arg}" ]; then
        my_eval_sample_root="$(resolve_path_best_effort "${output_root_arg}")"
    else
        my_eval_sample_root="$(infer_sample_root "${ckpt_root_arg}")"
    fi
    if [ -z "${my_eval_sample_root}" ]; then
        echo "ERROR: --run-my-eval requires --ckpt-root/--ckpt or --output-root so sample root can be inferred" >&2
        exit 2
    fi
    if [ -z "${my_eval_output_root}" ]; then
        sample_tag="$(basename "${my_eval_sample_root}")"
        my_eval_output_root="${project_root}/eval/my_eval_${sample_tag}"
    fi

    my_eval_cmd=(bash infer/t2av/my_eval/run_my_eval.sh
        --sample-root "${my_eval_sample_root}"
        --eval-output-root "${my_eval_output_root}"
        --cfg "${my_eval_cfg}"
        --dispatch-mode "${my_eval_dispatch_mode}"
        --scan-workers "${my_eval_scan_workers}"
    )
    if [ "${my_eval_skip_completed}" -eq 1 ]; then
        my_eval_cmd+=(--skip-completed)
    fi
    if [ -n "${validation_jsonl_arg}" ]; then
        my_eval_cmd+=(--valid-jsonl "$(resolve_path_best_effort "${validation_jsonl_arg}")")
    fi
    if [ -n "${steps_arg}" ] && [[ ! "${steps_arg}" =~ (^|[,\[:space:]])latest($|[,\[:space:]]) ]]; then
        my_eval_cmd+=(--steps "${steps_arg}")
    fi
    if [ "${#experiments_arg[@]}" -gt 0 ]; then
        my_eval_cmd+=(--experiments "${experiments_arg[@]}")
    fi
    my_eval_cmd+=("${my_eval_extra[@]}")

    if [ -n "${PET_NNODES:-}" ] && [ "${PET_NNODES}" -gt 1 ] 2>/dev/null; then
        export PET_MASTER_PORT="$(bump_port "${PET_MASTER_PORT}")"
    else
        export MASTER_PORT="$(bump_port "${MASTER_PORT}")"
    fi

    echo "===================== start post-validation my_eval ====================="
    echo "my_eval sample root : ${my_eval_sample_root}"
    echo "my_eval output root : ${my_eval_output_root}"
    echo "Executing: ${my_eval_cmd[*]}"
    if [ -z "${gpu_monitor_dir_arg}" ]; then
        timing_output_dir="$(resolve_path_best_effort "${my_eval_output_root}")/_timing"
    fi
    eval_gpu_monitor_csv=""
    start_gpu_monitor "eval" "${timing_output_dir}"
    eval_gpu_monitor_pid="${gpu_monitor_last_pid}"
    eval_gpu_monitor_csv="${gpu_monitor_last_file}"
    eval_start_epoch="$(date +%s)"
    "${my_eval_cmd[@]}"
    eval_end_epoch="$(date +%s)"
    stop_gpu_monitor "${eval_gpu_monitor_pid}"
    pipeline_timing_path="${my_eval_output_root}/pipeline_timing.json"
    write_pipeline_timing \
        "${pipeline_timing_path}" \
        "${inference_start_epoch}" "${inference_end_epoch}" \
        "${eval_start_epoch}" "${eval_end_epoch}"
    run_timing_postprocess \
        "${pipeline_timing_path}" \
        "${timing_output_dir}" \
        "${inference_gpu_monitor_csv}" "${eval_gpu_monitor_csv}"
elif [ "${dry_run_arg}" -eq 1 ] && [ "${run_my_eval}" -eq 1 ]; then
    echo "Skipping my_eval because --dry-run was passed."
fi
