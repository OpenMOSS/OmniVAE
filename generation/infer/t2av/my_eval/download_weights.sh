#!/usr/bin/env bash
# One-shot downloader for the 3 weights my_eval needs that are not yet on disk.
#
# Run this on a node that has outbound HTTPS (login / dev node). The target
# paths are written under the HuggingFace release directory so compute nodes can
# reuse the same model/data bundle.
#
# Each download:
#   * skips if the file already exists with non-zero size,
#   * uses wget -c so an interrupted transfer can be resumed by re-running,
#   * tries a primary URL first, then a fallback mirror if reachable,
#   * verifies a minimum file size at the end.
#
# Usage:
#   bash generation/infer/t2av/my_eval/download_weights.sh
#   bash generation/infer/t2av/my_eval/download_weights.sh --only imagebind
#   bash generation/infer/t2av/my_eval/download_weights.sh --only s3fd,panns
set -uo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
my_eval_dir="$(cd "$(dirname "${script_path}")" && pwd)"
project_root="$(cd "${my_eval_dir}/../../../.." && pwd)"

VERSE_BENCH_ROOT="${MY_EVAL_VERSE_BENCH_ROOT:-${project_root}/generation/evaluation/verse_bench}"
if [[ -f "${VERSE_BENCH_ROOT}/scripts/common.sh" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "${VERSE_BENCH_ROOT}/scripts/common.sh"
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
MODELS_PATH_R="${MODELS_PATH:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/verse_models}"
TORCH_HOME_R="${TORCH_HOME:-${OPEN_SOURCE_T2AV_EVAL_MODELS}/torch_cache}"
EVAL_ROOT="${project_root}/generation/evaluation"

ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "${script_path}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ----------------------------------------------------------------------
# Registry: id  |  target_path  |  min_size_bytes  |  url1 [url2 ...]
# ----------------------------------------------------------------------
declare -a IDS=(imagebind s3fd panns)

declare -A TARGET=(
    # NOTE: ${TORCH_HOME}/checkpoints/ (no /hub) -- this is where the vendored
    # imagebind_model.imagebind_huge() looks. torch.hub's official default has
    # an extra /hub layer; we ignore that to match the consumer code.
    [imagebind]="${TORCH_HOME_R}/checkpoints/imagebind_huge.pth"
    [s3fd]="${OPEN_SOURCE_T2AV_EVAL_MODELS}/lip_sync/s3fd_face_detector/sfd_face.pth"
    [panns]="${OPEN_SOURCE_T2AV_EVAL_MODELS}/audio_quality_metrics/audio_is_clap/pann_home/Cnn14_mAP=0.431.pth"
)
declare -A MIN_SIZE=(
    [imagebind]=4000000000     # ~4.5 GB
    [s3fd]=80000000            # ~85 MB
    [panns]=290000000          # ~310 MB
)
declare -A LABEL=(
    [imagebind]="ImageBind huge (~4.5 GB)"
    [s3fd]="S3FD face detector (~85 MB)"
    [panns]="PANNs Cnn14 mAP=0.431 (~310 MB)"
)

# URLs in order of preference. wget tries the first; if it returns non-zero,
# the next URL is tried.
URLS_imagebind=(
    "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
)
URLS_s3fd=(
    # CORRECT format: VGG/extras/loc/conf keys -- matches MOVA's S3FDNet.
    # MOVA's download_weights.sh uses this as the primary mirror.
    "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth"
    # HuggingFace mirror of the same checkpoint
    "https://huggingface.co/camenduru/wav2lip/resolve/main/sfd_face.pth"
    # NOTE: do NOT use https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
    # That file uses a DIFFERENT key naming (conv1_1/conv6_2_mbox_*) and is
    # incompatible with MOVA's S3FDNet -- loading it crashes with
    # "Missing key(s) in state_dict: vgg.0.weight, ...".
)
URLS_panns=(
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    # Hugging Face mirrors of the same checkpoint (preferred when zenodo is blocked)
    "https://huggingface.co/qiuqiangkong/panns/resolve/main/Cnn14_mAP%3D0.431.pth"
    "https://hf-mirror.com/qiuqiangkong/panns/resolve/main/Cnn14_mAP%3D0.431.pth"
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
have_aria2c() { command -v aria2c >/dev/null 2>&1; }
have_wget()   { command -v wget   >/dev/null 2>&1; }
have_curl()   { command -v curl   >/dev/null 2>&1; }

if ! have_wget && ! have_curl && ! have_aria2c; then
    echo "ERROR: need at least one of wget / curl / aria2c on PATH." >&2
    exit 1
fi

human_size() {
    local b="$1"
    awk -v b="$b" 'BEGIN{
        u="B KB MB GB TB"; split(u, a, " "); i=1;
        while (b>=1024 && i<5) { b/=1024; i++ }
        printf "%.1f%s", b, a[i];
    }'
}

download_one() {
    local target="$1"; shift
    local min_size="$1"; shift
    local urls=("$@")

    mkdir -p "$(dirname "${target}")"

    # Already there and big enough?
    if [[ -s "${target}" ]]; then
        local sz
        sz="$(stat -c%s "${target}" 2>/dev/null || stat -f%z "${target}")"
        if (( sz >= min_size )); then
            echo "  [skip] already exists: ${target}  ($(human_size "${sz}"))"
            return 0
        else
            echo "  [partial] existing file is only $(human_size "${sz}"); resuming..."
        fi
    fi

    for url in "${urls[@]}"; do
        echo "  -> trying ${url}"
        if have_aria2c; then
            aria2c -c -x 8 -s 8 --allow-overwrite=true --auto-file-renaming=false \
                   --max-tries=5 --connect-timeout=30 \
                   -d "$(dirname "${target}")" -o "$(basename "${target}")" \
                   "${url}" && return 0 || true
        elif have_wget; then
            wget -c --tries=5 --timeout=60 --connect-timeout=30 \
                 -O "${target}" "${url}" && return 0 || true
        else
            # curl: -C - to resume, --retry-connrefused for transient errors
            curl -L -C - --retry 5 --retry-connrefused --connect-timeout 30 \
                 -o "${target}" "${url}" && return 0 || true
        fi
        echo "  -- failed; trying next mirror"
    done

    echo "  [FAIL] all mirrors timed out / refused for ${target}" >&2
    return 1
}

verify() {
    local target="$1"
    local min_size="$2"
    if [[ ! -s "${target}" ]]; then
        echo "  [FAIL] ${target} missing or empty"
        return 1
    fi
    local sz
    sz="$(stat -c%s "${target}" 2>/dev/null || stat -f%z "${target}")"
    if (( sz < min_size )); then
        echo "  [FAIL] ${target} is only $(human_size "${sz}"), expected at least $(human_size "${min_size}")"
        return 1
    fi
    echo "  [ok] ${target}  ($(human_size "${sz}"))"
    return 0
}

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
echo "============================================"
echo "  my_eval one-shot weight downloader"
echo "============================================"
echo "MODELS_PATH : ${MODELS_PATH_R}"
echo "TORCH_HOME  : ${TORCH_HOME_R}"
echo "EVAL_ROOT   : ${EVAL_ROOT}"
echo "RELEASE_ROOT: ${OPEN_SOURCE_ROOT}"
echo "Tools       : aria2c=$(have_aria2c && echo yes || echo no), wget=$(have_wget && echo yes || echo no), curl=$(have_curl && echo yes || echo no)"
echo "============================================"

if [[ -n "${ONLY}" ]]; then
    IFS=',' read -r -a ONLY_LIST <<<"${ONLY// /}"
    # filter IDS
    FILTERED=()
    for id in "${IDS[@]}"; do
        for o in "${ONLY_LIST[@]}"; do
            if [[ "${id}" == "${o}" ]]; then
                FILTERED+=("${id}")
                break
            fi
        done
    done
    IDS=("${FILTERED[@]}")
fi

failures=()
for id in "${IDS[@]}"; do
    echo ""
    echo "==[ ${id} ]== ${LABEL[$id]}"
    echo "    target: ${TARGET[$id]}"
    var="URLS_${id}[@]"
    if ! download_one "${TARGET[$id]}" "${MIN_SIZE[$id]}" "${!var}"; then
        failures+=("${id}")
        continue
    fi
    verify "${TARGET[$id]}" "${MIN_SIZE[$id]}" || failures+=("${id}")
done

echo ""
echo "============================================"
echo "  Summary"
echo "============================================"
if [[ ${#failures[@]} -eq 0 ]]; then
    echo "All requested downloads complete."
    echo "Re-run bash generation/infer/t2av/my_eval/check_weights.sh to confirm."
    exit 0
else
    echo "FAILED for: ${failures[*]}"
    echo ""
    echo "If the network is blocking these hosts, run on a node that can reach them"
    echo "and the files will land on the shared filesystem automatically."
    exit 1
fi
