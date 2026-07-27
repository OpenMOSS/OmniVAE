#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERSE_CACHE_DIR="${VERSE_CACHE_DIR:-${ROOT_DIR}/.cache/verse-bench-cache}"
VERSE_CONDA_ROOT="${VERSE_CONDA_ROOT:-${ROOT_DIR}/.cache/conda}"
VERSE_CONDA_ENV="${VERSE_CONDA_ENV:-verse-bench}"
VERSE_MODELS_DIR="${VERSE_MODELS_DIR:-${ROOT_DIR}/models}"
VERSE_BENCH_DATA_DIR="${VERSE_BENCH_DATA_DIR:-${ROOT_DIR}/verse_bench}"
VERSE_DOWNLOAD_DATASET="${VERSE_DOWNLOAD_DATASET:-1}"
VERSE_PREFETCH_AUX="${VERSE_PREFETCH_AUX:-1}"
VERSE_HF_ENDPOINT="${VERSE_HF_ENDPOINT:-https://huggingface.co}"
VERSE_PIP_INDEX_URL="${VERSE_PIP_INDEX_URL:-https://pypi.org/simple}"
VERSE_DINOV3_REPO="${VERSE_DINOV3_REPO:-tao-hunter/dinov3-vitl16-pretrain-lvd1689m}"
T2AV_EVAL_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
T2AV_COMPASS_DIR="${T2AV_COMPASS_DIR:-${T2AV_EVAL_ROOT}/T2AV-Compass}"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${VERSE_CONDA_ROOT}/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${VERSE_CONDA_ROOT}/pkgs}"
export HF_HOME="${HF_HOME:-${VERSE_CACHE_DIR}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${VERSE_CACHE_DIR}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VERSE_CACHE_DIR}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VERSE_CACHE_DIR}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${VERSE_CACHE_DIR}/matplotlib}"
export TMPDIR="${TMPDIR:-${VERSE_CACHE_DIR}/tmp}"
export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-${VERSE_CACHE_DIR}/insightface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${VERSE_CACHE_DIR}/modelscope}"
export HF_ENDPOINT="${HF_ENDPOINT:-${VERSE_HF_ENDPOINT}}"
export MODELS_PATH="${MODELS_PATH:-${VERSE_MODELS_DIR}}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-${VERSE_PIP_INDEX_URL}}"
unset PIP_EXTRA_INDEX_URL

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

VERSE_ENV_PREFIX="${CONDA_ENVS_PATH%%:*}/${VERSE_CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-${VERSE_ENV_PREFIX}/bin/python}"
export PYTHON_BIN

ensure_cache_layout() {
  mkdir -p \
    "${VERSE_CACHE_DIR}" \
    "${CONDA_ENVS_PATH}" \
    "${CONDA_PKGS_DIRS}" \
    "${HF_HOME}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${MPLCONFIGDIR}" \
    "${TMPDIR}" \
    "${INSIGHTFACE_HOME}" \
    "${MODELSCOPE_CACHE}" \
    "${VERSE_MODELS_DIR}" \
    "${VERSE_BENCH_DATA_DIR}"
}

ensure_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    local candidate
    for candidate in \
      "${VERSE_CONDA_EXE:-}" \
      "/opt/conda/bin/conda" \
      "${HOME}/miniforge3/bin/conda" \
      "${HOME}/miniconda3/bin/conda" \
      "${HOME}/anaconda3/bin/conda"; do
      if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        export PATH="$(dirname "${candidate}"):${PATH}"
        break
      fi
    done
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is required to create ${VERSE_ENV_PREFIX}" >&2
    exit 1
  fi
}

conda_env_exists() {
  [[ -x "${VERSE_ENV_PREFIX}/bin/python" ]]
}

create_env_if_missing() {
  ensure_conda
  if conda_env_exists; then
    return 0
  fi
  conda create -y -p "${VERSE_ENV_PREFIX}" python=3.10
}

conda_run() {
  local cmd="$1"
  shift
  if [[ ! -x "${VERSE_ENV_PREFIX}/bin/python" ]]; then
    echo "ERROR: Verse-Bench env is missing: ${VERSE_ENV_PREFIX}" >&2
    echo "Run: bash ${ROOT_DIR}/setup_verse_bench.sh" >&2
    exit 1
  fi
  case "${cmd}" in
    python)
      "${VERSE_ENV_PREFIX}/bin/python" "$@"
      ;;
    pip)
      "${VERSE_ENV_PREFIX}/bin/python" -m pip "$@"
      ;;
    *)
      PATH="${VERSE_ENV_PREFIX}/bin:${PATH}" "${cmd}" "$@"
      ;;
  esac
}

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${name}" >&2
    exit 1
  fi
}

copy_file_or_keep() {
  local source="$1"
  local target="$2"
  if [[ ! -s "${source}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${target}")"
  if [[ -f "${target}" && ! -L "${target}" && -s "${target}" ]]; then
    return 0
  fi
  rm -f "${target}"
  cp -aL "${source}" "${target}"
}

copy_dir_or_keep() {
  local source="$1"
  local target="$2"
  if [[ ! -d "${source}" ]]; then
    return 0
  fi
  if [[ -d "${target}" && ! -L "${target}" ]] && find "${target}" -mindepth 1 -print -quit | grep -q .; then
    return 0
  fi
  rm -rf "${target}"
  mkdir -p "${target}"
  cp -aL "${source}/." "${target}/"
}

download_url_if_missing() {
  local url="$1"
  local target="$2"
  if [[ -s "${target}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${target}")"
  local tmp="${target}.tmp"
  local aria2_cmd=""
  if command -v aria2c >/dev/null 2>&1; then
    aria2_cmd="$(command -v aria2c)"
  elif [[ -x "${VERSE_ENV_PREFIX}/bin/aria2c" ]]; then
    aria2_cmd="${VERSE_ENV_PREFIX}/bin/aria2c"
  fi
  if [[ -n "${aria2_cmd}" ]]; then
    "${aria2_cmd}" -c -x "${VERSE_ARIA2_CONNECTIONS:-8}" -s "${VERSE_ARIA2_CONNECTIONS:-8}" \
      --allow-overwrite=true --auto-file-renaming=false \
      -d "$(dirname "${target}")" -o "$(basename "${tmp}")" "${url}"
  else
    curl -L --fail --retry 3 -C - -o "${tmp}" "${url}"
  fi
  mv "${tmp}" "${target}"
}

reuse_local_verse_weights() {
  copy_file_or_keep \
    "${T2AV_COMPASS_DIR}/.cache/t2av-cache/weights/audiobox-aesthetics/checkpoint.pt" \
    "${VERSE_MODELS_DIR}/audiobox-aesthetics/checkpoint.pt"
  copy_file_or_keep \
    "${T2AV_COMPASS_DIR}/t2av-compass/Objective/Similarity/Synchformer-main/logs/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt" \
    "${VERSE_MODELS_DIR}/24-01-04T16-39-21.pt"
  copy_file_or_keep \
    "${T2AV_COMPASS_DIR}/t2av-compass/Objective/Similarity/LatentSync/checkpoints/auxiliary/syncnet_v2.model" \
    "${VERSE_MODELS_DIR}/syncnet_v2.model"
  copy_file_or_keep \
    "${T2AV_COMPASS_DIR}/t2av-compass/Objective/Video/aesthetic-predictor-v2-5/models/aesthetic_predictor_v2_5.pth" \
    "${VERSE_MODELS_DIR}/aesthetic_predictor_v2_5.pth"

  local siglip_snapshot="${T2AV_COMPASS_DIR}/.cache/t2av-cache/huggingface/transformers/models--google--siglip-so400m-patch14-384/snapshots"
  if [[ -d "${siglip_snapshot}" ]]; then
    local snapshot
    for snapshot in "${siglip_snapshot}"/*; do
      if [[ -d "${snapshot}" ]]; then
        copy_dir_or_keep "${snapshot}" "${VERSE_MODELS_DIR}/siglip-so400m-patch14-384"
        break
      fi
    done
  fi
}

download_hf_repo() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  mkdir -p "${local_dir}"
  conda_run hf download "${repo_id}" \
    --local-dir "${local_dir}" \
    --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
    --max-workers "${VERSE_HF_MAX_WORKERS:-8}" \
    "$@"
}

download_public_verse_models() {
  reuse_local_verse_weights

  if [[ ! -s "${VERSE_MODELS_DIR}/630k-audioset-fusion-best.pt" ]]; then
    download_url_if_missing \
      "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt" \
      "${VERSE_MODELS_DIR}/630k-audioset-fusion-best.pt"
  fi

  if [[ ! -s "${VERSE_MODELS_DIR}/ckpt_koniq10k.pt" ]]; then
    download_url_if_missing \
      "https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt" \
      "${VERSE_MODELS_DIR}/ckpt_koniq10k.pt"
  fi

  if [[ ! -s "${VERSE_MODELS_DIR}/raft-things.pth" ]]; then
    download_url_if_missing \
      "https://huggingface.co/ddrfan/RAFT/resolve/main/raft-things.pth" \
      "${VERSE_MODELS_DIR}/raft-things.pth"
  fi

  if [[ ! -s "${VERSE_MODELS_DIR}/SenseVoiceSmall/am.mvn" || ! -s "${VERSE_MODELS_DIR}/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model" || ! -s "${VERSE_MODELS_DIR}/SenseVoiceSmall/config.yaml" || ! -s "${VERSE_MODELS_DIR}/SenseVoiceSmall/configuration.json" ]]; then
    download_hf_repo FunAudioLLM/SenseVoiceSmall "${VERSE_MODELS_DIR}/SenseVoiceSmall" \
      --include \
        "am.mvn" \
        "chn_jpn_yue_eng_ko_spectok.bpe.model" \
        "config.yaml" \
        "configuration.json"
    rm -f "${VERSE_MODELS_DIR}/SenseVoiceSmall/requirements.txt"
  fi
  if [[ ! -s "${VERSE_MODELS_DIR}/SenseVoiceSmall/model.pt" ]]; then
    download_url_if_missing \
      "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main/model.pt" \
      "${VERSE_MODELS_DIR}/SenseVoiceSmall/model.pt"
  fi

  if [[ ! -s "${VERSE_MODELS_DIR}/roberta-base/config.json" || ! -s "${VERSE_MODELS_DIR}/roberta-base/merges.txt" || ! -s "${VERSE_MODELS_DIR}/roberta-base/tokenizer.json" || ! -s "${VERSE_MODELS_DIR}/roberta-base/tokenizer_config.json" || ! -s "${VERSE_MODELS_DIR}/roberta-base/vocab.json" ]]; then
    download_hf_repo FacebookAI/roberta-base "${VERSE_MODELS_DIR}/roberta-base" \
      --include \
        "config.json" \
        "merges.txt" \
        "special_tokens_map.json" \
        "tokenizer.json" \
        "tokenizer_config.json" \
        "vocab.json"
  fi
  if [[ ! -s "${VERSE_MODELS_DIR}/roberta-base/model.safetensors" ]]; then
    download_url_if_missing \
      "https://huggingface.co/FacebookAI/roberta-base/resolve/main/model.safetensors" \
      "${VERSE_MODELS_DIR}/roberta-base/model.safetensors"
  fi

  if [[ ! -s "${VERSE_MODELS_DIR}/dinov3-vitl16-pretrain-lvd1689m/config.json" || ! -s "${VERSE_MODELS_DIR}/dinov3-vitl16-pretrain-lvd1689m/preprocessor_config.json" ]]; then
    download_hf_repo "${VERSE_DINOV3_REPO}" \
      "${VERSE_MODELS_DIR}/dinov3-vitl16-pretrain-lvd1689m" \
      --include \
        "config.json" \
        "preprocessor_config.json"
  fi
  if [[ ! -s "${VERSE_MODELS_DIR}/dinov3-vitl16-pretrain-lvd1689m/model.safetensors" ]]; then
    download_url_if_missing \
      "https://huggingface.co/${VERSE_DINOV3_REPO}/resolve/main/model.safetensors" \
      "${VERSE_MODELS_DIR}/dinov3-vitl16-pretrain-lvd1689m/model.safetensors"
  fi

  if [[ ! -s "${VERSE_MODELS_DIR}/siglip-so400m-patch14-384/config.json" || ! -s "${VERSE_MODELS_DIR}/siglip-so400m-patch14-384/preprocessor_config.json" ]]; then
    download_hf_repo google/siglip-so400m-patch14-384 \
      "${VERSE_MODELS_DIR}/siglip-so400m-patch14-384" \
      --include \
        "config.json" \
        "preprocessor_config.json" \
        "special_tokens_map.json" \
        "tokenizer.json" \
        "tokenizer_config.json" \
        "vocab.txt"
  fi
  if [[ ! -s "${VERSE_MODELS_DIR}/siglip-so400m-patch14-384/model.safetensors" ]]; then
    download_url_if_missing \
      "https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/model.safetensors" \
      "${VERSE_MODELS_DIR}/siglip-so400m-patch14-384/model.safetensors"
  fi
}

try_download_official_verse_models() {
  if [[ "${VERSE_SKIP_OFFICIAL_MODELS:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    local hf_identity
    hf_identity="$(conda_run hf auth whoami 2>&1 || true)"
    if [[ -z "${hf_identity}" || "${hf_identity}" == "Not logged in"* ]]; then
      echo "WARN: skipping official gated Verse-Bench model repo because no Hugging Face login/token is available." >&2
      return 0
    fi
  fi
  local output
  set +e
  output="$(conda_run hf download zuoweizwzw/Verse-Bench-Models \
    --local-dir "${VERSE_MODELS_DIR}" \
    --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
    --max-workers "${VERSE_HF_MAX_WORKERS:-8}" \
    --include \
      "24-01-04T16-39-21.pt" \
      "630k-audioset-fusion-best.pt" \
      "SenseVoiceSmall/*" \
      "aesthetic_predictor_v2_5.pth" \
      "audiobox-aesthetics/checkpoint.pt" \
      "ckpt_koniq10k.pt" \
      "dinov3-vitl16-pretrain-lvd1689m/*" \
      "raft-things.pth" \
      "roberta-base/config.json" \
      "roberta-base/merges.txt" \
      "roberta-base/model.safetensors" \
      "roberta-base/special_tokens_map.json" \
      "roberta-base/tokenizer.json" \
      "roberta-base/tokenizer_config.json" \
      "roberta-base/vocab.json" \
      "siglip-so400m-patch14-384/*" \
      "syncnet_v2.model" 2>&1)"
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "WARN: official Verse-Bench model repo was not downloaded; falling back to public/local sources." >&2
    echo "WARN: ${output}" >&2
  fi
  rm -f "${VERSE_MODELS_DIR}/SenseVoiceSmall/requirements.txt"
  return 0
}

verse_required_model_files() {
  local models_dir="${1:-${VERSE_MODELS_DIR}}"
  printf '%s\n' \
    "${models_dir}/24-01-04T16-39-21.pt" \
    "${models_dir}/630k-audioset-fusion-best.pt" \
    "${models_dir}/SenseVoiceSmall/am.mvn" \
    "${models_dir}/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model" \
    "${models_dir}/SenseVoiceSmall/config.yaml" \
    "${models_dir}/SenseVoiceSmall/configuration.json" \
    "${models_dir}/SenseVoiceSmall/model.pt" \
    "${models_dir}/aesthetic_predictor_v2_5.pth" \
    "${models_dir}/audiobox-aesthetics/checkpoint.pt" \
    "${models_dir}/ckpt_koniq10k.pt" \
    "${models_dir}/dinov3-vitl16-pretrain-lvd1689m/config.json" \
    "${models_dir}/dinov3-vitl16-pretrain-lvd1689m/model.safetensors" \
    "${models_dir}/dinov3-vitl16-pretrain-lvd1689m/preprocessor_config.json" \
    "${models_dir}/raft-things.pth" \
    "${models_dir}/roberta-base/config.json" \
    "${models_dir}/roberta-base/merges.txt" \
    "${models_dir}/roberta-base/model.safetensors" \
    "${models_dir}/roberta-base/tokenizer.json" \
    "${models_dir}/roberta-base/tokenizer_config.json" \
    "${models_dir}/roberta-base/vocab.json" \
    "${models_dir}/siglip-so400m-patch14-384/config.json" \
    "${models_dir}/siglip-so400m-patch14-384/model.safetensors" \
    "${models_dir}/siglip-so400m-patch14-384/preprocessor_config.json" \
    "${models_dir}/syncnet_v2.model"
}

verse_model_files_present() {
  local models_dir="${1:-${VERSE_MODELS_DIR}}"
  local path
  while IFS= read -r path; do
    if [[ ! -s "${path}" ]]; then
      return 1
    fi
  done < <(verse_required_model_files "${models_dir}")
}

require_verse_model_files() {
  local models_dir="${1:-${VERSE_MODELS_DIR}}"
  local missing=0
  local path
  while IFS= read -r path; do
    if [[ ! -s "${path}" ]]; then
      echo "ERROR: required Verse-Bench model file is missing: ${path}" >&2
      missing=1
    fi
  done < <(verse_required_model_files "${models_dir}")
  if [[ ${missing} -ne 0 ]]; then
    echo "ERROR: model setup is incomplete. If the fallback sources fail, login with a token that has access to zuoweizwzw/Verse-Bench-Models and rerun setup." >&2
    exit 1
  fi
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "${path}"
  fi
}

write_filtered_requirements() {
  local output="${VERSE_CACHE_DIR}/requirements/requirements-no-torch.txt"
  mkdir -p "$(dirname "${output}")"
  grep -Ev '^(torch|torchvision|torchaudio|opencv-python|pyiqa)([<>=~! ].*)?$' "${ROOT_DIR}/requirements.txt" >"${output}"
  printf '%s\n' "${output}"
}

install_python_requirements() {
  local marker="${VERSE_CACHE_DIR}/markers/${VERSE_CONDA_ENV}-requirements-v5.ready"
  mkdir -p "$(dirname "${marker}")"
  if [[ -f "${marker}" ]]; then
    return 0
  fi

  conda_run python -m pip install --upgrade pip
  conda_run pip install "setuptools<81" wheel packaging ninja
  conda_run pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

  local filtered_requirements
  filtered_requirements="$(write_filtered_requirements)"
  conda_run pip install -r "${filtered_requirements}"
  conda_run pip install numpy==1.26.4 opencv-python-headless==4.10.0.84
  conda_run pip install --no-deps \
    pyiqa==0.1.14.1 \
    facexlib==0.3.0 \
    openai-clip==1.0.1
  conda_run pip install --no-build-isolation --no-deps basicsr==1.4.2
  conda_run pip install \
    aesthetic-predictor-v2-5==2024.12.18.1 \
    audiobox-aesthetics==0.0.4 \
    hear21passt==0.0.26 \
    h5py==3.15.1 \
    torchlibrosa==0.1.0 \
    wget==3.2 \
    addict==2.4.0 \
    lmdb==1.7.5 \
    filterpy==1.4.5 \
    future==1.0.0 \
    icecream==2.1.8 \
    lpips==0.1.4 \
    tensorboard==2.20.0 \
    gdown==5.2.0 \
    yapf==0.43.0 \
    pyloudnorm==0.1.1 \
    python_speech_features==0.6 \
    opencv-python-headless==4.10.0.84 \
    av==12.3.0 \
    onnxruntime-gpu \
    yt-dlp

  if command -v conda >/dev/null 2>&1; then
    conda install -y -p "${VERSE_ENV_PREFIX}" -c conda-forge ffmpeg aria2 || true
  fi

  touch "${marker}"
}

download_verse_models() {
  local marker="${VERSE_CACHE_DIR}/markers/models-v2.ready"
  mkdir -p "$(dirname "${marker}")"
  if [[ -f "${marker}" ]]; then
    if verse_model_files_present "${VERSE_MODELS_DIR}"; then
      return 0
    fi
    echo "WARN: model marker exists, but required files are incomplete. Repairing model directory." >&2
    rm -f "${marker}"
  fi

  try_download_official_verse_models
  download_public_verse_models
  require_verse_model_files "${VERSE_MODELS_DIR}"

  touch "${marker}"
}

download_verse_dataset_metadata() {
  if [[ "${VERSE_DOWNLOAD_DATASET}" != "1" ]]; then
    return 0
  fi

  local marker="${VERSE_CACHE_DIR}/markers/dataset-metadata-v1.ready"
  mkdir -p "$(dirname "${marker}")"
  if [[ -f "${marker}" ]]; then
    return 0
  fi

  conda_run hf download dorni/Verse-Bench \
    --repo-type dataset \
    --local-dir "${VERSE_BENCH_DATA_DIR}" \
    --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
    --max-workers "${VERSE_HF_MAX_WORKERS:-8}"

  touch "${marker}"
}

prefetch_auxiliary_weights() {
  if [[ "${VERSE_PREFETCH_AUX}" != "1" ]]; then
    return 0
  fi

  local marker="${VERSE_CACHE_DIR}/markers/auxiliary-v1.ready"
  mkdir -p "$(dirname "${marker}")"
  if [[ -f "${marker}" ]]; then
    return 0
  fi

  conda_run python "${SCRIPT_DIR}/prefetch_auxiliary.py"
  touch "${marker}"
}

bootstrap_verse_runtime() {
  ensure_cache_layout
  create_env_if_missing
  install_python_requirements
  download_verse_models
  download_verse_dataset_metadata
  prefetch_auxiliary_weights
}
