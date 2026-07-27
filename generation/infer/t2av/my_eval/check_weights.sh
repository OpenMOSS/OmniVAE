#!/usr/bin/env bash
# Inventory every model checkpoint that the 8 my_eval metric tasks load. Prints
# a per-weight present/missing report plus exact download commands for whatever
# is missing.
#
# Usage:
#   bash generation/infer/t2av/my_eval/check_weights.sh
#
# Output sections:
#   1. Resolved paths (MODELS_PATH, TORCH_HOME, etc.)
#   2. Per-weight status table
#   3. Concrete download commands for each missing file
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
my_eval_dir="$(cd "$(dirname "${script_path}")" && pwd)"
project_root="$(cd "${my_eval_dir}/../../../.." && pwd)"

VERSE_BENCH_ROOT="${MY_EVAL_VERSE_BENCH_ROOT:-${project_root}/generation/evaluation/verse_bench}"
_USER_MODELS_PATH_SET="${MODELS_PATH+x}"
_USER_TORCH_HOME_SET="${TORCH_HOME+x}"
_USER_MY_EVAL_VERSE_MODELS_SET="${MY_EVAL_VERSE_MODELS+x}"
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
if [[ ! -d "${OPEN_SOURCE_T2AV_EVAL_MODELS}" && -d "${project_root}/open_source/eval_models/t2av" ]]; then
    OPEN_SOURCE_T2AV_EVAL_MODELS="${project_root}/open_source/eval_models/t2av"
fi
if [[ -d "${OPEN_SOURCE_T2AV_EVAL_MODELS}" ]]; then
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

# Resolved defaults; everything reachable by the metric tasks ends up under one
# of these roots.
MODELS_PATH_R="${MODELS_PATH:-${VERSE_BENCH_ROOT}/models}"
TORCH_HOME_R="${TORCH_HOME:-${HOME}/.cache/torch}"
EVAL_ROOT="${project_root}/generation/evaluation"

export MODELS_PATH_R TORCH_HOME_R EVAL_ROOT PROJECT_ROOT="${project_root}"

echo "============================================"
echo "  my_eval weight inventory"
echo "============================================"
echo "PROJECT_ROOT      : ${project_root}"
echo "MODELS_PATH       : ${MODELS_PATH_R}"
echo "TORCH_HOME        : ${TORCH_HOME_R}"
echo "EVAL_ROOT         : ${EVAL_ROOT}"
echo "VERSE_BENCH_ROOT  : ${VERSE_BENCH_ROOT}"
echo "OPEN_SOURCE_T2AV  : ${OPEN_SOURCE_T2AV_EVAL_MODELS}"
echo "MY_EVAL_PE_AV     : ${MY_EVAL_PE_AV_MODEL_DIR:-<unset>}"
echo "MY_EVAL_CLAP_DIR  : ${MY_EVAL_AUDIO_IS_CLAP_DIR:-<unset>}"
echo "MY_EVAL_DNSMOS    : ${MY_EVAL_DNSMOS_DIR:-<unset>}"
echo "============================================"
echo ""

python3 - <<'PY'
"""Check every weight the 8 metrics consume."""
from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

MODELS_PATH = Path(os.environ["MODELS_PATH_R"])
TORCH_HOME = Path(os.environ["TORCH_HOME_R"])
EVAL_ROOT = Path(os.environ["EVAL_ROOT"])
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
PE_AV_MODEL_DIR = Path(os.environ.get(
    "MY_EVAL_PE_AV_MODEL_DIR",
    "",
))
SYNCHFORMER_CKPT = Path(os.environ.get(
    "MY_EVAL_SYNCHFORMER_CKPT",
    str(MODELS_PATH / "24-01-04T16-39-21.pt"),
))
SYNCNET_MODEL = Path(os.environ.get(
    "MY_EVAL_SYNCNET_MODEL",
    str(EVAL_ROOT / "models/wav2lip/evaluation/syncnet_python/data/syncnet_v2.model"),
))
S3FD_WEIGHT = Path(os.environ.get(
    "MY_EVAL_S3FD_WEIGHT",
    str(EVAL_ROOT / "models/wav2lip/evaluation/syncnet_python/detectors/s3fd/weights/sfd_face.pth"),
))
AUDIO_IS_CLAP_DIR = Path(os.environ.get(
    "MY_EVAL_AUDIO_IS_CLAP_DIR",
    str(EVAL_ROOT / "metrics/audio_is_clap"),
))
PANNS_HOME = Path(os.environ.get(
    "MY_EVAL_PANNS_HOME",
    str(AUDIO_IS_CLAP_DIR / "pann_home"),
))
DNSMOS_DIR = Path(os.environ.get(
    "MY_EVAL_DNSMOS_DIR",
    str(EVAL_ROOT / "metrics/dnsmos"),
))
AUDIOBOX_CKPT = Path(os.environ.get(
    "MY_EVAL_AUDIOBOX_CKPT",
    str(MODELS_PATH / "audiobox-aesthetics/checkpoint.pt"),
))


def fmt_size(p: Path) -> str:
    try:
        sz = p.stat().st_size
    except OSError:
        return "0B"
    for unit in ["B", "KB", "MB", "GB"]:
        if sz < 1024:
            return f"{sz:.1f}{unit}"
        sz /= 1024
    return f"{sz:.1f}TB"


# Each entry: name, task using it, list of candidate paths (any present == OK),
# notes, and how to download (a curl/wget that lands the file at the FIRST
# candidate path).
WEIGHTS = [
    # ---- pe_av ----
    {
        "name": "facebook/pe-av-large model",
        "task": "pe_av (PE-TV/TA/TAV + cosine)",
        "candidates": [
            PE_AV_MODEL_DIR / "model.safetensors",
            PE_AV_MODEL_DIR / "pytorch_model.bin",
            PE_AV_MODEL_DIR / "model.safetensors.index.json",
            PE_AV_MODEL_DIR / "pytorch_model.bin.index.json",
        ],
        "size_hint": "~8.4 GB",
        "download": dedent("""\
            # Preferred: use the already mirrored local HF model path.
            # If missing, download on a node with outbound HTTPS:
            mkdir -p {target_dir}
            huggingface-cli download facebook/pe-av-large \\
                --local-dir {target_dir} \\
                --local-dir-use-symlinks False

            # Or point evaluation at another complete local copy:
            export MY_EVAL_PE_AV_MODEL_DIR=/path/to/facebook/pe-av-large
        """),
    },
    # ---- av_sync_imagebind ----
    {
        "name": "Synchformer state_dict",
        "task": "av_sync_imagebind (DeSync)",
        "candidates": [
            SYNCHFORMER_CKPT,
            EVAL_ROOT / "metrics/av_quality/weights/synchformer_state_dict.pth",
            MODELS_PATH / "24-01-04T16-39-21.pt",  # Verse-Bench alias
        ],
        "size_hint": "~927 MB",
        "download": dedent("""\
            # Option A: official mirror used by MOVA download_weights.sh
            mkdir -p {target_dir}
            wget -O {target_dir}/synchformer_state_dict.pth \\
                https://a3s.fi/synchformer_data/sync_models/24-01-04T16-39-21/synchformer_state_dict.pth

            # Option B: reuse Verse-Bench's HF download
            cd {verse_bench_root}
            bash setup_verse_bench.sh   # will fill models/24-01-04T16-39-21.pt
        """),
    },
    {
        "name": "ImageBind huge",
        "task": "av_sync_imagebind (IB-AV + IB-TV)",
        # The vendored imagebind in MOVA looks at ${TORCH_HOME}/checkpoints/
        # (no /hub layer). We also accept ${TORCH_HOME}/hub/checkpoints/
        # because that's where torch.hub defaults to; if the file lives in the
        # hub path you'll want to symlink it into the canonical (first) one.
        "candidates": [
            TORCH_HOME / "checkpoints/imagebind_huge.pth",
            TORCH_HOME / "hub/checkpoints/imagebind_huge.pth",
        ],
        "size_hint": "~4.5 GB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/imagebind_huge.pth \\
                https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
        """),
    },
    # ---- lip_sync ----
    {
        "name": "SyncNet v2",
        "task": "lip_sync (LSE-D/LSE-C)",
        "candidates": [
            SYNCNET_MODEL,
            EVAL_ROOT / "models/wav2lip/evaluation/syncnet_python/data/syncnet_v2.model",
            MODELS_PATH / "syncnet_v2.model",
        ],
        "size_hint": "~50 MB",
        "download": dedent("""\
            mkdir -p {target_dir}
            # Original mirror (used by Wav2Lip)
            wget -O {target_dir}/syncnet_v2.model \\
                'https://www.robots.ox.ac.uk/~vgg/data/syncnet/data/syncnet_v2.model'
        """),
    },
    {
        "name": "S3FD face detector (sfd_face.pth)",
        "task": "lip_sync (face detection)",
        "candidates": [
            S3FD_WEIGHT,
            EVAL_ROOT / "models/wav2lip/evaluation/syncnet_python/detectors/s3fd/weights/sfd_face.pth",
        ],
        "size_hint": "~85 MB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/sfd_face.pth \\
                'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'
        """),
    },
    # ---- audio_clap ----
    {
        "name": "LAION-CLAP 630k-audioset-fusion-best",
        "task": "audio_clap (CLAP score)",
        "candidates": [
            AUDIO_IS_CLAP_DIR / "clap_ckpt/630k-audioset-fusion-best.pt",
            EVAL_ROOT / "metrics/audio_is_clap/clap_ckpt/630k-audioset-fusion-best.pt",
            MODELS_PATH / "630k-audioset-fusion-best.pt",
        ],
        "size_hint": "~2.4 GB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/630k-audioset-fusion-best.pt \\
                https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt
        """),
    },
    {
        "name": "Roberta-base (text encoder for CLAP)",
        "task": "audio_clap",
        "candidates": [
            MODELS_PATH / "roberta-base/model.safetensors",
            MODELS_PATH / "roberta-base/pytorch_model.bin",
            EVAL_ROOT / "models/roberta-base/model.safetensors",
            EVAL_ROOT / "models/roberta-base/pytorch_model.bin",
        ],
        "size_hint": "~500 MB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/model.safetensors \\
                https://huggingface.co/FacebookAI/roberta-base/resolve/main/model.safetensors
            # If you also need tokenizer files (config.json, tokenizer.json, vocab.json,
            # merges.txt, special_tokens_map.json, tokenizer_config.json), grab them
            # from the same Hugging Face repo.
        """),
    },
    # ---- speech_wer ----
    {
        "name": "SenseVoiceSmall ASR model",
        "task": "speech_wer",
        "candidates": [
            MODELS_PATH / "SenseVoiceSmall/model.pt",
        ],
        "size_hint": "~900 MB",
        "download": dedent("""\
            cd {verse_bench_root}
            bash setup_verse_bench.sh   # will fill models/SenseVoiceSmall/
        """),
    },
    {
        "name": "FSMN VAD model",
        "task": "speech_wer",
        "candidates": [
            MODELS_PATH / "speech_fsmn_vad_zh-cn-16k-common-pytorch/model.pt",
        ],
        "size_hint": "~2 MB",
        "download": dedent("""\
            cd {verse_bench_root}
            bash setup_verse_bench.sh   # will fill models/speech_fsmn_vad_zh-cn-16k-common-pytorch/
        """),
    },
    # ---- video_aesthetic (AS = Aesthetic + MusiQ + ManiQA) ----
    {
        "name": "Aesthetic predictor v2.5 head",
        "task": "video_aesthetic (Aesthetic)",
        "candidates": [
            MODELS_PATH / "aesthetic_predictor_v2_5.pth",
        ],
        "size_hint": "~50 KB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/aesthetic_predictor_v2_5.pth \\
                https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/aesthetic_predictor_v2_5.pth
            # Alternative: pip install aesthetic-predictor-v2-5 already ships the head
            # inside its package; you can also symlink the in-package copy here.
        """),
    },
    {
        "name": "SigLIP so400m-patch14-384 (vision encoder)",
        "task": "video_aesthetic (Aesthetic backbone)",
        "candidates": [
            MODELS_PATH / "siglip-so400m-patch14-384/model.safetensors",
        ],
        "size_hint": "~3.5 GB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/model.safetensors \\
                https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/model.safetensors
            # The companion config.json + preprocessor_config.json must also be at
            # {target_dir}/. setup_verse_bench.sh already pulls them.
        """),
    },
    {
        "name": "ManiQA Koniq-10k",
        "task": "video_aesthetic (ManiQA)",
        "candidates": [
            MODELS_PATH / "ckpt_koniq10k.pt",
        ],
        "size_hint": "~210 MB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/ckpt_koniq10k.pt \\
                https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt
        """),
    },
    {
        "name": "pyiqa MusiQ cache",
        "task": "video_aesthetic (MusiQ)",
        "candidates": [
            TORCH_HOME / "hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth",
        ],
        "size_hint": "~104 MB",
        "download": dedent("""\
            cd {verse_bench_root}
            bash setup_verse_bench.sh   # prefetches pyiqa MusiQ into TORCH_HOME
        """),
    },
    # ---- identity_dino / video_motion ----
    {
        "name": "DINOv3 ViT-L/16",
        "task": "identity_dino",
        "candidates": [
            MODELS_PATH / "dinov3-vitl16-pretrain-lvd1689m/model.safetensors",
        ],
        "size_hint": "~1.2 GB",
        "download": dedent("""\
            cd {verse_bench_root}
            bash setup_verse_bench.sh   # will fill models/dinov3-vitl16-pretrain-lvd1689m/
        """),
    },
    {
        "name": "RAFT Things",
        "task": "video_motion",
        "candidates": [
            MODELS_PATH / "raft-things.pth",
        ],
        "size_hint": "~21 MB",
        "download": dedent("""\
            cd {verse_bench_root}
            bash setup_verse_bench.sh   # will fill models/raft-things.pth
        """),
    },
    # ---- audio_fd_kl ----
    {
        "name": "PaSST HEAR21 cache",
        "task": "audio_fd_kl (KL)",
        "candidates": [
            TORCH_HOME / "hub/checkpoints/passt-s-f128-p16-s10-ap.476-swa.pt",
        ],
        "size_hint": "~329 MB",
        "download": dedent("""\
            cd {verse_bench_root}
            bash setup_verse_bench.sh   # prefetches PaSST into TORCH_HOME
        """),
    },
    # ---- audio_box ----
    {
        "name": "audiobox-aesthetics checkpoint",
        "task": "audio_box (CE/CU/PC/PQ)",
        "candidates": [
            AUDIOBOX_CKPT,
            MODELS_PATH / "audiobox-aesthetics/checkpoint.pt",
        ],
        "size_hint": "~530 MB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O {target_dir}/checkpoint.pt \\
                https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt
            # If unreachable, the audiobox_aesthetics pip package ships a default
            # checkpoint that initialize_predictor() can auto-load.
        """),
    },
    # ---- audio_dnsmos (ONNX, committed in repo) ----
    {
        "name": "DNSMOS sig_bak_ovr.onnx",
        "task": "audio_dnsmos",
        "candidates": [
            DNSMOS_DIR / "DNSMOS/sig_bak_ovr.onnx",
            EVAL_ROOT / "metrics/dnsmos/DNSMOS/sig_bak_ovr.onnx",
        ],
        "size_hint": "~1.5 MB",
        "download": dedent("""\
            # Already committed in the MOVA repo; if missing, restore from git:
            cd {project_root}
            git checkout HEAD -- generation/evaluation/metrics/dnsmos/DNSMOS/
        """),
    },
    {
        "name": "DNSMOS model_v8.onnx (P808 head)",
        "task": "audio_dnsmos (P808_MOS)",
        "candidates": [
            DNSMOS_DIR / "DNSMOS/model_v8.onnx",
            EVAL_ROOT / "metrics/dnsmos/DNSMOS/model_v8.onnx",
        ],
        "size_hint": "~3 MB",
        "download": dedent("""\
            # Same as sig_bak_ovr.onnx -- committed in the MOVA repo.
            cd {project_root}
            git checkout HEAD -- generation/evaluation/metrics/dnsmos/DNSMOS/
        """),
    },
    # ---- audio_is ----
    {
        "name": "PANNs Cnn14 (32 kHz, mAP=0.431)",
        "task": "audio_is",
        "candidates": [
            PANNS_HOME / "Cnn14_mAP=0.431.pth",
            EVAL_ROOT / "metrics/audio_is_clap/pann_home/Cnn14_mAP=0.431.pth",
            Path.home() / "panns_data/Cnn14_mAP=0.431.pth",
        ],
        "size_hint": "~310 MB",
        "download": dedent("""\
            mkdir -p {target_dir}
            wget -O '{target_dir}/Cnn14_mAP=0.431.pth' \\
                'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
            # The cluster's outbound HTTPS to zenodo.org is currently blocked --
            # download from a node with internet (login node usually has it) and
            # rsync the file to this path. Once present, panns_inference will skip
            # the download.
        """),
    },
    # ---- audio_amplitude: no model needed ----
]


def status(p: Path) -> tuple[bool, str]:
    if not p.exists():
        return False, "missing"
    if not p.is_file():
        return False, "not a regular file"
    if p.stat().st_size == 0:
        return False, "0 bytes"
    return True, fmt_size(p)


def color(ok: bool) -> tuple[str, str]:
    return ("\033[32m", "\033[0m") if ok else ("\033[31m", "\033[0m")


missing = []
print(f"{'#':<3} {'STATUS':<8} {'WEIGHT':<45} {'SIZE / WHERE':<28} TASK")
print("-" * 120)
for idx, w in enumerate(WEIGHTS, 1):
    found_path = None
    found_status = "missing"
    for cand in w["candidates"]:
        ok, info = status(cand)
        if ok:
            found_path = cand
            found_status = info
            break
    is_ok = found_path is not None
    c_on, c_off = color(is_ok)
    tag = "OK" if is_ok else "MISS"
    where = (str(found_path).replace(str(PROJECT_ROOT) + "/", "") if found_path
             else w.get("size_hint", ""))
    print(f"{idx:<3} {c_on}{tag:<8}{c_off} {w['name']:<45} {where:<28} {w['task']}")
    if not is_ok:
        missing.append(w)

print("-" * 120)
print(f"\nTotal: {len(WEIGHTS)} weights checked, {len(missing)} missing.\n")

if missing:
    print("=" * 80)
    print(f"  DOWNLOAD INSTRUCTIONS FOR {len(missing)} MISSING WEIGHT(S)")
    print("=" * 80)
    for w in missing:
        target = w["candidates"][0]
        cmd = w["download"].format(
            target_dir=str(target.parent),
            project_root=str(PROJECT_ROOT),
            verse_bench_root=str(PROJECT_ROOT / "generation/evaluation/verse_bench"),
        )
        print(f"\n--- {w['name']}  ({w['task']})  [{w.get('size_hint', '?')}]")
        print(f"    target: {target}")
        print()
        for line in cmd.rstrip().split("\n"):
            print(f"    {line}")
    print()
    print("=" * 80)
    print("  Cluster nodes cannot reach the public internet (zenodo / HF / GitHub).")
    print("  Run the above commands on a login / dev node that has outbound HTTPS,")
    print("  then make sure the shared filesystem mount is visible to compute nodes.")
    print("=" * 80)
PY
