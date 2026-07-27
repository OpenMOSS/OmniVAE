#!/usr/bin/env bash
# Install the handful of extra pip packages that the my_eval toolkit needs on
# top of the Verse-Bench conda env created by setup_verse_bench.sh.
#
# Run this AFTER setup_verse_bench.sh has produced
# generation/evaluation/verse_bench/.cache/conda/envs/verse-bench/.
#
# Idempotent: pip skips packages that are already installed.
set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
my_eval_dir="$(cd "$(dirname "${script_path}")" && pwd)"
project_root="$(cd "${my_eval_dir}/../../../.." && pwd)"

VERSE_BENCH_ROOT="${MY_EVAL_VERSE_BENCH_ROOT:-${project_root}/generation/evaluation/verse_bench}"
if [[ ! -f "${VERSE_BENCH_ROOT}/scripts/common.sh" ]]; then
    echo "ERROR: Verse-Bench/scripts/common.sh not found at ${VERSE_BENCH_ROOT}" >&2
    echo "       Set MY_EVAL_VERSE_BENCH_ROOT or run setup_verse_bench.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1090,SC1091
source "${VERSE_BENCH_ROOT}/scripts/common.sh"

if [[ ! -x "${VERSE_ENV_PREFIX}/bin/python" ]]; then
    echo "ERROR: verse-bench env missing at ${VERSE_ENV_PREFIX}" >&2
    echo "       Run: bash ${VERSE_BENCH_ROOT}/setup_verse_bench.sh" >&2
    exit 1
fi

PIP="${VERSE_ENV_PREFIX}/bin/python -m pip"

echo "[my_eval setup] python: ${VERSE_ENV_PREFIX}/bin/python"
echo "[my_eval setup] installing missing pip packages..."

# panns-inference: used by audio_is.py (PANNs Cnn14 Inception Score).
${PIP} install --no-deps panns-inference

# pytorchvideo.data needs the full iopath + fvcore stack. Their transitive
# dependencies (portalocker, tabulate, termcolor, yacs, pyyaml, parameterized)
# are small CPU-only packages -- safe to install with pip's default resolver,
# they will not touch torch/torchvision. We THEN install iopath/fvcore/
# pytorchvideo themselves with --no-deps so pip cannot try to upgrade torch.
${PIP} install \
    portalocker \
    tabulate \
    termcolor \
    yacs \
    pyyaml \
    parameterized
${PIP} install --no-deps iopath fvcore
${PIP} install --no-deps pytorchvideo

# audiotools (preferred over pyloudnorm for LUFS; falls back gracefully if
# unavailable). MOVA prefers it. Optional.
${PIP} install --no-deps descript-audiotools || echo "(skipping audiotools; pyloudnorm fallback)"

# soundfile is required by utils/audio_video.load_wav_mono. Already in
# verse-bench requirements but ensure.
${PIP} install --no-deps soundfile

# PE-AV support is available in transformers 5.x and imports timm components.
${PIP} install "transformers==5.3.0"
${PIP} install --no-deps "timm>=1.0.0"

# scenedetect is imported by MOVA's wav2lip/evaluation/syncnet_python/run_pipeline.py
# (used by the lip_sync task for scene cut detection inside the face track).
# MUST pin to <0.6 because that version removed the ``video_manager`` module
# which MOVA's pipeline imports directly. Click is the only runtime dep.
${PIP} install click
${PIP} install --no-deps "scenedetect<0.6"

# SyncNet imports python_speech_features.mfcc. The repo also carries a local
# compatibility module, but installing the package keeps direct imports working.
${PIP} install --no-deps "python_speech_features>=0.6"

echo "[my_eval setup] verifying critical imports..."
export PROJECT_ROOT="${project_root}"
"${VERSE_ENV_PREFIX}/bin/python" - <<'PY'
import importlib
import sys

# torchvision 0.17+ removed ``torchvision.transforms.functional_tensor``;
# pytorchvideo 0.1.5 (and imagebind via pv_transforms) still imports it.
# Provide a thin shim that re-exports torchvision.transforms.functional so the
# old import path keeps working. The handful of functions pytorchvideo touches
# (rgb_to_grayscale, etc.) live under the new path.
try:
    import torchvision.transforms.functional as _tvF
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _tvF)
except Exception as _exc:
    print(f"  warn  torchvision.transforms.functional shim: {_exc}")

EXTRA_PATHS = []
def add(p):
    if p not in sys.path:
        sys.path.insert(0, p)

import os
project_root = os.environ["PROJECT_ROOT"] if "PROJECT_ROOT" in os.environ else None
# Best-effort path bootstrap: paths that my_eval tasks themselves add at runtime.
# NOTE: deliberately omit generation/evaluation/models/pytorchvideo here.
# The vendored copy lacks the ``data/`` submodule that imagebind/data.py needs;
# prepending it to sys.path would shadow the pip-installed pytorchvideo and
# cause ``ModuleNotFoundError: pytorchvideo.data`` even when pip install
# succeeded.
candidates = [
    "generation/evaluation/models/imagebind",
    "generation/evaluation/models/clap/src",
    "generation/evaluation/models/clap/src/laion_clap",
    "generation/evaluation/metrics/audio_is_clap",
    "generation/evaluation/metrics/av_quality",
    "generation/evaluation/metrics/dnsmos",
    "generation/evaluation/metrics/audio_amplitude",
    "generation/evaluation/models/wav2lip/evaluation/syncnet_python",
    "generation/evaluation/verse_bench",
]
if project_root:
    for c in candidates:
        add(os.path.join(project_root, c))

checks = [
    # core torch stack
    ("torch", True),
    ("torchaudio", True),
    ("torchvision", True),
    ("transformers", True),
    ("transformers.models.pe_audio_video", True),
    ("timm", True),
    # general I/O + DSP
    ("librosa", True),
    ("pyloudnorm", True),
    ("onnxruntime", True),
    ("av", True),
    ("moviepy.editor", True),
    ("pyiqa", True),
    # Verse-Bench-ported metrics
    ("aesthetic_predictor_v2_5", True),
    ("audiobox_aesthetics", True),
    # PANNs Inception Score
    ("panns_inference", True),
    # ImageBind + its transitive dep chain (the place we keep tripping over)
    ("portalocker", True),
    ("iopath.common.file_io", True),
    ("fvcore.common.config", True),
    ("pytorchvideo.transforms", True),
    ("pytorchvideo.data.clip_sampling", True),
    ("pytorchvideo.data.encoded_video", True),
    ("imagebind.data", True),
    ("imagebind.models.imagebind_model", True),
    # MOVA CLAP path
    ("laion_clap", True),
    ("clap_module.factory", True),
    # MOVA lip_sync pipeline: needs the OLD scenedetect API (video_manager
    # module). scenedetect>=0.6 will pass the top-level import but FAIL this
    # specific submodule import, which is exactly what MOVA needs.
    ("scenedetect.video_manager", True),
    ("python_speech_features", True),
]
errors = []
for mod, required in checks:
    try:
        m = importlib.import_module(mod)
        # Print the resolved package location for the more fragile ones so the
        # user can see whether site-packages or the vendored copy is being
        # used.
        if mod.startswith("pytorchvideo") or mod.startswith("imagebind"):
            origin = getattr(m, "__file__", "<unknown>")
            print(f"  ok    {mod}  ({origin})")
        else:
            print(f"  ok    {mod}")
    except Exception as exc:
        msg = f"  FAIL  {mod}: {exc.__class__.__name__}: {exc}"
        print(msg)
        if required:
            errors.append(msg)
if errors:
    print("\nSome required imports failed. Re-run setup_my_eval_deps.sh or install manually.")
    sys.exit(1)
print("\nAll critical imports OK.")

try:
    from transformers.models.pe_audio_video import PeAudioVideoModel, PeAudioVideoProcessor  # noqa: F401
    print("  ok    PeAudioVideoModel + PeAudioVideoProcessor")
except Exception as exc:
    print(f"  FAIL  PE-AV classes: {exc.__class__.__name__}: {exc}")
    sys.exit(1)
PY

echo "[my_eval setup] done."
