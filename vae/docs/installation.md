# Installation

This guide installs the Python environment for OmniVAE training. Commands are
written for Linux GPU machines and assume you start from the repository root.

## 1. System Checks

Check that the machine sees NVIDIA GPUs:

```bash
nvidia-smi
```

Recommended system tools:

```bash
# Ubuntu/Debian example
sudo apt-get update
sudo apt-get install -y git ffmpeg build-essential libsndfile1
```

On managed clusters, install these with the site package manager or load the
equivalent modules. `ffmpeg` is useful for video/audio I/O, and a compiler is
needed by a few optional metric packages.

## 2. Create A Python Environment

Python 3.10 is recommended. The package metadata allows Python 3.9+, but the
current smoke checks were run with Python 3.10.

```bash
conda create -n omnivae python=3.10 -y
conda activate omnivae

python -m pip install --upgrade pip setuptools wheel
```

If you use `venv` instead of conda:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch

Install PyTorch first so the CUDA wheel matches your driver and cluster setup.
Use the official PyTorch selector for the exact command:

<https://pytorch.org/get-started/locally/>

Common pip examples:

```bash
# CUDA wheel example; replace cu126 with the selector value for your machine.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8 wheel example, if your driver/platform requires it.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only fallback for import/smoke checks
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify the install:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
PY
```

For real training, `cuda available` should be `True`.

## 4. Install OmniVAE

From the repository root:

```bash
python -m pip install -e .
```

Equivalent requirements-file install:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Use the second form when you want dependencies installed explicitly before the
editable package.

## 5. Managed Cluster PyTorch

If your cluster already provides a PyTorch/CUDA module, load it first and avoid
letting pip replace it:

```bash
module load cuda
module load pytorch

python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY

grep -vE '^(torch|torchvision|torchaudio)([<>= ]|$)' requirements.txt > /tmp/omnivae_requirements_no_torch.txt
python -m pip install -r /tmp/omnivae_requirements_no_torch.txt
python -m pip install -e . --no-deps
```

This keeps the site-provided PyTorch build and installs only the remaining
OmniVAE dependencies.

## 6. Optional Dependencies

Install optional metric packages only if you need the corresponding evaluation:

```bash
# PESQ/STOI and speaker-sim evaluation helpers
python -m pip install -e ".[metrics]"

# torchcodec video decoder path; decord remains the fallback decoder
python -m pip install -e ".[video]"

# Debugger support for --debug
python -m pip install -e ".[dev]"
```

Speaker similarity also needs a WavLM-style checkpoint path:

```bash
export OMNIVAE_SPEAKER_SIM_MODEL=/path/to/wavlm_checkpoint
```

DAC-style Mel/STFT metrics use `descript-audiotools`, which is installed by
the `metrics` extra above. DAC-style ViSQOL requires Google's compiled Python
binding. ViSQOL does not publish a standard pip package for this binding; build
the `python:visqol_lib_py.so` target from the public ViSQOL repository and
install the extension, generated `visqol.pb2` modules, and both scoring models
into the active environment. Verify it with:

```bash
python - <<'PY'
from pathlib import Path
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2

model_dir = Path(visqol_lib_py.__file__).resolve().parent / "model"
assert (model_dir / "libsvm_nu_svr_model.txt").is_file()
assert (model_dir / "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite").is_file()
print("ViSQOL binding and models: OK")
PY
```

If ViSQOL is unavailable, the rest of the DAC metrics can still be run with
`COMPUTE_VISQOL=0` or `--no-compute_visqol`.

## 7. Semantic Distillation Assets

Stage 2 and Stage 3 semantic distillation require the Qwen3 AV encoder service
assets. The default expected location is:

```text
$OMNIVAE_CKPT_ROOT/qwen3_avencoder_service
```

If the semantic encoder lives elsewhere:

```bash
export OMNIVAE_SEMANTIC_MODEL=/path/to/qwen3_avencoder_service
export SEMANTIC_MODEL_PATH=$OMNIVAE_SEMANTIC_MODEL
```

Local semantic distillation imports service code from the encoder directory
(`encoder_service/src/...`) and uses `qwen_omni_utils`. Install the dependency
set required by that encoder package before running distillation stages.

For an external encoder server, pass the service URL through the training
config or CLI:

```bash
--semantic_api_url http://host:port
```

## 8. Configure Paths

The launch scripts set repo-relative defaults automatically:

```bash
export OMNIVAE_REPO_ROOT=$PWD
export OMNIVAE_CKPT_ROOT=$PWD/ckpts
export OMNIVAE_DATA_ROOT=$PWD/data
export OMNIVAE_EXP_ROOT=$PWD/exp
export OMNIVAE_SEMANTIC_MODEL=$OMNIVAE_CKPT_ROOT/qwen3_avencoder_service
```

Override only the paths that live outside the repo:

```bash
export OMNIVAE_DATA_ROOT=/path/to/data
export OMNIVAE_CKPT_ROOT=/path/to/ckpts
export OMNIVAE_EXP_ROOT=/path/to/experiments
```

Optional launcher hooks:

```bash
export OMNIVAE_CONDA_ENV=omnivae      # train_local.sh activates this env
export OMNIVAE_ENV_SH=/path/to/env.sh # sourced before training
export TORCHRUN_BIN=torchrun          # override torchrun binary if needed
```

## 9. Verify The Environment

Run import and CLI checks:

```bash
python -m compileall -q omnivae scripts
PYTHONPATH=$PWD python -m omnivae.train.train_audio_video_vae --help
```

Run the lightweight smoke tests:

```bash
python scripts/tools/smoke_test_variable_negatives.py
python scripts/tools/smoke_intra_seg_xattn_head.py
```

Verify recipe expansion without launching distributed training:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/recipes/stage1_video_recon.sh
OMNIVAE_DRY_RUN=1 bash scripts/recipes/stage1_audio_recon.sh
```

Stage 3 and Stage 4 dry-runs require checkpoint variables:

```bash
VIDEO_CKPT=/path/to/video_ckpt \
AUDIO_CKPT=/path/to/audio_ckpt \
OMNIVAE_DRY_RUN=1 bash scripts/recipes/stage3_av_align.sh

PRETRAINED_AUDIO_CKPT=/path/to/audio_ckpt \
PRETRAINED_DISC_CKPT=/path/to/disc_ckpt \
OMNIVAE_DRY_RUN=1 bash scripts/recipes/stage4_audio_decoder_ft.sh
```

## Troubleshooting

`torch.cuda.is_available()` is `False`:
Check that the NVIDIA driver is visible through `nvidia-smi`, then reinstall
PyTorch with a CUDA wheel supported by that driver.

`No video decoder available`:
Install `decord` or `torchcodec`, and make sure `ffmpeg` is available.

`ImportError: pesq`:
Install optional metrics with `python -m pip install -e ".[metrics]"`.

`ImportError: qwen_omni_utils` or `No module named src...`:
Install the semantic encoder service dependencies and confirm
`OMNIVAE_SEMANTIC_MODEL` points to the `qwen3_avencoder_service` directory.

`ImportError: libGL.so.1` while importing OpenCV:
Install the system OpenGL package, for example `sudo apt-get install libgl1`,
or replace `opencv-python` with `opencv-python-headless` in headless images.
