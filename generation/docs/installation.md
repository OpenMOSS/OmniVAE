# Installation

Create the downstream generation environment independently from the OmniVAE tokenizer environment.

```bash
cd OmniVAE-Generation
conda env create -f environment.yml
conda activate omnivae-generation
```

For pip-only installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.10.0+cu126 torchvision==0.25.0+cu126 \
  torchaudio==2.10.0+cu126 torchcodec==0.10.0+cu126
pip install -r requirements.txt -r requirements_audio.txt
pip install -e .
```

The CUDA 12.6 wheel stack above was smoke-tested on an RTX 4090 with
driver 550.163.01. Avoid leaving `torch` unconstrained: on this machine,
plain `pip install -r requirements.txt` resolved to CUDA 13 wheels that could
not initialize CUDA with the installed driver. `torchcodec==0.10.0+cu126` was
also required with the system FFmpeg 4.4 libraries; `torchcodec==0.11.1+cu126`
failed to load in this environment.

If PyPI access is slow, use the Tsinghua mirror for non-PyTorch packages:

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements.txt -r requirements_audio.txt
```

## Required System Tools

- CUDA-capable PyTorch environment for training.
- `ffmpeg` on `PATH` for video/audio muxing and preview export.
- Hugging Face cache access for the text encoder and scheduler, unless all dependencies are already cached locally.

## OmniVAE Checkpoint

Train or download an OmniVAE checkpoint first. The downstream recipes expect:

```bash
export OMNIVAE_CKPT=/path/to/omnivae/checkpoints/Trainer_00084000/state_dict.pt
```

The native Wan2.2 VAE implementation used by OmniVAE video decoding is bundled
under `generation/opensora/infer/wan2_2vae`. If you intentionally want to use a
different implementation, set `OMNIGEN_WAN_VAE_REPO` to a source tree that
contains `opensora/infer/wan2_2vae`.

## Environment Variables

- `OMNIGEN_ENV_SH`: optional shell file to source before launching.
- `OMNIGEN_CONDA_ENV`: optional conda env to activate in launcher scripts.
- `OMNIGEN_PYTHON`: explicit Python interpreter for distributed launch.
- `OMNIGEN_HF_HOME`: optional Hugging Face cache location.
- `OMNIGEN_TORCH_HOME`: optional Torch cache location.
- `OMNIGEN_TORCHINDUCTOR_CACHE_DIR`: optional TorchInductor cache location.
- `OMNIGEN_DRY_RUN=1`: print launch commands without starting training or inference.
