# Installation

OmniVAE-Sync is an extension package. Install the base OmniVAE package first,
because Sync imports `omnivae.models.*` for the video/audio VAE encoders and
contrastive pooling head.

## Conda Environment

You can reuse the OmniVAE environment if it already has a compatible PyTorch,
CUDA, ffmpeg, and torchvision/torchaudio stack. The Sync code adds lightweight
training/logging dependencies such as `wandb`, `matplotlib`, `pandas`, and
`timm`.

```bash
conda create -n omnivae python=3.10 -y
conda activate omnivae

# Install PyTorch wheels matching your CUDA version, then:
cd /path/to/OmniVAE
pip install -e .

cd /path/to/OmniVAE-Sync
pip install -e .
```

For a source checkout with both repositories side by side, this is also valid:

```bash
cd /path/to/OmniVAE
pip install -e .

cd /path/to/OmniVAE-Sync
pip install -e .
```

## Verification

```bash
python -m compileall -q omnivae_sync
python - <<'PY'
from omnivae_sync.model.sync_model_vae import SynchformerVAE
from omnivae_sync.training.train_sync import train
print("omnivae-sync import ok")
PY
```

## Runtime Paths

Set data and output locations with environment variables:

```bash
export OMNIVAE_SYNC_VIDEOS=/path/to/videos
export OMNIVAE_SYNC_VGGSOUND_META=/path/to/vggsound.csv
export OMNIVAE_SYNC_SPLITS=/path/to/splits
export OMNIVAE_SYNC_EXP_ROOT=/path/to/experiments
```

The config uses these variables through OmegaConf `oc.env` resolvers.
