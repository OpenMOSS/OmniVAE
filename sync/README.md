# OmniVAE-Sync

This repository contains the synchronization probing code for OmniVAE audio-video
latent features. It trains a lightweight Sync head on frozen OmniVAE video/audio
VAE encoders and evaluates audio-video offset prediction.

## Documentation

- [Installation](docs/installation.md): environment setup and dependency notes.
- [Training](docs/training.md): Sync-VAE training and resume workflow.
- [Inference](docs/inference.md): single-video offset prediction.
- [Evaluation](docs/evaluation.md): test-only evaluation and metrics.
- [Checkpoints](docs/checkpoints.md): required OmniVAE and Sync checkpoint layout.

## Repository Layout

- `omnivae_sync/model/sync_model_vae.py`: OmniVAE-backed Sync model.
- `omnivae_sync/training/train_sync.py`: DDP training and test-only loop.
- `omnivae_sync/train.py`: public training entrypoint.
- `omnivae_sync/infer/predict_offset.py`: single-video inference entrypoint.
- `omnivae_sync/eval/evaluate.py`: evaluation entrypoint.
- `configs/sync_24fps_nonspeech_vae.yaml`: 24 fps Sync-VAE config.
- `scripts/train/train_sync_vae.sh`: torchrun training wrapper.
- `scripts/infer/predict_offset.sh`: inference wrapper.
- `scripts/eval/eval_sync.sh`: evaluation wrapper.

## Path Convention

No machine-specific paths are required in the repo. Override these environment
variables for your data and outputs:

```bash
export OMNIVAE_SYNC_VIDEOS=/path/to/vggsound_or_training_videos
export OMNIVAE_SYNC_VGGSOUND_META=/path/to/vggsound.csv
export OMNIVAE_SYNC_SPLITS=/path/to/split_files
export OMNIVAE_SYNC_EXP_ROOT=/path/to/experiments
```

The Sync model depends on OmniVAE weights and config files. Set them explicitly:

```bash
export VAE_CKPT=/path/to/omnivae/state_dict.pt
export VAE_CFG=/path/to/omnivae/config.yaml
```

## Quick Start

Install OmniVAE first, then this extension:

```bash
cd /path/to/OmniVAE
pip install -e .

cd /path/to/OmniVAE-Sync
pip install -e .
```

Dry-run the training command:

```bash
OMNIVAE_SYNC_DRY_RUN=1 bash scripts/train/train_sync_vae.sh \
  --vae_pretrained "$VAE_CKPT" \
  --av_vae_config "$VAE_CFG" \
  --fps 24 \
  --crop_len_sec 6 \
  --skip_temporal_pool False \
  --audio_merge 1 \
  --suffix recon_distill_avclip
```

Launch training:

```bash
bash scripts/train/train_sync_vae.sh \
  --vae_pretrained "$VAE_CKPT" \
  --av_vae_config "$VAE_CFG" \
  --fps 24 \
  --crop_len_sec 6 \
  --skip_temporal_pool False \
  --audio_merge 1 \
  --suffix recon_distill_avclip
```

Run single-video inference:

```bash
bash scripts/infer/predict_offset.sh \
  --ckpt_path /path/to/sync_model.pt \
  --cfg_path /path/to/cfg-sync.yaml \
  --vid_path /path/to/video.mp4 \
  --offset_sec 0.0
```

Run evaluation:

```bash
bash scripts/eval/eval_sync.sh \
  --ckpt_path /path/to/sync_model.pt \
  --config configs/sync_24fps_nonspeech_vae.yaml \
  --nproc 2
```

See the docs for dataset format, checkpoint expectations, and common overrides.
