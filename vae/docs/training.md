# Training Recipes

This guide describes the cleaned OmniVAE training flow. All commands assume
you are in the repository root.

## Environment

Install the environment first:

```bash
python -m pip install -e .
```

See [installation.md](installation.md) for CUDA/PyTorch selection, managed
cluster setup, optional metrics, semantic encoder dependencies, and verification
commands.

The launch scripts infer the repository root and set default local paths:

```bash
export OMNIVAE_CKPT_ROOT=${OMNIVAE_CKPT_ROOT:-$PWD/ckpts}
export OMNIVAE_DATA_ROOT=${OMNIVAE_DATA_ROOT:-$PWD/data}
export OMNIVAE_EXP_ROOT=${OMNIVAE_EXP_ROOT:-$PWD/exp}
export OMNIVAE_SEMANTIC_MODEL=${OMNIVAE_SEMANTIC_MODEL:-$OMNIVAE_CKPT_ROOT/qwen3_avencoder_service}
```

Optional environment hooks:

```bash
export OMNIVAE_ENV_SH=/path/to/env.sh      # optional source before launch
export OMNIVAE_CONDA_ENV=omnivae           # optional conda activation
export TORCHRUN_BIN=torchrun                   # optional torchrun override
```

Use dry-run mode to verify recipe expansion without launching `torchrun`:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/recipes/stage1_video_recon.sh
```

## Stage 1: Reconstruction Training

Video VAE reconstruction:

```bash
bash scripts/recipes/stage1_video_recon.sh
```

Audio VAE reconstruction:

```bash
bash scripts/recipes/stage1_audio_recon.sh
```

## Stage 2: Reconstruction + Uni-Modal Semantic Distillation

Set the semantic encoder path if it is not in the default checkpoint root:

```bash
export SEMANTIC_MODEL_PATH=/path/to/qwen3_avencoder_service
```

Video VAE with semantic distillation:

```bash
bash scripts/recipes/stage2_video_distill.sh
```

Audio VAE with semantic distillation:

```bash
bash scripts/recipes/stage2_audio_distill.sh
```

## Stage 3: Reconstruction + Semantic + Cross-Modal Alignment

This stage loads separately trained video and audio VAE checkpoints. The
contrastive head is intentionally initialized from scratch unless you pass
`--pretrained_contrastive_checkpoint` yourself.

```bash
export VIDEO_CKPT=/path/to/stage2_video/checkpoints/Trainer_00130000
export AUDIO_CKPT=/path/to/stage2_audio/checkpoints/Trainer_00264000
export FREEZE_STEPS=20000

bash scripts/recipes/stage3_av_align.sh
```

## Stage 4: Audio Decoder Fine-Tuning

This stage freezes the audio encoder and fine-tunes the decoder with audio
discriminator losses.

```bash
export PRETRAINED_AUDIO_CKPT=/path/to/stage3/checkpoints/Trainer_00084000
export PRETRAINED_DISC_CKPT=/path/to/audio_disc/checkpoints/Trainer_00250000

bash scripts/recipes/stage4_audio_decoder_ft.sh
```

## Direct Wrapper Usage

You can still call the wrapper directly:

```bash
bash scripts/train_local.sh \
  configs/audio_video_vae/omnivae_recon_distill_wan22.yaml \
  --batch_size 1 \
  --num_frames 121 \
  --no_semantic_distill
```

Use `--gpus 0,1,2,3` for local GPU selection. Multi-node runs use the
`PET_*` variables already supported by the original launcher.
