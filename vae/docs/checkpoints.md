# Checkpoint Layout

The public configs use environment variables instead of absolute paths. The
default checkpoint root is:

```bash
$OMNIVAE_CKPT_ROOT
```

By default this resolves to `./ckpts` under the repository root.

## Required Pretrained Assets

Expected layout:

```text
ckpts/
  audio_video_ckpt/
    Wan2.2-VAE/
      Wan2.2_VAE.pth
    dac_vae_48khz/
      vae_128d_48k.pth
  qwen3_avencoder_service/
    ...
```

The configs reference these paths:

```yaml
model:
  video:
    pretrained_model_name_or_path: $OMNIVAE_CKPT_ROOT/audio_video_ckpt/Wan2.2-VAE/Wan2.2_VAE.pth
  audio:
    pretrained_model_name_or_path: $OMNIVAE_CKPT_ROOT/audio_video_ckpt/dac_vae_48khz/vae_128d_48k.pth
```

If your assets live elsewhere, either set `OMNIVAE_CKPT_ROOT` or pass the
corresponding CLI override.

## Stage Checkpoints

Stage 3 needs trained video/audio checkpoints:

```bash
export VIDEO_CKPT=/path/to/video/checkpoints/Trainer_00130000
export AUDIO_CKPT=/path/to/audio/checkpoints/Trainer_00264000
```

Stage 4 needs the stage 3 checkpoint and an audio discriminator checkpoint:

```bash
export PRETRAINED_AUDIO_CKPT=/path/to/stage3/checkpoints/Trainer_00084000
export PRETRAINED_DISC_CKPT=/path/to/audio_disc/checkpoints/Trainer_00250000
```

Checkpoint paths may point to a trainer checkpoint directory or to a checkpoint
file when supported by the loader.
