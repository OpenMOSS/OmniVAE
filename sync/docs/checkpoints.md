# Checkpoints

OmniVAE-Sync uses two kinds of checkpoints:

- OmniVAE checkpoint: provides frozen video/audio VAE weights and, optionally,
  contrastive pooling head weights.
- Sync checkpoint: produced by this repository and used for resume, evaluation,
  and inference.

## OmniVAE Inputs

Training needs:

```bash
export VAE_CKPT=/path/to/omnivae/checkpoints/state_dict.pt
export VAE_CFG=/path/to/omnivae/checkpoints/config.yaml
```

`VAE_CFG` is the AudioVideoVAE config. It must contain `model.video`,
`model.audio`, and `model.contrastive` sections compatible with the checkpoint.

Launch scripts pass these into:

```text
av_vae_config=$VAE_CFG
vae_pretrained=$VAE_CKPT
```

## Optional Split VAE Inputs

If video and audio VAE weights come from separate checkpoints, use:

```bash
bash scripts/train/train_sync_vae.sh \
  --video_vae_pretrained /path/to/video_state_dict.pt \
  --audio_vae_pretrained /path/to/audio_state_dict.pt \
  --video_av_vae_config /path/to/video_config.yaml \
  --audio_av_vae_config /path/to/audio_config.yaml
```

When both split checkpoints are provided, `--load_contrastive_head_from_external`
controls whether modality-specific contrastive-head submodules are loaded from
those external checkpoints.

## Sync Outputs

Training writes:

```text
${OMNIVAE_SYNC_EXP_ROOT}/sync_models/<run_name>/
  cfg-<run_name>.yaml
  log-<run_name>.log
  <run_name>.pt
  <run_name>_best.pt
  <run_name>_latest.pt
```

Use `<run_name>_latest.pt` for resume and `<run_name>.pt` or
`<run_name>_best.pt` for evaluation/inference, depending on your model
selection policy.

## Checkpoint Format

Sync checkpoints are dictionaries with at least:

```text
args
epoch
metrics
model
optimizer
scaler
lr_scheduler
```

The inference entrypoint can load the config from `args` when `--cfg_path` is
not passed.
