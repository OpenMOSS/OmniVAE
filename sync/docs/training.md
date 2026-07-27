# Training

The released training path is `train_avsync_model_vae`. It freezes OmniVAE
video/audio VAE encoders, optionally freezes the contrastive pooling head, and
trains `vproj`, `aproj`, and the Sync transformer/head for offset prediction.

## Data

The default config expects VGGSound-style videos plus `vggsound.csv` and split
files:

```bash
export OMNIVAE_SYNC_VIDEOS=/path/to/video_reencoded_24fps_48k
export OMNIVAE_SYNC_VGGSOUND_META=/path/to/vggsound.csv
export OMNIVAE_SYNC_SPLITS=/path/to/splits
```

The split directory should contain files such as:

```text
vggsound_train.txt
vggsound_valid.txt
vggsound_test.txt
```

Each line is a clip id without `.mp4`; the dataset resolves it under
`OMNIVAE_SYNC_VIDEOS`.

## Launch

```bash
export VAE_CKPT=/path/to/omnivae/checkpoints/state_dict.pt
export VAE_CFG=/path/to/omnivae/checkpoints/config.yaml
export OMNIVAE_SYNC_EXP_ROOT=/path/to/sync_experiments

bash scripts/train/train_sync_vae.sh \
  --config configs/sync_24fps_nonspeech_vae.yaml \
  --vae_pretrained "$VAE_CKPT" \
  --av_vae_config "$VAE_CFG" \
  --fps 24 \
  --crop_len_sec 6 \
  --skip_temporal_pool False \
  --audio_merge 1 \
  --suffix recon_distill_avclip \
  --freeze_contrastive_head True
```

Use `--nproc 2` to force two local GPUs. If omitted, the wrapper uses all
visible GPUs.

## Common Overrides

The shell wrapper forwards common arguments into OmegaConf dotlist overrides.
You can also call the Python entrypoint directly:

```bash
torchrun --standalone --nproc_per_node=2 -m omnivae_sync.train \
  config=configs/sync_24fps_nonspeech_vae.yaml \
  run_name=my_sync_run \
  logging.logdir=/path/to/logs \
  logging.use_wandb=False \
  training.base_batch_size=1 \
  training.num_workers=4 \
  av_vae_config="$VAE_CFG" \
  vae_pretrained="$VAE_CKPT" \
  data.crop_len_sec=6 \
  data.target_fps=24 \
  model.params.source_vfps=24 \
  model.params.model_target_vfps=24 \
  model.params.skip_temporal_pool=False \
  model.params.audio_merge_factor=1
```

For smoke tests, use a small dataset ratio and skip the final test pass:

```bash
torchrun --standalone --nproc_per_node=1 -m omnivae_sync.train \
  config=configs/sync_24fps_nonspeech_vae.yaml \
  run_name=smoke_sync \
  logging.logdir=./outputs/smoke \
  training.num_epochs=1 \
  training.base_batch_size=1 \
  training.num_workers=0 \
  data.dataset.params.size_ratio=0.00002 \
  skip_test=True \
  av_vae_config="$VAE_CFG" \
  vae_pretrained="$VAE_CKPT"
```

## Resume

`scripts/train/train_sync_vae.sh` has auto-resume enabled by default. It looks
under `${OMNIVAE_SYNC_EXP_ROOT}/sync_models` for the latest directory matching
the computed run name and prefers `*_latest.pt`. Disable it with:

```bash
bash scripts/train/train_sync_vae.sh --auto_resume False ...
```

Manual resume:

```bash
bash scripts/train/train_sync_vae.sh --ckpt_path /path/to/run_latest.pt ...
```
