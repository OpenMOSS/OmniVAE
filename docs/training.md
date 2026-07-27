# Training

Training entrypoints are split by module.

## VAE

```bash
cd /path/to/OmniVAE/vae

bash scripts/recipes/stage1_video_recon.sh
bash scripts/recipes/stage1_audio_recon.sh
bash scripts/recipes/stage2_video_distill.sh
bash scripts/recipes/stage2_audio_distill.sh
bash scripts/recipes/stage3_av_align.sh
bash scripts/recipes/stage4_audio_decoder_ft.sh
```

The VAE JSONL metadata format is documented in `vae/examples/metadata`.
Relevant fields are usually `video_path`, `audio_path`, and an optional prompt
or caption field.

## Generation

Generation is trained in four stages:

```bash
cd /path/to/OmniVAE/generation

bash scripts/recipes/stage1_t2i.sh
bash scripts/recipes/stage2_t2v.sh
bash scripts/recipes/stage3_t2a.sh
bash scripts/recipes/stage4_t2av.sh
```

Use the VAE paths under `OMNIVAE_RELEASE_ROOT/models/vae` or your own newly
trained VAE checkpoints.

## VAE-Sync

```bash
cd /path/to/OmniVAE/sync
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release

bash scripts/train/train_sync_vae.sh \
  --config configs/sync_24fps_nonspeech_vae.yaml \
  --vae_pretrained "${OMNIVAE_RELEASE_ROOT}/models/vae/audio_video/recon_distill_avclip/state_dict.pt" \
  --av_vae_config "${OMNIVAE_RELEASE_ROOT}/models/vae/audio_video/recon_distill_avclip/config.yaml" \
  --fps 24 \
  --crop_len_sec 6 \
  --skip_temporal_pool False \
  --audio_merge 1 \
  --suffix recon_distill_avclip
```

For the public no-extra-tail setup, use `--crop_len_sec 6` and do not add a
separate tail-retention duration.
