# HuggingFace Release Layout

The HuggingFace directory is expected to be:

```text
omnivae_release/
  manifest.json
  models/
    text_encoder/
      Qwen3.5-0.8B-Base/
    vae/
      video_only/
      audio_only/
      audio_video/
    dit/
      t2av/
  eval/
    data/
    models/
```

Set:

```bash
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release
```

All model package metadata uses paths relative to this directory, for example
`models/vae/audio_video/recon_distill_avclip` or
`models/text_encoder/Qwen3.5-0.8B-Base`.

## Text Encoders

The release bundle includes the frozen Qwen3.5 text encoder directory used by
the T2AV configs:

- `models/text_encoder/Qwen3.5-0.8B-Base`

These are third-party dependency weights, not additional OmniVAE checkpoints.
It is kept in the bundle so the released inference and evaluation commands
can run with the same text encoder settings used during validation.

## VAE Models

- `models/vae/video_only/recon`
- `models/vae/video_only/recon_distill`
- `models/vae/audio_only/recon`
- `models/vae/audio_only/recon_distill`
- `models/vae/audio_video/recon_avclip`
- `models/vae/audio_video/recon_distill_avclip`
- `models/vae/audio_only/recon_avclip_ft_decoder`
- `models/vae/audio_only/recon_distill_avclip_ft_decoder`

For T2AV AVCLIP experiment families, the audio branch uses the
`*_ft_decoder` checkpoint by default.

## DiT Models

The public release contains only T2AV generation packages:

```text
t2av: t2av_recon, t2av_recon_distill, t2av_recon_avclip, t2av_recon_distill_avclip
```

Each package is inference-only and excludes optimizer, LR scheduler,
dataloader, random state, and training-state snapshots.
