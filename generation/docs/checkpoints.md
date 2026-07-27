# Checkpoints

Recommended layout:

```text
checkpoints/
  omnivae/
    av_vae/
      state_dict.pt

  generation/
    t2i/
      checkpoint-00495000/
        transformer/
        metadata.json
    t2v/
      checkpoint-00350000/
        transformer/
        metadata.json
    t2a/
      checkpoint-00195000/
        transformer/
        metadata.json
    t2av/
      checkpoint-00045000/
        video/
        audio/
        bridges/
        metadata.json
      final/
        transformer_video/
        transformer_audio/
        bridges/
        metadata.json
```

Inference-only export layout:

```text
model_ckpts/
  manifest.json
  t2i/
    t2i_recon/
      transformer/
      tokenizer/
      scheduler/
      resolved_config.json
      metadata.json
  t2av/
    t2av_recon_avclip/
      transformer_video/
      transformer_audio/
      bridges/
      tokenizer/
      scheduler/
      resolved_config.json
      metadata.json
```

Create the inference-only packages with:

```bash
python scripts/export_inference_checkpoints.py \
  --output-root /path/to/model_ckpts \
  --overwrite
```

The export keeps only files needed for inference/evaluation. It deliberately
does not copy root-level `optimizer.bin`, training `scheduler.bin`,
`random_states_*.pkl`, `dataloader_state_rank*.bin`, `.checkpoint_complete`, or
the accelerator root `model.safetensors`. The `scheduler/` directory is kept
because it is the diffusion scheduler used at inference time.

Inference and evaluation code accepts both trainer-saved `checkpoint-XXXXXXXX`
directories and these inference-only package directories. Inference-only
packages report checkpoint step `0` internally so downstream code does not rely
on an encoded training step.

Training recipes use these environment variables:

- `OMNIVAE_CKPT`: shared OmniVAE checkpoint.
- `OMNIVAE_VIDEO_CKPT`: video branch override for T2AV.
- `OMNIVAE_AUDIO_CKPT`: audio branch override for T2A/T2AV.
- `T2I_TRANSFORMER_CKPT`: stage-1 transformer used to initialize T2V.
- `T2V_TRANSFORMER_CKPT`: stage-2 transformer used to initialize T2AV.
- `T2A_TRANSFORMER_CKPT`: stage-3 transformer used to initialize T2AV.

The public VAE type is `omnivae`; `univae` remains accepted only as a backward-compatible alias.
