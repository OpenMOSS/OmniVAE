# Inference

Set the release root first:

```bash
cd /path/to/OmniVAE
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release
```

## VAE Reconstruction

```bash
cd vae
MODE=video \
CHECKPOINT="${OMNIVAE_RELEASE_ROOT}/models/vae/video_only/recon/state_dict.pt" \
CONFIG="${OMNIVAE_RELEASE_ROOT}/models/vae/video_only/recon/config.yaml" \
INPUT_JSONL=/path/to/video_inputs.jsonl \
OUTPUT_DIR=../outputs/vae/video_recon \
bash scripts/infer/run_reconstruct.sh
```

Use `MODE=audio` with an audio VAE checkpoint, or `MODE=audio_video` with an
AV VAE checkpoint.

## T2AV

```bash
cd /path/to/OmniVAE

bash generation/scripts/av/validate_checkpoints.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --ckpt-root "${OMNIVAE_RELEASE_ROOT}/models/dit/t2av" \
  --validation-jsonl "${OMNIVAE_RELEASE_ROOT}/eval/data/t2av/versebench_minimal/versebench_t2av_infer_minimal.jsonl" \
  --experiments t2av_recon t2av_recon_distill_avclip \
  --steps 200000 \
  --cfg 4 \
  --types set3-large \
  --max-examples 3 \
  --output-root ../outputs/t2av/inference
```
