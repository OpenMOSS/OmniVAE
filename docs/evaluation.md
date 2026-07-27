# Evaluation

Set the release root:

```bash
cd /path/to/OmniVAE
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release
```

## VAE

Video reconstruction metrics:

```bash
cd vae
CHECKPOINT_LIST=/path/to/video_checkpoints.txt \
VIDEO_EVAL_JSONL=/path/to/video_eval.jsonl \
bash scripts/eval/run_video_recon_multi_ckpt.sh
```

rFVD uses `OMNIVAE_RELEASE_ROOT/eval/models/vae/fvd/i3d_torchscript.pt` when
available.

Audio reconstruction metrics:

```bash
cd vae
CHECKPOINT_LIST=/path/to/audio_checkpoints.txt \
AUDIO_EVAL_JSONL=/path/to/audio_eval.jsonl \
bash scripts/eval/run_audio_recon_multi_ckpt.sh
```

## T2AV Release Validation

This validates `t2av_recon` and `t2av_recon_distill_avclip` on VerseBench
`set3-large` with CFG 4.

```bash
cd /path/to/OmniVAE
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release

bash scripts/release_launchers/run_t2av_release_compare_distributed.sh
```

Defaults:

- CFG: `4`
- step: `200000`
- data: `eval/data/t2av/versebench_minimal/versebench_t2av_infer_minimal.jsonl`
- type filter: `set3-large`
- examples: all filtered rows
- output: `../test_output/t2av_release_compare_set3_large`
- resume: enabled by default

The launcher runs inference and `my_eval`, then compares the generated metric
JSONs to a reference directory if available. If the reference directory is
absent, comparison is skipped without failing the validation run.
