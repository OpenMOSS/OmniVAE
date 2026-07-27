# Reconstruction Evaluation

This document describes the cleaned OmniVAE reconstruction evaluation flow.
It replaces the internal evaluation scripts with repo-relative entrypoints.

## Script Mapping

Original internal scripts map to these public scripts:

```text
audio_eval              -> scripts/eval/run_audio_recon_multi_ckpt.sh
video_eval (no rFVD)    -> scripts/eval/run_video_recon_multi_ckpt.sh
video_rfvd_eval         -> scripts/eval/run_video_eval_rfvd_only.sh
```

The Python entrypoints are:

```text
omnivae.eval.reconstruction.audio_recon
omnivae.eval.reconstruction.video_recon
omnivae.eval.reconstruction.video_rfvd_only
```

## Example Data

Small example media and metadata are included:

```text
examples/eval/media/example_audio.wav
examples/eval/media/example_video.mp4
examples/eval/audio_recon.jsonl
examples/eval/video_recon.jsonl
examples/eval/checkpoints.txt
```

The JSONL format is intentionally minimal:

```json
{"audio_path": "media/example_audio.wav", "prompt": "Example audio clip."}
{"video_path": "media/example_video.mp4", "prompt": "Example video clip."}
```

Relative media paths are resolved first against `OMNIVAE_DATA_ROOT` when set,
then against the JSONL file directory.

## Checkpoint List

Create a text file with one checkpoint per line:

```text
$OMNIVAE_EXP_ROOT/stage2_video/checkpoints/Trainer_00130000/state_dict.pt
$OMNIVAE_EXP_ROOT/stage2_audio/checkpoints/Trainer_00250000/state_dict.pt
```

Blank lines and lines starting with `#` are ignored. Checkpoint paths may be
absolute, repo-relative, or environment-variable based. If `--config` is not
provided, the Python entrypoints try to infer:

```text
<experiment>/config.yaml
```

from the usual training checkpoint layout:

```text
<experiment>/checkpoints/Trainer_xxxxxxxx/state_dict.pt
```

The shell wrappers pass the public config by default:

```text
configs/audio_video_vae/omnivae_recon_distill_wan22.yaml
```

Override with:

```bash
export EVAL_CONFIG=/path/to/config.yaml
```

## Audio Reconstruction

Dry-run first:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/eval/run_audio_recon_multi_ckpt.sh
```

Run on your data/checkpoints:

```bash
export CHECKPOINT_LIST=/path/to/audio_checkpoints.txt
export AUDIO_EVAL_JSONL=/path/to/audio_eval.jsonl
export OUTPUT_DIR=$OMNIVAE_EXP_ROOT/eval/audio_recon

bash scripts/eval/run_audio_recon_multi_ckpt.sh
```

Useful overrides:

```bash
export BATCH_SIZE=4
export MAX_EXAMPLES=32
export MAX_DURATION=10
export AUDIO_SAMPLE_RATE=24000
export COMPUTE_STOI=1
export COMPUTE_PESQ=1
export INFERENCE_DTYPE=float32  # default
```

Select the evaluation domain so the appropriate reconstruction metrics are
enabled automatically:

```bash
# LibriSpeech or another speech set: adds WavLM speaker similarity.
export EVALUATION_DOMAIN=speech
export OMNIVAE_SPEAKER_SIM_MODEL=/path/to/wavlm_large_finetune.pth
bash scripts/eval/run_audio_recon_multi_ckpt.sh

# AudioSet: adds DAC-style Mel, STFT, and ViSQOL Audio metrics.
export EVALUATION_DOMAIN=audio
bash scripts/eval/run_audio_recon_multi_ckpt.sh

# MUSDB18 or another music set: same DAC metric recipe.
export EVALUATION_DOMAIN=music
bash scripts/eval/run_audio_recon_multi_ckpt.sh
```

The reconstruction forward pass uses FP32 by default. BF16 remains available
as an explicit opt-in with `INFERENCE_DTYPE=bfloat16` on CUDA.

Default audio metrics:

```text
avg_l1
avg_snr
avg_mel_loss
avg_stft_sc
avg_stft_mag
avg_stft_dist
```

For `EVALUATION_DOMAIN=audio` or `music`, the default metrics additionally
include the DAC public evaluation recipe at 44.1 kHz by default:

```text
avg_dac_mel_distance_44100
avg_dac_stft_distance_44100
avg_dac_visqol_audio_44100
```

To additionally evaluate at another rate, pass for example
`--dac_sample_rates 22050 44100`.

Lower is better for Mel/STFT distance; higher is better for ViSQOL. The
default `--visqol_argument_order dac` preserves the public DAC evaluation
script's call order. Use `--visqol_argument_order standard` only when standard
reference/degraded ordering is desired instead of exact DAC comparability.

For `EVALUATION_DOMAIN=speech`, the default metrics additionally include
`avg_speaker_similarity` (higher is better). It is computed from 16 kHz mono
audio with the WavLM Large speaker-verification checkpoint. Full per-file
scores are written to `no_ema/speaker_similarity.json`.

Domain defaults can be overridden with `COMPUTE_DAC_METRICS=0/1`,
`COMPUTE_VISQOL=0/1`, and `COMPUTE_SPEAKER_SIMILARITY=0/1`.

Optional metrics:

```text
avg_stoi      requires python -m pip install -e ".[metrics]"
avg_pesq_wb   requires 16 kHz audio and python -m pip install -e ".[metrics]"
avg_pesq_nb   requires 16 kHz audio and python -m pip install -e ".[metrics]"
```

DAC Mel/STFT requires `descript-audiotools` (included in `.[metrics]`).
ViSQOL requires Google's compiled Python binding; see
`docs/installation.md`. Speaker similarity requires `stopes`, CUDA for the
recommended runtime, and `OMNIVAE_SPEAKER_SIM_MODEL`.

Outputs:

```text
<OUTPUT_DIR>/<run_name>/no_ema/gt/*.wav
<OUTPUT_DIR>/<run_name>/no_ema/recon/*.wav
<OUTPUT_DIR>/<run_name>/no_ema/metrics.json
<OUTPUT_DIR>/<run_name>/no_ema/speaker_similarity.json  # speech only
<OUTPUT_DIR>/<run_name>/results.json
<OUTPUT_DIR>/summary.json
```

## Video Reconstruction

Dry-run first:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/eval/run_video_recon_multi_ckpt.sh
```

Run on your data/checkpoints:

```bash
export CHECKPOINT_LIST=/path/to/video_checkpoints.txt
export VIDEO_EVAL_JSONL=/path/to/video_eval.jsonl
export DATASET_NAME=ucf101
export OUTPUT_DIR=$OMNIVAE_EXP_ROOT/eval/video_recon

bash scripts/eval/run_video_recon_multi_ckpt.sh
```

24 FPS reconstruction settings:

```bash
export NUM_FRAMES=121
export TARGET_FPS=24
export RESOLUTION=256
```

Useful overrides:

```bash
export MAX_EXAMPLES=8
export COMPUTE_LPIPS=1
export NO_TORCHCODEC=1  # force decord fallback
```

Default video metrics:

```text
l1
psnr
ssim
```

Optional metric:

```text
lpips  requires COMPUTE_LPIPS=1
```

Outputs:

```text
<OUTPUT_DIR>/<run_name>/no_ema/video_recon/gt/*.mp4
<OUTPUT_DIR>/<run_name>/no_ema/video_recon/recon/*.mp4
<OUTPUT_DIR>/<run_name>/no_ema/video_recon/.progress/rank0.jsonl
<OUTPUT_DIR>/<run_name>/no_ema/video_recon/metrics.json
<OUTPUT_DIR>/<run_name>/results.json
<OUTPUT_DIR>/summary.json
```

The `.progress/rank0.jsonl` file records the sample order used by rFVD.

## rFVD Only

rFVD is computed after video reconstruction, using the saved `gt/` and
`recon/` videos.

Dry-run:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/eval/run_video_eval_rfvd_only.sh
```

Run:

```bash
export RFVD_OUTPUT_ROOTS=$OMNIVAE_EXP_ROOT/eval/video_recon
export RFVD_BATCH_SIZE=2
export I3D_TORCHSCRIPT_PT=/path/to/i3d_torchscript.pt  # recommended
export RFVD_GROUP_JSONLS=/path/to/ucf-101-test-all.jsonl:/path/to/panda-70m-test-all.jsonl

bash scripts/eval/run_video_eval_rfvd_only.sh
```

When `RFVD_GROUP_JSONLS` is set, the rFVD post-processing also computes one
value per metadata JSONL (the filename stem is used as the group name), while
retaining the overall rFVD. The CSV contains `overall`, `ucf-101-test-all`,
and `panda-70m-test-all` rows, and `results.json` stores the same values under
`no_ema.video_recon.by_group`.

Multiple output roots use a colon-separated list:

```bash
export RFVD_OUTPUT_ROOTS="$OMNIVAE_EXP_ROOT/eval/video_recon:$OMNIVAE_EXP_ROOT/eval/video_recon_nf49"
```

Smoke-test controls:

```bash
export MAX_TASKS=1
export MAX_VIDEOS=8
```

Outputs:

```text
<RFVD_OUTPUT_ROOT>/rfvd_summary.csv
<RFVD_OUTPUT_ROOT>/<run_name>/results.json       # updated with rfvd/rfvd_count
<RFVD_OUTPUT_ROOT>/<run_name>/no_ema/video_recon/.rfvd_only/rfvd.json
<parent_of_RFVD_OUTPUT_ROOT>/rfvd_summary_all.csv
```

The I3D TorchScript weight is not stored in the repository. If
`I3D_TORCHSCRIPT_PT` is not provided, the underlying FVD helper may try to
download it at runtime.

## Direct Python Usage

Audio:

```bash
python -m omnivae.eval.reconstruction.audio_recon \
  --checkpoint_list examples/eval/checkpoints.txt \
  --config configs/audio_video_vae/omnivae_recon_distill_wan22.yaml \
  --input_jsonl examples/eval/audio_recon.jsonl \
  --output_dir $OMNIVAE_EXP_ROOT/eval/audio_recon \
  --dry_run
```

Video:

```bash
python -m omnivae.eval.reconstruction.video_recon \
  --checkpoint_list examples/eval/checkpoints.txt \
  --config configs/audio_video_vae/omnivae_recon_distill_wan22.yaml \
  --input_jsonl examples/eval/video_recon.jsonl \
  --output_dir $OMNIVAE_EXP_ROOT/eval/video_recon \
  --num_frames 121 \
  --target_fps 24 \
  --resolution 256 \
  --dry_run
```

rFVD:

```bash
python -m omnivae.eval.reconstruction.video_rfvd_only \
  --output_root $OMNIVAE_EXP_ROOT/eval/video_recon \
  --group_jsonl /path/to/ucf-101-test-all.jsonl \
  --group_jsonl /path/to/panda-70m-test-all.jsonl \
  --dry_run
```
