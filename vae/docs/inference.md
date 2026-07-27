# Inference

This guide describes how to run OmniVAE reconstruction inference on audio,
video, or audio-video inputs. Inference is intentionally separate from
evaluation: it generates reconstructed media and a manifest, but does not run
dataset-level metrics.

## Entrypoints

Shell wrapper:

```bash
scripts/infer/run_reconstruct.sh
```

Python entrypoint:

```bash
python -m omnivae.infer.reconstruct
```

Supported modes:

```text
audio  reconstruct audio only
video  reconstruct video only
av     reconstruct both video and audio from each sample
```

## Inputs

Use exactly one of:

```bash
--input_file /path/to/file.wav
--input_dir /path/to/media_dir
--input_jsonl /path/to/metadata.jsonl
```

JSONL records use the same public path convention as training and evaluation:

```json
{"audio_path": "media/example_audio.wav", "prompt": "Example audio clip."}
{"video_path": "media/example_video.mp4", "prompt": "Example video clip."}
{"video_path": "media/example_video.mp4", "audio_path": "media/example_audio.wav"}
```

Relative paths are resolved first against `OMNIVAE_DATA_ROOT`, then against
the JSONL file directory.

## Checkpoint And Config

Set a checkpoint and config explicitly:

```bash
export CHECKPOINT=/path/to/checkpoints/Trainer_00130000/state_dict.pt
export INFER_CONFIG=/path/to/config.yaml
```

If `INFER_CONFIG` is not set, the wrapper uses:

```text
configs/audio_video_vae/omnivae_recon_distill_wan22.yaml
```

The Python entrypoint can infer `config.yaml` from the standard training
checkpoint layout when `--config` is omitted:

```text
<experiment>/checkpoints/Trainer_xxxxxxxx/state_dict.pt
<experiment>/config.yaml
```

## Dry Run

Dry-run checks argument expansion, config paths, and input metadata without
loading the model or writing outputs:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/infer/run_reconstruct.sh
```

## Audio Inference

```bash
export MODE=audio
export CHECKPOINT=/path/to/audio_or_av_checkpoint/state_dict.pt
export INPUT_JSONL=/path/to/audio_inputs.jsonl
export OUTPUT_DIR=$OMNIVAE_EXP_ROOT/infer/audio

bash scripts/infer/run_reconstruct.sh
```

Useful overrides:

```bash
export AUDIO_SAMPLE_RATE=24000
export MAX_DURATION=10
export MAX_EXAMPLES=16
export SAVE_INPUTS=1
```

Outputs:

```text
<OUTPUT_DIR>/audio_recon/*.wav
<OUTPUT_DIR>/manifest.jsonl
<OUTPUT_DIR>/summary.json
```

## Video Inference

```bash
export MODE=video
export CHECKPOINT=/path/to/video_or_av_checkpoint/state_dict.pt
export INPUT_JSONL=/path/to/video_inputs.jsonl
export OUTPUT_DIR=$OMNIVAE_EXP_ROOT/infer/video

export NUM_FRAMES=121
export TARGET_FPS=24
export RESOLUTION=256

bash scripts/infer/run_reconstruct.sh
```

Useful overrides:

```bash
export MAX_EXAMPLES=8
export SAVE_INPUTS=1
export NO_TORCHCODEC=1
```

Outputs:

```text
<OUTPUT_DIR>/video_recon/*.mp4
<OUTPUT_DIR>/manifest.jsonl
<OUTPUT_DIR>/summary.json
```

## Audio-Video Inference

Use `MODE=av` when the checkpoint contains both branches and each sample has
video plus either `audio_path` or audio embedded in the video file.

```bash
export MODE=av
export CHECKPOINT=/path/to/av_checkpoint/state_dict.pt
export INPUT_JSONL=/path/to/av_inputs.jsonl
export OUTPUT_DIR=$OMNIVAE_EXP_ROOT/infer/av

bash scripts/infer/run_reconstruct.sh
```

Outputs:

```text
<OUTPUT_DIR>/video_recon/*.mp4
<OUTPUT_DIR>/audio_recon/*.wav
<OUTPUT_DIR>/manifest.jsonl
<OUTPUT_DIR>/summary.json
```

## Direct Python Usage

Audio:

```bash
python -m omnivae.infer.reconstruct \
  --mode audio \
  --checkpoint /path/to/state_dict.pt \
  --config configs/audio_video_vae/omnivae_recon_distill_wan22.yaml \
  --input_jsonl examples/eval/audio_recon.jsonl \
  --output_dir $OMNIVAE_EXP_ROOT/infer/audio
```

Video:

```bash
python -m omnivae.infer.reconstruct \
  --mode video \
  --checkpoint /path/to/state_dict.pt \
  --config configs/audio_video_vae/omnivae_recon_distill_wan22.yaml \
  --input_jsonl examples/eval/video_recon.jsonl \
  --output_dir $OMNIVAE_EXP_ROOT/infer/video \
  --num_frames 121 \
  --target_fps 24 \
  --resolution 256
```

Dry-run:

```bash
python -m omnivae.infer.reconstruct \
  --mode video \
  --checkpoint ckpts/example/checkpoints/Trainer_00010000/state_dict.pt \
  --config configs/audio_video_vae/omnivae_recon_distill_wan22.yaml \
  --input_jsonl examples/eval/video_recon.jsonl \
  --dry_run
```

