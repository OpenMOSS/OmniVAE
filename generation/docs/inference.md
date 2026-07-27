# Inference

The wrappers in `scripts/infer/` call the Python inference entrypoints with conservative defaults. They accept extra CLI arguments at the end.

Checkpoint arguments may point either to a `checkpoint-XXXXXXXX` directory or
to the exported `final/` directory from a completed training run.

## T2I

```bash
export T2I_CHECKPOINT=/path/to/t2i/checkpoint-00495000
export OMNIVAE_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt

bash scripts/infer/t2i.sh --save-image-previews 8
```

COCO FID requires:

```bash
export T2I_ANNOTATIONS_JSON=data/image/annotations/captions_val2014.json
export T2I_IMAGES_DIR=data/image/val2014
```

## T2V

```bash
export T2V_CHECKPOINT=/path/to/t2v/checkpoint-00350000
export T2V_PROMPT_MANIFEST=examples/prompts/t2v_prompts.jsonl
export OMNIVAE_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt

bash scripts/infer/t2v.sh
```

## T2A

```bash
export T2A_CHECKPOINT=/path/to/t2a/checkpoint-00195000

bash scripts/infer/t2a.sh
```

When an audio VAE override is needed, set the checkpoint experiment name explicitly:

```bash
export T2A_AUDIO_VAE_OVERRIDE_NAME=t2a_omnivae
export OMNIVAE_AUDIO_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt
bash scripts/infer/t2a.sh
```

## T2AV

```bash
export T2AV_CHECKPOINT=/path/to/t2av/checkpoint-00045000
export T2AV_PROMPT_MANIFEST=examples/prompts/t2av_valid.jsonl
export OMNIVAE_VIDEO_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt
export OMNIVAE_AUDIO_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt

bash scripts/infer/t2av.sh
```

For smoke inference, reduce `--num-inference-steps`, `--num-frames`,
`--height`, `--width`, and audio duration. For example, a one-step T2AV smoke
run can use `--num-inference-steps 1 --num-frames 9 --height 64 --width 64
--audio-duration-seconds 1.0`.

## Dry Run

```bash
OMNIGEN_DRY_RUN=1 bash scripts/infer/t2av.sh
```
