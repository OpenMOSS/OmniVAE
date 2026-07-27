# Training

The downstream generation recipe follows four stages. All scripts support `--gpus`, `--debug`, `--no_compile`, and extra passthrough arguments accepted by the Python entrypoints.

## Stage 1: T2I

```bash
export OMNIVAE_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt

bash scripts/recipes/stage1_t2i.sh
```

Useful overrides:

```bash
OMNIGEN_RUN_NAME=t2i_omnivae \
OMNIGEN_BATCH_SIZE=32 \
OMNIGEN_LR=1.0e-4 \
bash scripts/recipes/stage1_t2i.sh --gpus 0,1,2,3
```

## Stage 2: T2V

Warm-start from the stage-1 T2I transformer:

```bash
export OMNIVAE_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt
export T2I_TRANSFORMER_CKPT=/path/to/t2i/checkpoints/snapshots/checkpoint-00495000/transformer

bash scripts/recipes/stage2_t2v.sh
```

## Stage 3: T2A

```bash
export OMNIVAE_AUDIO_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt

bash scripts/recipes/stage3_t2a.sh
```

`OMNIVAE_AUDIO_CKPT` defaults to `OMNIVAE_CKPT` when not set.

## Stage 4: T2AV

Warm-start from the stage-2 T2V and stage-3 T2A transformers:

```bash
export OMNIVAE_VIDEO_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt
export OMNIVAE_AUDIO_CKPT=/path/to/omnivae/Trainer_00084000/state_dict.pt
export T2V_TRANSFORMER_CKPT=/path/to/t2v/checkpoints/snapshots/checkpoint-00350000/transformer
export T2A_TRANSFORMER_CKPT=/path/to/t2a/checkpoints/snapshots/checkpoint-00195000/transformer

bash scripts/recipes/stage4_t2av.sh
```

## Data

The public configs use relative placeholder paths:

```text
data/image/relaion/
data/video/train/video_caption.jsonl
data/audio/train/*.jsonl
data/av/train/*.jsonl
```

Either place data at those paths, create symlinks, or edit the corresponding YAML. Large datasets are intentionally not included.

## Dry Run

```bash
OMNIGEN_DRY_RUN=1 bash scripts/recipes/stage1_t2i.sh
```

Dry-run mode validates launcher argument construction without starting `torch.distributed.run`.

## Smoke Runs

For small local smoke tests with custom YAML files, call the lower-level
launchers directly:

```bash
bash scripts/audio/train.sh configs/visual/t2i.yaml --debug --gpus 0 --no_compile
bash scripts/audio/train.sh configs/visual/t2v.yaml --debug --gpus 0 --no_compile
bash scripts/audio/train.sh configs/audio/t2a.yaml --debug --gpus 0 --no_compile
bash scripts/av/train.sh configs/av/t2av.yaml --debug --gpus 0 --no_compile --muon_shard_across_ranks 1
```

The `scripts/recipes/stage*.sh` wrappers intentionally choose the public stage
configs and pass extra CLI overrides through to the launcher. If you need to
use a different config file for a one-step smoke run, use
`scripts/audio/train.sh` or `scripts/av/train.sh` as shown above.

On a single GPU, pass `--muon_shard_across_ranks 1` for stage 4 / T2AV. The
T2AV launcher default is `8`, which is meant for larger multi-rank jobs.
If memory is tight, reduce `OMNIGEN_BATCH_SIZE`, video frame count, image size,
and audio duration in the config before increasing gradient accumulation.
