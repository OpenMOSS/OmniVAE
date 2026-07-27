# Evaluation

Evaluation reuses the training loop in `training.run_test_only=True` mode. It
loads the checkpoint, runs the configured test split, and reports classification
metrics such as `accuracy_1`, `accuracy_5`, and median per-class accuracy.

## Command

```bash
bash scripts/eval/eval_sync.sh \
  --ckpt_path /path/to/sync_model.pt \
  --config configs/sync_24fps_nonspeech_vae.yaml \
  --nproc 2
```

The wrapper launches `torchrun -m omnivae_sync.eval.evaluate`.

## Common Overrides

Pass additional OmegaConf overrides after the known arguments:

```bash
bash scripts/eval/eval_sync.sh \
  --ckpt_path /path/to/sync_model.pt \
  --config configs/sync_24fps_nonspeech_vae.yaml \
  --nproc 2 \
  data.dataset.params.size_ratio=0.01 \
  training.base_batch_size=1 \
  training.num_workers=4 \
  logging.use_wandb=False
```

To evaluate a fixed metadata setup, point the environment variables to the
desired dataset before launch:

```bash
export OMNIVAE_SYNC_VIDEOS=/path/to/test/videos
export OMNIVAE_SYNC_VGGSOUND_META=/path/to/vggsound.csv
export OMNIVAE_SYNC_SPLITS=/path/to/splits
```

## Direct Entrypoint

```bash
torchrun --standalone --nproc_per_node=2 -m omnivae_sync.eval.evaluate \
  --config configs/sync_24fps_nonspeech_vae.yaml \
  --ckpt_path /path/to/sync_model.pt \
  --logdir ./outputs/eval \
  training.base_batch_size=1
```

The reported metrics are written to TensorBoard under the eval log directory
and also printed in the process log.
