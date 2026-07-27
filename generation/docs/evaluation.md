# Evaluation

Evaluation code is kept lightweight and tied to included inference outputs. The
repository ships small JSONL prompt sets under `examples/eval/` so command
construction and short qualitative sweeps can run without private datasets.

## T2I

`infer/image/run_eval.py` can generate samples and compute FID against COCO val2014:

```bash
python infer/image/run_eval.py \
  --config configs/visual/t2i.yaml \
  --checkpoint /path/to/t2i/checkpoint-00495000 \
  --annotations-json data/image/annotations/captions_val2014.json \
  --images-dir data/image/val2014 \
  --output-dir outputs/eval/t2i \
  --num-samples 5000
```

For a smoke FID run, use at least two samples. A single generated sample makes
the covariance estimate undefined in `torch-fidelity`.

## T2A

`infer/audio/run_eval.py` runs prompt-set generation from `configs/audio/t2a.yaml` and can compute WER for speech prompt sets. For one checkpoint:

```bash
python infer/audio/run_eval.py \
  --config configs/audio/t2a.yaml \
  --checkpoint /path/to/t2a/checkpoint-00195000 \
  --output-dir outputs/eval/t2a \
  --num-prompts 16
```

For checkpoint sweeps over a training root:

```bash
OMNIGEN_CONDA_ENV=omnivae-generation \
bash scripts/eval/sweep_t2a_ckpts.sh \
  --config configs/audio/t2a.yaml \
  --ckpt-root exp/t2a \
  --steps 195000,latest \
  --cfg 0,1,2,3,4,5 \
  --num-inference-steps 50 \
  --output-root outputs/eval/t2a \
  --eval-output-root outputs/eval/t2a_metrics \
  --tta-prompt-duration-seconds 10 \
  --trim-output-wav
```

`--ckpt-root` may point to a direct `checkpoint-XXXXXXXX` directory, one run
directory containing `checkpoints/snapshots/`, or a sweep root containing
multiple experiment directories. The wrapper maps `--tta-prompt-duration-seconds`
to fixed-duration prompt suffixes for the TTA/TTS validation sets.

The bundled T2A metrics are WER summaries and `report.html` from the generated
outputs. Dataset-specific AudioCaps-style metrics are not bundled because they
usually depend on separate evaluator repositories and environments. To attach
one, set `OMNIGEN_T2A_METRIC_CMD`; the command receives
`OMNIGEN_T2A_OUTPUT_ROOT` and `OMNIGEN_T2A_EVAL_OUTPUT_ROOT` in its environment.

When `--prompt-jsonl` is used, `--duration-seconds` is accepted as the manifest
mode duration alias for `--duration`.

## T2V

Use `infer/t2v/infer_t2v.py` to export videos, then run VBench if installed:

```bash
python infer/t2v/infer_t2v.py \
  --checkpoint-dir /path/to/t2v/checkpoint-00350000 \
  --prompt-manifest examples/prompts/t2v_prompts.jsonl \
  --output-dir outputs/eval/t2v

python infer/t2v/evaluate_vbench.py \
  --videos-dir outputs/eval/t2v \
  --output-dir outputs/eval/t2v_vbench
```

## T2AV

Use the T2AV inference script for qualitative and downstream metric evaluation:

```bash
python infer/t2av/infer_t2av.py \
  --ckpt /path/to/t2av/checkpoint-00045000 \
  --prompt-manifest examples/prompts/t2av_valid.jsonl \
  --output-dir outputs/eval/t2av \
  --modes joint_av video_only audio_only
```

External AV-sync or perceptual metrics can consume the generated `samples.jsonl` manifests.

For checkpoint sweeps:

```bash
OMNIGEN_CONDA_ENV=omnivae-generation \
bash scripts/eval/validate_t2av_checkpoints.sh \
  --gpus 0,1,2,3 \
  --ckpt-root exp/t2av \
  --validation-jsonl examples/eval/t2av_versebench_sample.jsonl \
  --steps 200000,190000,180000,170000 \
  --order desc \
  --cfg 0 \
  --output-root outputs/eval/t2av \
  --resume-inference \
  --experiments t2av_omnivae
```

This wrapper is the open-source equivalent of the internal multi-checkpoint AV
validation command. It writes the same validation sample layout as training:

```text
<output-root>/<experiment>/samples/step-XXXXXXXX/<mode>/cfg_<mode>/
```

`--ckpt-root` may also point to a run whose exported `final/metadata.json`
contains `global_step`; the validator treats that `final/` directory as the
checkpoint for that step.

The old `--run-my-eval`, `--eval-output-root`, `--resume-eval` and
`--my-eval-cfg` flags are accepted by the shell wrapper for migration
convenience. They do not pull private metric code into this repo. Set
`OMNIGEN_T2AV_METRIC_CMD` to run an external AV metric suite after inference;
the hook receives `OMNIGEN_T2AV_OUTPUT_ROOT`,
`OMNIGEN_T2AV_EVAL_OUTPUT_ROOT`, `OMNIGEN_T2AV_VALIDATION_JSONL`,
`OMNIGEN_T2AV_CKPT_ROOT`, `OMNIGEN_T2AV_MY_EVAL_CFG` and
`OMNIGEN_T2AV_RESUME_EVAL`.

For full benchmark runs, replace `examples/eval/t2av_versebench_sample.jsonl`
with your complete validation JSONL. The required fields are `av_caption`,
`type` and `index` by default; override them with `--text-field`,
`--type-field` and `--index-field` when needed.
