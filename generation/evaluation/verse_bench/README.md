# VerseBench: Benchmark code for Universe-1

## Local Independent Setup

This checkout has been configured to run independently from the other T2AV
benchmarks. The setup script keeps the conda environment, caches, model weights,
and Verse-Bench metadata under this directory:

```bash
cd /path/to/OmniVAE/generation/evaluation/verse_bench

bash setup_verse_bench.sh
bash scripts/check_verse_bench.sh
bash run_verse_bench.sh mini_testset eval_outputs/mini_smoke verse_bench models
```

For custom generated outputs, the input filenames must match the Verse-Bench
sample ids:

```bash
bash run_verse_bench.sh /path/to/input_dir /path/to/output_dir verse_bench models
```

For generated sample folders that contain `manifest.json` files, prepare a
symlinked evaluation subset first. This keeps the original experiment directory
untouched:

```bash
python scripts/prepare_custom_verse_eval.py \
  --samples-root /path/to/generated/t2av_samples \
  --metadata-jsonl /path/to/omnivae_release/eval/data/t2av/versebench_minimal/versebench_t2av_infer_minimal.jsonl \
  --output-root prepared/t2av_smoke \
  --experiment t2av_recon \
  --step step-00200000 \
  --mode joint_av \
  --cfg cfg_simple \
  --prompt-field av_caption \
  --limit 1

bash run_verse_bench.sh \
  prepared/t2av_smoke/inputs \
  eval_outputs/t2av_smoke \
  prepared/t2av_smoke/verse_bench \
  models
```

For release validation of OmniVAE T2AV checkpoints, use the wrapper in the
repository root. It performs inference and then runs the bundled `my_eval`
pipeline:

```bash
cd /path/to/OmniVAE
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release

bash scripts/release_eval/t2av/run_release_t2av_eval_compare.sh \
  --mode run \
  --cfg 4 \
  --types set3-large \
  --experiments t2av_recon t2av_recon_distill_avclip \
  --output-root /path/to/output/t2av_release_compare
```

Local paths used by the independent setup:

```text
.cache/conda/envs/verse-bench   Python environment
.cache/verse-bench-cache        HF/torch/pip/modelscope/InsightFace caches
models                          Evaluation weights
verse_bench                     Verse-Bench metadata and references
eval_outputs                    Metric logs
```

If you have access to the official gated Hugging Face repo
`zuoweizwzw/Verse-Bench-Models`, login or set `HF_TOKEN` before running setup.
Without a token, `setup_verse_bench.sh` falls back to public sources and local
copies, then verifies the required files in `models`.

<div align="center">
  <a href="https://huggingface.co/zuoweizwzw/Verse-Bench-Models"><img src="https://img.shields.io/static/v1?label=Verse-Bench-Models&message=HuggingFace&color=yellow"></a>
  <a href="https://huggingface.co/datasets/dorni/Verse-Bench"><img src="https://img.shields.io/static/v1?label=Verse-Bench&message=HuggingFace&color=yellow"></a>
  <a href="https://dorniwang.github.io/UniVerse-1"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a> &ensp;
</div>

## Introduction

The benchmark evaluation code for Universe-1.

## Benchmark Datasets Download

```bash
hf download dorni/Verse-Bench --local-dir ./verse_bench
```

## Model Download

| Models           | 🤗 Hugging Face                                                          |
|------------------|--------------------------------------------------------------------------|
| VerseBenchModels | [VerseBenchModels](https://huggingface.co/zuoweizwzw/Verse-Bench-Models) |

download the pretrained evaluation models into ./models

## Model Usage

### 🔧 Dependencies and Installation

```bash
pip install -r requirements.txt
```

We tested the code on Python 3.10, PyTorch 2.6.0, and CUDA 12.4 on Ubuntu 22.04 LTS.

### 🚀 Inference Scripts

```bash
export MODELS_PATH=models
python calculate_metrics.py
```

Optional arguments:

```bash
  --input_dir # Your data for evaluation, the file names must match the names in our Verse-Bench datasets. Each test case must contains both .mp4 and .wav.
                        #(default: ./mini_testset for quick test.)
  --models_path # Path to the models. (default: ./models)

```
