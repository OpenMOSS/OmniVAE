# OmniVAE Generation

Downstream generation training code for OmniVAE tokenizers:

- text-to-image (`t2i`)
- text-to-video (`t2v`)
- text-to-audio (`t2a`)
- text-to-audio-video (`t2av`)

This repository is intentionally separate from the `OmniVAE` tokenizer repository. It consumes an OmniVAE checkpoint through CLI arguments or environment variables; it does not depend on private experiment paths.

## Quick Start

```bash
conda env create -f environment.yml
conda activate omnivae-generation
```

Set the VAE checkpoint produced by the OmniVAE repository:

```bash
export OMNIVAE_CKPT=/path/to/omnivae/checkpoints/Trainer_00084000/state_dict.pt
```

Dry-run the four training stages:

```bash
OMNIGEN_DRY_RUN=1 bash scripts/recipes/stage1_t2i.sh

export T2I_TRANSFORMER_CKPT=/path/to/t2i/checkpoint/transformer
OMNIGEN_DRY_RUN=1 bash scripts/recipes/stage2_t2v.sh

OMNIGEN_DRY_RUN=1 bash scripts/recipes/stage3_t2a.sh

export T2V_TRANSFORMER_CKPT=/path/to/t2v/checkpoint/transformer
export T2A_TRANSFORMER_CKPT=/path/to/t2a/checkpoint/transformer
OMNIGEN_DRY_RUN=1 bash scripts/recipes/stage4_t2av.sh
```

Dry-run the T2AV evaluation wrapper:

```bash
OMNIGEN_DRY_RUN=1 bash scripts/eval/validate_t2av_checkpoints.sh \
  --gpus 0,1 --ckpt-root /path/to/t2av_sweep \
  --validation-jsonl examples/eval/t2av_versebench_sample.jsonl
```

## Verified Notes

The README flow was smoke-tested with a fresh pip environment on an RTX 4090
using CUDA 12.6 PyTorch wheels. Install the PyTorch stack explicitly before
the remaining requirements:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.10.0+cu126 torchvision==0.25.0+cu126 \
  torchaudio==2.10.0+cu126 torchcodec==0.10.0+cu126
```

For slow PyPI access, install the non-PyTorch requirements with:

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements.txt -r requirements_audio.txt
```

Additional verification notes:

- The native Wan2.2 VAE loader is bundled at
  `generation/opensora/infer/wan2_2vae`. `OMNIGEN_WAN_VAE_REPO` is only needed
  when overriding it with another source tree.
- Inference and evaluation accept either trainer-saved `checkpoint-XXXXXXXX`
  directories or inference-only package directories from the HuggingFace
  release bundle.
- For single-GPU T2AV smoke runs, pass `--muon_shard_across_ranks 1`.
- If a smoke run OOMs, reduce batch size, video frames, image size, and audio
  duration first.
## Documentation

- [Installation](docs/installation.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [Evaluation](docs/evaluation.md)
- [Checkpoints](docs/checkpoints.md)

## Repository Layout

```text
configs/                 Public training configs
scripts/recipes/         Four-stage downstream training recipes
scripts/infer/           Thin inference wrappers
scripts/eval/            Checkpoint sweep and evaluation wrappers
infer/                   T2AV Python inference and eval code
omnivae_generation/      Python package
examples/                Small prompt metadata examples
```

## Path Policy

Configs use relative paths by default. For real runs, either edit the relevant config or pass checkpoint paths through the recipe environment variables documented in `docs/training.md`.

The public VAE type is `omnivae`. Older configs using `univae` are still accepted as a compatibility alias.

## Inference-Only Checkpoints

The public HuggingFace release already provides inference-only packages under
`models/dit/<task>/<name>/`. These packages exclude optimizer, scheduler,
dataloader, random state, and accelerator training state.
