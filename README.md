# OmniVAE

English | [简体中文](README_zh-CN.md)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://openmoss.github.io/OmniVAE.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2607.00000)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Assets-yellow)](https://huggingface.co/OpenMOSS-Team/OmniVAE)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

<p align="center">
  <a href="assets/architecture/arch.pdf">
    <img src="assets/architecture/arch.png" alt="OmniVAE architecture" width="100%">
  </a>
</p>

**OmniVAE** is a unified audio-video variational autoencoder. It uses separate
audio and video encoders to extract modality-specific features and aligns them
in a shared latent space, providing representations that are easier to align
for downstream audio-video generation. OmniVAE supports audio, video, and joint
audio-video encoding and reconstruction. Our final T2AV configuration is
`t2av_recon_distill_avclip`; all other configurations are baselines or
ablations included for comparison.

The repository contains three main modules: [`vae/`](vae/) provides OmniVAE
training, reconstruction, and evaluation; [`generation/`](generation/) provides
DiT training and T2AV inference; and [`sync/`](sync/) evaluates audio-video
synchronization information in frozen OmniVAE features. It also includes the
scripts, evaluation workflows, and documentation needed to run the experiments.

## Quick Start

### 1. Install the packages

Install a PyTorch/CUDA stack compatible with your machine, then install the
three local packages:

```bash
cd /path/to/OmniVAE

pip install -e vae
pip install -e generation
pip install -e sync
```

See [Installation](docs/installation.md) for environment and distributed-job
details.

### 2. Download the release assets

```bash
pip install -U huggingface_hub
huggingface-cli download OpenMOSS-Team/OmniVAE \
  --repo-type model \
  --local-dir /path/to/omnivae_release \
  --local-dir-use-symlinks False
```

Point all OmniVAE modules to the downloaded bundle:

```bash
cd /path/to/OmniVAE
source scripts/setup_release_root.sh /path/to/omnivae_release
```

This sets `OMNIVAE_RELEASE_ROOT` and `OPEN_SOURCE_ROOT`. You can also export
the variables manually:

```bash
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release
export OPEN_SOURCE_ROOT="${OMNIVAE_RELEASE_ROOT}"
```

### 3. Run a single-GPU T2AV smoke test

```bash
bash generation/scripts/av/validate_checkpoints.sh \
  --gpus 0 \
  --ckpt-root "${OMNIVAE_RELEASE_ROOT}/models/dit/t2av" \
  --validation-jsonl "${OMNIVAE_RELEASE_ROOT}/eval/data/t2av/versebench_minimal/versebench_t2av_infer_minimal.jsonl" \
  --experiments t2av_recon_distill_avclip \
  --steps 200000 \
  --cfg 4 \
  --types set3-large \
  --max-examples 1 \
  --output-root outputs/t2av_smoke
```

For multi-GPU inference and additional options, see
[Inference](docs/inference.md).

## Common Workflows

### Reconstruct with a VAE checkpoint

```bash
cd /path/to/OmniVAE/vae

MODE=video \
CHECKPOINT="${OMNIVAE_RELEASE_ROOT}/models/vae/video_only/recon/state_dict.pt" \
CONFIG="${OMNIVAE_RELEASE_ROOT}/models/vae/video_only/recon/config.yaml" \
INPUT_JSONL=/path/to/video_inputs.jsonl \
OUTPUT_DIR=../outputs/vae/video_recon \
bash scripts/infer/run_reconstruct.sh
```

Set `MODE=audio` for an audio VAE or `MODE=audio_video` for a joint
audio-video VAE. Input formats and further examples are documented in
[VAE inference](vae/docs/inference.md).

### Validate the T2AV release

The release launcher evaluates the public checkpoints on VerseBench
`set3-large` with CFG 4. It supports both local multi-GPU execution and an
already allocated PET/distributed job, and resumes by default.

```bash
cd /path/to/OmniVAE
source scripts/setup_release_root.sh /path/to/omnivae_release

bash scripts/release_launchers/run_t2av_release_compare_distributed.sh
```

See [Evaluation](docs/evaluation.md) for defaults, outputs, and metric
comparison behavior.

### Train a model

Training is organized by module and stage:

| Goal | Entrypoints | Guide |
|---|---|---|
| Train an OmniVAE | `vae/scripts/recipes/` | [VAE training](vae/docs/training.md) |
| Train T2I/T2V/T2A/T2AV models | `generation/scripts/recipes/` | [Generation training](generation/docs/training.md) |
| Train a synchronization probe | `sync/scripts/train/train_sync_vae.sh` | [Sync training](sync/docs/training.md) |

The unified [Training guide](docs/training.md) provides a shorter overview of
all three workflows.

### Evaluate reconstructions

VAE evaluation supports video reconstruction metrics, audio/music metrics
(Mel, STFT, and ViSQOL), and speech speaker similarity. Start with the unified
[Evaluation guide](docs/evaluation.md), then refer to
[VAE evaluation](vae/docs/evaluation.md) for metric-specific dependencies and
options.

## Release Asset Layout

All public scripts resolve models and evaluation data relative to
`OMNIVAE_RELEASE_ROOT`:

```text
omnivae_release/
├── manifest.json
├── models/
│   ├── vae/
│   ├── dit/
│   └── text_encoder/
└── eval/
    ├── data/
    └── models/
```

The release bundle is intentionally separate from the code repository. This
keeps the GitHub package lightweight and lets the same code work with either
released assets or locally trained checkpoints.

## Repository Layout

```text
OmniVAE/
├── vae/          # VAE training, reconstruction, and metrics
├── generation/   # Downstream generation training and T2AV inference
├── sync/         # Audio-video synchronization probing
├── scripts/      # Release setup and validation launchers
└── docs/         # Cross-module documentation
```

Each major component also contains its own README and detailed documentation.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/installation.md) | Package setup, release assets, and distributed environments |
| [Release Assets](docs/huggingface_release.md) | Checkpoint inventory and Hugging Face directory layout |
| [Inference](docs/inference.md) | VAE reconstruction and T2AV generation |
| [Training](docs/training.md) | VAE, generation, and synchronization entrypoints |
| [Evaluation](docs/evaluation.md) | Reconstruction metrics and T2AV release validation |

For module-specific details, use the documentation under [`vae/`](vae/),
[`generation/`](generation/), and [`sync/`](sync/).

## Citation

If you find OmniVAE useful, please cite it with the temporary entry below. The
paper URL and final citation will be updated after the paper is available.

```bibtex
@article{omnivae2026,
  title   = {OmniVAE: A Unified Audio-Video Variational Autoencoder for Multimodal Generation},
  author  = {Zhan, Jun and Contributors},
  journal = {arXiv preprint arXiv:2607.00000},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.00000}
}
```

## Star History

<!-- star-history:start -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/star-history/star-history-dark.svg">
  <img alt="Star history" src="assets/star-history/star-history-light.svg">
</picture>
<!-- star-history:end -->

## License

This repository is released under the [Apache-2.0 License](LICENSE). Bundled
third-party evaluation components retain their original licenses; see their
respective subdirectories for details.
