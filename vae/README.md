# OmniVAE

This repository contains the public OmniVAE code for audio-video VAE training,
reconstruction inference, and reconstruction evaluation.

## Documentation

- [Installation](docs/installation.md): environment setup, CUDA/PyTorch install, optional dependencies.
- [Training](docs/training.md): staged reconstruction, distillation, cross-modal alignment, and decoder fine-tuning.
- [Inference](docs/inference.md): reconstruct audio, video, or audio-video samples from a checkpoint.
- [Evaluation](docs/evaluation.md): multi-checkpoint reconstruction metrics and rFVD.
- [Checkpoints](docs/checkpoints.md): expected pretrained asset layout.

## Repository Layout

- `omnivae/train/train_audio_video_vae.py`: main training entrypoint.
- `omnivae/train/av_vae/`: trainer, CLI merge logic, losses, checkpoint logic.
- `omnivae/models/audio_video_vae/`: audio/video VAE, contrastive heads, discriminators.
- `omnivae/dataset/`: JSONL/file-based audio-video datasets.
- `configs/audio_video_vae/`: training configs with environment-variable paths.
- `scripts/train_local.sh`: repo-relative torchrun wrapper.
- `scripts/recipes/`: clean four-stage training recipes.
- `scripts/infer/`: reconstruction inference wrapper.
- `scripts/eval/`: reconstruction evaluation wrappers.
- `docs/installation.md`: environment setup and verification.
- `docs/training.md`: end-to-end training guide.
- `docs/inference.md`: reconstruction inference guide.
- `docs/evaluation.md`: reconstruction evaluation guide.
- `docs/checkpoints.md`: required checkpoint layout.

## Path Convention

No machine-specific path is required in the repo. The launch scripts set these
defaults automatically:

```bash
export OMNIVAE_REPO_ROOT=/path/to/OmniVAE
export OMNIVAE_CKPT_ROOT=$OMNIVAE_REPO_ROOT/ckpts
export OMNIVAE_DATA_ROOT=$OMNIVAE_REPO_ROOT/data
export OMNIVAE_EXP_ROOT=$OMNIVAE_REPO_ROOT/exp
export OMNIVAE_SEMANTIC_MODEL=$OMNIVAE_CKPT_ROOT/qwen3_avencoder_service
```

Override them when your data or weights live elsewhere:

```bash
export OMNIVAE_DATA_ROOT=/path/to/data
export OMNIVAE_CKPT_ROOT=/path/to/ckpts
export OMNIVAE_EXP_ROOT=/path/to/experiments
```

Config values support `$OMNIVAE_*`, `~`, absolute paths, and repo-relative
paths.

## Quick Start

Install the package:

```bash
cd /path/to/OmniVAE

# See docs/installation.md for CUDA/PyTorch and cluster setup details.
pip install -e .

# Optional extras for PESQ/STOI/speaker-sim metrics or torchcodec decoding:
# pip install -e ".[metrics,video]"

export OMNIVAE_CONDA_ENV=omnivae  # optional; omit if already activated
```

Training dry-run:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/recipes/stage1_video_recon.sh
```

Training:

```bash
bash scripts/recipes/stage1_video_recon.sh
bash scripts/recipes/stage1_audio_recon.sh
```

Inference dry-run:

```bash
OMNIVAE_DRY_RUN=1 MODE=video bash scripts/infer/run_reconstruct.sh
```

Inference:

```bash
export MODE=video
export CHECKPOINT=/path/to/checkpoints/Trainer_00130000/state_dict.pt
export INPUT_JSONL=/path/to/video_inputs.jsonl
bash scripts/infer/run_reconstruct.sh
```

Evaluation dry-run:

```bash
OMNIVAE_DRY_RUN=1 bash scripts/eval/run_video_recon_multi_ckpt.sh
```

Evaluation:

```bash
export CHECKPOINT_LIST=/path/to/video_checkpoints.txt
export VIDEO_EVAL_JSONL=/path/to/video_eval.jsonl
bash scripts/eval/run_video_recon_multi_ckpt.sh
```

Audio evaluation uses FP32 by default. Set `EVALUATION_DOMAIN=speech` for
WavLM speaker similarity, or `EVALUATION_DOMAIN=audio`/`music` for DAC-style
Mel distance, STFT distance, and ViSQOL:

```bash
# Speech evaluation.
EVALUATION_DOMAIN=speech \
OMNIVAE_SPEAKER_SIM_MODEL=/path/to/wavlm_large_finetune.pth \
CHECKPOINT_LIST=/path/to/audio_checkpoints.txt \
AUDIO_EVAL_JSONL=/path/to/speech_eval.jsonl \
bash scripts/eval/run_audio_recon_multi_ckpt.sh

# Audio/music evaluation.
EVALUATION_DOMAIN=audio \
CHECKPOINT_LIST=/path/to/audio_checkpoints.txt \
AUDIO_EVAL_JSONL=/path/to/audio_eval.jsonl \
bash scripts/eval/run_audio_recon_multi_ckpt.sh
```

DAC metrics use only 44.1 kHz by default. Additional rates can be requested
explicitly with `--dac_sample_rates 22050 44100`. The detailed metric/output
description is in [docs/evaluation.md](docs/evaluation.md).

See [docs/installation.md](docs/installation.md) for environment setup,
[docs/training.md](docs/training.md) for the full staged recipe,
[docs/inference.md](docs/inference.md) for reconstruction inference,
[docs/evaluation.md](docs/evaluation.md) for reconstruction evaluation, and
[docs/checkpoints.md](docs/checkpoints.md) for checkpoint placement.

## Data Format

Training and validation metadata are JSONL files. Minimal examples:

```json
{"video_path": "videos/example.mp4", "audio_path": "audio/example.wav", "prompt": "A person playing drums"}
{"video_path": "videos/example2.mp4", "prompt": "A cyclist passing a fountain"}
```

Paths may be absolute or relative to the metadata file's configured root,
depending on the dataset mode. See `examples/metadata/` for sample files.

## License And Attribution

See [LICENSE](LICENSE). Keep upstream license notices for third-party model
components and pretrained weights when redistributing derived artifacts.
