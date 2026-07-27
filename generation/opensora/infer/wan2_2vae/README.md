# WanVAE2.2 — Self-contained Inference Package

A standalone copy of Wan2.2 Video VAE that **does not depend on any other
opensora directory**. Useful for distributing/loading the pretrained video VAE
without dragging in the full training stack.

## File layout

```
opensora/infer/wan2_2vae/
├── __init__.py          # exposes WanVAE22Model, modules, distribution
├── modules.py           # CausalConv3d / ResidualBlock / AttentionBlock /
│                        # Encoder3d / Decoder3d / patchify ...
├── distributions.py     # DiagonalGaussianDistribution
├── model.py             # WanVAE22Model wrapper (pure nn.Module)
├── config.json          # default 48-dim WanVAE2.2 config (qk_norm=true)
├── infer.py             # CLI: reconstruct + PSNR / L1 / MSE evaluation
└── README.md
```

External dependencies: `torch`, `einops`, `torchvision` (only for video IO),
`tqdm`, `numpy`. **No `diffusers` / `opensora.dataset` imports.**

## Python API

```python
from opensora.infer.wan2_2vae import WanVAE22Model

# Option A: a directory containing config.json + a *.pth file
model = WanVAE22Model.from_pretrained("/path/to/ckpt_dir")

# Option B: a single .pth file (config.json sibling auto-detected,
# falls back to bundled default 48-dim config)
model = WanVAE22Model.from_pretrained("/path/to/video_vae.pth")

# Option C: build from config + load checkpoint manually
cfg = WanVAE22Model.load_config("/path/to/config.json")
cfg["qk_norm"] = True
model = WanVAE22Model.from_config(cfg)
model.init_from_ckpt("/path/to/video_vae.pth")

model = model.eval().cuda()

# video: (B, 3, T, H, W) in [-1, 1]
posterior = model.encode(video)                     # DiagonalGaussianDistribution
z = posterior.mode()                                # or posterior.sample()
recon = model.decode(z)                             # (B, 3, T', H', W')
```

`init_from_ckpt` accepts every checkpoint layout we currently produce:
- bare `state_dict`
- `{"state_dict": sd}`
- `{"state_dict": {"gen_model": sd}}`
- `{"ema_state_dict": sd}` (preferred unless `NOT_USE_EMA_MODEL=1`)
- BaseModel-style `{"state_dict": sd, "metadata": {...}}` (the format produced
  by `scripts/audio_video_vae/extract_video_vae.py`)

## CLI

Run as a Python module (preferred — works from anywhere):

```bash
# 1) random sanity check (no video IO needed)
python -m opensora.infer.wan2_2vae.infer \
    --pretrained_path /path/.../video_vae.pth \
    --random_input --num_frames 17 --resolution 256

# 2) single video reconstruction with PSNR / L1
python -m opensora.infer.wan2_2vae.infer \
    --pretrained_path /path/.../video_vae.pth \
    --input_video /path/to/sample.mp4 \
    --output_dir ./wan22_recon \
    --num_frames 33 --resolution 256 --target_fps 8

# 3) batch over a directory of mp4 files
python -m opensora.infer.wan2_2vae.infer \
    --pretrained_path /path/.../video_vae.pth \
    --input_dir /path/to/videos \
    --output_dir ./wan22_recon \
    --max_examples 16

# 4) JSONL input ({"video_path": "..."} per line; multi-jsonl supported)
python -m opensora.infer.wan2_2vae.infer \
    --pretrained_path /path/.../video_vae.pth \
    --input_jsonl /path/to/test1.jsonl /path/to/test2.jsonl \
    --output_dir ./wan22_recon

# 5) multi-GPU: one process per GPU via torchrun
#    - videos are stride-sharded across ranks
#    - PSNR / L1 / MSE are all-reduced and reported only on rank 0
torchrun --nproc_per_node=8 -m opensora.infer.wan2_2vae.infer \
    --pretrained_path /path/.../video_vae.pth \
    --input_jsonl panda70m-test.jsonl ucf101-test.jsonl \
    --output_dir ./wan22_recon \
    --num_frames 121 --target_fps 24 --resolution 256 \
    --max_examples 10000 --sample_posterior --dtype bfloat16
```

Or run the file directly:

```bash
python /path/to/OmniVAE/generation/opensora/infer/wan2_2vae/infer.py --random_input ...
```

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--qk_norm {auto,true,false}` | `auto` reads `ckpt['metadata']['qk_norm_filtered']` if available; otherwise defaults to `True`. |
| `--streaming`     | Use temporally-causal `streaming_inference=True` (lower memory, stricter causality). |
| `--sample_posterior` | Use `posterior.sample()` instead of `posterior.mode()` (adds Gaussian noise; matches the training-time behavior of `infer_audio_video_vae.py`). |
| `--dtype`        | `bfloat16` (default) / `float16` / `float32`. |
| `--no_save`      | Skip writing reconstructed mp4s; just compute metrics. |

### Outputs

When videos are saved, the script writes:

```
output_dir/
├── gt/<name>.mp4          # ground-truth clip after preprocessing
├── recon/<name>.mp4       # reconstructed clip
├── compare/<name>.mp4     # side-by-side (GT | Recon)
├── per_sample.json        # per-video PSNR/L1/MSE
└── results.json           # aggregate metrics
```

## Notes on `qk_norm`

Wan2.2 introduced RMSNorm on Q/K (`qk_norm=True`). When you extract a video
VAE with `scripts/audio_video_vae/extract_video_vae.py --filter-qk-norm`,
those norm tensors are dropped and you must build the model with
`qk_norm=False`. The script handles this automatically when `--qk_norm auto`
is set, by reading the metadata field `qk_norm_filtered` from the extracted
checkpoint.
