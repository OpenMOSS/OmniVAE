"""
Self-contained WanVAE2.2 inference / reconstruction script.

This script depends only on torch, torchvision, einops and tqdm. It does not
import anything from the rest of the `opensora` package.

Input modes (pick exactly one):
  --input_video PATH    single .mp4
  --input_dir   DIR     scan all .mp4 files under a directory (recursive)
  --input_jsonl FILE    JSONL with {"video_path": "/abs/to/video.mp4"} per line
  --random_input        sanity-check only, randn (1, 3, T, H, W) input

Loading rules:
  --pretrained_path  /path/to/file_or_dir
                     File: treated as the weight file.
                     Dir : looks for *.pth / *.pt / *.ckpt / *.safetensors / *.bin.
  --model_config     Optional config.json. Auto-resolved from:
                       (1) sibling of --pretrained_path (file mode), or
                       (2) inside --pretrained_path (dir mode), or
                       (3) bundled `config.json` in this directory.
  --qk_norm {auto,true,false}
                     auto: read ckpt['metadata']['qk_norm_filtered'] when present.

Examples (run from the project root or anywhere; this script is self-contained):

  # 1) random sanity check (no video IO needed)
  python -m opensora.infer.wan2_2vae.infer \
      --pretrained_path /path/.../video_vae.pth \
      --random_input --num_frames 17 --resolution 256

  # 2) single video reconstruction
  python -m opensora.infer.wan2_2vae.infer \
      --pretrained_path /path/.../video_vae.pth \
      --input_video /path/to/sample.mp4 \
      --output_dir ./wan22_recon

  # 3) batch over a directory of mp4s
  python -m opensora.infer.wan2_2vae.infer \
      --pretrained_path /path/.../video_vae.pth \
      --input_dir  /path/to/videos \
      --output_dir ./wan22_recon --max_examples 16

  # 4) multi-GPU (one process per GPU) via torchrun
  torchrun --nproc_per_node=8 -m opensora.infer.wan2_2vae.infer \
      --pretrained_path /path/.../video_vae.pth \
      --input_jsonl a.jsonl b.jsonl \
      --output_dir ./wan22_recon --num_frames 121 --target_fps 24
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm


_THIS_DIR = Path(__file__).resolve().parent

if __package__ in (None, ""):
    # Allow running this file directly via `python infer.py ...` by inserting
    # the parent of this directory into sys.path.
    parent = str(_THIS_DIR.parent.parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from opensora.infer.wan2_2vae.model import WanVAE22Model
else:
    from .model import WanVAE22Model


# --------------------------------------------------------------------------- #
# Distributed helpers                                                         #
# --------------------------------------------------------------------------- #
def setup_dist() -> Tuple[int, int, torch.device]:
    """Detect torchrun env. Returns (rank, world_size, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return rank, world_size, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Video IO (torchvision-based, self-contained)                                #
# --------------------------------------------------------------------------- #
def _read_video_torchvision(
    path: str,
    num_frames: int,
    resolution: Tuple[int, int],
    target_fps: Optional[float],
    sample_rate: int = 1,
    assumed_src_fps: float = 30.0,
) -> Tuple[torch.Tensor, dict]:
    """Read `num_frames` frames from a video, resize to `resolution`,
    return a tensor of shape (C, T, H, W) in [-1, 1] (float32).

    - Only decodes the time window we actually need (keeps CPU RAM in check
      for long source videos like Panda70m).
    - If `target_fps` is given, the source is uniformly downsampled to it,
      then we further keep one frame per `sample_rate`.
    - Falls back to taking the leading frames + repeating the last one if
      the source is shorter than required.
    """
    import gc
    try:
        import torchvision  # noqa: F401
        from torchvision.io import read_video
    except Exception as e:
        raise ImportError(
            "torchvision is required for video IO. Install with `pip install torchvision`."
        ) from e

    # Estimate the time window we need (with a safety margin), so that
    # read_video doesn't decode the entire 30s+ source clip into RAM.
    if target_fps and target_fps > 0:
        needed_seconds = (num_frames * max(1, sample_rate)) / target_fps
    else:
        needed_seconds = num_frames / max(1.0, assumed_src_fps)
    end_pts = needed_seconds * 1.5 + 0.5  # generous margin

    try:
        video, _audio, info = read_video(
            path, start_pts=0.0, end_pts=end_pts, pts_unit="sec"
        )
    except Exception:
        # Some codecs ignore end_pts; let the caller catch the OOM if any.
        video, _audio, info = read_video(path, pts_unit="sec")
    del _audio

    if video.ndim != 4 or video.shape[0] == 0:
        # Maybe the clip is shorter than `end_pts`; try a full decode.
        del video
        gc.collect()
        video, _audio, info = read_video(path, pts_unit="sec")
        del _audio
        if video.ndim != 4 or video.shape[0] == 0:
            raise RuntimeError(f"read_video failed for {path}, got shape {tuple(video.shape)}")

    src_fps = float(info.get("video_fps", 0.0)) or 0.0

    src_t = video.shape[0]
    if target_fps is not None and target_fps > 0 and src_fps > 0:
        stride = max(1, int(round(src_fps / target_fps)))
    else:
        stride = 1
    stride *= max(1, sample_rate)

    indices = list(range(0, src_t, stride))[:num_frames]
    if len(indices) == 0:
        raise RuntimeError(f"No frames decoded from {path}")
    if len(indices) < num_frames:
        # Try a full decode in case the windowed decode was too short.
        if end_pts > 0:
            del video
            gc.collect()
            video, _audio, _ = read_video(path, pts_unit="sec")
            del _audio
            src_t = video.shape[0]
            indices = list(range(0, src_t, stride))[:num_frames]
        if len(indices) < num_frames:
            indices = indices + [indices[-1]] * (num_frames - len(indices))

    video = video[indices].clone()                  # (T, H, W, C) uint8
    gc.collect()

    video = video.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W) uint8
    video = video.float()

    # Resize per-frame.
    c, t, h, w = video.shape
    target_h, target_w = resolution
    if (h, w) != (target_h, target_w):
        video = video.permute(1, 0, 2, 3)           # (T, C, H, W)
        video = F.interpolate(
            video, size=(target_h, target_w),
            mode="bilinear", align_corners=False, antialias=True,
        )
        video = video.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

    video = video / 127.5 - 1.0
    info_out = {
        "src_fps": src_fps,
        "src_num_frames": src_t,
        "stride": stride,
        "target_fps": target_fps,
    }
    return video, info_out


def _write_video_torchvision(
    tensor: torch.Tensor, path: Path, fps: int
) -> None:
    """Save a (C, T, H, W) float tensor in [-1, 1] to an mp4 at `fps`."""
    from torchvision.io import write_video

    if tensor.dim() != 4:
        raise ValueError(f"expected (C, T, H, W), got {tuple(tensor.shape)}")
    v = ((tensor.float().clamp(-1, 1) + 1) * 127.5).round()
    v = v.clamp(0, 255).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).contiguous()  # (T, H, W, C)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(path), v, fps=max(1, int(fps)), video_codec="libx264")


# --------------------------------------------------------------------------- #
# Model loading helpers                                                        #
# --------------------------------------------------------------------------- #
def _resolve_model_config(
    pretrained_path: Path, model_config: Optional[str]
) -> str:
    """Pick the most appropriate VAE config.json.

    Priority:
      1. explicit `--model_config`
      2. sibling / inside `pretrained_path` (only if it actually looks like
         a VAE config — has `dim_mult`. Otherwise it's likely an unrelated
         diffuser config that happened to live next to the weights, and we
         skip it with a warning.)
      3. bundled `config.json` next to this script.
    """
    if model_config and os.path.exists(model_config):
        return model_config

    if pretrained_path.is_file():
        side = pretrained_path.parent / "config.json"
    else:
        side = pretrained_path / "config.json"

    if side.exists():
        try:
            cfg_peek = WanVAE22Model.load_config(str(side))
        except Exception as e:
            cfg_peek = None
            logging.warning(f"Failed to read {side}: {e}; falling back.")
        if cfg_peek is not None and WanVAE22Model.looks_like_vae_config(cfg_peek):
            return str(side)
        elif cfg_peek is not None:
            logging.warning(
                f"{side} does not look like a WanVAE22 config "
                f"(missing 'dim_mult'); ignoring it and falling back to the "
                f"bundled default config."
            )

    bundled = _THIS_DIR / "config.json"
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError(
        f"Cannot resolve model_config. Tried --model_config={model_config}, "
        f"{side}, {bundled}."
    )


def _resolve_qk_norm(
    arg_value: str,
    ckpt_path: Path,
    cfg_dict: Dict[str, object],
) -> bool:
    """Decide whether the model uses qk_norm.

    Priority:
      1. explicit `--qk_norm true|false`
      2. ckpt['metadata']['qk_norm_filtered'] (set by extract_video_vae.py)
      3. `qk_norm` field inside the config.json
      4. default `False` (matches the open-source Wan2.2 VAE release)
    """
    if arg_value == "true":
        return True
    if arg_value == "false":
        return False

    if ckpt_path.is_file():
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception as e:
            logging.warning(f"[qk_norm=auto] failed to peek at ckpt: {e}")
            ckpt = None
        if isinstance(ckpt, dict):
            meta = ckpt.get("metadata") if isinstance(ckpt, dict) else None
            if isinstance(meta, dict) and "qk_norm_filtered" in meta:
                qk = not bool(meta["qk_norm_filtered"])
                logging.info(
                    f"[qk_norm=auto] inferred from ckpt metadata: filtered="
                    f"{meta['qk_norm_filtered']} -> qk_norm={qk}"
                )
                return qk

    if isinstance(cfg_dict, dict) and "qk_norm" in cfg_dict:
        qk = bool(cfg_dict["qk_norm"])
        logging.info(f"[qk_norm=auto] using value from config.json: qk_norm={qk}")
        return qk

    logging.info(
        "[qk_norm=auto] no signal from ckpt metadata or config; "
        "defaulting to False (matches open-source Wan2.2 VAE)."
    )
    return False


def load_model(
    pretrained_path: str,
    model_config: Optional[str],
    qk_norm: str,
    device: torch.device,
    dtype: torch.dtype,
) -> WanVAE22Model:
    pp = Path(pretrained_path)
    cfg_path = _resolve_model_config(pp, model_config)
    cfg_dict = WanVAE22Model.load_config(cfg_path)
    cfg_dict["qk_norm"] = _resolve_qk_norm(qk_norm, pp, cfg_dict)

    logging.info(
        f"Building WanVAE22Model from config={cfg_path}, qk_norm={cfg_dict['qk_norm']}"
    )
    model = WanVAE22Model.from_config(cfg_dict)

    if pp.is_file():
        logging.info(f"Loading weights from file: {pp}")
        model.init_from_ckpt(str(pp))
    else:
        weight_file = None
        for pattern in ("*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.bin"):
            files = sorted(pp.glob(pattern))
            if files:
                weight_file = files[-1]
                break
        if weight_file is None:
            raise FileNotFoundError(f"No weight file found under {pp}")
        logging.info(f"Loading weights from directory: {weight_file}")
        model.init_from_ckpt(str(weight_file))

    model = model.to(device).eval()
    model.requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"WanVAE22Model loaded: {n_params:.2f}M params, dtype={dtype}")
    return model


# --------------------------------------------------------------------------- #
# Reconstruction & metrics                                                     #
# --------------------------------------------------------------------------- #
def scan_mp4_files(input_dir: str) -> List[str]:
    return sorted(str(p) for p in Path(input_dir).rglob("*.mp4"))


def load_video_paths_from_jsonl(
    jsonl_paths: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """Read `video_path` field from one or more JSONL files.

    Returns:
        paths       - de-duplicated, ordered list of video paths
        path_to_grp - map of video_path -> group key (basename of jsonl
                      without extension), kept from the first occurrence.
    """
    paths: List[str] = []
    seen = set()
    path_to_group: Dict[str, str] = {}
    for jsonl_path in jsonl_paths:
        group = Path(jsonl_path).stem
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.warning(
                        f"JSONL {jsonl_path} line {line_no}: invalid JSON ({e})"
                    )
                    continue
                vp = item.get("video_path")
                if not vp:
                    logging.warning(
                        f"JSONL {jsonl_path} line {line_no}: missing 'video_path'"
                    )
                    continue
                if vp not in seen:
                    seen.add(vp)
                    paths.append(vp)
                    path_to_group[vp] = group
    return paths, path_to_group


def _psnr(x_01: torch.Tensor, y_01: torch.Tensor) -> float:
    mse = F.mse_loss(x_01, y_01).item()
    return 20.0 * math.log10(1.0 / (mse ** 0.5 + 1e-8))


@torch.no_grad()
def reconstruct_one(
    model: WanVAE22Model,
    video: torch.Tensor,                # (1, C, T, H, W) in [-1, 1]
    device: torch.device,
    dtype: torch.dtype,
    streaming: bool,
    sample_posterior: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    video = video.to(device)
    cast_dtype = None if dtype == torch.float32 else dtype
    if cast_dtype is not None and device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", dtype=cast_dtype)
    else:
        autocast_ctx = torch.amp.autocast("cuda", enabled=False)
    with autocast_ctx:
        posterior = model.encode(video, streaming_inference=streaming)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        recon = model.decode(latent, streaming_inference=streaming)

    video_01 = (video.float().clamp(-1, 1) + 1) / 2
    recon_01 = (recon.float().clamp(-1, 1) + 1) / 2
    metrics = {
        "l1":   F.l1_loss(recon_01, video_01).item(),
        "mse":  F.mse_loss(recon_01, video_01).item(),
        "psnr": _psnr(recon_01, video_01),
        "latent_shape": tuple(latent.shape),
        "recon_shape":  tuple(recon.shape),
    }
    return video, recon, metrics


@torch.no_grad()
def run_video_reconstruction(
    model: WanVAE22Model,
    video_paths: List[str],
    device: torch.device,
    num_frames: int,
    resolution: int,
    sample_rate: int,
    target_fps: float,
    dtype: torch.dtype,
    output_dir: Path,
    save_videos: bool,
    streaming: bool,
    sample_posterior: bool,
    path_to_group: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
    desc: str = "WanVAE22 Recon",
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    """Reconstruct each video and return raw sums + per-sample records.

    Returns:
        sums      - {group_name: {"count", "l1_sum", "psnr_sum", "mse_sum"}};
                    a special key "__total__" holds the aggregate over groups.
        per_sample- per-video PSNR/L1/MSE records (this rank only).
    """
    res = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
    if save_videos:
        gt_dir = output_dir / "gt"
        recon_dir = output_dir / "recon"
        compare_dir = output_dir / "compare"
        gt_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)

    save_fps = int(target_fps / sample_rate) if sample_rate > 0 else int(target_fps)

    sums: Dict[str, Dict[str, float]] = {
        "__total__": {"count": 0, "l1_sum": 0.0, "psnr_sum": 0.0, "mse_sum": 0.0}
    }
    per_sample: List[Dict[str, object]] = []
    pbar = tqdm(video_paths, desc=desc, disable=not show_progress)

    for vp in pbar:
        try:
            video, _ = _read_video_torchvision(
                vp, num_frames=num_frames, resolution=res,
                target_fps=target_fps, sample_rate=sample_rate,
            )
        except Exception as e:
            logging.warning(f"Skipping {vp}: {e}")
            continue
        video = video.unsqueeze(0)                          # (1, C, T, H, W)

        video, recon, metrics = reconstruct_one(
            model, video, device, dtype, streaming, sample_posterior,
        )

        group = (path_to_group or {}).get(vp, "default")
        for key in (group, "__total__"):
            s = sums.setdefault(
                key, {"count": 0, "l1_sum": 0.0, "psnr_sum": 0.0, "mse_sum": 0.0}
            )
            s["count"] += 1
            s["l1_sum"]  += metrics["l1"]
            s["psnr_sum"] += metrics["psnr"]
            s["mse_sum"]  += metrics["mse"]

        per_sample.append({
            "video": vp,
            "group": group,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        })

        if save_videos:
            name = Path(vp).stem
            gt_cpu = video[0].cpu()
            recon_cpu = recon[0].cpu()
            sub = group  # one sub-folder per dataset
            _write_video_torchvision(gt_cpu, gt_dir / sub / f"{name}.mp4", fps=save_fps)
            _write_video_torchvision(recon_cpu, recon_dir / sub / f"{name}.mp4", fps=save_fps)
            compare = torch.cat([gt_cpu, recon_cpu], dim=-1)  # (C, T, H, 2W)
            _write_video_torchvision(
                compare, compare_dir / sub / f"{name}.mp4", fps=save_fps
            )

        pbar.set_postfix(psnr=f"{metrics['psnr']:.2f}", l1=f"{metrics['l1']:.4f}")
        del video, recon
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    return sums, per_sample


def _reduce_sums(
    local_sums: Dict[str, Dict[str, float]],
    world_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Cross-rank reduction. Group keys are unioned across ranks, then each
    (count, l1_sum, psnr_sum, mse_sum) tuple is summed via dist.all_reduce.
    """
    if world_size <= 1:
        return local_sums

    all_groups: List[set] = [set() for _ in range(world_size)]
    dist.all_gather_object(all_groups, set(local_sums.keys()))
    union: List[str] = sorted(set().union(*all_groups))

    out: Dict[str, Dict[str, float]] = {}
    for g in union:
        s = local_sums.get(
            g, {"count": 0, "l1_sum": 0.0, "psnr_sum": 0.0, "mse_sum": 0.0}
        )
        buf = torch.tensor(
            [s["count"], s["l1_sum"], s["psnr_sum"], s["mse_sum"]],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        c = int(buf[0].item())
        out[g] = {
            "count": c,
            "l1_sum":   float(buf[1].item()),
            "psnr_sum": float(buf[2].item()),
            "mse_sum":  float(buf[3].item()),
        }
    return out


def _format_results(sums: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """Convert the reduced sums to the public results dict (with averages)."""
    by_group_avg: Dict[str, Dict[str, float]] = {}
    for g, s in sums.items():
        if g == "__total__":
            continue
        n = max(1, int(s["count"]))
        by_group_avg[g] = {
            "count": int(s["count"]),
            "psnr": s["psnr_sum"] / n,
            "l1":   s["l1_sum"]   / n,
            "mse":  s["mse_sum"]  / n,
        }
    t = sums.get("__total__", {"count": 0, "l1_sum": 0.0, "psnr_sum": 0.0, "mse_sum": 0.0})
    n = max(1, int(t["count"]))
    total = {
        "count": int(t["count"]),
        "psnr": t["psnr_sum"] / n if t["count"] else 0.0,
        "l1":   t["l1_sum"]   / n if t["count"] else 0.0,
        "mse":  t["mse_sum"]  / n if t["count"] else 0.0,
    }
    return {
        "total": total,
        "by_group": by_group_avg,
        # Promote total to top-level for back-compat with prior outputs.
        **{k: v for k, v in total.items() if k != "mse"},
    }


@torch.no_grad()
def run_random_sanity(
    model: WanVAE22Model,
    device: torch.device,
    dtype: torch.dtype,
    num_frames: int,
    resolution: int,
    streaming: bool,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    x = torch.rand(1, 3, num_frames, resolution, resolution, device=device) * 2 - 1
    logging.info(f"[sanity] input shape: {tuple(x.shape)}")
    _, _, metrics = reconstruct_one(
        model, x, device, dtype, streaming, sample_posterior=False,
    )
    logging.info(f"[sanity] latent shape: {metrics['latent_shape']}")
    logging.info(f"[sanity] recon  shape: {metrics['recon_shape']}")
    logging.info(
        f"[sanity] PSNR={metrics['psnr']:.2f} dB, L1={metrics['l1']:.4f}, "
        f"MSE={metrics['mse']:.6f}"
    )
    return {k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float))}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    rank, world_size, device = setup_dist()
    logging.basicConfig(
        level=logging.INFO if is_main(rank) else logging.WARNING,
        format=f"%(asctime)s [rank{rank}] [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Self-contained WanVAE22 (Wan2.2 Video VAE) reconstruction inference."
    )
    parser.add_argument("--pretrained_path", "-p", type=str, required=True,
                        help="Path to a .pth/.ckpt file or a directory.")
    parser.add_argument("--model_config", "-c", type=str, default=None,
                        help="Path to config.json. Auto-resolved if omitted.")
    parser.add_argument("--qk_norm", choices=["auto", "true", "false"], default="auto",
                        help="Whether the model uses qk_norm. 'auto' reads "
                             "ckpt['metadata']['qk_norm_filtered'] when present.")

    parser.add_argument("--input_video", type=str, default=None)
    parser.add_argument("--input_dir",   type=str, default=None)
    parser.add_argument("--input_jsonl", type=str, nargs="+", default=None,
                        help="One or more JSONL files. "
                             "Per-jsonl PSNR/L1 will be reported separately.")
    parser.add_argument("--random_input", action="store_true")

    parser.add_argument("--output_dir",  type=str, default="./wan22_recon")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_frames",  type=int, default=33)
    parser.add_argument("--resolution",  type=int, default=256)
    parser.add_argument("--target_fps",  type=float, default=8.0)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming_inference=True (lower mem).")
    parser.add_argument("--sample_posterior", action="store_true",
                        help="Use posterior.sample() instead of mode() (adds noise).")
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save reconstructed mp4s, only metrics.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if device.type == "cpu" and is_main(rank):
        logging.warning("CUDA unavailable; falling back to CPU (will be very slow).")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.dtype]

    inputs_specified = sum(int(bool(x)) for x in (
        args.input_video, args.input_dir, args.input_jsonl, args.random_input,
    ))
    if inputs_specified == 0:
        parser.error(
            "Specify one of --input_video / --input_dir / --input_jsonl / --random_input."
        )
    if inputs_specified > 1:
        parser.error(
            "--input_video / --input_dir / --input_jsonl / --random_input are mutually exclusive."
        )

    model = load_model(
        args.pretrained_path, args.model_config, args.qk_norm, device, dtype,
    )
    output_dir = Path(args.output_dir)

    if args.random_input:
        if is_main(rank):
            metrics = run_random_sanity(
                model, device, dtype,
                num_frames=args.num_frames, resolution=args.resolution,
                streaming=args.streaming, seed=args.seed,
            )
            if not args.no_save:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / "sanity_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                logging.info(f"Sanity metrics saved to {output_dir / 'sanity_metrics.json'}")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    path_to_group: Dict[str, str] = {}
    if args.input_video:
        video_paths = [args.input_video]
        source_desc = f"single video {args.input_video}"
    elif args.input_jsonl:
        video_paths, path_to_group = load_video_paths_from_jsonl(args.input_jsonl)
        source_desc = f"{len(args.input_jsonl)} JSONL file(s): {args.input_jsonl}"
    else:
        video_paths = scan_mp4_files(args.input_dir)
        source_desc = f"directory {args.input_dir}"
    if args.max_examples is not None and args.max_examples > 0:
        video_paths = video_paths[: args.max_examples]
    if is_main(rank):
        logging.info(f"Found {len(video_paths)} videos from {source_desc}")
    if not video_paths:
        if is_main(rank):
            logging.error("No video files found. Exiting.")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # Stride-shard across ranks for balanced load.
    local_paths = video_paths[rank::world_size]
    if is_main(rank):
        logging.info(
            f"World size = {world_size}. Rank 0 will process {len(local_paths)} of "
            f"{len(video_paths)} videos."
        )

    local_sums, local_per_sample = run_video_reconstruction(
        model, local_paths, device,
        num_frames=args.num_frames, resolution=args.resolution,
        sample_rate=args.sample_rate, target_fps=args.target_fps,
        dtype=dtype, output_dir=output_dir,
        save_videos=not args.no_save, streaming=args.streaming,
        sample_posterior=args.sample_posterior,
        path_to_group=path_to_group,
        show_progress=is_main(rank),
        desc=f"WanVAE22 Recon [rank0/{world_size}]",
    )

    reduced_sums = _reduce_sums(local_sums, world_size, device)
    metrics = _format_results(reduced_sums)

    # Gather per-sample records across ranks for a global json on rank 0.
    if world_size > 1:
        gathered: List[Optional[List[Dict[str, object]]]] = [None] * world_size
        dist.all_gather_object(gathered, local_per_sample)
        all_per_sample: List[Dict[str, object]] = []
        for chunk in gathered:
            all_per_sample.extend(chunk or [])
    else:
        all_per_sample = local_per_sample

    if is_main(rank):
        logging.info("=" * 60)
        logging.info("  WanVAE22 Reconstruction Results")
        logging.info("=" * 60)
        by_group = metrics.get("by_group", {}) or {}
        if by_group:
            for g, s in sorted(by_group.items()):
                logging.info(
                    f"  [{g:<20}] count={s['count']:<5d} "
                    f"PSNR={s['psnr']:.2f} dB  L1={s['l1']:.4f}  MSE={s['mse']:.6f}"
                )
            logging.info("-" * 60)
        logging.info(
            f"  [TOTAL]              count={metrics['count']:<5d} "
            f"PSNR={metrics['psnr']:.2f} dB  L1={metrics['l1']:.4f}"
        )
        logging.info("=" * 60)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        if not args.no_save:
            with open(output_dir / "per_sample.json", "w") as f:
                json.dump(all_per_sample, f, indent=2)
        logging.info(f"Results saved to {output_dir / 'results.json'}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
