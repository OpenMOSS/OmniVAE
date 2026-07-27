from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from accelerate import Accelerator
from torchcodec.decoders import VideoDecoder
from torchvision.utils import make_grid, save_image

from omnivae_generation.trainer.modeling import decode_latents_to_images, encode_images_to_latents
from omnivae_generation.trainer.utils import ensure_dir, save_json
from omnivae_generation.trainer.video_data import _close_decoder, _extract_torchcodec_metadata, _resize_and_crop_video


MCL_JCV_QP20_PATH_LIST = Path("examples/metadata/video_vae_validation.txt")


def _positive_int(value, default: int) -> int:
    if value is None:
        return int(default)
    parsed = int(value)
    return max(1, parsed)


def _optional_positive_int(value) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _prepare_pixel_values(pixel_values: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    source_dtype = pixel_values.dtype
    prepared = pixel_values.to(device, dtype=dtype, non_blocking=True)
    if source_dtype == torch.uint8:
        prepared.mul_(1.0 / 127.5).sub_(1.0)
    return prepared


def _to_01(pixel_values: torch.Tensor) -> torch.Tensor:
    return ((pixel_values.detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _crop_to_common_shape(inputs: torch.Tensor, reconstructions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != reconstructions.ndim:
        raise ValueError(
            "VAE validation expected reconstruction rank to match input rank, "
            f"got input={tuple(inputs.shape)} reconstruction={tuple(reconstructions.shape)}."
        )
    slices = [slice(None)] * inputs.ndim
    for dim in range(inputs.ndim):
        size = min(int(inputs.shape[dim]), int(reconstructions.shape[dim]))
        slices[dim] = slice(0, size)
    index = tuple(slices)
    return inputs[index], reconstructions[index]


def _mse_per_sample(inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    batch = int(inputs.shape[0])
    return (inputs.float() - reconstructions.float()).pow(2).reshape(batch, -1).mean(dim=1)


def _psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return 20.0 * math.log10(2.0) - 10.0 * torch.log10(mse.float().clamp_min(1.0e-12))


def _psnr_per_sample(inputs: torch.Tensor, reconstructions: torch.Tensor, *, video_aggregation: str) -> torch.Tensor:
    if inputs.ndim == 5 and video_aggregation == "frame_mean":
        batch = int(inputs.shape[0])
        frames = int(inputs.shape[2])
        mse_frame = (
            inputs.float()
            .sub(reconstructions.float())
            .pow(2)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, frames, -1)
            .mean(dim=2)
        )
        return _psnr_from_mse(mse_frame).mean(dim=1)
    return _psnr_from_mse(_mse_per_sample(inputs, reconstructions))


def _save_image_recon_grid(inputs: torch.Tensor, reconstructions: torch.Tensor, output_path: Path) -> torch.Tensor:
    count = int(inputs.shape[0])
    grid = make_grid(torch.cat([_to_01(inputs), _to_01(reconstructions)], dim=0).cpu(), nrow=count, padding=2)
    save_image(grid, output_path)
    return grid


def _video_to_btchw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor [B, C, T, H, W], got {tuple(video.shape)}.")
    return video.permute(0, 2, 1, 3, 4).contiguous()


def _make_video_recon_strip(inputs: torch.Tensor, reconstructions: torch.Tensor, *, max_frames: int) -> torch.Tensor:
    inputs = _video_to_btchw(_to_01(inputs))
    reconstructions = _video_to_btchw(_to_01(reconstructions))
    items = int(inputs.shape[0])
    frames = min(int(max_frames), int(inputs.shape[1]))
    strips = []
    for sample_index in range(items):
        panel = torch.cat([inputs[sample_index, :frames], reconstructions[sample_index, :frames]], dim=0)
        strips.append(make_grid(panel.cpu(), nrow=frames, padding=2))
    if len(strips) == 1:
        return strips[0]
    pad = torch.zeros((strips[0].shape[0], 2, strips[0].shape[2]), dtype=strips[0].dtype)
    grid = strips[0]
    for strip in strips[1:]:
        grid = torch.cat([grid, pad, strip], dim=1)
    return grid


def _make_video_recon_frames(inputs: torch.Tensor, reconstructions: torch.Tensor, *, max_frames: int) -> list:
    inputs = _video_to_btchw(_to_01(inputs))
    reconstructions = _video_to_btchw(_to_01(reconstructions))
    items = int(inputs.shape[0])
    frames = min(int(max_frames), int(inputs.shape[1]))
    rendered = []
    for frame_index in range(frames):
        panel = torch.cat([inputs[:, frame_index], reconstructions[:, frame_index]], dim=0)
        grid = make_grid(panel.cpu(), nrow=items, padding=2)
        if int(grid.shape[0]) == 1:
            grid = grid.repeat(3, 1, 1)
        elif int(grid.shape[0]) > 3:
            grid = grid[:3]
        rendered.append(
            grid.mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        )
    return rendered


def _slice_metadata(batch: dict, key: str, count: int) -> list[str]:
    values = batch.get(key)
    if values is None:
        return []
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return [str(value) for value in list(values)[:count]]


def _mean_or_nan(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _resolve_vae_validation_path_list(train_cfg: dict[str, Any]) -> Path | None:
    path_list = train_cfg.get("vae_validation_path_list")
    if path_list:
        return Path(str(path_list)).expanduser().resolve()

    source = str(train_cfg.get("vae_validation_data") or "").strip().lower().replace("-", "_")
    if source in {"mcl_jcv", "mcl_jcv_qp20", "mcljcv", "mcljcv_qp20"}:
        return MCL_JCV_QP20_PATH_LIST
    return None


def _read_video_path_list(path_list: Path, *, limit: int | None) -> list[Path]:
    if not path_list.is_file():
        raise FileNotFoundError(f"VAE validation video path list not found: {path_list}")
    paths: list[Path] = []
    for raw_line in path_list.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = (path_list.parent / path).resolve()
        paths.append(path)
        if limit is not None and len(paths) >= int(limit):
            break
    if not paths:
        raise RuntimeError(f"VAE validation video path list is empty: {path_list}")
    return paths


def _resolve_video_frame_size(config: dict) -> tuple[int, int]:
    train_cfg = config["train"]
    dataset_cfg = config.get("dataset", {})
    frame_size = (
        train_cfg.get("vae_validation_frame_size")
        or train_cfg.get("validation_frame_size")
        or dataset_cfg.get("frame_size")
    )
    if frame_size is None and dataset_cfg.get("image_size") is not None:
        size = int(dataset_cfg["image_size"])
        frame_size = [size, size]
    if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
        raise ValueError("VAE video validation requires train.vae_validation_frame_size or dataset.frame_size=[height, width].")
    height, width = int(frame_size[0]), int(frame_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"VAE video validation frame size must be positive, got {frame_size!r}.")
    return height, width


def _decode_center_video_clip(
    path: Path,
    *,
    frame_size: tuple[int, int],
    num_frames: int,
    num_ffmpeg_threads: int,
    pad_mode: str,
) -> torch.Tensor:
    decoder = VideoDecoder(
        path,
        dimension_order="NCHW",
        num_ffmpeg_threads=int(num_ffmpeg_threads),
        device="cpu",
        seek_mode="exact",
    )
    try:
        metadata = _extract_torchcodec_metadata(decoder)
        total_frames = int(metadata["num_frames"])
        if total_frames <= 0:
            raise RuntimeError(f"Video has no decodable frames: {path}")

        target_frames = max(1, int(num_frames))
        start = max(0, (total_frames - target_frames) // 2)
        end = min(total_frames, start + target_frames)
        decoded = decoder[int(start) : int(end)]
        if decoded.numel() == 0:
            raise RuntimeError(f"torchcodec returned an empty clip for {path}")

        width = int(metadata.get("width") or decoded.shape[-1])
        height = int(metadata.get("height") or decoded.shape[-2])
        sampled = decoded.permute(1, 0, 2, 3).contiguous()
        pixel_values = _resize_and_crop_video(
            sampled,
            source_height=height,
            source_width=width,
            target_height=int(frame_size[0]),
            target_width=int(frame_size[1]),
            center_crop=True,
            random_flip=False,
        )
        if pixel_values.dtype != torch.uint8:
            pixel_values = pixel_values.clamp(0, 255).to(torch.uint8)

        pad = target_frames - int(pixel_values.shape[1])
        if pad > 0:
            mode = str(pad_mode).strip().lower()
            if mode == "black":
                pad_value = 0
            elif mode == "white":
                pad_value = 255
            else:
                pad_value = 128
            fill = torch.full(
                (int(pixel_values.shape[0]), int(pad), int(pixel_values.shape[2]), int(pixel_values.shape[3])),
                int(pad_value),
                dtype=torch.uint8,
            )
            pixel_values = torch.cat([pixel_values, fill], dim=1)
        return pixel_values.contiguous()
    finally:
        _close_decoder(decoder)


def _build_video_path_list_batches(config: dict) -> list[dict] | None:
    train_cfg = config["train"]
    path_list = _resolve_vae_validation_path_list(train_cfg)
    if path_list is None:
        return None

    limit = _optional_positive_int(train_cfg.get("vae_validation_dataset_limit"))
    paths = _read_video_path_list(path_list, limit=limit)
    batch_size = _positive_int(train_cfg.get("vae_validation_batch_size"), default=1)
    frame_size = _resolve_video_frame_size(config)
    num_frames = _positive_int(train_cfg.get("vae_validation_num_frames"), default=17)
    num_ffmpeg_threads = _positive_int(train_cfg.get("vae_validation_num_ffmpeg_threads"), default=1)
    pad_mode = str(train_cfg.get("vae_validation_pad_mode") or "gray")

    batches: list[dict] = []
    for start in range(0, len(paths), batch_size):
        chunk_paths = paths[start : start + batch_size]
        clips = [
            _decode_center_video_clip(
                path,
                frame_size=frame_size,
                num_frames=num_frames,
                num_ffmpeg_threads=num_ffmpeg_threads,
                pad_mode=pad_mode,
            )
            for path in chunk_paths
        ]
        batches.append(
            {
                "pixel_values": torch.stack(clips),
                "image_ids": [f"{path.stem}:{start + offset}" for offset, path in enumerate(chunk_paths)],
                "image_paths": [str(path) for path in chunk_paths],
                "validation_source": str(train_cfg.get("vae_validation_data") or path_list.name),
                "validation_path_list": str(path_list),
            }
        )
    return batches


def _metric_names(config: dict) -> set[str]:
    raw_metrics = config["train"].get("vae_validation_metrics")
    if raw_metrics is None:
        raw_metrics = ["psnr"]
    return {str(metric).strip().lower() for metric in raw_metrics if str(metric).strip()}


def _build_lpips_metric(metrics: set[str], device: torch.device):
    if "lpips" not in metrics:
        return None, None
    try:
        import piq  # type: ignore
    except Exception:
        return None, None
    try:
        metric = piq.LPIPS(reduction="none").to(device=device)
        metric.eval()
        metric.requires_grad_(False)
    except Exception:
        metric = None
    return piq, metric


def _quality_metrics_per_sample(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    *,
    metrics: set[str],
    video_aggregation: str,
    piq_module,
    lpips_metric,
) -> dict[str, list[float]]:
    batch = int(inputs.shape[0])
    mse_values = _mse_per_sample(inputs, reconstructions)
    psnr_values = _psnr_per_sample(inputs, reconstructions, video_aggregation=video_aggregation)
    out = {
        "mse": [float(value) for value in mse_values.detach().cpu().tolist()],
        "psnr": [float(value) for value in psnr_values.detach().cpu().tolist()],
    }

    want_ssim = "ssim" in metrics
    want_lpips = "lpips" in metrics
    if not want_ssim and not want_lpips:
        return out

    inputs_01 = _to_01(inputs).to(device=inputs.device)
    recons_01 = _to_01(reconstructions).to(device=inputs.device)
    if inputs_01.ndim == 5:
        inputs_btchw = _video_to_btchw(inputs_01)
        recons_btchw = _video_to_btchw(recons_01)
        bsz, frames, channels, height, width = inputs_btchw.shape
        inputs_2d = inputs_btchw.reshape(int(bsz) * int(frames), int(channels), int(height), int(width))
        recons_2d = recons_btchw.reshape(int(bsz) * int(frames), int(channels), int(height), int(width))
    elif inputs_01.ndim == 4:
        bsz, channels, height, width = inputs_01.shape
        frames = 1
        inputs_2d = inputs_01
        recons_2d = recons_01
    else:
        raise ValueError(f"Quality metrics expect image/video tensors, got {tuple(inputs.shape)}.")

    if int(channels) == 1:
        inputs_2d = inputs_2d.repeat(1, 3, 1, 1)
        recons_2d = recons_2d.repeat(1, 3, 1, 1)

    if want_ssim:
        if piq_module is None:
            try:
                import piq as piq_module  # type: ignore
            except Exception:
                piq_module = None
        if piq_module is None:
            out["ssim"] = [float("nan")] * batch
        else:
            values = piq_module.ssim(recons_2d.float(), inputs_2d.float(), data_range=1.0, reduction="none")
            out["ssim"] = [
                float(value)
                for value in values.reshape(int(batch), int(frames)).mean(dim=1).detach().cpu().tolist()
            ]

    if want_lpips:
        if lpips_metric is None:
            out["lpips"] = [float("nan")] * batch
        else:
            values = lpips_metric(recons_2d.float(), inputs_2d.float()).reshape(-1)
            out["lpips"] = [
                float(value)
                for value in values.reshape(int(batch), int(frames)).mean(dim=1).detach().cpu().tolist()
            ]
    return out


def _log_to_trackers(
    accelerator: Accelerator,
    *,
    metrics: dict[str, float],
    image_grid: torch.Tensor,
    video_path: Path | None,
    fps: float,
    step: int,
) -> None:
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for key, value in metrics.items():
                tracker.writer.add_scalar(key, value, global_step=step)
            tracker.writer.add_image("validation/vae/reconstruction", image_grid, global_step=step)
        elif tracker.name == "wandb":
            import wandb

            image_array = (
                image_grid.detach()
                .float()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .add(0.5)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            payload = dict(metrics)
            payload["validation/vae/reconstruction"] = wandb.Image(image_array)
            if video_path is not None:
                payload["validation/vae/reconstruction_video"] = wandb.Video(
                    str(video_path),
                    fps=max(1, int(round(float(fps)))),
                    format="mp4",
                )
            tracker.log(payload, step=step)


@torch.no_grad()
def run_vae_validation(
    accelerator: Accelerator,
    config: dict,
    *,
    step: int,
    batch: dict,
    vae,
    vae_dtype: torch.dtype,
) -> None:
    interval = int(config.get("train", {}).get("vae_validation_steps") or 0)
    if interval <= 0 or not accelerator.is_main_process:
        return

    preview_items = _positive_int(config["train"].get("vae_validation_num_samples"), default=4)
    validation_batches = _build_video_path_list_batches(config)
    if validation_batches is None:
        pixel_values = batch.get("pixel_values")
        if not torch.is_tensor(pixel_values):
            return
        validation_batch = dict(batch)
        validation_batch["pixel_values"] = pixel_values[:preview_items]
        validation_batches = [validation_batch]
    if not validation_batches:
        return

    max_frames = _positive_int(config["train"].get("vae_validation_max_video_frames"), default=8)
    sample_mode = str(config["train"].get("vae_validation_sample_mode", "argmax")).strip().lower()
    if sample_mode not in {"sample", "argmax"}:
        raise ValueError(f"Unsupported train.vae_validation_sample_mode={sample_mode!r}; expected 'sample' or 'argmax'.")
    video_aggregation = str(config["train"].get("vae_validation_psnr_video_aggregation", "frame_mean")).strip().lower()
    if video_aggregation not in {"frame_mean", "global_mse"}:
        raise ValueError(
            "Unsupported train.vae_validation_psnr_video_aggregation="
            f"{video_aggregation!r}; expected 'frame_mean' or 'global_mse'."
        )
    requested_metrics = _metric_names(config)

    vae_model = accelerator.unwrap_model(vae, keep_torch_compile=False)
    vae_was_training = bool(vae_model.training)
    vae_model.eval()
    piq_module, lpips_metric = _build_lpips_metric(requested_metrics, accelerator.device)
    try:
        per_sample_records: list[dict[str, Any]] = []
        metric_values: dict[str, list[float]] = {"mse": [], "psnr": []}
        latent_sum = 0.0
        latent_sq_sum = 0.0
        latent_count = 0
        preview_inputs: list[torch.Tensor] = []
        preview_recons: list[torch.Tensor] = []
        input_shape = None
        reconstruction_shape = None
        latent_shape = None
        validation_source = validation_batches[0].get("validation_source")
        validation_path_list = validation_batches[0].get("validation_path_list")

        for validation_batch in validation_batches:
            pixel_values = validation_batch.get("pixel_values")
            if not torch.is_tensor(pixel_values):
                continue
            inputs = _prepare_pixel_values(pixel_values, device=accelerator.device, dtype=vae_dtype)
            latents = encode_images_to_latents(
                inputs,
                vae_model,
                update_stats=False,
                sample_mode=sample_mode,
            )
            reconstructions = decode_latents_to_images(
                latents.to(dtype=getattr(vae_model, "dtype", latents.dtype)),
                vae_model,
            )
            inputs, reconstructions = _crop_to_common_shape(inputs, reconstructions)

            quality = _quality_metrics_per_sample(
                inputs,
                reconstructions,
                metrics=requested_metrics,
                video_aggregation=video_aggregation,
                piq_module=piq_module,
                lpips_metric=lpips_metric,
            )
            for key, values in quality.items():
                metric_values.setdefault(key, []).extend(values)

            latents_float = latents.detach().float()
            latent_sum += float(latents_float.sum().cpu())
            latent_sq_sum += float(latents_float.pow(2).sum().cpu())
            latent_count += int(latents_float.numel())
            if input_shape is None:
                input_shape = [int(dim) for dim in inputs.shape]
                reconstruction_shape = [int(dim) for dim in reconstructions.shape]
                latent_shape = [int(dim) for dim in latents.shape]

            image_ids = _slice_metadata(validation_batch, "image_ids", int(inputs.shape[0]))
            image_paths = _slice_metadata(validation_batch, "image_paths", int(inputs.shape[0]))
            for sample_index in range(int(inputs.shape[0])):
                record = {
                    "image_id": image_ids[sample_index] if sample_index < len(image_ids) else "",
                    "image_path": image_paths[sample_index] if sample_index < len(image_paths) else "",
                }
                for key, values in quality.items():
                    record[key] = float(values[sample_index])
                per_sample_records.append(record)

            remaining_preview = int(preview_items) - sum(int(item.shape[0]) for item in preview_inputs)
            if remaining_preview > 0:
                preview_inputs.append(inputs[:remaining_preview].detach().cpu())
                preview_recons.append(reconstructions[:remaining_preview].detach().cpu())

        if not per_sample_records:
            return

        latent_mean = latent_sum / max(1, latent_count)
        latent_var = max(0.0, latent_sq_sum / max(1, latent_count) - latent_mean * latent_mean)
        metrics = {
            "validation/vae/mse": _mean_or_nan(metric_values.get("mse", [])),
            "validation/vae/psnr": _mean_or_nan(metric_values.get("psnr", [])),
            "validation/vae/latent_mean": float(latent_mean),
            "validation/vae/latent_std": float(math.sqrt(latent_var)),
        }
        for key, values in metric_values.items():
            if key in {"mse", "psnr"}:
                continue
            metrics[f"validation/vae/{key}"] = _mean_or_nan(values)

        sample_dir = ensure_dir(Path(config["experiment"]["output_dir"]) / "vae_samples" / f"step-{int(step):08d}")
        video_path = None
        preview_input_tensor = torch.cat(preview_inputs, dim=0)
        preview_recon_tensor = torch.cat(preview_recons, dim=0)
        if preview_input_tensor.ndim == 5:
            fps = float(
                config["train"].get("vae_validation_fps")
                or config["train"].get("validation_fps")
                or config["dataset"].get("target_fps")
                or 8.0
            )
            image_grid = _make_video_recon_strip(preview_input_tensor, preview_recon_tensor, max_frames=max_frames)
            save_image(image_grid, sample_dir / "reconstruction.png")
            video_path = sample_dir / "reconstruction.mp4"
            imageio.mimsave(
                video_path,
                _make_video_recon_frames(preview_input_tensor, preview_recon_tensor, max_frames=max_frames),
                fps=max(1.0, fps),
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        elif preview_input_tensor.ndim == 4:
            fps = 0.0
            image_grid = _save_image_recon_grid(preview_input_tensor, preview_recon_tensor, sample_dir / "reconstruction.png")
        else:
            raise ValueError(f"VAE validation expects image/video inputs, got {tuple(preview_input_tensor.shape)}.")

        save_json(
            sample_dir / "metrics.json",
            {
                "step": int(step),
                "sample_mode": sample_mode,
                "validation_source": validation_source,
                "validation_path_list": validation_path_list,
                "psnr_video_aggregation": video_aggregation,
                "num_samples": int(len(per_sample_records)),
                "num_preview_samples": int(preview_input_tensor.shape[0]),
                "input_shape": input_shape,
                "reconstruction_shape": reconstruction_shape,
                "latent_shape": latent_shape,
                "mse_per_sample": [float(record["mse"]) for record in per_sample_records],
                "psnr_per_sample": [float(record["psnr"]) for record in per_sample_records],
                "metrics": metrics,
                "per_sample": per_sample_records,
            },
        )
        _log_to_trackers(
            accelerator,
            metrics=metrics,
            image_grid=image_grid,
            video_path=video_path,
            fps=fps,
            step=int(step),
        )
    finally:
        vae_model.train(vae_was_training)
        if hasattr(vae_model, "clear_cache"):
            vae_model.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
