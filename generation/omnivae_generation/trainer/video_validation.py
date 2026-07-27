from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from diffusers import FlowMatchEulerDiscreteScheduler

from omnivae_generation.trainer.data import maybe_format_chat_prompt
from omnivae_generation.trainer.modeling import configure_scheduler_prediction_target, decode_latents_to_images, encode_prompts
from omnivae_generation.trainer.utils import ensure_dir, flatten_gathered_record_chunks, save_json

# Per-sample latent seed offset; parallel validation uses one generator per prompt (differs from legacy single-generator sequence).
_VALIDATION_SAMPLE_SEED_STRIDE = 100_003


def apply_zimage_cfg(pos: torch.Tensor, neg: torch.Tensor, guidance_scale: float, cfg_normalization=False) -> torch.Tensor:
    pred = pos + float(guidance_scale) * (pos - neg)
    if cfg_normalization and float(cfg_normalization) > 0.0:
        ori_pos_norm = torch.linalg.vector_norm(pos)
        new_pos_norm = torch.linalg.vector_norm(pred)
        max_new_norm = ori_pos_norm * float(cfg_normalization)
        if new_pos_norm > max_new_norm:
            pred = pred * (max_new_norm / new_pos_norm)
    return pred


def _coerce_validation_frame_size(config: dict) -> tuple[int, int]:
    frame_size = config["train"].get("validation_frame_size") or config["dataset"].get("frame_size")
    if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
        raise ValueError("Video validation requires train.validation_frame_size or dataset.frame_size=[height, width].")
    height, width = int(frame_size[0]), int(frame_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Video validation frame size must be positive, got {frame_size!r}.")
    return height, width


def _get_vae_spatial_scale(vae) -> int:
    scale = getattr(getattr(vae, "config", None), "scale_factor_spatial", None)
    if scale is None:
        block_out_channels = getattr(getattr(vae, "config", None), "block_out_channels", None)
        scale = 2 ** (len(block_out_channels) - 1) if block_out_channels is not None else 8
    scale = int(scale)
    if scale <= 0:
        raise ValueError(f"Invalid VAE spatial scale for video validation: {scale!r}.")
    return scale


def _get_vae_temporal_scale(vae) -> int:
    scale = int(getattr(getattr(vae, "config", None), "scale_factor_temporal", 1) or 1)
    if scale <= 0:
        raise ValueError(f"Invalid VAE temporal scale for video validation: {scale!r}.")
    return scale


def _video_validation_latent_shape(config: dict, vae) -> tuple[int, int, int, int, int, int, float]:
    height, width = _coerce_validation_frame_size(config)
    num_frames = int(config["train"].get("validation_num_frames") or config["dataset"].get("num_frames") or 1)
    fps = float(config["train"].get("validation_fps") or config["dataset"].get("target_fps") or 8.0)
    if num_frames <= 0:
        raise ValueError(f"Video validation num_frames must be positive, got {num_frames!r}.")
    if fps <= 0.0:
        raise ValueError(f"Video validation fps must be positive, got {fps!r}.")

    spatial_scale = _get_vae_spatial_scale(vae)
    temporal_scale = _get_vae_temporal_scale(vae)
    if height % spatial_scale != 0 or width % spatial_scale != 0:
        raise ValueError(
            f"Video validation frame_size {(height, width)!r} must be divisible by VAE spatial scale {spatial_scale}."
        )
    latent_frames = 1 + (num_frames - 1) // temporal_scale
    latent_height = height // spatial_scale
    latent_width = width // spatial_scale
    return num_frames, latent_frames, height, width, latent_height, latent_width, fps


def _normalize_negative_prompts(negative_prompts, count: int) -> list[str]:
    if negative_prompts is None:
        return [""] * count
    if isinstance(negative_prompts, str):
        return [negative_prompts] * count
    negative_prompts = list(negative_prompts)
    if len(negative_prompts) == 1 and count > 1:
        return negative_prompts * count
    if len(negative_prompts) > count:
        raise ValueError(
            f"validation_negative_prompts has more entries than validation_prompts "
            f"({len(negative_prompts)} > {count})."
        )
    if len(negative_prompts) < count:
        # Pad with empty string (default unconditional branch) when the list was not updated after adding prompts.
        negative_prompts = negative_prompts + [""] * (count - len(negative_prompts))
    return [str(item) for item in negative_prompts]


def _video_tensor_to_uint8_frames(video: torch.Tensor):
    frames = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).cpu().numpy()


def _load_uint8_frames_from_mp4(video_path: Path) -> np.ndarray:
    """Return ``(T, H, W, C)`` uint8 numpy array for TensorBoard."""
    frames = imageio.mimread(str(video_path), memtest=False)
    if not frames:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)


def _build_inference_scheduler(config: dict, scheduler, transformer_model, device: torch.device):
    inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    configure_scheduler_prediction_target(
        inference_scheduler,
        getattr(transformer_model, "_laion_predict_target", config["transformer"].get("predict_target", "v")),
    )
    inference_scheduler.set_timesteps(
        int(config["train"]["validation_num_inference_steps"]),
        device=device,
    )
    return inference_scheduler


def _log_video_samples_to_trackers(accelerator: Accelerator, records: list[dict], frames_by_sample: list, *, fps: float, step: int) -> None:
    if not records or not frames_by_sample:
        return

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for record, frames in zip(records, frames_by_sample):
                video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
                tracker.writer.add_video(
                    f"validation/samples/sample-{int(record['sample_index']):02d}",
                    video_tensor,
                    global_step=step,
                    fps=fps,
                )
        elif tracker.name == "wandb":
            import wandb

            videos = [
                wandb.Video(
                    record["video_path"],
                    caption=record["prompt"],
                    fps=max(1, int(round(float(fps)))),
                    format="mp4",
                )
                for record in records
            ]
            tracker.log({"validation/samples": videos}, step=step)


def _validation_sample_seed(*, base_seed: int | None, sample_index: int) -> int | None:
    """Deterministic per-prompt seed; ``base_seed`` should match legacy ``train.seed + step``."""
    if base_seed is None:
        return None
    return int(base_seed) + int(sample_index) * _VALIDATION_SAMPLE_SEED_STRIDE


@torch.no_grad()
def run_video_validation(
    accelerator: Accelerator,
    config: dict,
    step: int,
    transformer,
    tokenizer,
    text_encoder,
    vae,
    scheduler,
) -> None:
    prompts = config["train"].get("validation_prompts") or []
    if not prompts:
        return
    if text_encoder is None:
        raise NotImplementedError("Video validation v1 requires a separate text encoder; qwen3_vl_dit is not supported.")

    transformer_model = accelerator.unwrap_model(transformer, keep_torch_compile=False)
    text_encoder_model = accelerator.unwrap_model(text_encoder, keep_torch_compile=False)
    vae_model = accelerator.unwrap_model(vae, keep_torch_compile=False)

    was_compiled = False
    if hasattr(transformer_model, "is_forward_compilation_enabled"):
        was_compiled = transformer_model.is_forward_compilation_enabled()
        if was_compiled:
            transformer_model.set_forward_compilation(False)

    transformer_was_training = transformer_model.training
    text_encoder_was_training = text_encoder_model.training
    vae_was_training = vae_model.training
    text_encoder_model.eval()
    vae_model.eval()
    transformer_model.eval()

    local_records: list[dict] = []
    try:
        from omnivae_generation.trainer.forward_transformer import build_forward_transformer

        (
            requested_num_frames,
            latent_frames,
            height,
            width,
            latent_height,
            latent_width,
            fps,
        ) = _video_validation_latent_shape(config, vae_model)

        train_patch_size = int(config["transformer"]["all_patch_size"][0])
        train_f_patch_size = int(config["transformer"]["all_f_patch_size"][0])
        forward_transformer = build_forward_transformer(
            transformer_model,
            transformer_model,
            train_patch_size=train_patch_size,
            train_f_patch_size=train_f_patch_size,
        )
        guidance_scale = float(config["train"].get("validation_guidance_scale", 4.0))
        cfg_normalization = config["train"].get("validation_cfg_normalization", False)
        cfg_truncation = config["train"].get("validation_cfg_truncation", 1.0)
        cfg_truncation = None if cfg_truncation is None else float(cfg_truncation)
        negative_prompts = _normalize_negative_prompts(config["train"].get("validation_negative_prompts"), len(prompts))
        base_seed = None if config["train"].get("seed") is None else int(config["train"]["seed"]) + int(step)

        sample_dir = ensure_dir(Path(config["experiment"]["output_dir"]) / "samples" / f"step-{step:08d}")
        num_processes = int(accelerator.num_processes)
        process_index = int(accelerator.process_index)
        shard_indices = [i for i in range(len(prompts)) if i % num_processes == process_index]

        for sample_index in shard_indices:
            prompt = prompts[sample_index]
            negative_prompt = negative_prompts[sample_index]
            sample_seed = _validation_sample_seed(base_seed=base_seed, sample_index=sample_index)
            sample_generator = None
            if sample_seed is not None:
                sample_generator = torch.Generator(device=accelerator.device).manual_seed(sample_seed)

            inference_scheduler = _build_inference_scheduler(config, scheduler, transformer_model, accelerator.device)
            formatted_prompt = maybe_format_chat_prompt(str(prompt), tokenizer)
            formatted_negative_prompt = maybe_format_chat_prompt(str(negative_prompt), tokenizer)
            prompt_embeds = encode_prompts(
                [formatted_prompt],
                tokenizer,
                text_encoder_model,
                accelerator.device,
                int(config["text_encoder"]["max_sequence_length"]),
                cache_enabled=bool(config["text_encoder"].get("cache_enabled", False)),
            )
            negative_prompt_embeds = encode_prompts(
                [formatted_negative_prompt],
                tokenizer,
                text_encoder_model,
                accelerator.device,
                int(config["text_encoder"]["max_sequence_length"]),
                cache_enabled=bool(config["text_encoder"].get("cache_enabled", False)),
            )

            latents = torch.randn(
                (1, int(config["transformer"]["in_channels"]), latent_frames, latent_height, latent_width),
                generator=sample_generator,
                device=accelerator.device,
                dtype=torch.float32,
            )
            for timestep_value in inference_scheduler.timesteps:
                timestep = timestep_value.expand(latents.shape[0])
                model_timesteps = (
                    float(inference_scheduler.config.num_train_timesteps) - timestep
                ) / float(inference_scheduler.config.num_train_timesteps)
                model_timesteps = model_timesteps.to(device=accelerator.device, dtype=torch.float32)
                t_norm = float(model_timesteps[0].item())
                current_guidance_scale = guidance_scale
                if cfg_truncation is not None and cfg_truncation <= 1.0 and t_norm > cfg_truncation:
                    current_guidance_scale = 0.0

                apply_cfg = current_guidance_scale > 0.0
                if apply_cfg:
                    latent_model_input = latents.repeat(2, 1, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = model_timesteps.repeat(2)
                else:
                    latent_model_input = latents
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = model_timesteps

                model_pred, _ = forward_transformer(
                    latent_model_input.to(dtype=getattr(transformer_model, "dtype", latents.dtype)),
                    timestep_model_input,
                    prompt_embeds_model_input,
                )
                if apply_cfg:
                    pos_pred = model_pred[:1].float()
                    neg_pred = model_pred[1:].float()
                    model_pred = apply_zimage_cfg(pos_pred, neg_pred, current_guidance_scale, cfg_normalization)
                else:
                    model_pred = model_pred.float()
                model_pred = -model_pred
                latents = inference_scheduler.step(
                    model_pred.to(torch.float32),
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]
                latents = latents.to(torch.float32)

            video = decode_latents_to_images(
                latents.to(dtype=getattr(vae_model, "dtype", latents.dtype)),
                vae_model,
            )[0]
            frames = _video_tensor_to_uint8_frames(video)
            video_path = sample_dir / f"sample-{sample_index:02d}.mp4"
            imageio.mimsave(video_path, list(frames), fps=fps, codec="libx264", quality=8, macro_block_size=None)
            local_records.append(
                {
                    "sample_index": sample_index,
                    "video_path": str(video_path),
                    "prompt": str(prompt),
                    "negative_prompt": str(negative_prompt),
                    "requested_num_frames": requested_num_frames,
                    "decoded_num_frames": int(video.shape[1]),
                    "latent_frames": latent_frames,
                    "height": height,
                    "width": width,
                    "fps": fps,
                    "num_inference_steps": int(config["train"]["validation_num_inference_steps"]),
                    "guidance_scale": guidance_scale,
                    "cfg_normalization": cfg_normalization,
                    "cfg_truncation": cfg_truncation,
                    "sample_seed": sample_seed,
                    "base_seed": base_seed,
                }
            )

        accelerator.wait_for_everyone()
        gathered = gather_object(local_records)
        flat_records = flatten_gathered_record_chunks(gathered)
        if accelerator.is_main_process:
            by_idx = {int(rec["sample_index"]): rec for rec in flat_records}
            records = [by_idx[k] for k in sorted(by_idx.keys())]
            save_json(
                sample_dir / "sample_manifest.json",
                {
                    "step": step,
                    "samples": records,
                    "num_inference_steps": int(config["train"]["validation_num_inference_steps"]),
                    "guidance_scale": guidance_scale,
                    "base_seed": base_seed,
                    "validation_sample_seed_stride": _VALIDATION_SAMPLE_SEED_STRIDE,
                },
            )
            frames_by_sample = [_load_uint8_frames_from_mp4(Path(rec["video_path"])) for rec in records]
            _log_video_samples_to_trackers(accelerator, records, frames_by_sample, fps=fps, step=step)
    finally:
        if was_compiled:
            transformer_model.set_forward_compilation(True)
        transformer_model.train(transformer_was_training)
        text_encoder_model.train(text_encoder_was_training)
        vae_model.train(vae_was_training)
        if hasattr(vae_model, "clear_cache"):
            vae_model.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
