from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import make_grid, save_image

from omnivae_generation.trainer.data import maybe_format_chat_prompt
from omnivae_generation.trainer.modeling import build_validation_pipeline
from omnivae_generation.trainer.qwen3_vl_dit import Qwen3VLDiffusionTransformer, tokenize_prompt_payloads
from omnivae_generation.trainer.utils import ensure_dir, flatten_gathered_record_chunks, save_json
from omnivae_generation.trainer.video_validation import run_video_validation

# Match video validation per-prompt RNG stride for parallel image validation.
_VALIDATION_SAMPLE_SEED_STRIDE = 100_003


def save_image_grid(images, output_path: Path, nrow: int = 2) -> None:
    tensors = [pil_to_tensor(image).float() / 255.0 for image in images]
    grid = make_grid(tensors, nrow=nrow)
    save_image(grid, output_path)


def run_validation(
    accelerator: Accelerator,
    config: dict,
    step: int,
    transformer,
    tokenizer,
    text_encoder,
    vae,
    scheduler,
) -> None:
    if int(config["train"].get("validation_steps") or 0) <= 0:
        return

    dataset_type = str(config.get("dataset", {}).get("type", "imagenet")).strip().lower()
    if dataset_type == "video_jsonl":
        run_video_validation(
            accelerator=accelerator,
            config=config,
            step=step,
            transformer=transformer,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
        )
        return
    if dataset_type == "audio_jsonl":
        from omnivae_generation.trainer.audio_validation import run_audio_validation

        run_audio_validation(
            accelerator=accelerator,
            config=config,
            step=step,
            transformer=transformer,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
        )
        return

    prompts = config["train"].get("validation_prompts") or []
    if not prompts:
        return

    transformer_model = accelerator.unwrap_model(transformer, keep_torch_compile=False)
    was_compiled = False
    if hasattr(transformer_model, "is_forward_compilation_enabled"):
        was_compiled = transformer_model.is_forward_compilation_enabled()
        if was_compiled:
            transformer_model.set_forward_compilation(False)
    text_encoder_model = (
        accelerator.unwrap_model(text_encoder, keep_torch_compile=False) if text_encoder is not None else None
    )
    vae_model = accelerator.unwrap_model(vae, keep_torch_compile=False)

    pipeline = build_validation_pipeline(
        transformer=transformer_model,
        tokenizer=tokenizer,
        text_encoder=text_encoder_model,
        vae=vae_model,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device)

    base_seed = None if config["train"].get("seed") is None else int(config["train"]["seed"]) + int(step)
    sample_dir = ensure_dir(Path(config["experiment"]["output_dir"]) / "samples")
    num_processes = int(accelerator.num_processes)
    process_index = int(accelerator.process_index)
    shard_indices = [i for i in range(len(prompts)) if i % num_processes == process_index]

    local_records: list[dict] = []
    for idx in shard_indices:
        prompt = prompts[idx]
        sample_generator = None
        if base_seed is not None:
            sample_generator = torch.Generator(device=accelerator.device).manual_seed(
                base_seed + idx * _VALIDATION_SAMPLE_SEED_STRIDE
            )
        if isinstance(transformer_model, Qwen3VLDiffusionTransformer):
            formatted_prompt = maybe_format_chat_prompt(prompt, tokenizer)
            prompt_payloads = tokenize_prompt_payloads(
                [formatted_prompt],
                tokenizer,
                device=accelerator.device,
                max_sequence_length=int(
                    config["transformer"].get(
                        "max_sequence_length",
                        config.get("text_encoder", {}).get("max_sequence_length", 512),
                    )
                ),
            )
            negative_prompt_payloads = tokenize_prompt_payloads(
                [maybe_format_chat_prompt("", tokenizer)],
                tokenizer,
                device=accelerator.device,
                max_sequence_length=int(
                    config["transformer"].get(
                        "max_sequence_length",
                        config.get("text_encoder", {}).get("max_sequence_length", 512),
                    )
                ),
            )
            result = pipeline(
                prompt=None,
                prompt_embeds=prompt_payloads,
                negative_prompt_embeds=negative_prompt_payloads,
                height=config["dataset"]["image_size"],
                width=config["dataset"]["image_size"],
                num_inference_steps=config["train"]["validation_num_inference_steps"],
                guidance_scale=config["train"]["validation_guidance_scale"],
                generator=sample_generator,
                max_sequence_length=int(
                    config["transformer"].get(
                        "max_sequence_length",
                        config.get("text_encoder", {}).get("max_sequence_length", 512),
                    )
                ),
            )
        else:
            result = pipeline(
                prompt=prompt,
                height=config["dataset"]["image_size"],
                width=config["dataset"]["image_size"],
                num_inference_steps=config["train"]["validation_num_inference_steps"],
                guidance_scale=config["train"]["validation_guidance_scale"],
                generator=sample_generator,
                max_sequence_length=config["text_encoder"]["max_sequence_length"],
            )
        image = result.images[0]
        per_path = sample_dir / f"sample-{idx:04d}.png"
        image.save(per_path)
        local_records.append({"sample_index": idx, "image_path": str(per_path), "prompt": str(prompt)})

    accelerator.wait_for_everyone()
    gathered = gather_object(local_records)
    flat_records = flatten_gathered_record_chunks(gathered)
    if accelerator.is_main_process:
        by_idx = {int(rec["sample_index"]): rec for rec in flat_records}
        ordered = [by_idx[k] for k in sorted(by_idx.keys())]
        images = [Image.open(rec["image_path"]).convert("RGB") for rec in ordered]
        grid_path = sample_dir / f"step-{step:08d}.png"
        save_image_grid(images, grid_path, nrow=max(1, min(2, len(images))))
        save_json(
            sample_dir / f"step-{step:08d}.json",
            {
                "step": step,
                "image_path": str(grid_path),
                "prompts": prompts,
                "num_inference_steps": config["train"]["validation_num_inference_steps"],
                "guidance_scale": config["train"]["validation_guidance_scale"],
                "seed": base_seed,
                "validation_sample_seed_stride": _VALIDATION_SAMPLE_SEED_STRIDE,
                "height": config["dataset"]["image_size"],
                "width": config["dataset"]["image_size"],
            },
        )

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_image(
                    "validation/samples",
                    make_grid(
                        [pil_to_tensor(image).float() / 255.0 for image in images],
                        nrow=max(1, min(2, len(images))),
                    ),
                    global_step=step,
                )
            elif tracker.name == "wandb":
                tracker.log_images({"validation/samples": images}, step=step)
    del pipeline
    if transformer_model is not None and was_compiled:
        transformer_model.set_forward_compilation(True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
