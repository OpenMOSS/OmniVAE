from __future__ import annotations

import gc
import itertools
import json
import os
from pathlib import Path

from omnivae_generation.trainer.runtime_env import ensure_hf_home

ensure_hf_home()

import torch
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import omnivae_generation.trainer.zimage_training as zt
from omnivae_generation.trainer.profiler import build_profiler


logger = get_logger(__name__)


def _get_loss_spike_debug_config(output_dir: Path) -> tuple[float | None, Path]:
    threshold_text = os.environ.get("OMNIGEN_DEBUG_LOSS_SPIKE_THRESHOLD", "10").strip()
    if not threshold_text:
        return None, output_dir / "debug_loss_spikes.jsonl"
    try:
        threshold = float(threshold_text)
    except ValueError:
        logger.warning("Ignoring invalid OMNIGEN_DEBUG_LOSS_SPIKE_THRESHOLD=%r", threshold_text)
        return None, output_dir / "debug_loss_spikes.jsonl"
    if threshold <= 0:
        return None, output_dir / "debug_loss_spikes.jsonl"

    debug_path_text = os.environ.get("OMNIGEN_DEBUG_LOSS_SPIKE_FILE", "debug_loss_spikes.jsonl").strip()
    debug_path = Path(debug_path_text or "debug_loss_spikes.jsonl").expanduser()
    if not debug_path.is_absolute():
        debug_path = output_dir / debug_path
    return threshold, debug_path


def _per_sample_scalar(tensor: torch.Tensor, batch_size: int) -> list[float]:
    """Reduce an arbitrary-shape per-sample tensor (e.g. ``[B]`` or ``[B,1,1]``)
    to a 1D Python list of length ``batch_size``."""
    values = tensor.detach().float()
    if values.numel() == 0:
        return []
    if values.dim() == 0:
        return [float(values.item())] * int(batch_size)
    flat = values.reshape(values.shape[0], -1).mean(dim=1)
    return [float(x) for x in flat.cpu().tolist()]


def _maybe_dump_loss_spike(
    *,
    accelerator,
    debug_threshold: float | None,
    debug_path: Path,
    global_step: int,
    averaged_loss: float,
    per_sample_loss: torch.Tensor,
    batch: dict,
    diffusion_batch,
) -> None:
    """Persist debug info for a step whose loss exceeds ``debug_threshold``.

    The trigger fires when *either* (a) the world-averaged loss exceeds the
    threshold, or (b) any single sample on any rank exceeds it. Each rank's
    payload includes a ``samples`` array zipping ``audio_path`` /
    ``image_path`` / ``image_id`` with that sample's loss, model timestep,
    sigma, weighting and an ``is_spike`` flag (``loss > threshold``), so
    downstream tooling can pinpoint the offending samples instead of seeing
    only a batch-averaged number.
    """
    if debug_threshold is None:
        return

    local_per_sample = per_sample_loss.detach().float().reshape(-1)
    local_max = float(local_per_sample.max().item()) if local_per_sample.numel() > 0 else 0.0

    if (
        int(accelerator.num_processes) > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        max_buf = torch.tensor([local_max], device=accelerator.device, dtype=torch.float32)
        torch.distributed.all_reduce(max_buf, op=torch.distributed.ReduceOp.MAX)
        global_max = float(max_buf.item())
    else:
        global_max = local_max

    triggered_avg = averaged_loss > debug_threshold
    triggered_sample = global_max > debug_threshold
    if not triggered_avg and not triggered_sample:
        return
    triggers: list[str] = []
    if triggered_sample:
        triggers.append("per_sample_max")
    if triggered_avg:
        triggers.append("averaged_loss")

    batch_size = int(diffusion_batch.batch_size)
    per_sample_list = [float(x) for x in local_per_sample.cpu().tolist()]
    timesteps = _per_sample_scalar(diffusion_batch.model_timesteps, batch_size)
    sigmas = _per_sample_scalar(diffusion_batch.sigmas, batch_size)
    weighting_per_sample = _per_sample_scalar(diffusion_batch.weighting, batch_size)

    audio_paths = list(batch.get("audio_paths", []))
    image_paths = list(batch.get("image_paths", []))
    image_ids = list(batch.get("image_ids", []))

    n_rows = max(
        len(per_sample_list),
        len(audio_paths),
        len(image_paths),
        len(image_ids),
        batch_size,
    )

    def _at(seq: list, i: int):
        return seq[i] if i < len(seq) else None

    samples: list[dict] = []
    for i in range(n_rows):
        loss_val = _at(per_sample_list, i)
        sample: dict = {
            "index": i,
            "loss": loss_val,
            "timestep": _at(timesteps, i),
            "sigma": _at(sigmas, i),
            "weighting": _at(weighting_per_sample, i),
            "is_spike": (loss_val is not None and loss_val > debug_threshold),
        }
        ap = _at(audio_paths, i)
        if ap is not None:
            sample["audio_path"] = ap
        ip = _at(image_paths, i)
        if ip is not None:
            sample["image_path"] = ip
        iid = _at(image_ids, i)
        if iid is not None:
            sample["image_id"] = iid
        samples.append(sample)

    n_spike = sum(1 for s in samples if s.get("is_spike"))

    local_payload = {
        "rank": int(accelerator.process_index),
        "global_step": int(global_step),
        "local_loss_mean": float(local_per_sample.mean().item()) if local_per_sample.numel() > 0 else 0.0,
        "local_loss_max": local_max,
        "n_spike": int(n_spike),
        # legacy mirrors so older tooling that reads flat lists still works
        "image_ids": image_ids,
        "image_paths": image_paths,
        "audio_paths": audio_paths,
        "samples": samples,
    }

    gathered_payloads: list = [local_payload]
    if (
        int(accelerator.num_processes) > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        gathered_payloads = [None for _ in range(int(accelerator.num_processes))]
        torch.distributed.all_gather_object(gathered_payloads, local_payload)

    if accelerator.is_main_process:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "global_step": int(global_step),
                        "averaged_loss": float(averaged_loss),
                        "threshold": float(debug_threshold),
                        "global_max_per_sample_loss": float(global_max),
                        "triggered_by": triggers,
                        "ranks": gathered_payloads,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main(args=None, config=None):
    if args is None:
        args = zt.parse_train_args()
    if config is None:
        config = zt.load_train_config(args)
    zt.validate_training_config(config)

    runtime = zt.create_training_runtime(config)
    accelerator = runtime.accelerator
    paths = runtime.paths

    text_setup = zt.load_text_setup(config, accelerator)
    tokenizer = text_setup.tokenizer
    text_encoder = text_setup.text_encoder
    trains_separate_text_encoder = text_encoder is not None and bool(config["train"]["train_text_encoder"])

    vae, noise_scheduler, transformer, transformer_param_count = zt.load_base_models(
        config,
        text_setup.text_hidden_size,
    )
    zt.configure_model_modes(config, text_encoder, vae, train_vae=False)

    dataset, train_dataloader = zt.build_dataset_and_dataloader(
        config,
        tokenizer,
        accelerator,
        include_raw_pixel_values=False,
    )

    named_params_to_optimize: list[tuple[str, torch.nn.Parameter]] = []
    zt.append_trainable_named_parameters(named_params_to_optimize, transformer, "transformer")
    if trains_separate_text_encoder:
        zt.append_trainable_named_parameters(named_params_to_optimize, text_encoder, "text_encoder")

    optimizer = zt.build_main_optimizer(config, accelerator, named_params_to_optimize)
    lr_scheduler = zt.build_lr_scheduler(config, accelerator, optimizer)

    if trains_separate_text_encoder:
        transformer, text_encoder, optimizer, lr_scheduler = accelerator.prepare(
            transformer,
            text_encoder,
            optimizer,
            lr_scheduler,
        )
    else:
        transformer, optimizer, lr_scheduler = accelerator.prepare(
            transformer,
            optimizer,
            lr_scheduler,
        )
        if text_encoder is not None:
            text_encoder.to(accelerator.device, dtype=text_setup.text_encoder_dtype)

    finalized = zt.finalize_models_and_schedules(
        config,
        accelerator,
        text_encoder,
        text_setup.text_encoder_dtype,
        vae,
        noise_scheduler,
        transformer,
        train_dataloader,
        train_vae=False,
    )
    text_encoder = finalized.text_encoder

    if accelerator.is_main_process:
        zt.save_run_metadata(paths.output_dir, config, transformer_param_count, text_setup.text_hidden_size)

    logger.info(
        "Transformer params: %.4fB | dataset size: %s | epochs: %s | steps: %s",
        transformer_param_count / 1e9,
        len(dataset),
        finalized.num_train_epochs,
        int(config["train"]["max_train_steps"]),
    )

    resume_state = zt.restore_training_state(
        accelerator,
        train_dataloader,
        paths.snapshot_root,
        config["train"].get("resume_from_checkpoint"),
        num_update_steps_per_epoch=finalized.num_update_steps_per_epoch,
        persistent_checkpoint_root=paths.persistent_root,
    )
    global_step = resume_state.global_step
    first_epoch = resume_state.first_epoch
    if resume_state.checkpoint_path is not None:
        logger.info("Resumed from checkpoint %s at global step %s", resume_state.checkpoint_path, global_step)
        # logger.info may be swallowed if the root logger is left at WARNING; mirror to
        # stdout so the launch log unconditionally records that resume actually fired.
        if accelerator.is_main_process:
            print(
                f"[train_zimage] Resumed from checkpoint {resume_state.checkpoint_path} "
                f"at global step {global_step}",
                flush=True,
            )
    elif accelerator.is_main_process:
        requested = config["train"].get("resume_from_checkpoint")
        if requested:
            print(
                f"[train_zimage] resume_from_checkpoint={requested!r} requested but no usable "
                f"checkpoint was found under {paths.snapshot_root} (and persistent root "
                f"{paths.persistent_root}); starting from global step 0.",
                flush=True,
            )

    profiler = build_profiler(accelerator, config, paths.output_dir)
    if profiler.enabled:
        logger.info("Torch profiler chrome traces will be written to %s", profiler.trace_dir)
    profiler.start()

    max_train_steps = int(config["train"]["max_train_steps"])
    # `initial=global_step` keeps the displayed counter aligned with the true step
    # number after a resume; without it tqdm restarts the left-side count at 0
    # while only shrinking the right-side total, which has confused users into
    # thinking the resume failed.
    progress_bar = tqdm(
        initial=global_step,
        total=max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="train",
    )
    accumulated_loss = 0.0
    accumulated_loss_steps = 0
    latest_log_payload: dict[str, float] = {}
    loss_spike_debug_threshold, loss_spike_debug_path = _get_loss_spike_debug_config(paths.output_dir)

    def _optimizer_step_transformer_impl():
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if config.get("_no_compile", False):
        optimizer_step_transformer = _optimizer_step_transformer_impl
    else:
        optimizer_step_transformer = torch.compile(_optimizer_step_transformer_impl, fullgraph=False)

    forward_transformer = zt.build_forward_transformer(
        transformer,
        finalized.transformer_model,
        finalized.train_patch_size,
        finalized.train_f_patch_size,
    )

    for epoch in range(first_epoch, finalized.num_train_epochs):
        transformer.train()
        if trains_separate_text_encoder:
            text_encoder.train()

        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        else:
            logger.warning("train_dataloader.sampler does not have set_epoch method, skipping epoch set")

        for _, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if trains_separate_text_encoder:
                models_to_accumulate.append(text_encoder)

            with accelerator.accumulate(*models_to_accumulate):
                if zt.is_audio_dataset(config):
                    audio = batch["audio"].to(accelerator.device, dtype=finalized.vae_dtype, non_blocking=True)
                    with torch.no_grad():
                        latents = zt.encode_audio_to_latents(audio, vae).to(torch.float32)
                else:
                    pixel_values = zt.prepare_pixel_values_for_vae(
                        batch["pixel_values"],
                        accelerator.device,
                        finalized.vae_dtype,
                    )
                    with torch.no_grad():
                        latents = zt.encode_images_to_latents(pixel_values, vae).to(torch.float32)

                diffusion_batch = zt.prepare_diffusion_batch(
                    config,
                    noise_scheduler,
                    latents,
                    detach_target_latents=False,
                )
                prompt_embeds, _ = zt.prepare_prompt_embeddings(
                    config,
                    batch,
                    tokenizer,
                    text_encoder,
                    accelerator,
                    train_text_encoder=trains_separate_text_encoder,
                )

                model_pred, _ = forward_transformer(
                    diffusion_batch.noisy_latents,
                    diffusion_batch.model_timesteps,
                    prompt_embeds,
                )
                model_pred_for_loss = zt.adapt_model_prediction(
                    model_pred,
                    diffusion_batch.noisy_latents,
                    diffusion_batch.sigmas,
                    finalized.predict_target,
                )
                per_sample_loss = zt.compute_per_sample_denoising_loss(
                    diffusion_batch.weighting,
                    model_pred_for_loss,
                    diffusion_batch.target,
                )
                loss = per_sample_loss.mean()
                averaged_loss = float(
                    accelerator.gather(per_sample_loss.detach()).mean().item()
                )
                _maybe_dump_loss_spike(
                    accelerator=accelerator,
                    debug_threshold=loss_spike_debug_threshold,
                    debug_path=loss_spike_debug_path,
                    global_step=global_step + 1,
                    averaged_loss=averaged_loss,
                    per_sample_loss=per_sample_loss,
                    batch=batch,
                    diffusion_batch=diffusion_batch,
                )
                accumulated_loss += averaged_loss / config["train"]["gradient_accumulation_steps"]

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder.parameters())
                        if trains_separate_text_encoder
                        else transformer.parameters()
                    )
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, config["train"]["max_grad_norm"])

                optimizer_step_transformer()
                latest_log_payload = {"train/loss": float(loss.detach().item())}

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accumulated_loss_steps += 1

                if global_step % config["train"]["log_every_steps"] == 0:
                    mean_loss = accumulated_loss / max(1, accumulated_loss_steps)
                    log_payload = dict(latest_log_payload)
                    log_payload["train/loss"] = mean_loss
                    log_payload["train/lr"] = lr_scheduler.get_last_lr()[0]
                    if grad_norm is not None:
                        log_payload["train/grad_norm"] = float(grad_norm)
                    accelerator.log(log_payload, step=global_step)
                    accumulated_loss = 0.0
                    accumulated_loss_steps = 0

                snapshot_every = int(config["train"].get("snapshot_checkpointing_steps") or 0)
                snapshots_limit = config["train"].get("snapshots_total_limit")
                if snapshot_every > 0 and global_step % snapshot_every == 0:
                    zt.save_managed_checkpoint(
                        accelerator=accelerator,
                        checkpoint_root=paths.snapshot_root,
                        checkpoint_kind="snapshot",
                        checkpoints_limit=snapshots_limit,
                        train_dataloader=train_dataloader,
                        process_index=accelerator.process_index,
                        config=config,
                        global_step=global_step,
                        transformer_param_count=transformer_param_count,
                        transformer=transformer,
                        tokenizer=tokenizer,
                        scheduler=noise_scheduler,
                        train_text_encoder=trains_separate_text_encoder,
                        text_encoder=text_encoder,
                    )

                persistent_every = int(config["train"].get("persistent_checkpointing_steps") or 0)
                persistent_limit = config["train"].get("persistent_total_limit")
                if persistent_every > 0 and global_step % persistent_every == 0:
                    zt.save_managed_checkpoint(
                        accelerator=accelerator,
                        checkpoint_root=paths.persistent_root,
                        checkpoint_kind="persistent",
                        checkpoints_limit=persistent_limit,
                        train_dataloader=train_dataloader,
                        process_index=accelerator.process_index,
                        config=config,
                        global_step=global_step,
                        transformer_param_count=transformer_param_count,
                        transformer=transformer,
                        tokenizer=tokenizer,
                        scheduler=noise_scheduler,
                        train_text_encoder=trains_separate_text_encoder,
                        text_encoder=text_encoder,
                    )

                if zt.should_run_validation(config, global_step):
                    accelerator.wait_for_everyone()
                    zt.run_validation(
                        accelerator=accelerator,
                        config=config,
                        step=global_step,
                        transformer=transformer,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        vae=vae,
                        scheduler=noise_scheduler,
                    )
                    accelerator.wait_for_everyone()

                if zt.should_run_vae_validation(config, global_step):
                    accelerator.wait_for_everyone()
                    zt.run_vae_validation(
                        accelerator=accelerator,
                        config=config,
                        step=global_step,
                        batch=batch,
                        vae=vae,
                        vae_dtype=finalized.vae_dtype,
                    )
                    accelerator.wait_for_everyone()

                if runtime.manual_gc_every_steps is not None and global_step % runtime.manual_gc_every_steps == 0:
                    gc.collect()

                progress_bar.set_postfix(loss=f"{loss.detach().float().item():.4f}")

            profiler.step()

            if global_step >= int(config["train"]["max_train_steps"]):
                break

        if global_step >= int(config["train"]["max_train_steps"]):
            break

    profiler.stop()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        zt.export_checkpoint_artifacts(
            accelerator=accelerator,
            output_dir=paths.output_dir / "final",
            transformer=transformer,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            config=config,
            global_step=global_step,
            transformer_param_count=transformer_param_count,
            train_text_encoder=trains_separate_text_encoder,
            text_encoder=text_encoder,
            checkpoint_kind=None,
        )

    zt.finish_training(runtime.manual_gc_every_steps, accelerator)


if __name__ == "__main__":
    main()
