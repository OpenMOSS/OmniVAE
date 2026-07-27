from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import gather_object
from diffusers import FlowMatchEulerDiscreteScheduler

from omnivae_generation.trainer.audio_duration import make_duration_estimator_from_validation_config
from omnivae_generation.trainer.audio_task_prefix import (
    KIND_TTA,
    apply_task_prefix,
    resolve_task_kind,
)
from omnivae_generation.trainer.audio_wer import (
    WhisperEnAsr,
    load_prompt_set,
    score_records,
    transcribe_records,
)
from omnivae_generation.trainer.data import maybe_format_chat_prompt
from omnivae_generation.trainer.modeling import configure_scheduler_prediction_target, encode_prompts
from omnivae_generation.trainer.utils import ensure_dir
from omnivae_generation.trainer.video_validation import apply_zimage_cfg


_TYPE_SLUG_RE = re.compile(r"[^0-9a-z]+")
_PROMPT_TERMINAL_PUNCT_RE = re.compile(r"""[.!?。！？]['")\]\}”’]*$""")


def _slug(label: Any) -> str:
    text = str(label).strip().lower()
    text = _TYPE_SLUG_RE.sub("_", text).strip("_")
    return text or "unknown"


def _build_validation_prompt_text(
    text: str,
    *,
    duration_seconds: float,
    duration_precision: int,
    append_duration_suffix: bool,
) -> str:
    if not append_duration_suffix:
        return str(text)
    text = str(text).rstrip()
    if text and not _PROMPT_TERMINAL_PUNCT_RE.search(text):
        text = f"{text}."
    fmt = f"{{:.{max(0, int(duration_precision))}f}}"
    return f"{text} duration: {fmt.format(float(duration_seconds))}s"


# TTA random-duration policy: half the prompts use a fixed 8-second target,
# the other half use a uniform integer pick from [3, 15]. Training keeps the
# latent/audio window fixed and pads shorter clips; validation/inference should
# therefore normally keep the latent window fixed too, use this value only in
# the ``duration: X.Xs`` suffix, then trim saved TTA wavs to the target length.
_TTA_FIXED_DURATION_SECONDS: float = 8.0
_TTA_RANDOM_MIN_SECONDS: int = 3
_TTA_RANDOM_MAX_SECONDS: int = 15
_TTA_FIXED_PROBABILITY: float = 0.5


def _pick_tta_duration_seconds(seed: int) -> float:
    """Deterministic TTA duration draw seeded by ``seed`` (per-prompt).

    50% chance to return ``_TTA_FIXED_DURATION_SECONDS`` (8.0s); otherwise a
    uniform integer from ``[_TTA_RANDOM_MIN_SECONDS, _TTA_RANDOM_MAX_SECONDS]``
    inclusive. Determinism keeps reruns at the same step reproducible while
    the seed bleed-through (``base_seed + step + ...``) keeps successive steps
    diverse.
    """
    rng = random.Random(int(seed))
    if rng.random() < _TTA_FIXED_PROBABILITY:
        return float(_TTA_FIXED_DURATION_SECONDS)
    return float(rng.randint(_TTA_RANDOM_MIN_SECONDS, _TTA_RANDOM_MAX_SECONDS))


def _resolve_prompt_sets(val_cfg: dict) -> list[dict]:
    """Resolve `validation.prompt_sets` with back-compat for `prompts_jsonl_path`.

    Returns a list of normalized set dicts:
        {name, path, format, text_field, type_field, index_field,
         wer_normalization, num_prompts, compute_wer, task_kind}
    Sets with empty `path` are filtered out.

    ``compute_wer`` (default True) lets a set opt out of Whisper transcription
    + WER scoring; useful for environmental-audio (TTA) prompts where ASR
    output is meaningless.

    ``task_kind`` (optional, default ``None`` -> auto-resolve from per-entry
    ``type``) tags a whole prompt set as TTS / TTA / legacy so the task-prefix
    template pool can be picked correctly. Required when the per-entry
    ``type`` field carries non-task labels (``basetts_valid`` uses
    ``"Questions"``/``"Statements"``; the metalst loader hardcodes
    ``"all"``).
    """
    raw_sets = val_cfg.get("prompt_sets")
    if raw_sets:
        items = list(raw_sets)
    else:
        legacy_path = val_cfg.get("prompts_jsonl_path")
        if not legacy_path:
            return []
        items = [
            {
                "name": "default",
                "path": legacy_path,
                "format": "jsonl",
                "text_field": val_cfg.get("prompt_text_field", "text"),
                "type_field": val_cfg.get("type_field", "type"),
                "index_field": val_cfg.get("index_field", "index"),
                "wer_normalization": "simple",
            }
        ]

    resolved: list[dict] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path") or "").strip()
        if not path:
            continue
        task_kind_raw = raw.get("task_kind")
        task_kind = (
            str(task_kind_raw).strip().lower()
            if task_kind_raw is not None and str(task_kind_raw).strip()
            else None
        )
        resolved.append(
            {
                "name": str(raw.get("name") or Path(path).stem or "set"),
                "path": path,
                "format": str(raw.get("format") or "").strip().lower(),
                "text_field": str(raw.get("text_field", val_cfg.get("prompt_text_field", "text"))),
                "type_field": str(raw.get("type_field", val_cfg.get("type_field", "type"))),
                "index_field": str(raw.get("index_field", val_cfg.get("index_field", "index"))),
                "wer_normalization": str(raw.get("wer_normalization", "simple")).strip().lower(),
                "num_prompts": raw.get("num_prompts", val_cfg.get("num_prompts")),
                "compute_wer": bool(raw.get("compute_wer", True)),
                "task_kind": task_kind,
            }
        )
    return resolved


def _build_inference_scheduler(
    config: dict,
    scheduler,
    transformer_model,
    device: torch.device,
    num_inference_steps: int,
):
    inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    configure_scheduler_prediction_target(
        inference_scheduler,
        getattr(transformer_model, "_laion_predict_target", config["transformer"].get("predict_target", "v")),
    )
    inference_scheduler.set_timesteps(int(num_inference_steps), device=device)
    return inference_scheduler


def _audio_vae_hop_length(vae_model, audio_vae_cfg: dict) -> int:
    hop_length = getattr(vae_model, "hop_length", None)
    if hop_length is None:
        hop_length = audio_vae_cfg.get("hop_length")
    if hop_length is None:
        raise ValueError("Cannot resolve audio VAE hop_length; configure audio_vae.hop_length.")
    return int(hop_length)


def _index_tag(index_label: Any) -> str:
    if isinstance(index_label, (int, float)):
        return f"{int(index_label):02d}"
    return str(index_label)


def _is_legacy_single_set(prompt_sets: list[dict]) -> bool:
    """Legacy path layout: keep no <set_name>/ layer when caller used the old
    `prompts_jsonl_path` field (single resolved set named 'default')."""
    return len(prompt_sets) == 1 and prompt_sets[0].get("name") == "default"


def _log_audio_samples_to_trackers(
    accelerator: Accelerator,
    grouped_records: dict[str, dict[str, list[dict]]],
    *,
    sample_rate: int,
    step: int,
    set_prefix: bool,
) -> None:
    """Emit audio samples (and optional WER scalars) to trackers.

    grouped_records maps set_name -> {type_label -> [record, ...]}.
    """
    if not grouped_records:
        return

    def _audio_key(set_name: str, type_slug: str, index_tag: str) -> str:
        if set_prefix:
            return f"validation/audio/{_slug(set_name)}/{type_slug}/{index_tag}"
        return f"validation/audio/{type_slug}/{index_tag}"

    def _text_key(set_name: str, type_slug: str, index_tag: str) -> str:
        if set_prefix:
            return f"validation/text/{_slug(set_name)}/{type_slug}/{index_tag}"
        return f"validation/text/{type_slug}/{index_tag}"

    def _wandb_key(set_name: str, type_slug: str) -> str:
        if set_prefix:
            return f"validation/audio/{_slug(set_name)}/{type_slug}"
        return f"validation/audio/{type_slug}"

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for set_name, by_type in grouped_records.items():
                for type_label, records in by_type.items():
                    type_slug = _slug(type_label)
                    for record in records:
                        index_tag = _index_tag(record["index"])
                        wave = torch.from_numpy(record["wave"]).unsqueeze(0)
                        tracker.writer.add_audio(
                            _audio_key(set_name, type_slug, index_tag),
                            wave,
                            global_step=step,
                            sample_rate=int(sample_rate),
                        )
                        caption = f"[{set_name}/{type_label}] #{record['index']} | {record['text']}"
                        tracker.writer.add_text(
                            _text_key(set_name, type_slug, index_tag),
                            caption,
                            global_step=step,
                        )
        elif tracker.name == "wandb":
            import wandb

            payload: dict[str, list] = {}
            for set_name, by_type in grouped_records.items():
                for type_label, records in by_type.items():
                    type_slug = _slug(type_label)
                    payload[_wandb_key(set_name, type_slug)] = [
                        wandb.Audio(
                            record["wave"],
                            sample_rate=int(sample_rate),
                            caption=f"[{set_name}/{type_label}] #{record['index']} | {record['text']}",
                        )
                        for record in records
                    ]
            if payload:
                tracker.log(payload, step=step)


def _log_wer_to_trackers(
    accelerator: Accelerator,
    wer_results: dict[str, dict],
    *,
    step: int,
    set_prefix: bool,
    max_log_per_set: int,
) -> None:
    if not wer_results:
        return

    def _scalar_key(set_name: str, suffix: str) -> str:
        if set_prefix:
            return f"validation/wer/{_slug(set_name)}/{suffix}"
        return f"validation/wer/{suffix}"

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for set_name, result in wer_results.items():
                summary = result["summary"]
                tracker.writer.add_scalar(
                    _scalar_key(set_name, "mean"), float(summary["mean_wer"]), global_step=step
                )
                tracker.writer.add_scalar(
                    _scalar_key(set_name, "mean_below_50"),
                    float(summary["mean_wer_below_50"]),
                    global_step=step,
                )
                tracker.writer.add_scalar(
                    _scalar_key(set_name, "n_above_50"),
                    int(summary["n_above_50"]),
                    global_step=step,
                )
                for type_label, mean_wer in summary.get("per_type_mean_wer", {}).items():
                    tracker.writer.add_scalar(
                        _scalar_key(set_name, f"{_slug(type_label)}/mean"),
                        float(mean_wer),
                        global_step=step,
                    )
        elif tracker.name == "wandb":
            import wandb

            payload: dict[str, Any] = {}
            for set_name, result in wer_results.items():
                summary = result["summary"]
                payload[_scalar_key(set_name, "mean")] = float(summary["mean_wer"])
                payload[_scalar_key(set_name, "mean_below_50")] = float(summary["mean_wer_below_50"])
                payload[_scalar_key(set_name, "n_above_50")] = int(summary["n_above_50"])
                for type_label, mean_wer in summary.get("per_type_mean_wer", {}).items():
                    payload[_scalar_key(set_name, f"{_slug(type_label)}/mean")] = float(mean_wer)

                if max_log_per_set > 0 and result["per_record"]:
                    table = wandb.Table(
                        columns=["type", "index", "wer", "ref_norm", "hyp_norm", "text", "hyp"]
                    )
                    for record in result["per_record"][:max_log_per_set]:
                        table.add_data(
                            str(record.get("type")),
                            str(record.get("index")),
                            float(record.get("wer", 0.0)),
                            str(record.get("ref_norm", "")),
                            str(record.get("hyp_norm", "")),
                            str(record.get("text", "")),
                            str(record.get("hyp", "")),
                        )
                    table_key = (
                        f"validation/wer_table/{_slug(set_name)}" if set_prefix else "validation/wer_table"
                    )
                    payload[table_key] = table
            if payload:
                tracker.log(payload, step=step)


@torch.no_grad()
def _generate_one_set(
    *,
    accelerator: Accelerator,
    config: dict,
    set_cfg: dict,
    transformer_model,
    text_encoder_model,
    vae_model,
    scheduler,
    forward_transformer,
    negative_prompt_embeds,
    tokenizer,
    base_seed: int | None,
    step: int,
    seed_offset: int,
    duration_seconds: float,
    duration_precision: int,
    append_duration_suffix: bool,
    num_inference_steps: int,
    guidance_scale: float,
    cfg_normalization: bool,
    in_channels: int,
    t_latent: int,
    max_seq_len: int,
    cache_enabled: bool,
    duration_seconds_for_text: Callable[[str], float] | None = None,
    tta_prompt_duration_seconds: float | None = None,
    tta_random_duration: bool = False,
    resize_latent_for_tta: bool = False,
    sample_rate: int | None = None,
    hop_length: int | None = None,
    task_prefix_enabled: bool = False,
    progress_desc: str | None = None,
) -> list[dict]:
    """Run text->latent->wav for the prompts in one set on the local rank.

    Returns the local rank's list of records; collation/gathering happens in
    the caller so we can hold all sets in memory consistently.

    ``duration_seconds_for_text`` is an optional callable used **only** to fill
    the ``duration: X.Xs`` suffix in the text prompt (i.e. the value the model
    sees as a soft length hint). It does NOT change ``t_latent`` — the latent
    shape stays at ``(1, in_channels, t_latent)`` so we keep the unconditional
    branch + scheduler trajectory aligned with training. When ``None`` (the
    default and what training-time validation passes), every prompt uses
    ``duration_seconds`` exactly as before, so behavior is unchanged.

    ``tta_random_duration`` (default False) is the TTA-only override: for
    prompts whose ``type`` is ``"tta"``, draw a per-prompt target duration
    from :func:`_pick_tta_duration_seconds` (50% 8s / 50% int in [3,15]) and
    rewrite the prompt's ``duration: X.Xs`` suffix to that length.

    ``tta_prompt_duration_seconds`` (default None) overrides the above random
    policy for TTA prompt sets: keep the latent window fixed, write this
    duration into the prompt suffix, then trim saved TTA wavs to this duration.

    ``resize_latent_for_tta`` (default False) controls whether the random
    duration *also* resizes the latent ``t_latent`` so the generated wave
    actually matches the suffix. Default is OFF so inference matches training
    (training pads short clips out to the full ``num_audio_samples`` window,
    so the latent shape is fixed and the model learns to emit silence after
    the prompted ``X.Xs`` mark). Set to True to force the latent to shrink
    to the prompted length; requires ``sample_rate`` and ``hop_length`` to
    recompute the latent shape, and degrades to "suffix only" silently when
    either is missing.

    ``sample_rate`` / ``hop_length`` are optional VAE descriptors used only
    when ``resize_latent_for_tta`` is enabled (see above). External callers
    that don't opt in can keep ignoring them.

    ``task_prefix_enabled`` (default False) wraps each prompt with a randomly
    picked instruction-style template from
    :mod:`omnivae_generation.trainer.audio_task_prefix` based on the resolved task kind (TTS
    vs TTA, see :func:`resolve_task_kind`). The pick is seeded from
    ``base_seed + step + seed_offset + global_idx + 31337`` so reruns at the
    same step are reproducible. When False (the default and what external
    inference scripts pass unless they opt in) prompts are left as-is, so
    behavior is unchanged.

    ``progress_desc`` (default None) optionally enables a tqdm progress bar
    over ``local_entries`` on the **main process only** (so multi-rank runs
    don't print overlapping bars). Training-time validation does not pass
    this argument, so behavior is unchanged.
    """
    entries = load_prompt_set(set_cfg)
    if not entries:
        return []

    rank = int(accelerator.process_index)
    world_size = max(1, int(accelerator.num_processes))
    local_entries = entries[rank::world_size]

    iterable = local_entries
    if progress_desc and accelerator.is_main_process:
        try:
            from tqdm.auto import tqdm

            iterable = tqdm(
                local_entries,
                desc=progress_desc,
                leave=False,
                dynamic_ncols=True,
            )
        except ImportError:
            iterable = local_entries

    local_results: list[dict] = []
    can_resize_latent = (
        bool(tta_random_duration)
        and bool(resize_latent_for_tta)
        and sample_rate is not None
        and hop_length is not None
        and int(sample_rate) > 0
        and int(hop_length) > 0
    )
    for entry in iterable:
        global_idx = int(entry["global_idx"])
        type_label = entry["type"]
        index_label = entry["index"]
        text = entry["text"]
        # Single source of truth for which task pool we belong to: prefer the
        # set-level ``task_kind`` (yaml), fall back to per-entry ``type ==
        # "tta"``, else default to TTS. Used by both TTA random-duration and
        # task-prefix template picks below.
        resolved_kind = resolve_task_kind(set_cfg, entry)
        kind_is_tta = resolved_kind == KIND_TTA

        # Per-prompt latent length defaults to the set-level ``t_latent``;
        # only the TTA random-duration branch may shorten/lengthen it below.
        effective_t_latent = int(t_latent)

        if kind_is_tta and tta_prompt_duration_seconds is not None:
            target_duration_seconds = min(float(tta_prompt_duration_seconds), float(duration_seconds))
        elif tta_random_duration and kind_is_tta:
            # Seed offset (+9173) keeps the duration RNG decoupled from the
            # noise RNG so two prompts sharing a seed don't lock-step their
            # length to their noise pattern.
            duration_seed_base = (
                int(base_seed) + int(step) + int(seed_offset) + global_idx + 9173
                if base_seed is not None
                else int(step) * 1000003 + global_idx + 9173
            )
            target_duration_seconds = min(
                _pick_tta_duration_seconds(duration_seed_base),
                float(duration_seconds),
            )
            if can_resize_latent:
                effective_t_latent = max(
                    1,
                    int(round(target_duration_seconds * float(sample_rate) / float(hop_length))),
                )
        elif duration_seconds_for_text is not None:
            target_duration_seconds = float(duration_seconds_for_text(text))
        else:
            target_duration_seconds = float(duration_seconds)

        prompt_text = str(text)
        if task_prefix_enabled:
            # Independent RNG from duration / noise (different additive
            # offset). Seeded so reruns at the same step are reproducible.
            template_seed = (
                int(base_seed) + int(step) + int(seed_offset) + global_idx + 31337
                if base_seed is not None
                else int(step) * 6151 + global_idx + 31337
            )
            template_rng = random.Random(template_seed)
            prompt_text = apply_task_prefix(resolved_kind, prompt_text, rng=template_rng)

        prompt_with_suffix = _build_validation_prompt_text(
            prompt_text,
            duration_seconds=target_duration_seconds,
            duration_precision=duration_precision,
            append_duration_suffix=append_duration_suffix,
        )
        formatted_prompt = maybe_format_chat_prompt(prompt_with_suffix, tokenizer)
        prompt_embeds = encode_prompts(
            [formatted_prompt],
            tokenizer,
            text_encoder_model,
            accelerator.device,
            max_seq_len,
            cache_enabled=cache_enabled,
        )

        seed = (
            None
            if base_seed is None
            else int(base_seed) + int(step) + seed_offset + global_idx
        )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(seed)

        inference_scheduler = _build_inference_scheduler(
            config,
            scheduler,
            transformer_model,
            accelerator.device,
            num_inference_steps,
        )

        latents = torch.randn(
            (1, in_channels, effective_t_latent),
            generator=generator,
            device=accelerator.device,
            dtype=torch.float32,
        )

        for timestep_value in inference_scheduler.timesteps:
            timestep = timestep_value.expand(latents.shape[0])
            model_timesteps = (
                float(inference_scheduler.config.num_train_timesteps) - timestep
            ) / float(inference_scheduler.config.num_train_timesteps)
            model_timesteps = model_timesteps.to(device=accelerator.device, dtype=torch.float32)

            latent_model_input = latents.repeat(2, 1, 1)
            prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
            timestep_model_input = model_timesteps.repeat(2)

            model_pred, _ = forward_transformer(
                latent_model_input.to(dtype=getattr(transformer_model, "dtype", latents.dtype)),
                timestep_model_input,
                prompt_embeds_model_input,
            )
            pos_pred = model_pred[:1].float()
            neg_pred = model_pred[1:].float()
            cfg_pred = apply_zimage_cfg(pos_pred, neg_pred, guidance_scale, cfg_normalization)
            cfg_pred = -cfg_pred
            latents = inference_scheduler.step(
                cfg_pred.to(torch.float32),
                timestep_value,
                latents,
                return_dict=False,
            )[0]
            latents = latents.to(torch.float32)

        val_cfg = config.get("validation") or {}
        offloaded_for_decode = False
        if (
            bool(val_cfg.get("offload_text_and_transformer_before_audio_decode", False))
            and accelerator.device.type == "cuda"
        ):
            transformer_model.to("cpu")
            text_encoder_model.to("cpu")
            offloaded_for_decode = True
            torch.cuda.empty_cache()

        audio = vae_model.decode(
            latents.to(dtype=getattr(vae_model, "dtype", latents.dtype))
        )
        if offloaded_for_decode:
            transformer_model.to(accelerator.device)
            text_encoder_model.to(accelerator.device)
        wave = audio[0, 0].detach().float().clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        generated_duration_seconds = (
            float(wave.shape[0]) / float(sample_rate)
            if sample_rate is not None and int(sample_rate) > 0
            else None
        )

        local_results.append(
            {
                "global_idx": global_idx,
                "type": str(type_label),
                "index": index_label,
                "text": str(text),
                "task_kind": resolved_kind,
                "prompt_with_suffix": prompt_with_suffix,
                "wave": wave,
                "seed": seed,
                "target_duration_seconds": target_duration_seconds,
                "generated_duration_seconds": generated_duration_seconds,
            }
        )

        del prompt_embeds, latents, model_pred, audio, cfg_pred, pos_pred, neg_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return local_results


def _persist_set_outputs(
    *,
    set_cfg: dict,
    sorted_results: list[dict],
    sample_dir: Path,
    use_set_layer: bool,
    sample_rate: int,
    duration_seconds: float,
    num_inference_steps: int,
    guidance_scale: float,
    num_log_per_type: int,
) -> tuple[Path, dict[str, list[dict]]]:
    """Write wavs + meta.jsonl for one set; return the set_dir and the
    per-type slice that should also be surfaced through trackers."""
    set_name = str(set_cfg["name"])
    if use_set_layer:
        set_dir = ensure_dir(sample_dir / _slug(set_name))
    else:
        set_dir = sample_dir
    meta_path = set_dir / "meta.jsonl"

    type_order: list[str] = []
    type_records: dict[str, list[dict]] = {}
    for record in sorted_results:
        type_label = record["type"]
        type_records.setdefault(type_label, []).append(record)
        if type_label not in type_order:
            type_order.append(type_label)

    grouped_for_tracker: dict[str, list[dict]] = {}
    with meta_path.open("w", encoding="utf-8") as meta_f:
        for type_label in type_order:
            type_dir = ensure_dir(set_dir / _slug(type_label))
            for record in type_records[type_label]:
                wav_path = type_dir / f"{_index_tag(record['index'])}.wav"
                wave_np = record["wave"]
                if str(record.get("task_kind") or "").lower() == KIND_TTA:
                    target = record.get("target_duration_seconds")
                    if target is not None and int(sample_rate) > 0:
                        target_samples = max(1, int(round(float(target) * int(sample_rate))))
                        if target_samples < int(wave_np.shape[-1]):
                            wave_np = wave_np[:target_samples]
                wave_t = torch.from_numpy(wave_np).unsqueeze(0)
                torchaudio.save(str(wav_path), wave_t, sample_rate=int(sample_rate))
                record["wav_path"] = str(wav_path)
                saved_duration_seconds = float(wave_np.shape[-1]) / float(sample_rate)

                meta_f.write(
                    json.dumps(
                        {
                            "set": set_name,
                            "global_idx": record["global_idx"],
                            "type": type_label,
                            "index": record["index"],
                            "text": record["text"],
                            "task_kind": record.get("task_kind"),
                            "prompt_with_suffix": record["prompt_with_suffix"],
                            "wav_path": str(wav_path),
                            "duration_seconds": duration_seconds,
                            "target_duration_seconds": float(
                                record.get("target_duration_seconds", duration_seconds)
                            ),
                            "generated_duration_seconds": record.get(
                                "generated_duration_seconds"
                            ),
                            "saved_duration_seconds": saved_duration_seconds,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "seed": record["seed"],
                        }
                    )
                    + "\n"
                )

            if num_log_per_type > 0:
                grouped_for_tracker[type_label] = type_records[type_label][:num_log_per_type]

    return set_dir, grouped_for_tracker


def _load_asr_if_enabled(
    val_cfg: dict,
    accelerator: Accelerator,
) -> WhisperEnAsr | None:
    """Construct a (lazy) Whisper ASR on every rank when validation.wer is on.

    Returns ``None`` when WER is disabled or ``model_path`` is empty (we still
    log a warning on main rank in the second case to mirror the previous
    behavior). Lazy-loaded: ``from_pretrained`` only fires on the first
    ``transcribe`` call from each rank.
    """
    wer_cfg = val_cfg.get("wer") or {}
    if not bool(wer_cfg.get("enabled", False)):
        return None

    model_path = str(wer_cfg.get("model_path") or "").strip()
    if not model_path:
        if accelerator.is_main_process:
            from accelerate.logging import get_logger

            get_logger(__name__).warning(
                "validation.wer.enabled=True but model_path is empty; skipping WER."
            )
        return None

    language = str(wer_cfg.get("language", "en")).strip().lower()
    if language != "en":
        raise NotImplementedError(
            f"validation.wer currently only supports language='en' (got {language!r}); "
            "extend trainer/audio_wer.py for zh."
        )

    return WhisperEnAsr(
        model_path=model_path,
        device=accelerator.device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        target_sample_rate=int(wer_cfg.get("target_sample_rate", 16000)),
        local_files_only=bool(wer_cfg.get("local_files_only", False)),
    )


def _score_and_persist_wer_for_set(
    *,
    set_cfg: dict,
    sorted_records: list[dict],
    set_dir: Path,
) -> dict[str, Any] | None:
    """Run jiwer scoring + write per-set wer.jsonl / wer_summary.json.

    ``sorted_records`` must already carry a ``hyp`` field for each entry
    (populated by ``transcribe_records`` on the rank that owns the wave, then
    delivered through ``gather_object``). Returns the ``{per_record, summary}``
    dict so the caller can collect tracker payloads, or ``None`` for an empty
    set.
    """
    if not sorted_records:
        return None
    mode = str(set_cfg.get("wer_normalization", "simple")).lower()
    result = score_records(sorted_records, mode=mode)

    wer_jsonl = set_dir / "wer.jsonl"
    with wer_jsonl.open("w", encoding="utf-8") as fh:
        for record in result["per_record"]:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (set_dir / "wer_summary.json").write_text(
        json.dumps(
            {
                "set": set_cfg["name"],
                "wer_normalization": mode,
                **result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


@torch.no_grad()
def run_audio_validation(
    accelerator: Accelerator,
    config: dict,
    step: int,
    transformer,
    tokenizer,
    text_encoder,
    vae,
    scheduler,
) -> None:
    if text_encoder is None:
        raise NotImplementedError(
            "Audio validation v1 requires a separate text encoder; qwen3_vl_dit is not supported."
        )
    val_cfg = config.get("validation") or {}
    prompt_sets = _resolve_prompt_sets(val_cfg)
    if not prompt_sets:
        return

    use_set_layer = not _is_legacy_single_set(prompt_sets)
    set_prefix = use_set_layer

    dataset_cfg = config.get("dataset", {})
    sample_rate = int(dataset_cfg.get("sample_rate", 48000))
    duration_precision = int(dataset_cfg.get("duration_precision", 1))
    append_duration_suffix = bool(dataset_cfg.get("append_duration_suffix", True))

    duration_seconds = float(val_cfg.get("duration_seconds", 30.0))
    num_inference_steps = int(val_cfg.get("num_inference_steps", 25))
    guidance_scale = float(val_cfg.get("guidance_scale", 5.0))
    cfg_normalization = val_cfg.get("cfg_normalization", False)
    seed_offset = int(val_cfg.get("seed_offset", 7))
    base_seed = config["train"].get("seed")
    # TTA validation: 50% prob 8s, otherwise integer in [3,15]. Default ON for
    # TTA prompts; opt out per-run with ``validation.tta_random_duration: false``.
    tta_random_duration = bool(val_cfg.get("tta_random_duration", True))
    raw_tta_prompt_duration = val_cfg.get("tta_prompt_duration_seconds")
    tta_prompt_duration_seconds = (
        None if raw_tta_prompt_duration is None else float(raw_tta_prompt_duration)
    )
    # Whether the TTA random draw also resizes the latent. Default OFF so
    # inference matches training (training pads to a fixed 30s window, so the
    # latent shape never shrinks — only the ``duration: X.Xs`` text suffix
    # varies). Set ``validation.tta_random_duration_resize_latent: true`` to
    # opt back into the older behavior that physically shortens the latent to
    # the prompted length.
    resize_latent_for_tta = bool(val_cfg.get("tta_random_duration_resize_latent", False))
    # Task-prefix wrapping mirrors training by default: read
    # ``dataset.task_prefix_enabled`` (default True) so validation uses the
    # same modality signal the model was trained with. Override per-run with
    # ``validation.task_prefix_enabled``.
    task_prefix_enabled = bool(
        val_cfg.get(
            "task_prefix_enabled",
            dataset_cfg.get("task_prefix_enabled", True),
        )
    )

    # Per-prompt duration estimator. Defaults to F5-TTS-style byte-based
    # estimation (matches the offline inference sweep) so that the
    # ``duration: X.Xs`` suffix tracks each prompt's actual text length
    # instead of always claiming ``duration_seconds``. Set
    # ``validation.prompt_duration.strategy: fixed`` in the YAML to recover
    # the legacy behavior (every prompt gets ``duration_seconds`` verbatim).
    duration_estimator, prompt_duration_summary = (
        make_duration_estimator_from_validation_config(
            val_cfg,
            fixed_duration=duration_seconds,
            default_strategy="f5",
        )
    )
    if accelerator.is_main_process:
        block = prompt_duration_summary
        strat = block["strategy"]
        if strat == "fixed":
            extra = f"-> {block['latent_duration_seconds']:.2f}s for every prompt"
        elif strat == "auto":
            auto = block["auto"]
            extra = (
                f"words_per_second={auto['words_per_second']}, "
                f"margin={auto['margin_seconds']}s, "
                f"clamp=[{block['clamp_min_seconds']}, {block['clamp_max_seconds']}]s"
            )
        else:  # 'f5'
            f5 = block["f5"]
            extra = (
                f"bytes_per_second={f5['bytes_per_second']}, "
                f"v={f5['local_speed']} "
                f"(short<{f5['short_text_threshold_bytes']}B -> {f5['short_text_local_speed']}), "
                f"clamp=[{block['clamp_min_seconds']}, {block['clamp_max_seconds']}]s"
            )
        print(f"[validation] prompt_duration.strategy={strat}  ({extra})")
        if tta_random_duration:
            mode = "resize_latent" if resize_latent_for_tta else "suffix_only"
            print(
                f"[validation] tta_random_duration=on  "
                f"(p={_TTA_FIXED_PROBABILITY:.2f}@{_TTA_FIXED_DURATION_SECONDS:.0f}s, "
                f"else int[{_TTA_RANDOM_MIN_SECONDS},{_TTA_RANDOM_MAX_SECONDS}]s; "
                f"mode={mode})"
            )
        if tta_prompt_duration_seconds is not None:
            print(
                f"[validation] tta_prompt_duration_seconds={tta_prompt_duration_seconds:.2f}s "
                "(fixed TTA prompt suffix; saved TTA wavs are trimmed to this duration)"
            )
        print(
            f"[validation] task_prefix_enabled={'on' if task_prefix_enabled else 'off'} "
            "(per-prompt instruction template, deterministic seed)"
        )

    transformer_model = accelerator.unwrap_model(transformer, keep_torch_compile=False)
    text_encoder_model = accelerator.unwrap_model(text_encoder, keep_torch_compile=False)
    vae_model = accelerator.unwrap_model(vae, keep_torch_compile=False)

    hop_length = _audio_vae_hop_length(vae_model, config.get("audio_vae", {}))
    t_latent = max(1, int(round(duration_seconds * sample_rate / hop_length)))

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

    asr: WhisperEnAsr | None = None
    try:
        from omnivae_generation.trainer.forward_transformer import build_forward_transformer

        train_patch_size = int(config["transformer"]["all_patch_size"][0])
        train_f_patch_size = int(config["transformer"]["all_f_patch_size"][0])
        forward_transformer = build_forward_transformer(
            transformer_model,
            transformer_model,
            train_patch_size=train_patch_size,
            train_f_patch_size=train_f_patch_size,
        )

        max_seq_len = int(config["text_encoder"]["max_sequence_length"])
        cache_enabled = bool(config["text_encoder"].get("cache_enabled", False))
        in_channels = int(config["transformer"]["in_channels"])
        empty_prompt_text = maybe_format_chat_prompt("", tokenizer)
        negative_prompt_embeds = encode_prompts(
            [empty_prompt_text],
            tokenizer,
            text_encoder_model,
            accelerator.device,
            max_seq_len,
            cache_enabled=cache_enabled,
        )

        run_name = (
            str(config.get("wandb", {}).get("run_name") or "")
            or str(config.get("experiment", {}).get("name") or "")
            or "default"
        )
        run_slug = _slug(run_name)
        sample_dir = ensure_dir(
            Path(config["experiment"]["output_dir"])
            / "samples"
            / run_slug
            / f"step-{step:08d}"
        )

        # All ranks construct the ASR (lazy: real from_pretrained fires on first
        # transcribe). When wer is disabled / model_path empty this returns None
        # and the per-rank transcribe step is skipped.
        asr = _load_asr_if_enabled(val_cfg, accelerator)

        wer_results: dict[str, dict] = {}
        grouped_for_tracker: dict[str, dict[str, list[dict]]] = {}
        num_log_per_type = max(0, int(val_cfg.get("num_log_per_type", 2)))

        for set_cfg in prompt_sets:
            local_results = _generate_one_set(
                accelerator=accelerator,
                config=config,
                set_cfg=set_cfg,
                transformer_model=transformer_model,
                text_encoder_model=text_encoder_model,
                vae_model=vae_model,
                scheduler=scheduler,
                forward_transformer=forward_transformer,
                negative_prompt_embeds=negative_prompt_embeds,
                tokenizer=tokenizer,
                base_seed=base_seed,
                step=step,
                seed_offset=seed_offset,
                duration_seconds=duration_seconds,
                duration_precision=duration_precision,
                append_duration_suffix=append_duration_suffix,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                cfg_normalization=cfg_normalization,
                in_channels=in_channels,
                t_latent=t_latent,
                max_seq_len=max_seq_len,
                cache_enabled=cache_enabled,
                duration_seconds_for_text=duration_estimator,
                tta_prompt_duration_seconds=tta_prompt_duration_seconds,
                tta_random_duration=tta_random_duration,
                resize_latent_for_tta=resize_latent_for_tta,
                sample_rate=sample_rate,
                hop_length=hop_length,
                task_prefix_enabled=task_prefix_enabled,
            )

            # Per-rank ASR: every rank transcribes its own slice on its own GPU
            # before we hit the gather barrier. Wave never leaves its owner.
            # Per-set `compute_wer=false` (e.g. TTA environmental audio) skips
            # Whisper to avoid noisy WER curves.
            set_compute_wer = bool(set_cfg.get("compute_wer", True))
            if asr is not None and set_compute_wer and local_results:
                transcribe_records(local_results, asr=asr, sample_rate=sample_rate)

            accelerator.wait_for_everyone()
            gathered = gather_object(local_results)

            if accelerator.is_main_process:
                by_idx: dict[int, dict] = {}
                for item in gathered:
                    by_idx[int(item["global_idx"])] = item
                sorted_results = [by_idx[key] for key in sorted(by_idx.keys())]

                set_dir, set_grouped = _persist_set_outputs(
                    set_cfg=set_cfg,
                    sorted_results=sorted_results,
                    sample_dir=sample_dir,
                    use_set_layer=use_set_layer,
                    sample_rate=sample_rate,
                    duration_seconds=duration_seconds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_log_per_type=num_log_per_type,
                )
                if set_grouped:
                    grouped_for_tracker[str(set_cfg["name"])] = set_grouped

                if asr is not None and set_compute_wer:
                    set_result = _score_and_persist_wer_for_set(
                        set_cfg=set_cfg,
                        sorted_records=sorted_results,
                        set_dir=set_dir,
                    )
                    if set_result is not None:
                        wer_results[str(set_cfg["name"])] = set_result

        if accelerator.is_main_process:
            _log_audio_samples_to_trackers(
                accelerator,
                grouped_for_tracker,
                sample_rate=sample_rate,
                step=step,
                set_prefix=set_prefix,
            )

            if wer_results:
                max_log_per_set = int((val_cfg.get("wer") or {}).get("max_log_per_set", 20))
                _log_wer_to_trackers(
                    accelerator,
                    wer_results,
                    step=step,
                    set_prefix=set_prefix,
                    max_log_per_set=max_log_per_set,
                )
    finally:
        # Every rank that lazy-loaded Whisper must drop it before training
        # resumes; otherwise ~1.6GB of fp16 weights stay resident across steps.
        if asr is not None:
            asr.unload()
        if was_compiled:
            transformer_model.set_forward_compilation(True)
        transformer_model.train(transformer_was_training)
        text_encoder_model.train(text_encoder_was_training)
        vae_model.train(vae_was_training)
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
