"""PE-AV large text/video/audio alignment metrics.

Uses facebook/pe-av-large to score the generated sample against its caption:

* PE-TV  -- dot(video_embeds, text_video_embeds)
* PE-TA  -- dot(audio_embeds, text_audio_embeds)
* PE-TAV -- dot(audio_video_embeds, text_audio_video_embeds)
* PE-TV-cosine  -- cosine(video_embeds, text_video_embeds)
* PE-TA-cosine  -- cosine(audio_embeds, text_audio_embeds)
* PE-TAV-cosine -- cosine(audio_video_embeds, text_audio_video_embeds)

These are paired per-sample scores, i.e. the diagonal of the retrieval
similarity matrix described in the PE-AV model card.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from my_eval.utils.audio_video import load_video_rgb_array, load_wav_mono
from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PE_AV_MODEL_DIR = Path(os.environ.get("MY_EVAL_PE_AV_MODEL_DIR", ""))

_MODEL_CACHE: Dict[str, tuple[torch.nn.Module, Any]] = {}


def _resolve_model_dir() -> Path:
    raw = (
        os.environ.get("MY_EVAL_PE_AV_MODEL_DIR")
        or os.environ.get("PE_AV_MODEL_DIR")
        or str(DEFAULT_PE_AV_MODEL_DIR)
    )
    return Path(raw).expanduser()


def _resolve_device(local_rank: int) -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device


def _resolve_dtype(device: torch.device) -> torch.dtype:
    raw = os.environ.get("MY_EVAL_PE_AV_DTYPE", "bf16").strip().lower()
    if device.type == "cpu" or raw in {"fp32", "float32"}:
        return torch.float32
    if raw in {"fp16", "float16"}:
        return torch.float16
    return torch.bfloat16


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _load_model(model_dir: Path, device: torch.device, dtype: torch.dtype) -> tuple[torch.nn.Module, Any]:
    from transformers.models.pe_audio_video import PeAudioVideoModel, PeAudioVideoProcessor

    print(f"[pe_av] loading PE-AV from {model_dir} on {device} dtype={dtype}", flush=True)
    processor = PeAudioVideoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    try:
        model = PeAudioVideoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=dtype,
        )
    except ImportError as exc:
        raise RuntimeError(
            "pe-av-large needs a recent timm version. Run: python -m pip install -U timm"
        ) from exc
    model = model.to(device).eval()
    _cuda_sync(device)
    return model, processor


def _get_model(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    reuse_models: bool,
) -> tuple[torch.nn.Module, Any, float]:
    key = f"{model_dir.resolve()}|{device}|{dtype}"
    if reuse_models and key in _MODEL_CACHE:
        model, processor = _MODEL_CACHE[key]
        return model, processor, 0.0
    started_at = time.time()
    model, processor = _load_model(model_dir, device, dtype)
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = (model, processor)
    return model, processor, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: List[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    device = _resolve_device(local_rank)
    dtype = _resolve_dtype(device)
    _, _, elapsed = _get_model(_resolve_model_dir(), device, dtype, reuse_models=True)
    log(rank, f"[pe_av] preload complete model_load={elapsed:.3f}s")
    return {"model_load_elapsed_sec": elapsed}


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _prompt_for_record(rec: Dict[str, Any]) -> str:
    return str(
        rec.get("prompt")
        or rec.get("av_caption")
        or rec.get("formatted_prompt")
        or rec.get("video_prompt")
        or ""
    ).strip()


def _empty_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _prompt_for_record(rec)
    return {
        "PE-TV": float("nan"),
        "PE-TA": float("nan"),
        "PE-TAV": float("nan"),
        "PE-TV-cosine": float("nan"),
        "PE-TA-cosine": float("nan"),
        "PE-TAV-cosine": float("nan"),
        "caption_video": float("nan"),
        "caption_audio": float("nan"),
        "caption_audio_video_joint": float("nan"),
        "caption_video_cosine": float("nan"),
        "caption_audio_cosine": float("nan"),
        "caption_audio_video_joint_cosine": float("nan"),
        "prompt": prompt,
        "video_path": rec.get("video_path"),
        "audio_path": rec.get("audio_path"),
    }


def _prepare_batch(processor: Any, records: List[Dict[str, Any]]) -> Any:
    videos = [load_video_rgb_array(str(rec["video_path"])) for rec in records]
    audios = [load_wav_mono(str(rec["audio_path"]), 48000)[0].astype(np.float32, copy=False) for rec in records]
    texts = [_prompt_for_record(rec) for rec in records]
    return processor(
        videos=videos,
        audio=audios,
        text=texts,
        audio_kwargs={"return_tensors": "pt", "sampling_rate": 48000},
        text_kwargs={"return_tensors": "pt", "padding": True, "truncation": True},
        videos_kwargs={"return_tensors": "pt"},
    )


def _output_attr(outputs: Any, *names: str) -> torch.Tensor:
    for name in names:
        value = getattr(outputs, name, None)
        if value is not None:
            return value
    raise AttributeError(f"PE-AV output missing all of: {', '.join(names)}")


def _paired_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() * right.float()).sum(dim=-1)


def _paired_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    return (left * right).sum(dim=-1)


def _compute_batch(
    model: torch.nn.Module,
    processor: Any,
    records: List[Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Dict[str, float]]:
    valid = [rec for rec in records if _prompt_for_record(rec)]
    if not valid:
        return {}

    inputs = _prepare_batch(processor, valid).to(device)
    _cuda_sync(device)
    with torch.inference_mode(), torch.autocast(
        device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        outputs = model(**inputs)
    _cuda_sync(device)

    video_embeds = _output_attr(outputs, "video_embeds", "visual_embeds")
    audio_embeds = _output_attr(outputs, "audio_embeds")
    audio_video_embeds = _output_attr(outputs, "audio_video_embeds", "audio_visual_embeds")
    text_video_embeds = _output_attr(outputs, "text_video_embeds", "visual_text_embeds")
    text_audio_embeds = _output_attr(outputs, "text_audio_embeds", "audio_text_embeds")
    text_audio_video_embeds = _output_attr(
        outputs,
        "text_audio_video_embeds",
        "audio_video_text_embeds",
        "audio_visual_text_embeds",
    )

    pe_tv = _paired_dot(video_embeds, text_video_embeds).detach().float().cpu().tolist()
    pe_ta = _paired_dot(audio_embeds, text_audio_embeds).detach().float().cpu().tolist()
    pe_tav = _paired_dot(audio_video_embeds, text_audio_video_embeds).detach().float().cpu().tolist()
    pe_tv_cos = _paired_cosine(video_embeds, text_video_embeds).detach().float().cpu().tolist()
    pe_ta_cos = _paired_cosine(audio_embeds, text_audio_embeds).detach().float().cpu().tolist()
    pe_tav_cos = _paired_cosine(
        audio_video_embeds,
        text_audio_video_embeds,
    ).detach().float().cpu().tolist()

    out: Dict[str, Dict[str, float]] = {}
    for rec, tv, ta, tav, tv_cos, ta_cos, tav_cos in zip(
        valid,
        pe_tv,
        pe_ta,
        pe_tav,
        pe_tv_cos,
        pe_ta_cos,
        pe_tav_cos,
    ):
        out[str(rec["file_stem"])] = {
            "PE-TV": float(tv),
            "PE-TA": float(ta),
            "PE-TAV": float(tav),
            "PE-TV-cosine": float(tv_cos),
            "PE-TA-cosine": float(ta_cos),
            "PE-TAV-cosine": float(tav_cos),
        }
    return out


def run_task(
    rank: int,
    local_rank: int,
    world_size: int,
    target_dir: Path,
    manifest: Dict[str, Any],
    skip_completed: bool = True,
    metric_keys: List[str] | None = None,
    reuse_models: bool = False,
    **_: Any,
) -> Dict[str, float]:
    model_load_elapsed_sec = 0.0
    metric_keys = metric_keys or [
        "PE-TV",
        "PE-TA",
        "PE-TAV",
        "PE-TV-cosine",
        "PE-TA-cosine",
        "PE-TAV-cosine",
    ]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[pe_av] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    pending = [
        rec for rec in my_records
        if not (skip_completed and already_done(target_dir, "pe_av", rec["file_stem"], metric_keys))
    ]
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    device = _resolve_device(local_rank)
    dtype = _resolve_dtype(device)
    model, processor, model_load_elapsed_sec = _get_model(
        _resolve_model_dir(),
        device,
        dtype,
        reuse_models=reuse_models,
    )
    batch_size = max(1, int(os.environ.get("MY_EVAL_PE_AV_BATCH_SIZE", "2")))
    log(rank, f"[pe_av] batch_size={batch_size} dtype={dtype}")

    done = 0
    for batch in _chunks(pending, batch_size):
        try:
            scores = _compute_batch(model, processor, batch, device, dtype)
        except Exception as exc:
            if len(batch) == 1:
                rec = batch[0]
                log(rank, f"[pe_av] failed for {rec['file_stem']}: {exc}")
                scores = {}
            else:
                log(rank, f"[pe_av] batch failed ({len(batch)} samples): {exc}; retry one-by-one")
                scores = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for rec in batch:
                    try:
                        scores.update(_compute_batch(model, processor, [rec], device, dtype))
                    except Exception as one_exc:
                        log(rank, f"[pe_av] failed for {rec['file_stem']}: {one_exc}")

        for rec in batch:
            payload = _empty_payload(rec)
            if rec["file_stem"] in scores:
                payload.update(scores[rec["file_stem"]])
                payload["caption_video"] = payload["PE-TV"]
                payload["caption_audio"] = payload["PE-TA"]
                payload["caption_audio_video_joint"] = payload["PE-TAV"]
                payload["caption_video_cosine"] = payload["PE-TV-cosine"]
                payload["caption_audio_cosine"] = payload["PE-TA-cosine"]
                payload["caption_audio_video_joint_cosine"] = payload["PE-TAV-cosine"]
            write_per_sample(target_dir, "pe_av", rec["file_stem"], payload)
            done += 1
        if done % 20 == 0:
            log(rank, f"  pe_av {done}/{len(pending)} (batch_size={batch_size})")

    if not reuse_models:
        del model
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
