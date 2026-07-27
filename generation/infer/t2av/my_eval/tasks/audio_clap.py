"""CLAP text-audio similarity (LAION-CLAP).

Loads only the CLAP text/audio model from MOVA's audio_is_clap assets; the
scores match the MOVA toolkit but avoid constructing the unrelated Cnn14 IS
model. Audio is loaded directly from ``record["audio_path"]`` (no ffmpeg
extraction) at 48 kHz mono.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
AUDIO_IS_CLAP_CODE_DIR = REPO_ROOT / "generation" / "evaluation" / "metrics" / "audio_is_clap"
AUDIO_IS_CLAP_ASSET_DIR = Path(
    os.environ.get("MY_EVAL_AUDIO_IS_CLAP_DIR", str(AUDIO_IS_CLAP_CODE_DIR))
).expanduser()
CLAP_SRC_DIR = REPO_ROOT / "generation" / "evaluation" / "models" / "clap" / "src"
CLAP_LAION_DIR = CLAP_SRC_DIR / "laion_clap"


def _ensure_pythonpath() -> None:
    for p in (str(AUDIO_IS_CLAP_CODE_DIR), str(CLAP_SRC_DIR), str(CLAP_LAION_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_pythonpath()

from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample
from my_eval.utils.audio_video import load_wav_mono

_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_clap_ckpt() -> Path:
    candidates = [AUDIO_IS_CLAP_ASSET_DIR / "clap_ckpt" / "630k-audioset-fusion-best.pt"]
    models_path = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if models_path:
        candidates.append(Path(models_path).expanduser() / "630k-audioset-fusion-best.pt")
    candidates.append(AUDIO_IS_CLAP_CODE_DIR / "clap_ckpt" / "630k-audioset-fusion-best.pt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "CLAP checkpoint not found; tried: " + ", ".join(str(p) for p in candidates)
    )


def _load_clap_model(device: torch.device):
    """Load only CLAP. MOVA's setup_models also builds Cnn14, which CLAP does not use."""
    import laion_clap
    from clap_module.factory import load_state_dict
    from transformers import RobertaModel, RobertaTokenizer

    print(f"[audio_clap] loading CLAP on {device}", flush=True)
    clap_model = laion_clap.CLAP_Module(enable_fusion=True, device=str(device))

    roberta_candidates = []
    models_path = os.environ.get("MY_EVAL_VERSE_MODELS") or os.environ.get("MODELS_PATH")
    if models_path:
        roberta_candidates.append(Path(models_path).expanduser() / "roberta-base")
    roberta_candidates.append(AUDIO_IS_CLAP_CODE_DIR.parent.parent / "models" / "roberta-base")
    roberta_local = next((path for path in roberta_candidates if path.is_dir()), None)
    if roberta_local is not None:
        print(f"[audio_clap] using local roberta-base: {roberta_local}", flush=True)
        tokenizer = RobertaTokenizer.from_pretrained(str(roberta_local), local_files_only=True)
        text_encoder = RobertaModel.from_pretrained(str(roberta_local), local_files_only=True).to(device)
    else:
        print("[audio_clap] local roberta-base not found; using transformers cache", flush=True)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        text_encoder = RobertaModel.from_pretrained("roberta-base").to(device)

    clap_model.tokenize = tokenizer
    clap_model.model.text_branch = text_encoder
    ckpt = _resolve_clap_ckpt()
    print(f"[audio_clap] using CLAP checkpoint: {ckpt}", flush=True)
    pkg = load_state_dict(str(ckpt))
    pkg.pop("text_branch.embeddings.position_ids", None)
    clap_model.model.load_state_dict(pkg, strict=False)
    clap_model.eval()
    return clap_model


def _resolve_device(local_rank: int) -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device


def _get_clap_model(device: torch.device, *, reuse_models: bool) -> tuple[Any, float]:
    key = str(device)
    if reuse_models and key in _MODEL_CACHE:
        return _MODEL_CACHE[key], 0.0
    started_at = time.time()
    model = _load_clap_model(device)
    elapsed = time.time() - started_at
    if reuse_models:
        _MODEL_CACHE[key] = model
    return model, elapsed


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: List[str] | None = None,
    **_: Any,
) -> Dict[str, float]:
    device = _resolve_device(local_rank)
    _, elapsed = _get_clap_model(device, reuse_models=True)
    log(rank, f"[audio_clap] preload complete model_load={elapsed:.3f}s")
    return {"model_load_elapsed_sec": elapsed}


def _load_audio_48k_mono(audio_path: str) -> np.ndarray:
    audio, _ = load_wav_mono(audio_path, 48000)
    return audio.astype(np.float32, copy=False)


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def _prompt_for_record(rec: Dict[str, Any]) -> str:
    return (rec.get("audio_prompt") or rec.get("prompt") or "").strip()


def _empty_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {"CLAP": float("nan"), "prompt": _prompt_for_record(rec)}


def _audio_tensor_for_clap(audio_path: str, device: torch.device) -> torch.Tensor:
    import pyloudnorm as pyln
    from eval_audio_is_clap import float32_to_int16, int16_to_float32  # type: ignore

    audio = pyln.normalize.peak(_load_audio_48k_mono(audio_path), -1.0)
    audio = audio.reshape(1, -1)
    audio = int16_to_float32(float32_to_int16(audio)).squeeze(0)
    return torch.from_numpy(audio).float().to(device)


def _compute_clap_batch(
    clap_model,
    text_embs: Dict[str, torch.Tensor],
    batch: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:
    valid = [rec for rec in batch if _prompt_for_record(rec)]
    if not valid:
        return {}
    audio_tensors = [_audio_tensor_for_clap(rec["audio_path"], device) for rec in valid]
    with torch.no_grad():
        audio_embs = clap_model.get_audio_embedding_from_data(audio_tensors, use_tensor=True).to(device)
        text_batch = torch.stack([text_embs[_prompt_for_record(rec)].to(device) for rec in valid], dim=0)
        vals = torch.nn.functional.cosine_similarity(audio_embs, text_batch, dim=1, eps=1e-8)
    return {
        rec["file_stem"]: float(val)
        for rec, val in zip(valid, vals.detach().float().cpu().tolist())
    }


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
    metric_keys = metric_keys or ["CLAP"]
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[audio_clap] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    device = _resolve_device(local_rank)

    pending = [
        rec for rec in my_records
        if not (skip_completed and already_done(target_dir, "audio_clap", rec["file_stem"], metric_keys))
    ]
    if not pending:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    clap_model, model_load_elapsed_sec = _get_clap_model(device, reuse_models=reuse_models)
    clap_device = device
    batch_size = int(os.environ.get("MY_EVAL_CLAP_BATCH_SIZE", "16"))

    # Pre-compute text embeddings for unique prompts (one prompt per record).
    unique_prompts: List[str] = []
    seen = set()
    for rec in pending:
        prompt = _prompt_for_record(rec)
        if prompt and prompt not in seen:
            seen.add(prompt)
            unique_prompts.append(prompt)

    text_embs: Dict[str, torch.Tensor] = {}
    if unique_prompts:
        log(rank, f"[audio_clap] precomputing {len(unique_prompts)} unique prompt embeddings")
        for prompt_batch in _chunks(unique_prompts, batch_size):
            with torch.no_grad():
                embs = clap_model.get_text_embedding(prompt_batch, use_tensor=True)
            for prompt, emb in zip(prompt_batch, embs):
                text_embs[prompt] = emb

    done = 0
    for batch in _chunks(pending, batch_size):
        try:
            scores = _compute_clap_batch(clap_model, text_embs, batch, clap_device)
        except Exception as exc:
            if len(batch) == 1:
                rec = batch[0]
                log(rank, f"[audio_clap] failed for {rec['file_stem']}: {exc}")
                write_per_sample(target_dir, "audio_clap", rec["file_stem"], _empty_payload(rec))
                done += 1
                continue
            log(rank, f"[audio_clap] batch failed ({len(batch)} samples): {exc}; retry one-by-one")
            scores = {}
            for rec in batch:
                try:
                    scores.update(_compute_clap_batch(clap_model, text_embs, [rec], clap_device))
                except Exception as one_exc:
                    log(rank, f"[audio_clap] failed for {rec['file_stem']}: {one_exc}")

        for rec in batch:
            payload = _empty_payload(rec)
            if rec["file_stem"] in scores:
                payload["CLAP"] = scores[rec["file_stem"]]
            write_per_sample(target_dir, "audio_clap", rec["file_stem"], payload)
            done += 1
        if done % 50 == 0:
            log(rank, f"  audio_clap {done}/{len(pending)} (batch_size={max(1, batch_size)})")

    if not reuse_models:
        del clap_model
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    return {"model_load_elapsed_sec": model_load_elapsed_sec}
