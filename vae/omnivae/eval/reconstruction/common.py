from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml

from omnivae.models.audio_video_vae import AudioVideoVAE
from omnivae.train.av_vae.utils import _expand_known_path_vars, resolve_config_paths


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    raw = _expand_known_path_vars(str(path))
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir or repo_root()) / p
    return p.resolve()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = resolve_config_paths(cfg)
    return cfg


def checkpoint_state_path(checkpoint: str | Path) -> Path:
    path = resolve_path(checkpoint)
    if path.is_dir():
        state_path = path / "state_dict.pt"
        if state_path.is_file():
            return state_path
    return path


def infer_config_from_checkpoint(checkpoint: str | Path) -> Optional[Path]:
    path = checkpoint_state_path(checkpoint)
    candidates: List[Path] = []
    if path.is_file() and path.name == "state_dict.pt":
        # .../<run>/checkpoints/Trainer_x/state_dict.pt
        candidates.append(path.parent.parent.parent / "config.yaml")
    if path.is_dir():
        # .../<run>/checkpoints/Trainer_x
        candidates.append(path.parent.parent / "config.yaml")
    candidates.append(path.parent / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def require_config(config: Optional[str], checkpoint: str | Path) -> Path:
    if config:
        path = resolve_path(config)
    else:
        path = infer_config_from_checkpoint(checkpoint)
        if path is None:
            raise FileNotFoundError(
                "Could not infer config.yaml from checkpoint. Pass --config explicitly."
            )
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    return path


def _extract_state_dict(ckpt: Any, use_ema: bool = False) -> Dict[str, torch.Tensor]:
    if use_ema and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        ema = ckpt["ema_state_dict"]
        if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
            shadow = ema["shadow"]
            if all(isinstance(v, torch.Tensor) for v in shadow.values()):
                return shadow

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "module", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]

    raise KeyError(
        "Could not find model weights in checkpoint. Expected model_state_dict, "
        "state_dict, module, model, ema_state_dict.shadow, or a flat state dict."
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model_weights(
    model: AudioVideoVAE,
    checkpoint: str | Path,
    *,
    use_ema: bool = False,
    map_location: str | torch.device = "cpu",
) -> Dict[str, int]:
    state_path = checkpoint_state_path(checkpoint)
    if not state_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {state_path}")
    ckpt = torch.load(state_path, map_location=map_location)
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt, use_ema=use_ema))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "missing": len(missing),
        "unexpected": len(unexpected),
        "loaded_tensors": len(state_dict),
    }


def build_reconstruction_model(
    cfg: Dict[str, Any],
    *,
    modality: str,
) -> AudioVideoVAE:
    model_cfg = cfg.get("model", {})
    video_cfg = deepcopy(model_cfg.get("video", {}))
    audio_cfg = deepcopy(model_cfg.get("audio", {}))

    # Reconstruction eval does not need contrastive, LLM, or distill heads.
    return AudioVideoVAE(
        video_vae_kwargs=video_cfg,
        audio_vae_kwargs=audio_cfg,
        contrastive_kwargs={},
        llm_kwargs={},
        distill_kwargs={},
        skip_video_vae=(modality == "audio"),
        skip_audio_vae=(modality == "video"),
    )


def read_jsonl(path: str | Path, *, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    jsonl_path = resolve_path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def read_checkpoint_list(values: Iterable[str], file_path: Optional[str] = None) -> List[str]:
    checkpoints = [str(v) for v in values if str(v).strip()]
    if file_path:
        path = resolve_path(file_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                checkpoints.append(line)
    return checkpoints


def sanitize_name(value: str) -> str:
    value = value.replace("/", "_").replace(" ", "_").replace(":", "_")
    value = value.replace(",", "").replace("=", "")
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def run_name_for_checkpoint(index: int, checkpoint: str | Path) -> str:
    path = checkpoint_state_path(checkpoint)
    trainer = path.parent.name if path.name == "state_dict.pt" else path.stem
    run_dir = path.parent.parent.parent if path.name == "state_dict.pt" else path.parent
    exp_name = run_dir.name[:110]
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return sanitize_name(f"{index:02d}_{trainer}_{exp_name}_{digest}")


def setup_logging(log_file: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(out)
