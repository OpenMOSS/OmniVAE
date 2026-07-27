"""OmniVAE checkpoint adapter for DiT training.

Lets you point ``vae.type=omnivae`` (or ``audio_vae.type=omnivae``) at a
``Trainer_xxxxx/state_dict.pt`` produced by the OmniVAE ``OmniVAE``
training loop and pull a single-modality VAE out for DiT training, without
reaching into ``AudioVideoVAE`` / contrastive_head / llm_caption_head.

Layout reminder (OmniVAE training ckpt):

.. code-block:: python

    {
        "model_state_dict": {
            "video_vae.<...>": tensor,        # WanVAE22Model
            "audio_vae.<...>": tensor,        # DAC (continuous mode)
            "contrastive_head.<...>": ...,    # ignored here
            "llm_caption_head.<...>": ...,    # ignored here
        },
        "ema_state_dict": {"shadow": {...}},  # optional
        "config": {... original training yaml ...},
    }

Public surface:

* :func:`load_univae_ckpt` -- one-shot ckpt parser (LRU cached so that
  ``branch=both`` does not pay ``torch.load`` twice).
* :func:`build_univae_video_vae` -- returns :class:`Wan2_2_NativeVAE` with
  weights loaded from the ``video_vae.*`` slice.
* :func:`build_univae_audio_vae` -- returns :class:`DAC` with weights loaded
  from the ``audio_vae.*`` slice (continuous mode by default).
* :func:`attach_companion` -- store a dormant sibling branch on the primary
  VAE without registering it as an ``nn.Module`` submodule (so optimizer /
  EMA / ``.to(device)`` skip it).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from omnivae_generation.trainer.vae.dac_vae import (
    DAC,
    _extract_audio_state_dict,
    _infer_dac_kwargs_from_state_dict,
)
from omnivae_generation.trainer.vae.wan2_2_native import (
    Wan2_2_NativeVAE,
    _import_wan_vae22_pieces,
    _resolve_native_config,
)


logger = logging.getLogger(__name__)


# Architecture kwargs the OmniVAE training yaml exposes for the audio branch
# (matches AudioVideoVAE.__init__ keys -- see OmniVAE/opensora/models/
# audio_video_vae/model.py:184-208).
_AUDIO_KWARGS = (
    "encoder_dim",
    "encoder_rates",
    "latent_dim",
    "decoder_dim",
    "decoder_rates",
    "n_codebooks",
    "codebook_size",
    "codebook_dim",
    "quantizer_dropout",
    "continuous",
)


# --------------------------------------------------------------------------- #
# Loading + sd splitting                                                      #
# --------------------------------------------------------------------------- #
def _resolve_ckpt_path(path: str | Path) -> Path:
    p = Path(str(path)).expanduser().resolve()
    if p.is_dir():
        # Mirror infer_audio_video_vae.py: directory -> directory/state_dict.pt
        candidate = p / "state_dict.pt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"OmniVAE ckpt directory {p} does not contain a state_dict.pt file."
            )
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"OmniVAE ckpt not found: {p}")
    return p


@lru_cache(maxsize=4)
def _load_univae_raw(path_str: str, use_ema: bool) -> Dict[str, Any]:
    """Cached raw loader. Keyed by (resolved abs path, use_ema)."""
    path = Path(path_str)
    logger.info("OmniVAE: loading checkpoint %s (use_ema=%s)", path, use_ema)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"OmniVAE ckpt at {path} is not a dict (got {type(ckpt).__name__})."
        )

    # Pull the flat tensor dict out, preferring the canonical OmniVAE keys.
    sd = None
    for key in ("model_state_dict", "state_dict", "module", "model"):
        v = ckpt.get(key)
        if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
            sd = v
            break
    if sd is None:
        # Last resort: maybe ckpt itself is a flat sd (rare for OmniVAE training output).
        if all(isinstance(t, torch.Tensor) for t in ckpt.values()):
            sd = ckpt  # type: ignore[assignment]
    if sd is None:
        raise KeyError(
            f"OmniVAE ckpt at {path} does not contain a recognizable state dict "
            f"(looked for keys ['model_state_dict','state_dict','module','model'])."
        )

    if use_ema:
        ema = ckpt.get("ema_state_dict")
        if isinstance(ema, dict) and ema.get("shadow"):
            shadow = ema["shadow"]
            n_overrides = sum(1 for k in sd.keys() if k in shadow)
            logger.info(
                "OmniVAE: applying EMA shadow on %d / %d params (other params "
                "fall back to model_state_dict).",
                n_overrides, len(sd),
            )
            sd = {k: shadow.get(k, v) for k, v in sd.items()}
        else:
            logger.warning(
                "OmniVAE: use_ema=True but ckpt has no ema_state_dict.shadow; "
                "falling back to model_state_dict."
            )

    video_sd: Dict[str, torch.Tensor] = {}
    audio_sd: Dict[str, torch.Tensor] = {}
    other_count = 0
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        if k.startswith("video_vae."):
            video_sd[k[len("video_vae."):]] = v
        elif k.startswith("audio_vae."):
            audio_sd[k[len("audio_vae."):]] = v
        else:
            # contrastive_head.* / llm_caption_head.* etc. -- ignored on purpose.
            other_count += 1

    train_config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}

    logger.info(
        "OmniVAE: split state_dict -> video=%d keys, audio=%d keys, dropped=%d keys",
        len(video_sd), len(audio_sd), other_count,
    )
    return {
        "video_sd": video_sd,
        "audio_sd": audio_sd,
        "train_config": train_config,
        "ckpt_path": str(path),
    }


def load_univae_ckpt(path: str | Path, *, use_ema: bool = False) -> Dict[str, Any]:
    """Load and split a OmniVAE training checkpoint.

    Returns a dict ``{"video_sd", "audio_sd", "train_config", "ckpt_path"}``.
    Repeated calls with the same ``(path, use_ema)`` re-use a cached parse so
    that ``branch=both`` does not pay ``torch.load`` twice.
    """
    resolved = _resolve_ckpt_path(path)
    return _load_univae_raw(str(resolved), bool(use_ema))


# --------------------------------------------------------------------------- #
# Video branch: WanVAE22Model + Wan2_2_NativeVAE wrapper                      #
# --------------------------------------------------------------------------- #
def _resolve_video_model_config(
    train_config: Dict[str, Any],
    override: Optional[str],
) -> str:
    """Pick the path to the WanVAE22 ``config.json`` that describes the video
    branch architecture. Priority: explicit override > train_config field >
    sibling next to OmniVAE ckpt > bundled default in opensora.infer.wan2_2vae.
    """
    if override:
        candidate = str(override)
        if Path(candidate).exists():
            return candidate
        # Fall through to _resolve_native_config which has additional fallbacks.

    yaml_video = train_config.get("model", {}).get("video", {})
    yaml_cfg = yaml_video.get("model_config")
    if yaml_cfg and isinstance(yaml_cfg, str) and Path(yaml_cfg).exists():
        return yaml_cfg

    # _resolve_native_config takes a (path, model_config_override) and returns
    # a usable config.json path. We hand it a clearly-nonexistent placeholder
    # so its sibling-of-ckpt branch short-circuits ("/<sentinel>/config.json"
    # never exists), and it falls through to the bundled default.
    sentinel = Path("/__univae_nonexistent_sentinel__")
    return _resolve_native_config(sentinel, override or yaml_cfg)


def _resolve_video_qk_norm(
    arg_value: str,
    train_config: Dict[str, Any],
    cfg_dict: Dict[str, Any],
) -> bool:
    arg_value = (arg_value or "auto").strip().lower()
    if arg_value == "true":
        return True
    if arg_value == "false":
        return False

    yaml_video = train_config.get("model", {}).get("video", {})
    if "qk_norm" in yaml_video:
        qk = bool(yaml_video["qk_norm"])
        logger.info("OmniVAE video VAE: qk_norm taken from train_config.model.video (%s)", qk)
        return qk

    if isinstance(cfg_dict, dict) and "qk_norm" in cfg_dict:
        qk = bool(cfg_dict["qk_norm"])
        logger.info("OmniVAE video VAE: qk_norm taken from config.json (%s)", qk)
        return qk

    logger.info(
        "OmniVAE video VAE: no qk_norm signal from train_config / config.json; "
        "defaulting to False."
    )
    return False


def build_univae_video_vae(
    ckpt_loaded: Dict[str, Any],
    *,
    model_config_override: Optional[str] = None,
    qk_norm: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    deterministic_posterior: bool = False,
) -> Wan2_2_NativeVAE:
    """Construct a :class:`Wan2_2_NativeVAE` whose inner ``WanVAE22Model`` is
    populated from the ``video_vae.*`` slice of a OmniVAE training ckpt.
    """
    video_sd = ckpt_loaded["video_sd"]
    train_config = ckpt_loaded["train_config"]
    if not video_sd:
        raise ValueError(
            "OmniVAE ckpt does not contain a video branch (no 'video_vae.*' "
            "keys). Cannot build the video VAE."
        )

    WanVAE22Model = _import_wan_vae22_pieces()
    cfg_path = _resolve_video_model_config(train_config, model_config_override)
    cfg_dict = WanVAE22Model.load_config(cfg_path)
    cfg_dict["qk_norm"] = _resolve_video_qk_norm(qk_norm, train_config, cfg_dict)
    cfg_dict.setdefault("deterministic_posterior", deterministic_posterior)

    logger.info(
        "OmniVAE video VAE: building WanVAE22Model from config=%s (qk_norm=%s)",
        cfg_path, cfg_dict["qk_norm"],
    )
    inner = WanVAE22Model.from_config(cfg_dict)

    result = inner.load_state_dict(video_sd, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    logger.info(
        "OmniVAE video VAE: load_state_dict -> missing=%d, unexpected=%d",
        len(missing), len(unexpected),
    )
    if missing:
        logger.warning("OmniVAE video VAE: missing keys (first 8): %s", missing[:8])
    if unexpected:
        logger.warning("OmniVAE video VAE: unexpected keys (first 8): %s", unexpected[:8])

    inner = inner.eval().requires_grad_(False)
    if torch_dtype != torch.float32:
        inner = inner.to(torch_dtype)
    return Wan2_2_NativeVAE(inner)


# --------------------------------------------------------------------------- #
# Audio branch: DAC                                                           #
# --------------------------------------------------------------------------- #
def _resolve_audio_kwargs(
    train_config: Dict[str, Any],
    audio_sd: Dict[str, torch.Tensor],
    overrides: Optional[Dict[str, Any]],
    sample_rate: Optional[int],
) -> Dict[str, Any]:
    """Decide the DAC ``__init__`` kwargs. Priority:
    yaml overrides > train_config['model']['audio'] > inferred-from-tensors.
    """
    inferred = _infer_dac_kwargs_from_state_dict(audio_sd)
    yaml_audio = train_config.get("model", {}).get("audio", {})

    kwargs: Dict[str, Any] = {}
    for key in _AUDIO_KWARGS:
        if key in inferred and inferred[key] is not None:
            kwargs[key] = inferred[key]
    for key in _AUDIO_KWARGS:
        if key in yaml_audio and yaml_audio[key] is not None:
            kwargs[key] = yaml_audio[key]
    if overrides:
        for key in _AUDIO_KWARGS:
            v = overrides.get(key)
            if v is not None:
                kwargs[key] = v

    # OmniVAE's AudioVideoVAE defaults DAC to continuous mode (model.py:193).
    kwargs.setdefault("continuous", True)

    sr = sample_rate
    if not sr or int(sr) <= 0:
        sr = (
            yaml_audio.get("sample_rate")
            or yaml_audio.get("audio_sample_rate")
            or train_config.get("audio_sample_rate")
            or 48000
        )
    kwargs["sample_rate"] = int(sr)
    return kwargs


def build_univae_audio_vae(
    ckpt_loaded: Dict[str, Any],
    *,
    sample_rate: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    torch_dtype: torch.dtype = torch.float32,
) -> DAC:
    """Construct a :class:`DAC` populated from the ``audio_vae.*`` slice of a
    OmniVAE training ckpt.
    """
    audio_sd = ckpt_loaded["audio_sd"]
    train_config = ckpt_loaded["train_config"]
    if not audio_sd:
        raise ValueError(
            "OmniVAE ckpt does not contain an audio branch (no 'audio_vae.*' "
            "keys). Cannot build the audio VAE."
        )

    kwargs = _resolve_audio_kwargs(train_config, audio_sd, overrides, sample_rate)
    logger.info("OmniVAE audio VAE: building DAC with kwargs=%s", kwargs)
    vae = DAC(**kwargs)

    result = vae.load_state_dict(audio_sd, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    logger.info(
        "OmniVAE audio VAE: load_state_dict -> missing=%d, unexpected=%d",
        len(missing), len(unexpected),
    )
    if missing:
        logger.warning("OmniVAE audio VAE: missing keys (first 8): %s", missing[:8])
    if unexpected:
        logger.warning("OmniVAE audio VAE: unexpected keys (first 8): %s", unexpected[:8])

    vae = vae.eval().requires_grad_(False)
    return vae.to(dtype=torch_dtype)


# --------------------------------------------------------------------------- #
# branch=both -- attach a dormant sibling without registering it as a submodule
# --------------------------------------------------------------------------- #
_COMPANION_ATTR = "_univae_companion"


def attach_companion(primary: nn.Module, companion: nn.Module) -> None:
    """Attach ``companion`` to ``primary`` via plain ``__dict__`` assignment.

    PyTorch's ``nn.Module.__setattr__`` automatically registers submodules,
    which would cause:
      * ``optimizer.parameters()`` to pick up the companion's params
      * ``accelerator.prepare`` / ``vae.to(device)`` to move it to GPU
      * EMA / DDP wrappers to track its weights

    We want none of that for the dormant branch -- it should sit in CPU RAM
    until something explicitly opts in. Bypassing ``__setattr__`` keeps the
    companion as a regular Python attribute.
    """
    primary.__dict__[_COMPANION_ATTR] = companion


def get_companion(primary: nn.Module) -> Optional[nn.Module]:
    """Retrieve the dormant sibling branch attached via :func:`attach_companion`,
    or ``None`` if no companion was attached.
    """
    return primary.__dict__.get(_COMPANION_ATTR)


__all__ = [
    "load_univae_ckpt",
    "build_univae_video_vae",
    "build_univae_audio_vae",
    "attach_companion",
    "get_companion",
]
