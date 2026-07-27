"""Pretrained-checkpoint loader and split-saver for the joint AV model.

* ``load_pretrained_branches`` warm-starts each ``ZImageTransformer2DModel``
  branch from its single-modality checkpoint. Bridge parameters stay at
  their (already-zero) initialisation so the joint forward starts
  byte-equivalent to "video alone" + "audio alone".

* ``save_split_branches`` mirrors ``omnivae_generation.trainer.modeling.save_checkpoint_artifacts``
  but writes three separate sub-directories under the snapshot root::

      checkpoint-XXXXXX/
        transformer_video/    # video branch only (loadable by t2v trainer)
        transformer_audio/    # audio branch only (loadable by t2a trainer)
        bridges/              # safetensors blob with bridge.* params
        tokenizer/, scheduler/, metadata.json

  This keeps each branch independently reusable downstream (e.g. you can
  resume single-modality training from a joint-AV snapshot) without
  introducing a new "joint" checkpoint format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file as safetensors_save

from omnivae_generation.trainer.modeling import (
    _resolve_pretrained_transformer_dir,
    load_pretrained_transformer_weights,
)


logger = logging.getLogger(__name__)

BRIDGE_PARAM_PREFIX = "bridges."
VIDEO_PARAM_PREFIX = "video."
AUDIO_PARAM_PREFIX = "audio."

_BRIDGE_BLOB_NAME = "bridges.safetensors"


def _load_branch_strict(
    branch_module: torch.nn.Module,
    pretrained_path: str | Path,
    *,
    expected_config: dict | None,
    label: str,
) -> Path:
    """Load weights for a single branch.

    When ``expected_config`` is provided we go through the strict helper
    in ``omnivae_generation.trainer.modeling`` (validates the config keys before loading).
    Otherwise we load with ``strict=False`` and surface missing /
    unexpected keys as warnings -- useful when the pretrained checkpoint
    happens to omit non-trainable buffers like ``x_pad_token`` that the
    runtime patches stitch back in.
    """
    if expected_config is not None:
        return load_pretrained_transformer_weights(
            branch_module, pretrained_path, expected_config=expected_config,
        )

    from diffusers.models import modeling_utils

    transformer_dir = _resolve_pretrained_transformer_dir(pretrained_path)

    weights_file = None
    for weights_name in (modeling_utils.SAFETENSORS_WEIGHTS_NAME, modeling_utils.WEIGHTS_NAME):
        candidate = transformer_dir / weights_name
        if candidate.is_file():
            weights_file = candidate
            break
    if weights_file is None:
        raise FileNotFoundError(f"Could not find transformer weights under {transformer_dir}.")

    state_dict = modeling_utils.load_state_dict(str(weights_file), map_location="cpu")
    msg = branch_module.load_state_dict(state_dict, strict=False)
    missing = list(getattr(msg, "missing_keys", []))
    unexpected = list(getattr(msg, "unexpected_keys", []))
    if missing:
        logger.warning(
            "load_pretrained_branches[%s]: missing %d key(s) (kept init values): %s",
            label, len(missing), missing[:8],
        )
    if unexpected:
        logger.warning(
            "load_pretrained_branches[%s]: unexpected %d key(s) ignored: %s",
            label, len(unexpected), unexpected[:8],
        )
    return transformer_dir


def load_pretrained_branches(
    joint_model,                              # BridgedZImageJointModel
    *,
    pretrained_t2v: str | Path | None,
    pretrained_t2a: str | Path | None,
    expected_video_config: dict | None = None,
    expected_audio_config: dict | None = None,
) -> dict[str, str]:
    """Warm-start each branch from its single-modality checkpoint.

    Bridge parameters are *not* touched; they remain at their zero
    initialisation, so the joint model is identical to running each
    branch independently on step 0 (verified by
    ``BridgedZImageJointModel.assert_bridges_zero_initialised``).

    Returns ``{"video": <dir>, "audio": <dir>}`` with the resolved
    paths actually loaded (handy for a metadata trail in the run log).
    """
    loaded: dict[str, str] = {}
    if pretrained_t2v:
        print(f"[t2av:loader] loading video branch from {pretrained_t2v} ...", flush=True)
        video_dir = _load_branch_strict(
            joint_model.video,
            pretrained_t2v,
            expected_config=expected_video_config,
            label="video",
        )
        loaded["video"] = str(video_dir)
        logger.info("load_pretrained_branches: video branch loaded from %s", video_dir)
        print(f"[t2av:loader] video branch loaded from {video_dir}", flush=True)
    else:
        print("[t2av:loader] pretrained_t2v not provided; video branch keeps random init", flush=True)

    if pretrained_t2a:
        print(f"[t2av:loader] loading audio branch from {pretrained_t2a} ...", flush=True)
        audio_dir = _load_branch_strict(
            joint_model.audio,
            pretrained_t2a,
            expected_config=expected_audio_config,
            label="audio",
        )
        loaded["audio"] = str(audio_dir)
        logger.info("load_pretrained_branches: audio branch loaded from %s", audio_dir)
        print(f"[t2av:loader] audio branch loaded from {audio_dir}", flush=True)
    else:
        print("[t2av:loader] pretrained_t2a not provided; audio branch keeps random init", flush=True)

    joint_model.assert_bridges_zero_initialised()
    return loaded


def _split_state_dict_by_prefix(
    state_dict: dict[str, torch.Tensor], prefixes: Iterable[str]
) -> dict[str, dict[str, torch.Tensor]]:
    """Bucket ``state_dict`` keys by which ``prefix`` they start with.

    Keys matching ``prefix`` get re-keyed with the prefix stripped so
    each bucket can be loaded back into the corresponding *standalone*
    module (``ZImageTransformer2DModel`` / ``BridgeBlock``).
    """
    buckets: dict[str, dict[str, torch.Tensor]] = {prefix: {} for prefix in prefixes}
    for full_key, tensor in state_dict.items():
        for prefix in prefixes:
            if full_key.startswith(prefix):
                buckets[prefix][full_key[len(prefix):]] = tensor.detach().cpu().contiguous()
                break
    return buckets


def save_split_branches(
    output_dir: str | Path,
    *,
    joint_model,                              # BridgedZImageJointModel (unwrapped)
    tokenizer,
    scheduler,
    metadata: dict,
    save_text_encoder=None,
    save_video_vae=None,
    save_audio_vae=None,
) -> None:
    """Mirror ``save_checkpoint_artifacts`` but write three transformer
    sub-directories so each branch (and the bridge stack) can be loaded
    independently downstream.

    Layout::

        <output_dir>/
          transformer_video/diffusion_pytorch_model.safetensors + config.json
          transformer_audio/diffusion_pytorch_model.safetensors + config.json
          bridges/bridges.safetensors + bridge_config.json
          tokenizer/ ...
          scheduler/ ...
          [text_encoder/ ...]
          [vae/ ...]
          [audio_vae/ ...]
          metadata.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Video / audio branches: just call diffusers' save_pretrained on
    # the actual ``ZImageTransformer2DModel`` instance so config.json is
    # written correctly and the resulting directory looks identical to
    # what the single-modality trainers produce.
    joint_model.video.save_pretrained(output_path / "transformer_video", safe_serialization=True)
    joint_model.audio.save_pretrained(output_path / "transformer_audio", safe_serialization=True)

    # Bridges: hand-roll a safetensors blob + a small JSON descriptor.
    # We avoid using ``BridgeBlock.save_pretrained`` because BridgeBlock
    # is not a ConfigMixin (it's a plain nn.Module) and we don't want
    # to add ConfigMixin baggage just for this single sidecar.
    bridge_dir = output_path / "bridges"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    bridge_state: dict[str, torch.Tensor] = {}
    for name, tensor in joint_model.bridges.state_dict().items():
        bridge_state[name] = tensor.detach().cpu().contiguous()
    safetensors_save(bridge_state, str(bridge_dir / _BRIDGE_BLOB_NAME))
    bridge_descriptor = {
        "bridge_interval": int(joint_model.bridge_interval),
        "use_asymmetric_ati": bool(joint_model.use_asymmetric_ati),
        "a2v_window_size": int(joint_model.a2v_window_size),
        "n_bridges": int(len(joint_model.bridges)),
        "dim": int(joint_model.dim),
        "n_heads": int(joint_model.n_heads),
        "head_dim": int(joint_model.dim // joint_model.n_heads),
    }
    (bridge_dir / "bridge_config.json").write_text(
        json.dumps(bridge_descriptor, indent=2), encoding="utf-8"
    )

    tokenizer.save_pretrained(output_path / "tokenizer")
    scheduler.save_pretrained(output_path / "scheduler")
    if save_text_encoder is not None:
        save_text_encoder.save_pretrained(output_path / "text_encoder", safe_serialization=True)
    if save_video_vae is not None:
        save_video_vae.save_pretrained(output_path / "vae", safe_serialization=True)
    if save_audio_vae is not None:
        audio_vae_dir = output_path / "audio_vae"
        audio_vae_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(save_audio_vae, "save_pretrained"):
            save_audio_vae.save_pretrained(audio_vae_dir, safe_serialization=True)
        else:
            torch.save(save_audio_vae.state_dict(), audio_vae_dir / "audio_vae.pt")

    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_bridges_from_dir(joint_model, bridge_dir: str | Path) -> None:
    """Restore bridge weights produced by :func:`save_split_branches`.

    Used both for resume-from-checkpoint and for downstream eval scripts
    that want to load just the joint model without re-running training.
    """
    from safetensors.torch import load_file as safetensors_load

    bridge_path = Path(bridge_dir).expanduser().resolve()
    blob = bridge_path / _BRIDGE_BLOB_NAME
    if not blob.is_file():
        raise FileNotFoundError(f"Could not find bridge weights at {blob}")
    state = safetensors_load(str(blob))
    msg = joint_model.bridges.load_state_dict(state, strict=True)
    missing = list(getattr(msg, "missing_keys", []))
    unexpected = list(getattr(msg, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            f"load_bridges_from_dir: incompatible bridge state_dict at {bridge_path}: "
            f"missing={missing!r} unexpected={unexpected!r}"
        )
    logger.info("load_bridges_from_dir: bridges restored from %s", bridge_path)
