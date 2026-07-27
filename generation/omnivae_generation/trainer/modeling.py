from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from transformers import AutoModel, AutoTokenizer
from omnivae_generation.trainer.qwen3_vl_dit import (
    Qwen3VLDiffusionTransformer,
    is_qwen3_vl_dit_arch,
    load_qwen3_vl_text_config,
)

from omnivae_generation.trainer.runtime_patches import (
    patch_flux2_vae_for_zimage,
    raw_latents_to_training_layout,
    retrieve_latents,
    vae_encode_returns_training_latents,
    vae_uses_training_layout,
)
from omnivae_generation.trainer.vae import DAC, KeiVivit2VAE, Wan2_2_NativeVAE, Wan2_2_VAE


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def normalize_vae_type(value: str | None, *, default: str) -> str:
    vae_type = str(value or default).strip().lower()
    if vae_type == "omnivae":
        return "univae"
    return vae_type

PROMPT_EMBEDDING_CACHE_SIZE = 2048
VALID_PREDICT_TARGETS = {"v", "x0"}
TRANSFORMER_INIT_CONFIG_KEYS = (
    "in_channels",
    "out_channels",
    "all_patch_size",
    "all_f_patch_size",
    "dim",
    "n_layers",
    "n_refiner_layers",
    "n_heads",
    "n_kv_heads",
    "norm_eps",
    "qk_norm",
    "rope_theta",
    "t_scale",
    "axes_dims",
    "axes_lens",
    "cap_feat_dim",
)


class _PromptEmbeddingLRUCache:
    def __init__(self, maxsize: int = PROMPT_EMBEDDING_CACHE_SIZE) -> None:
        self.maxsize = maxsize
        self._items: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, prompt: str) -> Optional[torch.Tensor]:
        embedding = self._items.get(prompt)
        if embedding is None:
            return None
        self._items.move_to_end(prompt)
        return embedding

    def put(self, prompt: str, embedding: torch.Tensor) -> None:
        self._items[prompt] = embedding.detach().to(device="cpu").contiguous()
        self._items.move_to_end(prompt)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)


def resolve_dtype(name: Optional[str], fallback: torch.dtype = torch.float32) -> torch.dtype:
    if name is None:
        return fallback
    normalized = str(name).strip().lower()
    if normalized in {"auto", ""}:
        return fallback
    if normalized not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype specifier: {name}")
    return DTYPE_MAP[normalized]


def _iter_mid_block_attentions(vae: nn.Module) -> list[tuple[str, int, nn.Module]]:
    targets: list[tuple[str, int, nn.Module]] = []
    for branch_name in ("encoder", "decoder"):
        branch = getattr(vae, branch_name, None)
        mid_block = getattr(branch, "mid_block", None)
        attentions = getattr(mid_block, "attentions", None)
        if attentions is None:
            continue
        for index, attention in enumerate(attentions):
            if isinstance(attention, nn.Module):
                targets.append((branch_name, index, attention))
    return targets


def _patch_autoencoder_kl_mid_block_qk_rmsnorm_if_needed(vae: AutoencoderKL, *, enabled: bool) -> None:
    if not enabled:
        return

    from diffusers.models.normalization import RMSNorm

    targets = _iter_mid_block_attentions(vae)
    if len(targets) == 0:
        raise ValueError(
            f"mid_block_qk_rmsnorm=True but {vae.__class__.__name__} does not expose any mid_block.attentions."
        )

    for branch_name, index, attention in targets:
        if not hasattr(attention, "to_q") or not hasattr(attention, "to_k"):
            raise ValueError(
                f"mid_block_qk_rmsnorm expects diffusers-style Attention modules, but got "
                f"{vae.__class__.__name__}.{branch_name}.mid_block.attentions[{index}]={attention.__class__.__name__}."
            )

        norm_q = getattr(attention, "norm_q", None)
        norm_k = getattr(attention, "norm_k", None)
        if (norm_q is None) != (norm_k is None):
            raise ValueError(
                f"{vae.__class__.__name__}.{branch_name}.mid_block.attentions[{index}] has inconsistent q/k norms."
            )
        if norm_q is not None and norm_k is not None:
            continue

        heads = int(getattr(attention, "heads", 0))
        inner_dim = int(getattr(attention, "inner_dim", 0))
        if heads <= 0 or (inner_dim % heads) != 0:
            raise ValueError(
                f"Invalid attention geometry for {vae.__class__.__name__}.{branch_name}.mid_block.attentions[{index}]: "
                f"inner_dim={inner_dim}, heads={heads}."
            )

        head_dim = inner_dim // heads
        attention.norm_q = RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, bias=False)
        attention.norm_k = RMSNorm(head_dim, eps=1e-6, elementwise_affine=True, bias=False)


def _load_autoencoder_kl_with_optional_mid_block_qk_rmsnorm(
    *,
    model_name_or_path: str,
    subfolder: str | None,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    mid_block_qk_rmsnorm: bool,
):
    cfg_kwargs: dict[str, object] = {"local_files_only": local_files_only}
    if subfolder is not None:
        cfg_kwargs["subfolder"] = subfolder
    raw_config = AutoencoderKL.load_config(model_name_or_path, **cfg_kwargs)
    config_dict = dict(raw_config)
    config_mid_block_qk_rmsnorm = bool(config_dict.pop("mid_block_qk_rmsnorm", False))

    if not mid_block_qk_rmsnorm and not config_mid_block_qk_rmsnorm:
        return AutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    from diffusers.models import modeling_utils

    weights_file = None
    last_error: Exception | None = None
    for weights_name in (modeling_utils.SAFETENSORS_WEIGHTS_NAME, modeling_utils.WEIGHTS_NAME):
        try:
            weights_file = modeling_utils._get_model_file(
                model_name_or_path,
                weights_name=weights_name,
                subfolder=subfolder,
                local_files_only=local_files_only,
            )
            break
        except EnvironmentError as exc:
            last_error = exc
    if weights_file is None:
        if last_error is not None:
            raise last_error
        raise EnvironmentError(f"Could not locate AutoencoderKL weights for {model_name_or_path!r}.")

    checkpoint_state_dict = modeling_utils.load_state_dict(weights_file, map_location="cpu")
    checkpoint_mid_block_qk_rmsnorm = any(
        str(key).endswith(".norm_q.weight") or str(key).endswith(".norm_k.weight")
        for key in checkpoint_state_dict.keys()
    )
    resolved_mid_block_qk_rmsnorm = bool(
        mid_block_qk_rmsnorm or config_mid_block_qk_rmsnorm or checkpoint_mid_block_qk_rmsnorm
    )

    vae = AutoencoderKL.from_config(config_dict)
    _patch_autoencoder_kl_mid_block_qk_rmsnorm_if_needed(vae, enabled=resolved_mid_block_qk_rmsnorm)
    if resolved_mid_block_qk_rmsnorm:
        vae.register_to_config(mid_block_qk_rmsnorm=True)

    msg = vae.load_state_dict(checkpoint_state_dict, strict=True)
    missing_keys = list(getattr(msg, "missing_keys", []))
    unexpected_keys = list(getattr(msg, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Unexpected AutoencoderKL load_state_dict result: "
            f"missing_keys={missing_keys!r} unexpected_keys={unexpected_keys!r}"
        )

    return vae.to(dtype=torch_dtype)


def load_text_components(config: dict, torch_dtype: torch.dtype):
    if config.get("disable_qwen3_5_fast_path", False):
        from omnivae_generation.trainer.runtime_patches import patch_transformers_qwen3_5_disable_fast_path

        patch_transformers_qwen3_5_disable_fast_path()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config.get("trust_remote_code", False),
        local_files_only=config.get("local_files_only", False),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": config.get("trust_remote_code", False),
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": config.get("local_files_only", False),
    }
    attn_implementation = config.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    text_encoder = AutoModel.from_pretrained(config["model_name_or_path"], **model_kwargs)
    text_config = text_encoder.config.get_text_config()
    hidden_size = getattr(text_config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(
            "Could not determine the text hidden size from the text encoder config returned by "
            "`config.get_text_config()`."
        )

    return tokenizer, text_encoder, int(hidden_size)


def load_qwen3_vl_tokenizer_and_hidden_size(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        config["backbone_name_or_path"],
        trust_remote_code=config.get("trust_remote_code", False),
        local_files_only=config.get("local_files_only", False),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    configured_text_config = config.get("text_config")
    if configured_text_config is not None:
        hidden_size = configured_text_config.get("hidden_size")
    else:
        text_config = load_qwen3_vl_text_config(
            config["backbone_name_or_path"],
            trust_remote_code=config.get("trust_remote_code", False),
            local_files_only=config.get("local_files_only", False),
        )
        hidden_size = getattr(text_config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(
            "Could not determine the Qwen3-VL text hidden size from the configured backbone."
        )
    return tokenizer, int(hidden_size)

def load_vae(config: dict):
    vae_type = normalize_vae_type(config.get("type"), default="autoencoder_kl")
    torch_dtype = resolve_dtype(config.get("torch_dtype"), fallback=torch.float32)
    model_name_or_path = config["model_name_or_path"]
    local_files_only = config.get("local_files_only", False)

    if vae_type == "autoencoder_kl":
        return _load_autoencoder_kl_with_optional_mid_block_qk_rmsnorm(
            model_name_or_path=model_name_or_path,
            subfolder=config.get("subfolder"),
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            mid_block_qk_rmsnorm=bool(config.get("mid_block_qk_rmsnorm", False)),
        )

    if vae_type == "wan2_2_vae":
        normalized_path = str(model_name_or_path).strip().lower()
        if normalized_path.endswith((".pth", ".pt", ".ckpt", ".safetensors")) or "/blob/" in normalized_path:
            return Wan2_2_VAE.from_single_file(
                model_name_or_path,
                config=config.get("config_model_name_or_path"),
                subfolder=config.get("config_subfolder", "vae"),
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
                wan_chunk_mode=config.get("wan_chunk_mode", "cache"),
            )
        return Wan2_2_VAE.from_pretrained(
            model_name_or_path,
            subfolder=config.get("subfolder", "vae"),
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            wan_chunk_mode=config.get("wan_chunk_mode", "cache"),
        )

    if vae_type == "wan2_2_native_vae":
        # Loads the open-source Wan2.2 VAE in its native `.pth` format using
        # `WanVAE22Model` from OmniVAE/opensora/infer/wan2_2vae (the
        # same loader as that package's `infer.py`). Drop-in replacement for
        # the diffusers-based wan2_2_vae path; expects either a directory
        # holding {*.pth, config.json} or a single weight file.
        return Wan2_2_NativeVAE.from_native_ckpt(
            model_name_or_path,
            model_config=config.get("model_config"),
            qk_norm=str(config.get("qk_norm", "auto")),
            torch_dtype=torch_dtype,
            deterministic_posterior=bool(config.get("deterministic_posterior", False)),
        )

    if vae_type == "univae":
        # Drive DiT training directly from an OmniVAE training ckpt
        # (Trainer_xxxxx/state_dict.pt). Returns the video branch wrapped in
        # Wan2_2_NativeVAE; with branch="both" the audio branch is loaded too
        # and stashed on the wrapper as a dormant CPU companion (does NOT enter
        # the optimizer / EMA / accelerator.prepare). The implementation file
        # keeps the old univae name as a checkpoint-compatibility alias.
        from omnivae_generation.trainer.vae.univae import (
            attach_companion,
            build_univae_audio_vae,
            build_univae_video_vae,
            load_univae_ckpt,
        )

        branch = str(config.get("branch", "video")).strip().lower()
        if branch not in {"video", "both"}:
            raise ValueError(
                f"vae.type='omnivae' expects branch in {{'video','both'}}, got {branch!r}. "
                "(Use audio_vae.type='omnivae' for audio-primary training.)"
            )
        loaded = load_univae_ckpt(
            model_name_or_path,
            use_ema=bool(config.get("use_ema", False)),
        )
        primary = build_univae_video_vae(
            loaded,
            model_config_override=config.get("video_model_config") or config.get("model_config"),
            qk_norm=str(config.get("qk_norm", "auto")),
            torch_dtype=torch_dtype,
            deterministic_posterior=bool(config.get("deterministic_posterior", False)),
        )
        if branch == "both":
            companion = build_univae_audio_vae(
                loaded,
                sample_rate=config.get("audio_sample_rate") or config.get("sample_rate"),
                torch_dtype=torch_dtype,
            )
            attach_companion(primary, companion)
        return primary

    if vae_type == "kei_vivit2_vae":
        return KeiVivit2VAE.from_pretrained(
            model_name_or_path,
            subfolder=config.get("subfolder"),
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    if vae_type == "flux2_vae":
        from diffusers import AutoencoderKLFlux2

        vae = AutoencoderKLFlux2.from_pretrained(
            model_name_or_path,
            subfolder=config.get("subfolder", "vae"),
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        # Official FLUX.2 configs do not declare scaling/shift factors.
        # Keep an identity transform so generic VAE consumers keep working.
        if getattr(vae.config, "scaling_factor", None) is None:
            vae.config.scaling_factor = 1.0
        if getattr(vae.config, "shift_factor", None) is None:
            vae.config.shift_factor = 0.0
        patch_flux2_vae_for_zimage(vae)
        return vae

    raise ValueError(
        "Unsupported vae.type="
        f"{config.get('type')!r}. Expected one of: autoencoder_kl, wan2_2_vae, "
        "wan2_2_native_vae, omnivae, kei_vivit2_vae, flux2_vae."
    )


def load_scheduler(config: dict):
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        config["model_name_or_path"],
        subfolder=config.get("subfolder"),
        local_files_only=config.get("local_files_only", False),
    )


def load_audio_vae(config: dict):
    vae_type = normalize_vae_type(config.get("type"), default="dac")
    torch_dtype = resolve_dtype(config.get("torch_dtype"), fallback=torch.float32)

    if vae_type == "univae":
        # Drive DiT audio training directly from an OmniVAE training ckpt
        # (Trainer_xxxxx/state_dict.pt). Returns the audio branch (DAC,
        # continuous mode by default); with branch="both" the video branch is
        # also constructed and stashed on the DAC as a dormant CPU companion.
        from omnivae_generation.trainer.vae.univae import (
            attach_companion,
            build_univae_audio_vae,
            build_univae_video_vae,
            load_univae_ckpt,
        )

        branch = str(config.get("branch", "audio")).strip().lower()
        if branch not in {"audio", "both"}:
            raise ValueError(
                f"audio_vae.type='omnivae' expects branch in {{'audio','both'}}, got {branch!r}. "
                "(Use vae.type='omnivae' for video-primary training.)"
            )
        ckpt_path = config.get("model_path") or config.get("model_name_or_path")
        if not ckpt_path:
            raise ValueError(
                "audio_vae.type='omnivae' requires audio_vae.model_path "
                "(pointing at the OmniVAE state_dict.pt or its parent dir)."
            )
        loaded = load_univae_ckpt(
            ckpt_path,
            use_ema=bool(config.get("use_ema", False)),
        )
        overrides = {
            k: config[k]
            for k in (
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
            if k in config and config[k] is not None
        }
        sr = config.get("sample_rate")
        primary = build_univae_audio_vae(
            loaded,
            sample_rate=int(sr) if sr is not None else None,
            overrides=overrides,
            torch_dtype=torch_dtype,
        )
        if branch == "both":
            companion = build_univae_video_vae(
                loaded,
                model_config_override=config.get("video_model_config"),
                qk_norm=str(config.get("video_qk_norm", "auto")),
                torch_dtype=torch_dtype,
            )
            attach_companion(primary, companion)
        return primary

    if vae_type != "dac":
        raise ValueError(
            f"Unsupported audio_vae.type={config.get('type')!r}. Expected one of: 'dac', 'omnivae'."
        )
    model_path = config.get("model_path") or config.get("model_name_or_path")
    if not model_path:
        raise ValueError("audio_vae.type='dac' requires audio_vae.model_path.")
    path = str(Path(model_path).expanduser())

    try:
        vae = DAC.load(path)
    except Exception as e:
        # Fallback for ckpts that are not in audiotools `torch.package` format
        # (e.g. raw state_dict / "generator"-wrapped, extracted from OmniVAE).
        from omnivae_generation.trainer.vae.dac_vae import load_dac_from_path

        overrides = {
            k: config[k]
            for k in (
                "encoder_dim",
                "encoder_rates",
                "latent_dim",
                "decoder_dim",
                "decoder_rates",
                "continuous",
                "n_codebooks",
                "codebook_size",
                "codebook_dim",
                "quantizer_dropout",
            )
            if k in config and config[k] is not None
        }
        sample_rate = int(config.get("sample_rate", 44100))
        logger.warning(
            "DAC.load failed (%s: %s); falling back to manual state_dict loading from %s",
            type(e).__name__,
            e,
            path,
        )
        vae = load_dac_from_path(path, sample_rate=sample_rate, overrides=overrides)

        # Sanity-check yaml-declared shapes (warn-only, do not block training).
        declared_hop = config.get("hop_length")
        if declared_hop is not None and int(declared_hop) != int(vae.hop_length):
            logger.warning(
                "audio_vae.hop_length in yaml (%s) != DAC.hop_length inferred from "
                "ckpt (%s); using inferred.",
                declared_hop,
                int(vae.hop_length),
            )
        declared_latent = config.get("latent_channels") or config.get("latent_dim")
        if declared_latent is not None and int(declared_latent) != int(vae.latent_dim):
            logger.warning(
                "audio_vae latent dim in yaml (%s) != DAC.latent_dim inferred from "
                "ckpt (%s); using inferred.",
                declared_latent,
                int(vae.latent_dim),
            )

    return vae.to(dtype=torch_dtype)


def normalize_predict_target(predict_target: str | None) -> str:
    normalized = "v" if predict_target is None else str(predict_target).strip().lower()
    if normalized not in VALID_PREDICT_TARGETS:
        raise ValueError(
            f"Unsupported predict_target={predict_target!r}. Expected one of: {sorted(VALID_PREDICT_TARGETS)}."
        )
    return normalized


def adapt_model_prediction(
    model_prediction: torch.Tensor,
    noisy_sample: torch.Tensor,
    sigmas: torch.Tensor,
    predict_target: str | None,
) -> torch.Tensor:
    predict_target = normalize_predict_target(predict_target)
    if predict_target == "v":
        return model_prediction

    sigma_tensor = sigmas.to(device=model_prediction.device, dtype=torch.float32)
    sigma_tensor = sigma_tensor.clamp_min(torch.finfo(torch.float32).eps)
    return (model_prediction.float() - noisy_sample.float()) / sigma_tensor


def _sigmas_from_scheduler_timestep(
    scheduler,
    timestep,
    sample: torch.Tensor,
    per_token_timesteps: torch.Tensor | None = None,
) -> torch.Tensor:
    if per_token_timesteps is not None:
        sigma_tensor = per_token_timesteps.to(device=sample.device, dtype=torch.float32)
    elif torch.is_tensor(timestep):
        sigma_tensor = timestep.to(device=sample.device, dtype=torch.float32)
    else:
        sigma_tensor = torch.tensor(timestep, device=sample.device, dtype=torch.float32)

    sigma_tensor = sigma_tensor / float(scheduler.config.num_train_timesteps)
    while sigma_tensor.ndim < sample.ndim:
        sigma_tensor = sigma_tensor.unsqueeze(-1)
    return sigma_tensor


def configure_transformer_timestep_usage(transformer, use_timestep: bool):
    use_timestep = bool(use_timestep)
    original_forward = getattr(transformer, "_laion_original_forward", None)
    if original_forward is not None:
        transformer.forward = original_forward

    transformer._laion_use_timestep = use_timestep
    t_embedder = getattr(transformer, "t_embedder", None)
    if t_embedder is not None:
        t_embedder.requires_grad_(use_timestep)

    for collection_name in ("noise_refiner", "layers"):
        for layer in getattr(transformer, collection_name, ()):
            adaln_module = getattr(layer, "adaLN_modulation", None)
            if adaln_module is not None:
                adaln_module.requires_grad_(use_timestep)
            layer.modulation = bool(use_timestep and adaln_module is not None)

    final_layers = getattr(transformer, "all_final_layer", None)
    if final_layers is not None:
        for final_layer in final_layers.values():
            adaln_module = getattr(final_layer, "adaLN_modulation", None)
            if adaln_module is not None:
                adaln_module.requires_grad_(use_timestep)
            final_layer.modulation = bool(use_timestep and adaln_module is not None)
            final_layer._laion_disable_modulation = not final_layer.modulation

    return transformer


def configure_transformer_prediction_target(transformer, predict_target: str | None):
    transformer._laion_predict_target = normalize_predict_target(predict_target)
    return transformer


def _canonical_transformer_config_value(value):
    if isinstance(value, tuple):
        return [_canonical_transformer_config_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_transformer_config_value(item) for item in value]
    if isinstance(value, float):
        return float(value)
    return value


def build_transformer_init_expected_config(config: dict, *, cap_feat_dim: int) -> dict:
    expected = {
        key: config.get(key)
        for key in TRANSFORMER_INIT_CONFIG_KEYS
        if key != "cap_feat_dim"
    }
    expected["cap_feat_dim"] = int(cap_feat_dim)
    return expected


def _resolve_pretrained_transformer_dir(pretrained_model_name_or_path: str | Path) -> Path:
    from diffusers.models import modeling_utils

    root = Path(pretrained_model_name_or_path).expanduser()
    candidates = [root, root / "transformer", root / "final" / "transformer"]
    checked: list[Path] = []
    for candidate in candidates:
        checked.append(candidate)
        if not candidate.is_dir():
            continue
        if not (candidate / "config.json").is_file():
            continue
        if (candidate / modeling_utils.SAFETENSORS_WEIGHTS_NAME).is_file() or (
            candidate / modeling_utils.WEIGHTS_NAME
        ).is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve a diffusers transformer directory from "
        f"{pretrained_model_name_or_path!r}. Checked: {[str(path) for path in checked]}."
    )


def _validate_pretrained_transformer_config(transformer_dir: Path, expected_config: dict) -> None:
    pretrained_config = json.loads((transformer_dir / "config.json").read_text(encoding="utf-8"))
    mismatches = []
    for key in TRANSFORMER_INIT_CONFIG_KEYS:
        expected = _canonical_transformer_config_value(expected_config.get(key))
        actual = _canonical_transformer_config_value(pretrained_config.get(key))
        if actual != expected:
            mismatches.append(f"{key}: checkpoint={actual!r}, current={expected!r}")

    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise ValueError(
            f"Transformer warm-start config does not match current model for {transformer_dir}:\n  - {joined}"
        )


def load_pretrained_transformer_weights(
    transformer: nn.Module,
    pretrained_model_name_or_path: str | Path,
    *,
    expected_config: dict,
) -> Path:
    from diffusers.models import modeling_utils

    transformer_dir = _resolve_pretrained_transformer_dir(pretrained_model_name_or_path)
    _validate_pretrained_transformer_config(transformer_dir, expected_config)

    weights_file = None
    for weights_name in (modeling_utils.SAFETENSORS_WEIGHTS_NAME, modeling_utils.WEIGHTS_NAME):
        candidate = transformer_dir / weights_name
        if candidate.is_file():
            weights_file = candidate
            break
    if weights_file is None:
        raise FileNotFoundError(f"Could not find transformer weights under {transformer_dir}.")

    checkpoint_state_dict = modeling_utils.load_state_dict(weights_file, map_location="cpu")
    msg = transformer.load_state_dict(checkpoint_state_dict, strict=True)
    missing_keys = list(getattr(msg, "missing_keys", []))
    unexpected_keys = list(getattr(msg, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Unexpected transformer warm-start load_state_dict result: "
            f"missing_keys={missing_keys!r} unexpected_keys={unexpected_keys!r}"
        )
    return transformer_dir


def configure_scheduler_prediction_target(scheduler, predict_target: str | None):
    scheduler._laion_predict_target = normalize_predict_target(predict_target)

    wrapper = getattr(scheduler, "_laion_step_with_optional_prediction_target", None)
    if wrapper is None:
        original_step = scheduler.step

        def _step_with_optional_prediction_target(model_output, timestep, sample, *args, **kwargs):
            current_predict_target = normalize_predict_target(getattr(scheduler, "_laion_predict_target", "v"))
            if current_predict_target == "x0":
                sigmas = _sigmas_from_scheduler_timestep(
                    scheduler,
                    timestep,
                    sample,
                    per_token_timesteps=kwargs.get("per_token_timesteps"),
                )
                model_output = adapt_model_prediction(model_output, sample, sigmas, current_predict_target).to(
                    dtype=model_output.dtype
                )
            return original_step(model_output, timestep, sample, *args, **kwargs)

        scheduler._laion_original_step = original_step
        scheduler._laion_step_with_optional_prediction_target = _step_with_optional_prediction_target
        scheduler.step = _step_with_optional_prediction_target

    return scheduler


def build_transformer(config: dict, cap_feat_dim: int):
    if str(config.get("arch", "")).strip().lower() == "qwen3_vl_dit":
        transformer = Qwen3VLDiffusionTransformer(
            backbone_name_or_path=config["backbone_name_or_path"],
            all_patch_size=tuple(config["all_patch_size"]),
            all_f_patch_size=tuple(config["all_f_patch_size"]),
            in_channels=config["in_channels"],
            out_channels=config.get("out_channels"),
            t_scale=float(config.get("t_scale", 1000.0)),
            predict_target=config.get("predict_target", "v"),
            attn_implementation=config.get("attn_implementation", "flex_attention"),
            backbone_torch_dtype=config.get("backbone_torch_dtype"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            local_files_only=bool(config.get("local_files_only", False)),
            init_from_pretrained_backbone=bool(config.get("init_from_pretrained_backbone", True)),
            text_config=config.get("text_config"),
        )
        configure_transformer_prediction_target(transformer, config.get("predict_target", "v"))
        return transformer

    from omnivae_generation.trainer.runtime_patches import (
        patch_diffusers_zimage_forward_block_stacks,
        patch_diffusers_zimage_real_rope,
    )
    from diffusers.models.transformers.transformer_z_image import FinalLayer, ZImageTransformerBlock

    patch_diffusers_zimage_real_rope()
    patch_diffusers_zimage_forward_block_stacks()

    use_timestep = bool(config.get("use_timestep", True))
    previous_force_disable_modulation = getattr(ZImageTransformerBlock, "_laion_force_disable_modulation", False)
    previous_final_default_modulation = getattr(FinalLayer, "_laion_default_modulation", True)
    ZImageTransformerBlock._laion_force_disable_modulation = not use_timestep
    FinalLayer._laion_default_modulation = use_timestep

    try:
        transformer = ZImageTransformer2DModel(
            all_patch_size=tuple(config["all_patch_size"]),
            all_f_patch_size=tuple(config["all_f_patch_size"]),
            in_channels=config["in_channels"],
            dim=config["dim"],
            n_layers=config["n_layers"],
            n_refiner_layers=config["n_refiner_layers"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            norm_eps=config["norm_eps"],
            qk_norm=config["qk_norm"],
            cap_feat_dim=cap_feat_dim,
            rope_theta=config["rope_theta"],
            t_scale=config["t_scale"],
            axes_dims=config["axes_dims"],
            axes_lens=config["axes_lens"],
        )
    finally:
        ZImageTransformerBlock._laion_force_disable_modulation = previous_force_disable_modulation
        FinalLayer._laion_default_modulation = previous_final_default_modulation
    # diffusers' Z-Image leaves pad tokens as torch.empty(...); make them deterministic and finite.
    torch.nn.init.zeros_(transformer.x_pad_token)
    torch.nn.init.zeros_(transformer.cap_pad_token)
    transformer.x_pad_token.requires_grad_(False)
    transformer.cap_pad_token.requires_grad_(False)
    if getattr(transformer, "siglip_pad_token", None) is not None:
        torch.nn.init.zeros_(transformer.siglip_pad_token)
        transformer.siglip_pad_token.requires_grad_(False)
    configure_transformer_timestep_usage(transformer, use_timestep)
    configure_transformer_prediction_target(transformer, config.get("predict_target", "v"))
    return transformer


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _encode_prompt_batch(
    prompts: List[str],
    tokenizer,
    text_encoder,
    device: torch.device,
    max_sequence_length: int,
) -> List[torch.Tensor]:
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device).bool()

    hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeddings = []
    for batch_index in range(hidden_states.shape[0]):
        prompt_embeddings.append(hidden_states[batch_index][attention_mask[batch_index]])
    return prompt_embeddings


def _encode_prompt_tensor_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    text_encoder,
    device: torch.device,
) -> List[torch.Tensor]:
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError(
            "Pre-tokenized prompt tensors must be rank-2 "
            f"(got input_ids={tuple(input_ids.shape)}, attention_mask={tuple(attention_mask.shape)})."
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "Pre-tokenized prompt input_ids and attention_mask shapes must match "
            f"(got {tuple(input_ids.shape)} vs {tuple(attention_mask.shape)})."
        )

    input_ids = input_ids.to(device=device, dtype=torch.long, non_blocking=True)
    attention_mask = attention_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeddings = []
    for batch_index in range(hidden_states.shape[0]):
        prompt_embeddings.append(hidden_states[batch_index][attention_mask[batch_index]])
    return prompt_embeddings


def _get_prompt_embedding_cache(tokenizer, text_encoder, max_sequence_length: int) -> _PromptEmbeddingLRUCache:
    cache_key = (id(tokenizer), int(max_sequence_length))
    existing_cache_key = getattr(text_encoder, "_laion_prompt_embedding_cache_key", None)
    cache = getattr(text_encoder, "_laion_prompt_embedding_cache", None)
    if cache is None or existing_cache_key != cache_key:
        cache = _PromptEmbeddingLRUCache()
        text_encoder._laion_prompt_embedding_cache = cache
        text_encoder._laion_prompt_embedding_cache_key = cache_key
    return cache


def encode_prompts(
    prompts: List[str],
    tokenizer,
    text_encoder,
    device: torch.device,
    max_sequence_length: int,
    cache_enabled: bool = True,
):
    use_cache = bool(cache_enabled) and not torch.is_grad_enabled() and not text_encoder.training
    if not use_cache:
        return _encode_prompt_batch(prompts, tokenizer, text_encoder, device, max_sequence_length)

    cache = _get_prompt_embedding_cache(tokenizer, text_encoder, max_sequence_length)
    prompt_embeddings: List[Optional[torch.Tensor]] = [None] * len(prompts)
    missing_prompts: List[str] = []
    missing_prompt_indices: dict[str, List[int]] = {}

    for prompt_index, prompt in enumerate(prompts):
        cached_embedding = cache.get(prompt)
        if cached_embedding is not None:
            prompt_embeddings[prompt_index] = cached_embedding.to(device)
            continue

        if prompt not in missing_prompt_indices:
            missing_prompts.append(prompt)
            missing_prompt_indices[prompt] = []
        missing_prompt_indices[prompt].append(prompt_index)

    if missing_prompts:
        encoded_missing_prompts = _encode_prompt_batch(
            missing_prompts,
            tokenizer,
            text_encoder,
            device,
            max_sequence_length,
        )
        for prompt, encoded_prompt in zip(missing_prompts, encoded_missing_prompts):
            cache.put(prompt, encoded_prompt)
            for prompt_index in missing_prompt_indices[prompt]:
                prompt_embeddings[prompt_index] = encoded_prompt

    return [embedding for embedding in prompt_embeddings if embedding is not None]


def encode_tokenized_prompts(
    prompts: List[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    text_encoder,
    device: torch.device,
    max_sequence_length: int,
    cache_enabled: bool = True,
):
    if len(prompts) != int(input_ids.shape[0]):
        raise ValueError(
            "Pre-tokenized prompt batch size must match prompts length "
            f"(got {int(input_ids.shape[0])} vs {len(prompts)})."
        )
    use_cache = bool(cache_enabled) and not torch.is_grad_enabled() and not text_encoder.training
    if not use_cache:
        return _encode_prompt_tensor_batch(input_ids, attention_mask, text_encoder, device)

    cache = _get_prompt_embedding_cache(tokenizer, text_encoder, max_sequence_length)
    prompt_embeddings: List[Optional[torch.Tensor]] = [None] * len(prompts)
    missing_prompts: List[str] = []
    missing_prompt_indices: dict[str, List[int]] = {}
    missing_source_indices: list[int] = []

    for prompt_index, prompt in enumerate(prompts):
        cached_embedding = cache.get(prompt)
        if cached_embedding is not None:
            prompt_embeddings[prompt_index] = cached_embedding.to(device)
            continue

        if prompt not in missing_prompt_indices:
            missing_prompts.append(prompt)
            missing_prompt_indices[prompt] = []
            missing_source_indices.append(prompt_index)
        missing_prompt_indices[prompt].append(prompt_index)

    if missing_prompts:
        source_indices = torch.tensor(missing_source_indices, dtype=torch.long, device=input_ids.device)
        encoded_missing_prompts = _encode_prompt_tensor_batch(
            input_ids.index_select(0, source_indices),
            attention_mask.index_select(0, source_indices),
            text_encoder,
            device,
        )
        for prompt, encoded_prompt in zip(missing_prompts, encoded_missing_prompts):
            cache.put(prompt, encoded_prompt)
            for prompt_index in missing_prompt_indices[prompt]:
                prompt_embeddings[prompt_index] = encoded_prompt

    return [embedding for embedding in prompt_embeddings if embedding is not None]


def encode_images_to_latents(
    pixel_values: torch.Tensor,
    vae,
    *,
    update_stats: bool = False,
    sample_mode: str | None = None,
) -> torch.Tensor:
    encoded = vae.encode(pixel_values)
    if sample_mode is None:
        sample_mode = (
            "argmax"
            if not vae_encode_returns_training_latents(vae)
            and getattr(getattr(vae, "config", None), "_class_name", "") == "AutoencoderKLFlux2"
            else "sample"
        )
    latents = retrieve_latents(encoded, sample_mode=sample_mode)

    if vae_uses_training_layout(vae):
        if vae_encode_returns_training_latents(vae):
            return latents
        return raw_latents_to_training_layout(latents, vae, update_stats=update_stats)

    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(vae.config, "shift_factor", None)
    if shift_factor is None:
        latents = latents * scaling_factor
    else:
        latents = (latents - shift_factor) * scaling_factor
    return latents


def encode_audio_to_latents(audio: torch.Tensor, audio_vae) -> torch.Tensor:
    encoded = audio_vae.encode(audio)
    latents = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
    if hasattr(latents, "mode"):
        latents = latents.mode()
    if not torch.is_tensor(latents):
        raise TypeError(f"Audio VAE encode returned unsupported object: {type(latents)!r}")
    if latents.ndim != 3:
        raise ValueError(f"Expected audio latents with shape [B, C, T], got {tuple(latents.shape)}")
    return latents


def decode_latents_to_images(latents: torch.Tensor, vae) -> torch.Tensor:
    if vae_uses_training_layout(vae):
        return vae.decode(latents, return_dict=False)[0]
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(vae.config, "shift_factor", None)
    if shift_factor is None:
        latents = latents / scaling_factor
    else:
        latents = latents / scaling_factor + shift_factor
    return vae.decode(latents, return_dict=False)[0]


def build_validation_pipeline(transformer, tokenizer, text_encoder, vae, scheduler):
    inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    configure_scheduler_prediction_target(
        inference_scheduler,
        getattr(transformer, "_laion_predict_target", "v"),
    )
    return ZImagePipeline(
        scheduler=inference_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )


def save_checkpoint_artifacts(
    output_dir: str | Path,
    transformer,
    tokenizer,
    scheduler,
    metadata: dict,
    text_encoder=None,
    vae=None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(output_path / "transformer", safe_serialization=True)
    tokenizer.save_pretrained(output_path / "tokenizer")
    scheduler.save_pretrained(output_path / "scheduler")
    if text_encoder is not None:
        text_encoder.save_pretrained(output_path / "text_encoder", safe_serialization=True)
    if vae is not None:
        vae_output_path = output_path / "vae"
        vae.save_pretrained(vae_output_path, safe_serialization=True)
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
