from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_z_image import TimestepEmbedder
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from omnivae_generation.trainer.runtime_patches import _unpatchify_compact_x_tokens


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def is_qwen3_vl_dit_arch(config: dict) -> bool:
    transformer_cfg = config.get("transformer", {})
    return str(transformer_cfg.get("arch", "")).strip().lower() == "qwen3_vl_dit"


def load_qwen3_vl_text_config(
    backbone_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> Qwen3VLTextConfig:
    backbone_config = AutoConfig.from_pretrained(
        backbone_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    text_config = backbone_config.get_text_config()
    if not isinstance(text_config, Qwen3VLTextConfig):
        raise TypeError(
            "Expected Qwen3VLTextConfig from the configured backbone, got "
            f"{type(text_config).__name__}."
        )
    return text_config


def tokenize_prompt_payloads(
    prompts: list[str],
    tokenizer,
    *,
    device: torch.device,
    max_sequence_length: int,
) -> list[dict[str, torch.Tensor]]:
    prompt_payloads: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_sequence_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        attention_mask = encoded.attention_mask[0].to(dtype=torch.bool)
        valid_length = int(attention_mask.sum().item())
        input_ids = encoded.input_ids[0, :valid_length].to(device=device, dtype=torch.long)
        prompt_payloads.append({"input_ids": input_ids})
    return prompt_payloads


def prompt_token_tensors_to_payloads(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
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

    input_ids = input_ids.to(dtype=torch.long)
    attention_mask = attention_mask.to(dtype=torch.bool)
    prompt_payloads: list[dict[str, torch.Tensor]] = []
    for batch_index in range(input_ids.shape[0]):
        valid_length = int(attention_mask[batch_index].sum().item())
        prompt_payloads.append(
            {
                "input_ids": input_ids[batch_index, :valid_length].to(device=device, dtype=torch.long),
            }
        )
    return prompt_payloads


def _resolve_optional_dtype(name: str | None):
    if name is None:
        return None
    normalized = str(name).strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype specifier for Qwen3-VL backbone loading: {name!r}")
    return _DTYPE_MAP[normalized]


def _patchify_image(image: torch.Tensor, patch_size: int, f_patch_size: int):
    # Match diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel._patchify_image.
    p_h = patch_size
    p_w = patch_size
    p_f = f_patch_size
    channels, frames, height, width = image.size()
    frame_tokens = frames // p_f
    height_tokens = height // p_h
    width_tokens = width // p_w
    image = image.view(channels, frame_tokens, p_f, height_tokens, p_h, width_tokens, p_w)
    image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
        frame_tokens * height_tokens * width_tokens,
        p_f * p_h * p_w * channels,
    )
    return image, (frames, height, width), (frame_tokens, height_tokens, width_tokens)


def _build_text_like_position_ids(length: int, *, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.long)
    return positions.view(1, -1).expand(3, -1)


def _build_qwen3_vl_vision_position_ids(
    start_position: int,
    grid_thw: tuple[int, int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    # Match Qwen3VLModel.get_vision_position_ids, but we use a direct latent-patch grid
    # with no spatial/temporal merge because FLUX latent patches already define the token grid.
    grid_t, grid_h, grid_w = (int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2]))
    image_seq_length = grid_t * grid_h * grid_w
    position_width = torch.arange(start_position, start_position + grid_w, device=device).repeat(grid_h * grid_t)
    position_height = torch.arange(start_position, start_position + grid_h, device=device).repeat_interleave(
        grid_w * grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _extract_token_segment(
    hidden_states: torch.Tensor,
    start_offsets: torch.Tensor,
    lengths: torch.Tensor,
    max_segment_length: int,
) -> torch.Tensor:
    batch_size, _, hidden_dim = hidden_states.shape
    local_positions = torch.arange(max_segment_length, device=hidden_states.device).unsqueeze(0)
    gather_positions = start_offsets.unsqueeze(1) + local_positions
    gathered = hidden_states.gather(1, gather_positions.unsqueeze(-1).expand(batch_size, -1, hidden_dim))
    gathered_mask = local_positions < lengths.unsqueeze(1)
    return gathered * gathered_mask.unsqueeze(-1).to(gathered.dtype)


def _make_non_text_or_mask(non_text_mask: torch.Tensor):
    def _mask_mod(batch_idx, _head_idx, q_idx, kv_idx):
        q_non_text = non_text_mask[batch_idx, q_idx]
        kv_non_text = non_text_mask[batch_idx, kv_idx]
        return q_non_text | kv_non_text

    return _mask_mod


class Qwen3VLDiffusionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        *,
        backbone_name_or_path: str,
        all_patch_size: tuple[int, ...] = (2,),
        all_f_patch_size: tuple[int, ...] = (1,),
        in_channels: int = 16,
        out_channels: int | None = None,
        t_scale: float = 1000.0,
        predict_target: str = "v",
        attn_implementation: str = "flex_attention",
        backbone_torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        init_from_pretrained_backbone: bool = True,
        text_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if text_config is None:
            loaded_text_config = load_qwen3_vl_text_config(
                backbone_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            text_config_dict = loaded_text_config.to_dict()
        else:
            text_config_dict = deepcopy(text_config)

        self.register_to_config(text_config=text_config_dict, init_from_pretrained_backbone=False)
        text_config_dict["_attn_implementation"] = attn_implementation
        resolved_backbone_dtype = _resolve_optional_dtype(backbone_torch_dtype)
        self.text_config = Qwen3VLTextConfig(**text_config_dict)
        self.language_model = Qwen3VLTextModel(self.text_config)
        self.language_model.config._attn_implementation = attn_implementation

        if init_from_pretrained_backbone:
            pretrained_kwargs = {
                "trust_remote_code": trust_remote_code,
                "local_files_only": local_files_only,
                "low_cpu_mem_usage": True,
            }
            if resolved_backbone_dtype is not None:
                pretrained_kwargs["dtype"] = resolved_backbone_dtype
            pretrained_backbone = AutoModel.from_pretrained(
                backbone_name_or_path,
                **pretrained_kwargs,
            )
            if not hasattr(pretrained_backbone, "language_model"):
                raise AttributeError(
                    f"Backbone {backbone_name_or_path!r} does not expose a `language_model` module."
                )
            self.language_model.load_state_dict(pretrained_backbone.language_model.state_dict())
            del pretrained_backbone

        self.hidden_size = int(self.text_config.hidden_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(self.in_channels if out_channels is None else out_channels)
        self.all_patch_size = tuple(int(item) for item in all_patch_size)
        self.all_f_patch_size = tuple(int(item) for item in all_f_patch_size)
        self.t_scale = float(t_scale)
        self._laion_predict_target = str(predict_target).strip().lower()
        self._laion_forward_compilation_enabled = False

        all_patch_embed = {}
        all_patch_out = {}
        for patch_size, f_patch_size in zip(self.all_patch_size, self.all_f_patch_size):
            patch_dim = f_patch_size * patch_size * patch_size * self.in_channels
            out_patch_dim = f_patch_size * patch_size * patch_size * self.out_channels
            all_patch_embed[f"{patch_size}-{f_patch_size}"] = nn.Linear(patch_dim, self.hidden_size, bias=True)
            all_patch_out[f"{patch_size}-{f_patch_size}"] = nn.Linear(self.hidden_size, out_patch_dim, bias=True)

        self.all_patch_embed = nn.ModuleDict(all_patch_embed)
        self.all_patch_out = nn.ModuleDict(all_patch_out)
        self.timestep_embedder = TimestepEmbedder(self.hidden_size)
        if resolved_backbone_dtype is not None:
            self.to(dtype=resolved_backbone_dtype)

    def enable_gradient_checkpointing(self):
        self.language_model.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        self.language_model.gradient_checkpointing_disable()

    def set_forward_compilation(self, enabled: bool) -> None:
        self._laion_forward_compilation_enabled = bool(enabled)

    def is_forward_compilation_enabled(self) -> bool:
        return bool(self._laion_forward_compilation_enabled)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def _pack_inputs(
        self,
        all_image: list[torch.Tensor],
        timesteps: torch.Tensor,
        prompt_payloads: list[dict[str, torch.Tensor]],
        *,
        patch_size: int,
        f_patch_size: int,
    ):
        device = timesteps.device
        dtype = self.all_patch_embed[f"{patch_size}-{f_patch_size}"].weight.dtype

        seq_embeddings: list[torch.Tensor] = []
        seq_position_ids: list[torch.Tensor] = []
        text_mask_rows: list[torch.Tensor] = []
        image_sizes: list[tuple[int, int, int]] = []
        patch_starts: list[int] = []
        patch_lengths: list[int] = []

        for image, timestep, prompt_payload in zip(all_image, timesteps, prompt_payloads):
            input_ids = prompt_payload["input_ids"].to(device=device, dtype=torch.long)
            text_embeds = self.language_model.embed_tokens(input_ids).to(dtype=dtype)

            timestep_token = self.timestep_embedder(timestep.view(1) * self.t_scale).to(dtype=dtype)

            image_patches, image_size, patch_grid = _patchify_image(image.to(device=device, dtype=dtype), patch_size, f_patch_size)
            patch_tokens = self.all_patch_embed[f"{patch_size}-{f_patch_size}"](image_patches)

            text_length = int(text_embeds.shape[0])
            patch_length = int(patch_tokens.shape[0])
            patch_start = text_length + 1

            prefix_tokens = torch.cat([text_embeds, timestep_token], dim=0)
            prefix_position_ids = _build_text_like_position_ids(text_length + 1, device=device)
            patch_position_ids = _build_qwen3_vl_vision_position_ids(
                patch_start,
                patch_grid,
                device=device,
            )

            seq_embeddings.append(torch.cat([prefix_tokens, patch_tokens], dim=0))
            seq_position_ids.append(torch.cat([prefix_position_ids, patch_position_ids], dim=1))

            text_mask = torch.zeros((text_length + 1 + patch_length,), dtype=torch.bool, device=device)
            if text_length > 0:
                text_mask[:text_length] = True
            text_mask_rows.append(text_mask)

            image_sizes.append(image_size)
            patch_starts.append(patch_start)
            patch_lengths.append(patch_length)

        max_seq_len = max(int(row.shape[0]) for row in seq_embeddings)
        batch_size = len(seq_embeddings)

        inputs_embeds = torch.zeros((batch_size, max_seq_len, self.hidden_size), dtype=dtype, device=device)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
        position_ids = torch.zeros((3, batch_size, max_seq_len), dtype=torch.long, device=device)
        text_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)

        for index, (embeds, pos_ids, text_mask_row) in enumerate(zip(seq_embeddings, seq_position_ids, text_mask_rows)):
            seq_len = int(embeds.shape[0])
            inputs_embeds[index, :seq_len] = embeds
            attention_mask[index, :seq_len] = True
            position_ids[:, index, :seq_len] = pos_ids
            text_mask[index, :seq_len] = text_mask_row

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "non_text_mask": attention_mask & (~text_mask),
            "image_sizes": image_sizes,
            "patch_starts": torch.tensor(patch_starts, dtype=torch.long, device=device),
            "patch_lengths": torch.tensor(patch_lengths, dtype=torch.long, device=device),
        }

    def _run_language_backbone(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        non_text_mask: torch.Tensor,
    ):
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
        causal_mask = create_causal_mask(
            config=self.language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            or_mask_function=_make_non_text_or_mask(non_text_mask),
        )

        hidden_states = inputs_embeds
        position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids)

        for layer_index, decoder_layer in enumerate(self.language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.language_model.norm(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        timesteps: torch.Tensor,
        prompt_payloads: list[dict[str, torch.Tensor]],
        *,
        return_dict: bool = True,
        patch_size: int | None = None,
        f_patch_size: int | None = None,
        **_kwargs,
    ):
        if patch_size is None:
            patch_size = int(self.all_patch_size[0])
        if f_patch_size is None:
            f_patch_size = int(self.all_f_patch_size[0])
        if patch_size not in self.all_patch_size or f_patch_size not in self.all_f_patch_size:
            raise ValueError(
                f"Unsupported patch sizes patch_size={patch_size}, f_patch_size={f_patch_size}; "
                f"expected one of {list(zip(self.all_patch_size, self.all_f_patch_size))}."
            )

        packed = self._pack_inputs(
            hidden_states,
            timesteps,
            prompt_payloads,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )
        hidden_states = self._run_language_backbone(
            inputs_embeds=packed["inputs_embeds"],
            attention_mask=packed["attention_mask"],
            position_ids=packed["position_ids"],
            non_text_mask=packed["non_text_mask"],
        )

        max_patch_len = int(packed["patch_lengths"].max().item())
        patch_hidden_states = _extract_token_segment(
            hidden_states,
            packed["patch_starts"],
            packed["patch_lengths"],
            max_patch_len,
        )
        patch_tokens = self.all_patch_out[f"{patch_size}-{f_patch_size}"](patch_hidden_states)
        outputs = _unpatchify_compact_x_tokens(
            patch_tokens,
            packed["image_sizes"],
            patch_size,
            f_patch_size,
            self.out_channels,
        )

        if return_dict:
            return {"sample": outputs}
        return (outputs,)
