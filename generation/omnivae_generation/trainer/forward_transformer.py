from __future__ import annotations

import torch

from omnivae_generation.trainer.qwen3_vl_dit import Qwen3VLDiffusionTransformer


def build_forward_transformer(transformer, transformer_model, train_patch_size: int, train_f_patch_size: int):
    def forward_transformer(
        noisy_latents: torch.Tensor,
        model_timesteps: torch.Tensor,
        prompt_embeds,
    ):
        squeeze_frame_dim = False
        squeeze_audio_spatial_dims = False
        if noisy_latents.ndim == 3:
            model_input = noisy_latents.unsqueeze(-1).unsqueeze(-1)
            squeeze_audio_spatial_dims = True
        elif noisy_latents.ndim == 4:
            model_input = noisy_latents.unsqueeze(2)
            squeeze_frame_dim = True
        elif noisy_latents.ndim == 5:
            model_input = noisy_latents
        else:
            raise ValueError(
                "Expected latents with 3 dims [B, C, T], 4 dims [B, C, H, W], "
                f"or 5 dims [B, C, T, H, W], got ndim={noisy_latents.ndim}."
            )

        if isinstance(transformer_model, Qwen3VLDiffusionTransformer):
            outputs = transformer(
                list(model_input.unbind(dim=0)),
                model_timesteps,
                prompt_embeds,
                return_dict=False,
                patch_size=train_patch_size,
                f_patch_size=train_f_patch_size,
            )
            model_output = outputs[0]
            model_pred = torch.stack([item.float() for item in model_output], dim=0)
            if squeeze_frame_dim:
                model_pred = model_pred.squeeze(2)
            if squeeze_audio_spatial_dims:
                model_pred = model_pred.squeeze(-1).squeeze(-1)
            return model_pred, None

        packed_inputs = transformer_model.prepare_dense_inputs(
            list(model_input.unbind(dim=0)),
            prompt_embeds,
            train_patch_size,
            train_f_patch_size,
        )
        outputs = transformer(
            packed_inputs["x"],
            model_timesteps,
            packed_inputs["cap_feats"],
            return_dict=False,
            x_size=packed_inputs["x_size"],
            x_freqs=packed_inputs["x_freqs"],
            cap_freqs=packed_inputs["cap_freqs"],
            x_mask=packed_inputs["x_mask"],
            cap_mask=packed_inputs["cap_mask"],
            siglip_feats=packed_inputs["siglip_feats"],
            siglip_freqs=packed_inputs["siglip_freqs"],
            siglip_mask=packed_inputs["siglip_mask"],
            x_noise_tensor=packed_inputs["x_noise_tensor"],
            cap_noise_tensor=packed_inputs["cap_noise_tensor"],
            siglip_noise_tensor=packed_inputs["siglip_noise_tensor"],
            omni_mode=packed_inputs["omni_mode"],
            patch_size=train_patch_size,
            f_patch_size=train_f_patch_size,
        )
        model_output = outputs[0]
        model_pred = torch.stack([item.float() for item in model_output], dim=0)
        if squeeze_frame_dim:
            model_pred = model_pred.squeeze(2)
        if squeeze_audio_spatial_dims:
            model_pred = model_pred.squeeze(-1).squeeze(-1)
        return model_pred, None

    return forward_transformer
