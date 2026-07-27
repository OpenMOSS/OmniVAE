"""
Audio-Video VAE 联合模型

将 Video VAE (WanVAE) 和 Audio VAE (DAC) 包装在同一个 nn.Module 中，
支持分别加载预训练权重。
"""

import logging
import os
import zlib
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any

from omnivae.models.causalvideovae.model import WanVAEModel, WanVAE22Model

VIDEO_VAE_CLASSES = {
    "WanVAE": WanVAEModel,
    "WanVAE22": WanVAE22Model,
}

import sys
from omnivae.models.audio_vae_dac.dac import DAC as DACModel
from .contrastive import IntraSegCrossAttnHead, LatentAVContrastiveHead
from omnivae.train.av_vae.distill_loss import (
    ImageDistillProjector, VideoDistillProjector, AudioDistillProjector,
)


class _GradScale(torch.autograd.Function):
    """Identity in forward; scales the gradient by a constant in backward.

    Used to limit the magnitude of gradients flowing from the contrastive
    head back into shared encoders, without fully detaching the signal.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    if x is None:
        return x
    if scale == 1.0:
        return x
    if scale == 0.0:
        return x.detach()
    return _GradScale.apply(x, scale)


class AudioVideoVAE(nn.Module):
    """联合 Video VAE 和 Audio VAE 的模型"""

    def __init__(
        self,
        video_vae_kwargs: Dict[str, Any] = None,
        audio_vae_kwargs: Dict[str, Any] = None,
        contrastive_kwargs: Optional[Dict[str, Any]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        distill_kwargs: Optional[Dict[str, Any]] = None,
        skip_video_vae: bool = False,
        skip_audio_vae: bool = False,
    ):
        super().__init__()

        video_vae_kwargs = video_vae_kwargs or {}
        audio_vae_kwargs = audio_vae_kwargs or {}

        self.video_vae_kwargs = video_vae_kwargs
        self.audio_vae_kwargs = audio_vae_kwargs
        self.contrastive_kwargs = contrastive_kwargs or {}

        # === Video VAE ===
        self.video_vae = None
        self.video_temporal_downsample = 4
        self.video_spatial_downsample = 8
        self.video_latent_dim_actual = None

        if not skip_video_vae:
            video_model_name = video_vae_kwargs.get('model_name', 'WanVAE')
            video_model_config = video_vae_kwargs.get('model_config')
            video_pretrained_path = video_vae_kwargs.get('pretrained_model_name_or_path')
            video_qk_norm = video_vae_kwargs.get('qk_norm', False)
            if isinstance(video_pretrained_path, str) and video_pretrained_path.lower() in ("none", "null", ""):
                video_pretrained_path = None

            VideoVAEClass = VIDEO_VAE_CLASSES.get(video_model_name)
            if VideoVAEClass is None:
                raise ValueError(f"Unknown video_model_name '{video_model_name}'. Available: {list(VIDEO_VAE_CLASSES.keys())}")
            logging.info(f"Initializing Video VAE: {video_model_name}, {video_pretrained_path}")
            if video_pretrained_path:
                import os
                # Support three loading modes:
                #   1) HuggingFace / diffusers directory (contains config.json):
                #      → VideoVAEClass.from_pretrained(dir)
                #   2) Single checkpoint file (.pth/.ckpt/.pt/.safetensors):
                #      → build from model_config, then init_from_ckpt(file)
                #      (model_config is required to know the architecture)
                #   3) Directory containing config.json + a weight file:
                #      → same as (2) but auto-discover the weight file
                pretrained_path_str = str(video_pretrained_path)
                is_file = os.path.isfile(pretrained_path_str)
                ckpt_exts = (".pth", ".ckpt", ".pt", ".safetensors", ".bin")
                if is_file and pretrained_path_str.lower().endswith(ckpt_exts):
                    if not video_model_config:
                        # Try auto-discover config.json next to the checkpoint
                        _auto_cfg = os.path.join(os.path.dirname(pretrained_path_str), "config.json")
                        if os.path.exists(_auto_cfg):
                            video_model_config = _auto_cfg
                        else:
                            raise ValueError(
                                f"pretrained_model_name_or_path '{pretrained_path_str}' is a single "
                                f"checkpoint file but no model_config was provided and no config.json "
                                f"was found alongside it. Please set model.video.model_config in YAML."
                            )
                    logging.info(
                        f"Loading Video VAE from single ckpt: config={video_model_config}, "
                        f"ckpt={pretrained_path_str}"
                    )
                    if isinstance(video_model_config, str) and os.path.exists(video_model_config):
                        config_dict = VideoVAEClass.load_config(video_model_config)
                    elif isinstance(video_model_config, dict):
                        config_dict = video_model_config
                    else:
                        raise ValueError(
                            f"video_model_config must be a valid path or dict, got: "
                            f"{type(video_model_config)} / {video_model_config}"
                        )
                    if video_qk_norm:
                        config_dict['qk_norm'] = True
                    self.video_vae = VideoVAEClass.from_config(config_dict)
                    if not hasattr(self.video_vae, "init_from_ckpt"):
                        raise AttributeError(
                            f"{VideoVAEClass.__name__} does not implement init_from_ckpt(). "
                            f"Cannot load a single .pth/.ckpt file. Use a HF-format directory instead."
                        )
                    self.video_vae.init_from_ckpt(pretrained_path_str)
                else:
                    self.video_vae = VideoVAEClass.from_pretrained(pretrained_path_str)
            elif video_model_config:
                import os
                if isinstance(video_model_config, str) and os.path.exists(video_model_config):
                    config_dict = VideoVAEClass.load_config(video_model_config)
                    if video_qk_norm:
                        config_dict['qk_norm'] = True
                    self.video_vae = VideoVAEClass.from_config(config_dict)
                elif isinstance(video_model_config, dict):
                    if video_qk_norm:
                        video_model_config['qk_norm'] = True
                    self.video_vae = VideoVAEClass.from_config(video_model_config)
                else:
                    raise ValueError(f"video_model_config must be a path or dict, got: {type(video_model_config)}")
            else:
                self.video_vae = VideoVAEClass(qk_norm=video_qk_norm)

            self.video_temporal_downsample = getattr(self.video_vae, 'temporal_compress_factor', 4)
            self.video_spatial_downsample = getattr(self.video_vae, 'spatial_compress_factor', 8)
            self.video_latent_dim_actual = (
                getattr(self.video_vae, 'z_dim', None)
                or getattr(self.video_vae, 'embed_dim', None)
                or video_vae_kwargs.get('z_dim')
                or video_vae_kwargs.get('embed_dim')
            )
        else:
            logging.info("Video VAE: skipped (not needed by any active loss)")

        # === Audio VAE ===
        self.audio_vae = None
        self.audio_downsample = 1
        self.audio_latent_dim_actual = None

        audio_sample_rate = audio_vae_kwargs.get('sample_rate', audio_vae_kwargs.get('audio_sample_rate', 24000))
        self.audio_sample_rate = audio_sample_rate

        if not skip_audio_vae:
            audio_encoder_dim = audio_vae_kwargs.get('encoder_dim', 64)
            audio_encoder_rates = audio_vae_kwargs.get('encoder_rates', [2, 4, 8, 8])
            audio_latent_dim = audio_vae_kwargs.get('latent_dim')
            audio_decoder_dim = audio_vae_kwargs.get('decoder_dim', 1536)
            audio_decoder_rates = audio_vae_kwargs.get('decoder_rates', [8, 8, 4, 2])
            audio_n_codebooks = audio_vae_kwargs.get('n_codebooks', 9)
            audio_codebook_size = audio_vae_kwargs.get('codebook_size', 1024)
            audio_codebook_dim = audio_vae_kwargs.get('codebook_dim', 8)
            audio_quantizer_dropout = audio_vae_kwargs.get('quantizer_dropout', False)
            audio_continuous = audio_vae_kwargs.get('continuous', True)
            audio_pretrained_path = audio_vae_kwargs.get('pretrained_model_name_or_path')

            logging.info("Initializing Audio VAE (DAC continuous mode)")
            self.audio_vae = DACModel(
                encoder_dim=audio_encoder_dim,
                encoder_rates=audio_encoder_rates,
                latent_dim=audio_latent_dim,
                decoder_dim=audio_decoder_dim,
                decoder_rates=audio_decoder_rates,
                n_codebooks=audio_n_codebooks,
                codebook_size=audio_codebook_size,
                codebook_dim=audio_codebook_dim,
                quantizer_dropout=audio_quantizer_dropout,
                sample_rate=audio_sample_rate,
                continuous=audio_continuous,
            )
            if audio_pretrained_path:
                self._load_audio_pretrained(audio_pretrained_path)

            self.audio_downsample = self.audio_vae.hop_length
            self.audio_latent_dim_actual = self.audio_vae.latent_dim
        else:
            logging.info("Audio VAE: skipped (not needed by any active loss)")

        # === Contrastive Head ===
        self.use_contrastive = bool(self.contrastive_kwargs.get("enabled", False))
        self.contrastive_use_mean = bool(self.contrastive_kwargs.get("contrastive_use_mean", True))
        self.contrastive_grad_scale_video = float(
            self.contrastive_kwargs.get("contrastive_grad_scale_video", 1.0)
        )
        self.contrastive_grad_scale_audio = float(
            self.contrastive_kwargs.get("contrastive_grad_scale_audio", 1.0)
        )
        self.contrastive_head = None
        if self.use_contrastive:
            if self.video_latent_dim_actual is None:
                raise ValueError("Failed to infer video latent dim for contrastive training.")
            contrastive_cfg = dict(self.contrastive_kwargs)
            for _pop_key in ("enabled", "contrastive_use_mean",
                             "contrastive_grad_scale_video", "contrastive_grad_scale_audio",
                             "val_segment_num_negatives",
                             "val_segment_num_negative_videos", "val_global_num_negatives"):
                contrastive_cfg.pop(_pop_key, None)
            ctype = str(contrastive_cfg.pop("contrastive_type", "latent_seg"))

            _base_kwargs = dict(
                video_latent_dim=int(self.video_latent_dim_actual),
                audio_latent_dim=int(self.audio_latent_dim_actual),
                skip_first_video_latent_frame=True,
                video_temporal_compress_factor=int(self.video_temporal_downsample),
            )

            if ctype == "latent_seg":
                self.contrastive_head = LatentAVContrastiveHead(
                    **_base_kwargs, **contrastive_cfg,
                )
            elif ctype == "intra_seg_xattn":
                # Whitelist keys accepted by IntraSegCrossAttnHead so users can
                # keep the existing latent_seg YAML and only flip contrastive_type.
                _allowed_keys = {
                    "embed_dim", "nhead",
                    "self_attn_layers", "cross_attn_layers",
                    "spatial_merge_factor",
                    "max_spatial_h", "max_spatial_w",
                    "max_audio_tokens_per_seg",
                    "dim_feedforward", "dropout",
                    "init_scale", "clamp_scale_min", "clamp_scale_max",
                    "gather_for_loss",
                    "num_negatives", "num_negative_videos",
                    "same_long_video_priority",
                    "same_long_video_num_negatives",
                    "num_negatives_with_sibling",
                    "num_negatives_no_sibling",
                    "qk_norm",
                }
                filtered_cfg = {k: v for k, v in contrastive_cfg.items() if k in _allowed_keys}
                ignored = sorted(set(contrastive_cfg.keys()) - _allowed_keys)
                if ignored:
                    logging.info(
                        f"IntraSegCrossAttnHead: ignoring unused contrastive keys {ignored}"
                    )
                self.contrastive_head = IntraSegCrossAttnHead(
                    **_base_kwargs, **filtered_cfg,
                )
            else:
                raise ValueError(
                    f"Unknown contrastive_type: {ctype!r} "
                    f"(expected 'latent_seg' or 'intra_seg_xattn')."
                )

        # === LLM Caption Head ===
        self.llm_kwargs = llm_kwargs or {}
        self.use_llm = bool(self.llm_kwargs.get("enabled", False))
        self.llm_use_mean = bool(self.llm_kwargs.get("llm_use_mean", True))
        self.llm_caption_head = None
        if self.use_llm:
            raise NotImplementedError(
                "LLM caption training is not part of the public OmniVAE "
                "training boundary. Set loss.use_llm_caption=false."
            )

        # === Semantic Distillation Projectors ===
        self.distill_kwargs = distill_kwargs or {}
        self.use_distill = bool(self.distill_kwargs.get("enabled", False))
        self.distill_use_sampled = bool(self.distill_kwargs.get("distill_use_sampled", False))
        self.image_distill_proj: Optional[ImageDistillProjector] = None
        self.video_distill_proj: Optional[VideoDistillProjector] = None
        self.audio_distill_proj: Optional[AudioDistillProjector] = None
        if self.use_distill:
            proj_type = self.distill_kwargs.get("distill_proj_type", "conv")
            use_conv3d = bool(self.distill_kwargs.get("distill_use_conv3d", False))
            audio_distill_type = self.distill_kwargs.get("audio_distill_type", "d_axis")
            num_layers = int(self.distill_kwargs.get("distill_proj_layers", 1))
            hidden_dim_raw = self.distill_kwargs.get("distill_proj_hidden_dim")
            hidden_dim = int(hidden_dim_raw) if hidden_dim_raw is not None else None
            dim_schedule = self.distill_kwargs.get("distill_dim_schedule", "fixed")

            if self.video_latent_dim_actual is not None:
                video_spatial_factor = int(self.distill_kwargs.get("video_spatial_factor", 4))
                image_spatial_factor = int(self.distill_kwargs.get("image_spatial_factor", video_spatial_factor))
                temporal_agg_ratio = int(self.distill_kwargs.get("temporal_agg_ratio", 3))
                video_target_dim = int(self.distill_kwargs["video_target_dim"])
                image_target_dim = int(self.distill_kwargs.get("image_target_dim", video_target_dim))
                self.image_distill_proj = ImageDistillProjector(
                    in_dim=int(self.video_latent_dim_actual),
                    target_dim=image_target_dim,
                    spatial_factor=image_spatial_factor,
                    proj_type=proj_type,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    dim_schedule=dim_schedule,
                )
                proj_before_agg = bool(self.distill_kwargs.get("distill_proj_before_agg", True))
                self.video_distill_proj = VideoDistillProjector(
                    in_dim=int(self.video_latent_dim_actual),
                    target_dim=video_target_dim,
                    spatial_factor=video_spatial_factor,
                    temporal_agg_ratio=temporal_agg_ratio,
                    proj_type=proj_type,
                    use_conv3d=use_conv3d,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    proj_before_agg=proj_before_agg,
                    dim_schedule=dim_schedule,
                )
                logging.info(
                    f"Distill video projectors: "
                    f"image(in={self.video_latent_dim_actual}, spatial={image_spatial_factor}, target={image_target_dim}), "
                    f"video(in={self.video_latent_dim_actual}, spatial={video_spatial_factor}, "
                    f"temporal_agg={temporal_agg_ratio}, target={video_target_dim}), "
                    f"proj_type={proj_type}, use_conv3d={use_conv3d}, "
                    f"num_layers={num_layers}, hidden_dim={hidden_dim}, "
                    f"proj_before_agg={proj_before_agg}, dim_schedule={dim_schedule}"
                )

            if self.audio_latent_dim_actual is not None:
                audio_target_dim = int(self.distill_kwargs["audio_target_dim"])
                audio_proj_type = self.distill_kwargs.get("audio_proj_type", "linear")
                audio_num_layers = int(self.distill_kwargs.get("audio_proj_layers", num_layers))
                audio_hidden_dim_raw = self.distill_kwargs.get("audio_proj_hidden_dim")
                audio_hidden_dim = int(audio_hidden_dim_raw) if audio_hidden_dim_raw is not None else hidden_dim
                self.audio_distill_proj = AudioDistillProjector(
                    in_dim=int(self.audio_latent_dim_actual),
                    target_dim=audio_target_dim,
                    mode=audio_distill_type,
                    proj_type=audio_proj_type,
                    num_layers=audio_num_layers,
                    hidden_dim=audio_hidden_dim,
                    dim_schedule=dim_schedule,
                )
                logging.info(
                    f"Distill audio projector: "
                    f"in={self.audio_latent_dim_actual}, target={audio_target_dim}, "
                    f"type={audio_distill_type}, proj={audio_proj_type}, "
                    f"num_layers={audio_num_layers}, hidden_dim={audio_hidden_dim}, "
                    f"dim_schedule={dim_schedule}"
                )

        self.module_dtypes: Dict[str, Optional[torch.dtype]] = {}

        if self.video_vae is not None:
            logging.info(f"Video VAE: temporal_downsample={self.video_temporal_downsample}, "
                         f"spatial_downsample={self.video_spatial_downsample}, "
                         f"latent_dim={self.video_latent_dim_actual}")
        if self.audio_vae is not None:
            logging.info(f"Audio VAE: hop_length={self.audio_downsample}, "
                         f"latent_dim={self.audio_latent_dim_actual}, "
                         f"sample_rate={self.audio_sample_rate}")
        if self.use_contrastive:
            logging.info(
                f"AV contrastive head is enabled (use_mean={self.contrastive_use_mean}, "
                f"grad_scale_video={self.contrastive_grad_scale_video}, "
                f"grad_scale_audio={self.contrastive_grad_scale_audio})."
            )
        if self.use_llm:
            logging.info(f"LLM caption head is enabled (model={self.llm_kwargs.get('llm_model_path')}).")

    def _module_autocast(self, module_key: str):
        dtype = self.module_dtypes.get(module_key)
        if dtype is None or dtype == torch.float32:
            return torch.cuda.amp.autocast(enabled=False)
        return torch.cuda.amp.autocast(dtype=dtype)

    def _load_audio_pretrained(self, audio_ckpt: str):
        if os.path.isdir(audio_ckpt):
            candidate = os.path.join(audio_ckpt, "state_dict.pt")
            if os.path.exists(candidate):
                audio_ckpt = candidate
        logging.info(f"Loading Audio VAE weights from: {audio_ckpt}")
        ckpt = torch.load(audio_ckpt, map_location="cpu", weights_only=False)
        if "generator" in ckpt:
            state_dict = ckpt["generator"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        if "shadow" in state_dict and isinstance(state_dict["shadow"], dict):
            state_dict = state_dict["shadow"]
        has_audio_prefix = any(k.startswith("audio_vae.") for k in state_dict.keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            k = k.replace("module.", "")
            if k.startswith("audio_vae."):
                k = k[len("audio_vae."):]
            elif has_audio_prefix:
                continue
            if k.startswith("dac."):
                k = k[4:]
            new_state_dict[k] = v
        load_result = self.audio_vae.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Audio VAE load result - missing_keys: {len(load_result.missing_keys)}")
        logging.info(f"Audio VAE load result - unexpected_keys: {len(load_result.unexpected_keys)}")

    # 每个 rank 预留的 sentinel 命名空间大小：rank r 的 None 占用
    # [-1 - r*STRIDE, -2 - r*STRIDE, ...]，互不重叠。1<<20 = 1M 个 slot
    # 远大于任何现实 batch size，且 rank * STRIDE 不会溢出 int64。
    _NONE_SENTINEL_RANK_STRIDE: int = 1 << 20

    @staticmethod
    def _long_video_ids_to_int64(
        long_video_ids: List[Optional[str]],
        rank_offset: int = 0,
    ) -> torch.Tensor:
        """Convert per-sample long_video_id strings to a stable int64 tensor.

        - None / empty 映射为一个"独一份"的负数，保证：
            * 同一 rank 内彼此不等
            * 跨 rank all_gather 拼接后仍不等（用 rank_offset 把 sentinel
              的负数命名空间分给各 rank，避免不同 rank 的 None 撞成同一个
              负值，被 contrastive head 误判为 sibling）
            * 与正 id 的 crc32 区间不相等 -> 不会被误判为 sibling
        - 非 None 字符串用 crc32 哈希到正整数区间（2^32 以内），即便不同
          rank / 不同 dataloader worker 也能得到一致的 id。
        """
        stride = AudioVideoVAE._NONE_SENTINEL_RANK_STRIDE
        none_counter = -1 - int(rank_offset) * stride
        ids: List[int] = []
        for s in long_video_ids:
            if s is None or s == "":
                ids.append(none_counter)
                none_counter -= 1
            else:
                ids.append(int(zlib.crc32(str(s).encode("utf-8"))) & 0xFFFFFFFF)
        return torch.tensor(ids, dtype=torch.int64)

    def _all_gather_long_video_ids(
        self, long_video_ids_local: torch.Tensor, world_size: int,
    ) -> torch.Tensor:
        if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            return long_video_ids_local
        gather_for_loss = True
        if self.contrastive_head is not None:
            gather_for_loss = bool(getattr(self.contrastive_head, "gather_for_loss", True))
        if not gather_for_loss:
            return long_video_ids_local
        gathered = [torch.empty_like(long_video_ids_local) for _ in range(world_size)]
        dist.all_gather(gathered, long_video_ids_local.contiguous())
        return torch.cat(gathered, dim=0)

    def forward(
        self,
        video: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
        audio_lengths: Optional[torch.Tensor] = None,
        captions: Optional[list] = None,
        video_descriptions: Optional[list] = None,
        audio_descriptions: Optional[list] = None,
        sample_posterior: bool = True,
        skip_video_decoder: bool = False,
        skip_audio_decoder: bool = False,
        distill_target_shapes: Optional[Dict[str, tuple]] = None,
        long_video_ids: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, Dict]:
        results = {}

        # === Video VAE Forward ===
        if video is not None:
            with self._module_autocast('video_vae'):
                if skip_video_decoder:
                    video_posterior = self.video_vae.encode(video, streaming_inference=True)
                    video_recon = None
                else:
                    video_recon, video_posterior = self.video_vae(video, sample_posterior=sample_posterior)
                video_latent_mean = video_posterior.mode()
                if sample_posterior:
                    video_latent_sampled = video_posterior.sample()
                else:
                    video_latent_sampled = video_latent_mean
            results["video"] = {
                "recon": video_recon,
                "posterior": video_posterior,
                "latent": video_latent_sampled,
                "latent_mean": video_latent_mean,
            }

        # === Audio VAE Forward ===
        if audio is not None:
            with self._module_autocast('audio_vae'):
                audio_length = audio.shape[-1]
                audio_padded = self.audio_vae.preprocess(audio)
                audio_posterior, _, _, _, _ = self.audio_vae.encode(audio_padded)
                audio_latent_mean = audio_posterior.mode()
                if sample_posterior:
                    audio_latent_sampled = audio_posterior.sample()
                else:
                    audio_latent_sampled = audio_latent_mean
                if skip_audio_decoder:
                    audio_recon = None
                else:
                    audio_recon = self.audio_vae.decode(audio_latent_sampled)
                    audio_recon = audio_recon[..., :audio_length]
            results["audio"] = {
                "recon": audio_recon,
                "posterior": audio_posterior,
                "latent": audio_latent_sampled,
                "latent_mean": audio_latent_mean,
            }

        # === Contrastive ===
        if (self.use_contrastive and self.contrastive_head is not None
                and "video" in results and "audio" in results):
            with self._module_autocast('contrastive'):
                latent_key = "latent_mean" if self.contrastive_use_mean else "latent"
                audio_latent_lengths = self._compute_audio_latent_lengths(
                    audio_lengths=audio_lengths,
                    max_latent_length=results["audio"][latent_key].shape[-1],
                )
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
                v_lat_c = grad_scale(results["video"][latent_key], self.contrastive_grad_scale_video)
                a_lat_c = grad_scale(results["audio"][latent_key], self.contrastive_grad_scale_audio)

                # ===== Sibling-aware negative pool (collective-symmetric) =====
                # 注意：dist.all_gather 是集合通信，要求所有 rank 必须**对称**
                # 进入。早期实现把 all_gather 包在 `if long_video_ids is not None`
                # 里，但 collator 对全 None 的 batch 会省掉该 key，导致：
                #   - rank A 的 batch 里至少 1 个样本能解析出 long_video_id
                #     → long_video_ids 非空 → 进 all_gather 等其它 rank
                #   - rank B 的 batch 全部解析失败 → long_video_ids 为 None
                #     → 跳过 all_gather，直接走完 forward 进入下一 step
                # → 集合通信顺序错位 → 600s 后 NCCL watchdog 超时崩溃。
                # 现在改为：只要 same_long_video_priority 开关为 True（这是
                # 一个对所有 rank 都一致的模型属性），所有 rank 都对称地走
                # all_gather；本地没有 sibling 的样本用 sentinel 负数填充
                # （跨 rank 唯一，不会被误判为 sibling）。
                long_video_ids_pool: Optional[torch.Tensor] = None
                if getattr(self.contrastive_head, "same_long_video_priority", False):
                    B_local = int(v_lat_c.shape[0])
                    rank = (
                        dist.get_rank()
                        if dist.is_available() and dist.is_initialized()
                        else 0
                    )
                    if long_video_ids is None:
                        ids_str_list: List[Optional[str]] = [None] * B_local
                    elif len(long_video_ids) != B_local:
                        # Length mismatch: pad with None / truncate so that
                        # collective shapes still agree across ranks. This is
                        # purely defensive; the collator now guarantees
                        # len(long_video_ids) == B_local.
                        logging.warning(
                            f"long_video_ids length {len(long_video_ids)} != batch size "
                            f"{B_local}; padding/truncating to keep collective symmetric."
                        )
                        if len(long_video_ids) < B_local:
                            ids_str_list = list(long_video_ids) + [None] * (
                                B_local - len(long_video_ids)
                            )
                        else:
                            ids_str_list = list(long_video_ids[:B_local])
                    else:
                        ids_str_list = list(long_video_ids)
                    ids_local = self._long_video_ids_to_int64(
                        ids_str_list, rank_offset=rank,
                    ).to(v_lat_c.device)
                    long_video_ids_pool = self._all_gather_long_video_ids(
                        ids_local, world_size,
                    )

                results["contrastive"] = self.contrastive_head(
                    video_latent=v_lat_c,
                    audio_latent=a_lat_c,
                    audio_latent_lengths=audio_latent_lengths,
                    world_size=world_size,
                    long_video_ids_pool=long_video_ids_pool,
                )

        # === LLM Caption ===
        if (self.use_llm and self.llm_caption_head is not None
                and captions is not None
                and ("video" in results or "audio" in results)):
            with self._module_autocast('llm'):
                latent_key = "latent_mean" if self.llm_use_mean else "latent"
                video_lat = results["video"][latent_key] if "video" in results else None
                audio_lat = results["audio"][latent_key] if "audio" in results else None
                audio_latent_lengths = None
                if audio_lat is not None:
                    audio_latent_lengths = self._compute_audio_latent_lengths(
                        audio_lengths=audio_lengths,
                        max_latent_length=audio_lat.shape[-1],
                    )
                results["llm"] = self.llm_caption_head(
                    video_latent=video_lat,
                    audio_latent=audio_lat,
                    captions=captions,
                    video_descriptions=video_descriptions,
                    audio_descriptions=audio_descriptions,
                    audio_latent_lengths=audio_latent_lengths,
                )

        # === Semantic Distillation Projections ===
        if self.use_distill and distill_target_shapes is not None:
            img_shape = distill_target_shapes.get("image")   # (H_i, W_i, D_i)
            v_shape = distill_target_shapes.get("video")     # (T_v, H_v, W_v, D_v)
            a_shape = distill_target_shapes.get("audio")     # (T_a, D_a)
            _distill_lat_key = "latent" if self.distill_use_sampled else "latent_mean"

            if "video" in results:
                v_lat = results["video"][_distill_lat_key]

                if img_shape is not None and self.image_distill_proj is not None:
                    img_lat = v_lat[:, :, :1]
                    with self._module_autocast('distill'):
                        img_proj, img_pooled = self.image_distill_proj(
                            img_lat, target_h=img_shape[0], target_w=img_shape[1],
                        )
                    results["video"]["image_distill_proj"] = img_proj
                    results["video"]["image_distill_pooled"] = img_pooled

                if v_shape is not None and self.video_distill_proj is not None:
                    vid_lat = v_lat[:, :, 1:]
                    with self._module_autocast('distill'):
                        z_proj, z_pooled = self.video_distill_proj(
                            vid_lat, target_t=v_shape[0], target_h=v_shape[1], target_w=v_shape[2],
                        )
                    results["video"]["distill_proj"] = z_proj
                    results["video"]["distill_pooled"] = z_pooled

            if a_shape is not None and "audio" in results and self.audio_distill_proj is not None:
                a_lat = results["audio"][_distill_lat_key]
                with self._module_autocast('distill'):
                    z_proj_a = self.audio_distill_proj(a_lat, target_t=a_shape[0])
                results["audio"]["distill_proj"] = z_proj_a

        return results

    def _compute_audio_latent_lengths(self, audio_lengths, max_latent_length):
        if audio_lengths is None:
            return None
        audio_lengths = audio_lengths.to(dtype=torch.long)
        latent_lengths = (audio_lengths + self.audio_downsample - 1) // self.audio_downsample
        return latent_lengths.clamp_(min=1, max=max_latent_length)

    def encode_video(self, video, sample_posterior=True):
        if self.video_vae is None:
            return None, None
        posterior = self.video_vae.encode(video)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return latent, posterior

    def decode_video(self, latent):
        if self.video_vae is None:
            return None
        return self.video_vae.decode(latent)

    def encode_audio(self, audio, sample_posterior=True):
        if self.audio_vae is None:
            return None, None
        audio_padded = self.audio_vae.preprocess(audio)
        posterior, _, _, _, _ = self.audio_vae.encode(audio_padded)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return latent, posterior

    def decode_audio(self, latent, target_length=None):
        if self.audio_vae is None:
            return None
        audio = self.audio_vae.decode(latent)
        if target_length is not None:
            audio = audio[..., :target_length]
        return audio

    def enable_video_gradient_checkpointing(self):
        if self.video_vae is not None and hasattr(self.video_vae, 'enable_gradient_checkpointing'):
            self.video_vae.enable_gradient_checkpointing()

    def disable_video_gradient_checkpointing(self):
        if self.video_vae is not None and hasattr(self.video_vae, 'disable_gradient_checkpointing'):
            self.video_vae.disable_gradient_checkpointing()

    def get_video_encoder(self):
        if self.video_vae is not None and hasattr(self.video_vae, 'get_encoder'):
            return self.video_vae.get_encoder()
        return None

    def get_video_decoder(self):
        if self.video_vae is not None and hasattr(self.video_vae, 'get_decoder'):
            return self.video_vae.get_decoder()
        return None

    def get_video_last_layer(self):
        if self.video_vae is not None and hasattr(self.video_vae, 'get_last_layer'):
            return self.video_vae.get_last_layer()
        return None

    def get_video_encoder_last_layer(self):
        if self.video_vae is None:
            return None
        return self.video_vae.encoder.head[-1].weight

    def get_audio_encoder_last_layer(self):
        if self.audio_vae is None:
            return None
        return self.audio_vae.encoder.block[-1].weight

    def print_model_info(self):
        logging.info("=" * 80)
        logging.info("AudioVideoVAE Model Info")
        logging.info("=" * 80)

        video_total = 0
        if self.video_vae is not None:
            video_total = sum(p.numel() for p in self.video_vae.parameters())
            logging.info(f"Video VAE: model_name={self.video_vae_kwargs.get('model_name', 'WanVAE')}")
            logging.info(f"  temporal_downsample={self.video_temporal_downsample}, spatial_downsample={self.video_spatial_downsample}")
            logging.info(f"  latent_dim={self.video_latent_dim_actual}")
        else:
            logging.info("Video VAE: not loaded")

        audio_total = 0
        if self.audio_vae is not None:
            audio_total = sum(p.numel() for p in self.audio_vae.parameters())
            logging.info(f"Audio VAE: hop_length={self.audio_downsample}, latent_dim={self.audio_latent_dim_actual}")
        else:
            logging.info("Audio VAE: not loaded")

        contrastive_params = sum(p.numel() for p in self.contrastive_head.parameters()) if self.contrastive_head else 0
        llm_params = sum(p.numel() for p in self.llm_caption_head.parameters()) if self.llm_caption_head else 0
        distill_params = 0
        if self.image_distill_proj is not None:
            distill_params += sum(p.numel() for p in self.image_distill_proj.parameters())
        if self.video_distill_proj is not None:
            distill_params += sum(p.numel() for p in self.video_distill_proj.parameters())
        if self.audio_distill_proj is not None:
            distill_params += sum(p.numel() for p in self.audio_distill_proj.parameters())

        total_params = video_total + audio_total + contrastive_params + llm_params + distill_params
        logging.info(f"Total Parameters: {total_params / 1e6:.2f}M")
        if video_total > 0:
            logging.info(f"  Video VAE: {video_total / 1e6:.2f}M")
        if audio_total > 0:
            logging.info(f"  Audio VAE: {audio_total / 1e6:.2f}M")
        if contrastive_params > 0:
            logging.info(f"  Contrastive Head: {contrastive_params / 1e6:.2f}M")
        if llm_params > 0:
            logging.info(f"  LLM Caption Head: {llm_params / 1e6:.2f}M")
        if distill_params > 0:
            logging.info(f"  Distill Projectors: {distill_params / 1e6:.2f}M")
        logging.info("=" * 80)
