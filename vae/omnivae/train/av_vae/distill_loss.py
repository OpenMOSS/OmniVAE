"""
Semantic Distillation Loss — 语义蒸馏损失

支持两种语义特征来源：
  A. LocalSemanticEncoder — 在训练进程内加载 Qwen3-Omni 视觉/音频编码器，
     直接接收 tensor，无需网络服务。
  B. SemanticFeatureClient — 通过 HTTP API 上传 MP4 文件（保留向后兼容）。

损失函数：
  1. VF Loss (图像 / 视频)：Marginal Cosine Similarity + 可选 Marginal Distance Matrix
  2. d-axis Distillation Loss (音频)：沿时间轴逐特征维余弦相似度

投影方式 (proj_type):
  - "conv" (默认): Conv2d/Conv3d 投影，iREPA 风格
  - "linear": LayerNorm + Linear 投影，REPA 风格

空间归一化 (spatial_normalize):
  - iREPA 空间归一化，去除全局分量以增强空间对比度
"""

import io
import logging
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial normalization (iREPA)
# ---------------------------------------------------------------------------

def spatial_normalize(
    x: torch.Tensor,
    gamma: float = 0.7,
) -> torch.Tensor:
    """iREPA spatial normalization on teacher features.

    Subtracts a scaled mean and divides by std along the spatial dimension
    to enhance spatial contrast in the feature map.

    Args:
        x: (..., N, D) where N is the spatial dimension (H*W positions).
        gamma: mean subtraction scale, typically 0.6-0.8.
    Returns:
        Normalized x, same shape.
    """
    mean = x.mean(dim=-2, keepdim=True)
    x = x - gamma * mean
    x = x / (x.std(dim=-2, keepdim=True) + 1e-6)
    return x


# ---------------------------------------------------------------------------
# Multi-layer conv builder
# ---------------------------------------------------------------------------

def _build_multi_layer_conv(
    conv_cls,
    in_dim: int,
    target_dim: int,
    num_layers: int = 1,
    hidden_dim: Optional[int] = None,
    dim_schedule: str = "fixed",
) -> nn.Module:
    """Build a single- or multi-layer conv projection with GroupNorm + GELU.

    Args:
        dim_schedule:
            ``"fixed"``    — all hidden layers use *hidden_dim* (legacy default).
            ``"doubling"`` — dimensions double each layer from *in_dim* until
                             reaching *target_dim*; *num_layers* and *hidden_dim*
                             are ignored and the layer count is determined
                             automatically.
    """
    if dim_schedule == "doubling":
        dims = [in_dim]
        d = in_dim
        while d * 2 < target_dim:
            d = d * 2
            dims.append(d)
        dims.append(target_dim)

        layers: list = []
        for i in range(len(dims) - 1):
            layers.append(conv_cls(dims[i], dims[i + 1], 3, padding=1))
            if i < len(dims) - 2:
                ng = min(32, dims[i + 1] // 4) if dims[i + 1] >= 4 else 1
                layers += [nn.GroupNorm(ng, dims[i + 1]), nn.GELU()]
        return nn.Sequential(*layers)

    if num_layers == 1:
        return conv_cls(in_dim, target_dim, kernel_size=3, padding=1)
    h = hidden_dim or int(math.sqrt(in_dim * target_dim))
    num_groups = min(32, h // 4) if h >= 4 else 1
    layers = [conv_cls(in_dim, h, 3, padding=1), nn.GroupNorm(num_groups, h), nn.GELU()]
    for _ in range(num_layers - 2):
        layers += [conv_cls(h, h, 3, padding=1), nn.GroupNorm(num_groups, h), nn.GELU()]
    layers.append(conv_cls(h, target_dim, 3, padding=1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Local semantic encoder (Qwen3-Omni vision + audio towers)
# ---------------------------------------------------------------------------

def _ensure_encoder_service_importable(model_path: str) -> None:
    """Add the encoder_service source directory to sys.path if needed."""
    service_dir = str(Path(model_path).parent / "encoder_service")
    if not os.path.isdir(service_dir):
        service_dir = str(Path(model_path) / "encoder_service")
    if service_dir not in sys.path:
        sys.path.insert(0, service_dir)


class LocalSemanticEncoder:
    """Load Qwen3-Omni vision + audio encoders on-device for semantic distillation.

    Accepts raw training tensors — no file I/O or HTTP needed.
    The teacher models are frozen and run under ``torch.no_grad()``.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        encoder_fps: float = 4.0,
        encoder_resolution: int = 128,
        vision_layer: Optional[int] = None,
        audio_layer: Optional[int] = None,
    ):
        self.device = device
        self.encoder_fps = encoder_fps
        self.encoder_resolution = encoder_resolution
        self._vision_layer_idx = (int(vision_layer) - 1) if vision_layer else None
        self._audio_layer_idx = (int(audio_layer) - 1) if audio_layer else None

        _ensure_encoder_service_importable(model_path)

        os.environ.setdefault("QWEN_OMNI_MODEL_PATH", model_path)
        os.environ["ENCODER_DEVICE"] = str(device)

        from src.patches import apply_qwen_omni_vision_patches
        from src.torch_perf import apply_encoder_torch_runtime_settings
        from src.bootstrap import load_encoder_pair

        apply_qwen_omni_vision_patches()
        apply_encoder_torch_runtime_settings()

        logger.info("Loading Qwen3-Omni semantic encoders from %s onto %s ...", model_path, device)
        t0 = time.perf_counter()
        self.visual_encoder, self.audio_encoder = load_encoder_pair()
        logger.info("Semantic encoders loaded in %.1fs", time.perf_counter() - t0)
        logger.info(
            "LocalSemanticEncoder teacher layer selection: "
            "vision_layer=%s (blocks idx=%s), audio_layer=%s (layers idx=%s)",
            vision_layer if vision_layer else "last",
            self._vision_layer_idx if self._vision_layer_idx is not None else "last",
            audio_layer if audio_layer else "last",
            self._audio_layer_idx if self._audio_layer_idx is not None else "last",
        )

    def _video_tensor_to_pil_frames(
        self, video: torch.Tensor, fps: float, target_fps: float,
    ) -> list:
        """Convert (C, T, H, W) float tensor [-1,1] to PIL frames sampled at target_fps."""
        from PIL import Image as PILImage
        C, T, H, W = video.shape
        video_uint8 = ((video.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)

        group_size = max(1, round(fps / target_fps))
        center = group_size // 2
        indices = list(range(center, T, group_size))
        if len(indices) % 2 != 0:
            indices = indices[:-1]
        if len(indices) < 2:
            indices = [0, min(1, T - 1)]

        frames = []
        for i in indices:
            frame_np = video_uint8[:, i].permute(1, 2, 0).cpu().numpy()
            frames.append(PILImage.fromarray(frame_np))
        return frames

    def _image_tensor_to_pil(self, image: torch.Tensor):
        """Convert (C, H, W) float tensor [-1,1] to a single PIL Image."""
        from PIL import Image as PILImage
        img_uint8 = ((image.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return PILImage.fromarray(img_uint8.permute(1, 2, 0).cpu().numpy())

    def _preprocess_audio_from_tensor(self, audio_tensor: torch.Tensor, sample_rate: int) -> dict:
        """Build audio encoder input directly from tensor, bypassing file I/O.

        Qwen's process_audio_info accepts np.ndarray at 16kHz directly.
        """
        import librosa
        from qwen_omni_utils import process_mm_info

        wav_np = audio_tensor[0].float().cpu().numpy()  # (T_a,)
        if sample_rate != 16000:
            wav_np = librosa.resample(wav_np, orig_sr=sample_rate, target_sr=16000)

        conversation = [
            {"role": "user", "content": [{"type": "audio", "audio": wav_np}]},
        ]
        text = self.audio_encoder.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.audio_encoder.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False,
        )
        input_features = inputs.get("input_features")
        if input_features is None:
            raise ValueError("Processor did not return input_features from audio tensor.")
        return {
            "input_features": input_features,
            "feature_attention_mask": inputs.get("feature_attention_mask"),
        }

    @torch.no_grad()
    def extract_from_tensors(
        self,
        video: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
        video_fps: float = 24.0,
        audio_sample_rate: int = 48000,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Extract semantic features from raw training tensors.

        Args:
            video: (B, C, T, H, W), range [-1, 1]
            audio: (B, 1, T_a)
            video_fps: fps of the input video
            audio_sample_rate: audio sample rate

        Returns:
            dict with keys ``image_feat``, ``video_feat``, ``audio_feat``
            or None on failure.
        """
        out: Dict[str, torch.Tensor] = {}

        try:
            if video is not None and self.visual_encoder is not None:
                B = video.shape[0]
                T = video.shape[2]
                res = self.encoder_resolution

                # 仅当 T > 1 时才有 video clip 分支可跑;
                # T == 1 (image step) 时只产 image_feat。
                run_video_branch = T > 1

                image_inputs = []
                video_inputs = []
                for b in range(B):
                    first_frame = video[b, :, 0]
                    pil_img = self._image_tensor_to_pil(first_frame)
                    image_inputs.append(
                        self.visual_encoder.preprocess_image(pil_img)
                    )

                    if run_video_branch:
                        remaining = video[b, :, 1:]
                        pil_frames = self._video_tensor_to_pil_frames(
                            remaining, video_fps, self.encoder_fps,
                        )
                        video_inputs.append(
                            self.visual_encoder.preprocess_from_frames(
                                pil_frames,
                                target_height=res,
                                target_width=res,
                            )
                        )

                img_results = self.visual_encoder.forward_batch(
                    image_inputs, feature_mode="full",
                    layer_idx=self._vision_layer_idx,
                )
                image_feats = []
                for r in img_results:
                    flat, shape = r.features
                    image_feats.append(
                        torch.tensor(flat, dtype=torch.float32).reshape(shape)
                    )
                out["image_feat"] = torch.stack(image_feats, dim=0).to(self.device)

                if run_video_branch and video_inputs:
                    vid_results = self.visual_encoder.forward_batch(
                        video_inputs, feature_mode="full",
                        layer_idx=self._vision_layer_idx,
                    )
                    video_feats = []
                    for r in vid_results:
                        flat_v, shape_v = r.features
                        video_feats.append(
                            torch.tensor(flat_v, dtype=torch.float32).reshape(shape_v)
                        )
                    out["video_feat"] = torch.stack(video_feats, dim=0).to(self.device)

            if audio is not None and self.audio_encoder is not None:
                B = audio.shape[0]
                audio_inputs = []
                for b in range(B):
                    audio_inputs.append(
                        self._preprocess_audio_from_tensor(audio[b], audio_sample_rate)
                    )

                aud_results = self.audio_encoder.forward_batch(
                    audio_inputs, feature_mode="full",
                    layer_idx=self._audio_layer_idx,
                )

                audio_feats = []
                for flat_a, shape_a in aud_results:
                    audio_feats.append(
                        torch.tensor(flat_a, dtype=torch.float32).reshape(shape_a)
                    )
                out["audio_feat"] = torch.stack(audio_feats, dim=0).to(self.device)

        except Exception as e:
            logger.warning("LocalSemanticEncoder: extraction failed: %s", e, exc_info=True)
            return None

        return out if out else None

    @torch.no_grad()
    def extract_from_preprocessed(
        self,
        distill_first_frames: Optional[list] = None,
        distill_video_frames: Optional[list] = None,
        distill_audio_16k: Optional[list] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Extract semantic features from DataLoader-cached preprocessed data.

        This skips the GPU->CPU->PIL roundtrip by receiving PIL frames and
        16 kHz numpy audio directly from the DataLoader workers.

        Args:
            distill_first_frames: list of B PIL Images (first frame per sample).
            distill_video_frames: list of B lists-of-PIL-Images (remaining frames).
            distill_audio_16k: list of B numpy arrays at 16 kHz.
        """
        from qwen_omni_utils import process_mm_info

        out: Dict[str, torch.Tensor] = {}
        try:
            if (distill_first_frames is not None
                    and distill_video_frames is not None
                    and self.visual_encoder is not None):
                B = len(distill_first_frames)
                res = self.encoder_resolution

                image_inputs = []
                video_inputs = []
                for b in range(B):
                    if distill_first_frames[b] is None:
                        continue
                    image_inputs.append(
                        self.visual_encoder.preprocess_image(distill_first_frames[b])
                    )
                    video_inputs.append(
                        self.visual_encoder.preprocess_from_frames(
                            distill_video_frames[b],
                            target_height=res,
                            target_width=res,
                        )
                    )

                if image_inputs:
                    img_results = self.visual_encoder.forward_batch(
                        image_inputs, feature_mode="full",
                        layer_idx=self._vision_layer_idx,
                    )
                    image_feats = []
                    for r in img_results:
                        flat, shape = r.features
                        image_feats.append(
                            torch.tensor(flat, dtype=torch.float32).reshape(shape)
                        )
                    out["image_feat"] = torch.stack(image_feats, dim=0).to(self.device)

                if video_inputs:
                    vid_results = self.visual_encoder.forward_batch(
                        video_inputs, feature_mode="full",
                        layer_idx=self._vision_layer_idx,
                    )
                    video_feats = []
                    for r in vid_results:
                        flat_v, shape_v = r.features
                        video_feats.append(
                            torch.tensor(flat_v, dtype=torch.float32).reshape(shape_v)
                        )
                    out["video_feat"] = torch.stack(video_feats, dim=0).to(self.device)

            if distill_audio_16k is not None and self.audio_encoder is not None:
                audio_inputs = []
                for b, wav_np in enumerate(distill_audio_16k):
                    if wav_np is None:
                        continue
                    conversation = [
                        {"role": "user", "content": [{"type": "audio", "audio": wav_np}]},
                    ]
                    text = self.audio_encoder.processor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False,
                    )
                    audios, images, videos = process_mm_info(
                        conversation, use_audio_in_video=False,
                    )
                    inputs = self.audio_encoder.processor(
                        text=text, audio=audios, images=images, videos=videos,
                        return_tensors="pt", padding=True, use_audio_in_video=False,
                    )
                    input_features = inputs.get("input_features")
                    if input_features is None:
                        raise ValueError("Processor did not return input_features.")
                    audio_inputs.append({
                        "input_features": input_features,
                        "feature_attention_mask": inputs.get("feature_attention_mask"),
                    })

                if audio_inputs:
                    aud_results = self.audio_encoder.forward_batch(
                        audio_inputs, feature_mode="full",
                        layer_idx=self._audio_layer_idx,
                    )
                    audio_feats = []
                    for flat_a, shape_a in aud_results:
                        audio_feats.append(
                            torch.tensor(flat_a, dtype=torch.float32).reshape(shape_a)
                        )
                    out["audio_feat"] = torch.stack(audio_feats, dim=0).to(self.device)

        except Exception as e:
            logger.warning(
                "LocalSemanticEncoder: extract_from_preprocessed failed: %s",
                e, exc_info=True,
            )
            return None

        return out if out else None


# ---------------------------------------------------------------------------
# HTTP API client (backward-compatible)
# ---------------------------------------------------------------------------

def _resolve_api_url(api_url: str, max_wait: float = 300.0) -> str:
    """Resolve ``api_url`` to a concrete HTTP URL."""
    if not api_url.startswith("file://"):
        return api_url

    path = api_url[len("file://"):]
    rank = int(os.environ.get("RANK", 0))

    elapsed = 0.0
    while not os.path.isfile(path):
        if elapsed >= max_wait:
            raise FileNotFoundError(
                f"SemanticFeatureClient: worker list {path} not found after {max_wait}s. "
                "Is the encoder service running?"
            )
        if elapsed == 0:
            logging.info("SemanticFeatureClient: waiting for worker list %s ...", path)
        time.sleep(5)
        elapsed += 5

    with open(path) as f:
        urls = [line.strip() for line in f if line.strip()]
    if not urls:
        raise ValueError(f"SemanticFeatureClient: worker list {path} is empty.")

    chosen = urls[rank % len(urls)]
    logging.info(
        "SemanticFeatureClient: rank %d -> %s (from %d workers in %s)",
        rank, chosen, len(urls), path,
    )
    return chosen


class SemanticFeatureClient:
    """Upload MP4 files to an external semantic-feature extraction API (legacy)."""

    def __init__(self, api_url: str, timeout: float = 60.0, max_workers: int = 8):
        self.api_url = _resolve_api_url(api_url)
        self.timeout = timeout
        self.max_workers = max_workers
        self._session = requests.Session()

    def _extract_single(
        self,
        file_path: str,
        target_fps: float,
        resolution: int,
        audio_sample_rate: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            logging.warning(f"SemanticFeatureClient: cannot read {file_path}: {e}")
            return None

        try:
            resp = self._session.post(
                self.api_url,
                files={"file": (file_path, io.BytesIO(file_bytes), "video/mp4")},
                data={
                    "target_fps": str(target_fps),
                    "resolution": str(resolution),
                    "audio_sample_rate": str(audio_sample_rate),
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return pickle.loads(resp.content)
        except Exception as e:
            logging.warning(f"SemanticFeatureClient: API error for {file_path}: {e}")
            return None

    @torch.no_grad()
    def extract(
        self,
        file_paths: List[str],
        target_fps: float,
        resolution: int,
        audio_sample_rate: int,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not file_paths:
            return None

        results: List[Optional[Dict[str, np.ndarray]]] = [None] * len(file_paths)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(file_paths))) as pool:
            future_to_idx = {
                pool.submit(
                    self._extract_single, fp, target_fps, resolution, audio_sample_rate
                ): i
                for i, fp in enumerate(file_paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        if any(r is None for r in results):
            return None

        out: Dict[str, torch.Tensor] = {}

        video_feats = [r["video_feat"] for r in results if r.get("video_feat") is not None]
        if video_feats and len(video_feats) == len(results):
            out["video_feat"] = torch.from_numpy(np.stack(video_feats, axis=0)).to(device)

        audio_feats = [r["audio_feat"] for r in results if r.get("audio_feat") is not None]
        if audio_feats and len(audio_feats) == len(results):
            out["audio_feat"] = torch.from_numpy(np.stack(audio_feats, axis=0)).to(device)

        return out if out else None


# ---------------------------------------------------------------------------
# Projection modules
# ---------------------------------------------------------------------------

class ImageDistillProjector(nn.Module):
    """Project VAE first-frame latent to match the image encoder's feature space.

    Supports two projection modes:
      - ``conv``: Conv2d (supports multi-layer), iREPA style
      - ``linear``: k×k spatial merge → LayerNorm → Linear, REPA style
    """

    def __init__(self, in_dim: int, target_dim: int, spatial_factor: int = 4,
                 proj_type: str = "conv",
                 num_layers: int = 1, hidden_dim: Optional[int] = None,
                 dim_schedule: str = "fixed"):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.proj_type = proj_type

        if proj_type == "conv":
            self.conv = _build_multi_layer_conv(
                nn.Conv2d, in_dim, target_dim,
                num_layers=num_layers, hidden_dim=hidden_dim,
                dim_schedule=dim_schedule,
            )
        else:
            merged_dim = in_dim * spatial_factor * spatial_factor
            self.norm = nn.LayerNorm(merged_dim)
            self.proj = nn.Linear(merged_dim, target_dim)

    def forward(
        self,
        latent: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, D, 1, H, W) — VAE first-frame latent
        Returns:
            z_proj:   (B, H_tgt, W_tgt, D_target)
            z_pooled: (B, H_tgt, W_tgt, D_pooled)
        """
        B, D, _, H, W = latent.shape

        if self.proj_type == "conv":
            x = latent.squeeze(2)  # (B, D, H, W)
            x = self.conv(x)      # (B, D_target, H, W)
            if H != target_h or W != target_w:
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
            x = x.permute(0, 2, 3, 1)  # (B, H_tgt, W_tgt, D_target)
            return x, x  # conv mode: z_pooled = z_proj
        else:
            k = self.spatial_factor
            x = latent.squeeze(2).permute(0, 2, 3, 1)  # (B, H, W, D)
            x = x.reshape(B, H // k, k, W // k, k, D)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H // k, W // k, k * k * D)
            if (H // k) != target_h or (W // k) != target_w:
                x = x.permute(0, 3, 1, 2)
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                x = x.permute(0, 2, 3, 1)
            z_pooled = x
            z_proj = self.proj(self.norm(x))
            return z_proj, z_pooled


class VideoDistillProjector(nn.Module):
    """Project video VAE latent to match the semantic encoder's feature space.

    Projection is applied **before** temporal aggregation so that the conv
    operates at full temporal resolution (richer features, especially for
    Conv3d which can learn spatio-temporal patterns).

    Supports two projection modes:
      - ``conv``: Conv2d per-frame or Conv3d, iREPA style (supports multi-layer)
      - ``linear``: k×k spatial merge → LayerNorm → Linear, REPA style
    """

    def __init__(self, in_dim: int, target_dim: int, spatial_factor: int = 4,
                 temporal_agg_ratio: int = 3,
                 proj_type: str = "conv", use_conv3d: bool = False,
                 num_layers: int = 1, hidden_dim: Optional[int] = None,
                 proj_before_agg: bool = True,
                 dim_schedule: str = "fixed"):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_agg_ratio = temporal_agg_ratio
        self.proj_type = proj_type
        self.use_conv3d = use_conv3d
        self.proj_before_agg = proj_before_agg

        if proj_type == "conv":
            conv_cls = nn.Conv3d if use_conv3d else nn.Conv2d
            self.conv = _build_multi_layer_conv(
                conv_cls, in_dim, target_dim,
                num_layers=num_layers, hidden_dim=hidden_dim,
                dim_schedule=dim_schedule,
            )
        else:
            merged_dim = in_dim * spatial_factor * spatial_factor
            self.norm = nn.LayerNorm(merged_dim)
            self.proj = nn.Linear(merged_dim, target_dim)

    def _temporal_agg(self, x: torch.Tensor, target_t: int) -> torch.Tensor:
        """Aggregate temporal dim: group by ratio then adaptive pool to target_t.

        Args:
            x: (B, D, T, H, W)
        Returns:
            (B, D, T_target, H, W)
        """
        B, D, T, H, W = x.shape
        r = self.temporal_agg_ratio
        flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, T)
        if r > 1 and T >= r:
            usable = (T // r) * r
            grouped = flat[:, :, :usable].reshape(B * H * W, D, T // r, r)
            flat = grouped.mean(dim=-1)
        if flat.shape[-1] != target_t:
            flat = F.adaptive_avg_pool1d(flat, target_t)
        return flat.reshape(B, H, W, D, target_t).permute(0, 3, 4, 1, 2)

    def _forward_conv_proj_before_agg(self, latent, B, D, T, H, W, target_t, target_h, target_w):
        """Conv projection at full temporal resolution, then temporal aggregation."""
        if self.use_conv3d:
            x = self.conv(latent)
            D_out = x.shape[1]
            x = self._temporal_agg(x, target_t)
            if H != target_h or W != target_w:
                x = x.permute(0, 2, 1, 3, 4).reshape(B * target_t, D_out, H, W)
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                x = x.reshape(B, target_t, D_out, target_h, target_w)
            else:
                x = x.permute(0, 2, 1, 3, 4)
            return x.permute(0, 1, 3, 4, 2)
        else:
            x = latent.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W)
            x = self.conv(x)
            D_out = x.shape[1]
            x = x.reshape(B, T, D_out, H, W).permute(0, 2, 1, 3, 4)
            x = self._temporal_agg(x, target_t)
            if H != target_h or W != target_w:
                x = x.permute(0, 2, 1, 3, 4).reshape(B * target_t, D_out, H, W)
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                x = x.reshape(B, target_t, D_out, target_h, target_w)
            else:
                x = x.permute(0, 2, 1, 3, 4)
            return x.permute(0, 1, 3, 4, 2)

    def _forward_conv_agg_before_proj(self, latent, B, D, T, H, W, target_t, target_h, target_w):
        """Temporal aggregation first, then conv projection (legacy order)."""
        if self.use_conv3d:
            x = self._temporal_agg(latent, target_t)
            x = self.conv(x)
            D_out = x.shape[1]
            if H != target_h or W != target_w:
                x = x.permute(0, 2, 1, 3, 4).reshape(B * target_t, D_out, H, W)
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                x = x.reshape(B, target_t, D_out, target_h, target_w)
            else:
                x = x.permute(0, 2, 1, 3, 4)
            return x.permute(0, 1, 3, 4, 2)
        else:
            x = self._temporal_agg(latent, target_t)
            _, D_in, T_t, _, _ = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T_t, D_in, H, W)
            x = self.conv(x)
            D_out = x.shape[1]
            if H != target_h or W != target_w:
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                x = x.reshape(B, target_t, D_out, target_h, target_w)
            else:
                x = x.reshape(B, target_t, D_out, H, W)
            return x.permute(0, 1, 3, 4, 2)

    def forward(
        self,
        latent: torch.Tensor,
        target_t: int,
        target_h: int,
        target_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, D, T, H, W) — VAE video latent
        Returns:
            z_proj:   (B, T_target, H_target, W_target, D_target)
            z_pooled: (B, T_target, H_target, W_target, D_pooled)
        """
        B, D, T, H, W = latent.shape

        if self.proj_type == "conv":
            if self.proj_before_agg:
                x = self._forward_conv_proj_before_agg(latent, B, D, T, H, W, target_t, target_h, target_w)
            else:
                x = self._forward_conv_agg_before_proj(latent, B, D, T, H, W, target_t, target_h, target_w)
            return x, x
        else:
            k = self.spatial_factor
            x = latent.permute(0, 2, 3, 4, 1)  # (B, T, H, W, D)
            x = x.reshape(B, T, H // k, k, W // k, k, D)
            x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(
                B, T, H // k, W // k, k * k * D
            )
            Hs, Ws = H // k, W // k
            if Hs != target_h or Ws != target_w:
                x = x.permute(0, 4, 1, 2, 3)  # (B, k²D, T, Hs, Ws)
                x = x.reshape(B * (k * k * D), T, Hs, Ws)
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
                x = x.reshape(B, k * k * D, T, target_h, target_w)
                x = x.permute(0, 2, 3, 4, 1)  # (B, T, H_tgt, W_tgt, k²D)
            z_pooled_full = x  # (B, T, H_tgt, W_tgt, k²D)
            z_proj_full = self.proj(self.norm(x))  # (B, T, H_tgt, W_tgt, D_target)

            z_pooled_full_5d = z_pooled_full.permute(0, 4, 1, 2, 3)  # (B, k²D, T, H_tgt, W_tgt)
            z_proj_full_5d = z_proj_full.permute(0, 4, 1, 2, 3)      # (B, D_target, T, H_tgt, W_tgt)
            z_pooled_agg = self._temporal_agg(z_pooled_full_5d, target_t)
            z_proj_agg = self._temporal_agg(z_proj_full_5d, target_t)
            z_pooled = z_pooled_agg.permute(0, 2, 3, 4, 1)  # (B, T_tgt, H_tgt, W_tgt, k²D)
            z_proj = z_proj_agg.permute(0, 2, 3, 4, 1)      # (B, T_tgt, H_tgt, W_tgt, D_target)
            return z_proj, z_pooled


class AudioDistillProjector(nn.Module):
    """Project audio VAE latent to match the semantic encoder's feature space.

    Supports two projection types:
      - ``linear``: LayerNorm + Linear (original)
      - ``conv``:   Conv1d (supports multi-layer), projection before temporal pool

    Supports two output modes:
      - ``d_axis``: output (B, D_target, T_target) for d-axis distillation loss
      - ``t_axis``: output (B, T_target, D_target) for per-timestep cosine similarity loss
    """

    def __init__(self, in_dim: int, target_dim: int, mode: str = "d_axis",
                 proj_type: str = "linear",
                 num_layers: int = 1, hidden_dim: Optional[int] = None,
                 dim_schedule: str = "fixed"):
        super().__init__()
        self.mode = mode
        self.proj_type = proj_type

        if proj_type == "conv":
            self.conv = _build_multi_layer_conv(
                nn.Conv1d, in_dim, target_dim,
                num_layers=num_layers, hidden_dim=hidden_dim,
                dim_schedule=dim_schedule,
            )
        else:
            self.norm = nn.LayerNorm(in_dim)
            self.proj = nn.Linear(in_dim, target_dim)

    def forward(
        self, latent: torch.Tensor, target_t: int,
    ) -> torch.Tensor:
        """
        Args:
            latent: (B, D_a, T_l)
            target_t: target temporal length from semantic features
        Returns:
            d_axis mode: (B, D_target, T_target)
            t_axis mode: (B, T_target, D_target)
        """
        if self.proj_type == "conv":
            x = self.conv(latent)  # (B, D_target, T_l)
            x = F.adaptive_avg_pool1d(x, target_t)  # (B, D_target, T_target)
            if self.mode == "d_axis":
                return x  # (B, D_target, T_target)
            return x.permute(0, 2, 1)  # (B, T_target, D_target)
        else:
            x = F.adaptive_avg_pool1d(latent, target_t)  # (B, D_a, T_target)
            x = x.permute(0, 2, 1)  # (B, T_target, D_a)
            x = self.proj(self.norm(x))  # (B, T_target, D_target)
            if self.mode == "d_axis":
                return x.permute(0, 2, 1)  # (B, D_target, T_target)
            return x


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def marginal_cosine_similarity_loss(
    z_proj: torch.Tensor,
    f_target: torch.Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
    nonneg: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Marginal Cosine Similarity Loss.

    When ``margin > 0`` (VF Loss style): penalise positions where cosine
    similarity is below ``1 - margin``.

    When ``margin == 0`` (iREPA/REPA style): return the mean negative
    cosine similarity directly, or ``-log(sigmoid(cos_sim))`` if
    ``nonneg=True``.

    Works for video ``(B, T, H, W, D)``, image ``(B, H, W, D)``, and
    audio ``(B, T, D)`` — all non-batch dims are flattened.

    Args:
        z_proj:   projected VAE latent.
        f_target: semantic encoder features (same shape as z_proj).
        margin:   m₁. Default 0.0 (iREPA style, no margin).
        reduction: ``'mean'`` (default) averages over all positions and batch;
                   ``'sum'`` sums over spatial/temporal positions, averages
                   over batch — useful for combining image+video with proper
                   frame-count normalization downstream.
        nonneg: If True, use ``-log(sigmoid(cos_sim))`` instead of
                ``-cos_sim`` when ``margin == 0``, guaranteeing a non-negative
                loss value.  Required for adaptive loss balancing.
    Returns:
        ``(loss, mean_cos_sim)`` — scalar loss and the detached mean cosine
        similarity (in [-1, 1]) for monitoring.
    """
    z_flat = z_proj.reshape(z_proj.shape[0], -1, z_proj.shape[-1])
    f_flat = f_target.reshape(f_target.shape[0], -1, f_target.shape[-1])
    cos_sim = F.cosine_similarity(z_flat, f_flat, dim=-1)  # (B, N)
    mean_cos_sim = cos_sim.mean().detach()
    if margin > 0:
        per_pos = F.relu((1.0 - margin) - cos_sim)
    elif nonneg:
        per_pos = -torch.log(torch.sigmoid(cos_sim) + 1e-8)
    else:
        per_pos = -cos_sim
    if reduction == "sum":
        return per_pos.sum(dim=-1).mean(), mean_cos_sim
    return per_pos.mean(), mean_cos_sim


def marginal_distance_matrix_loss(
    z_pooled: torch.Tensor,
    f_target: torch.Tensor,
    margin: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """Marginal Distance Matrix Similarity Loss.

    Supports:
      - video: (B, T, H, W, D) — per-frame pairwise over H*W positions
      - image: (B, H, W, D) — pairwise over H*W positions

    Args:
        z_pooled: spatially/temporally aligned, *not* channel-projected VAE latent.
        f_target: semantic encoder features.
        margin:   m₂, default 0.25.
        reduction: ``'mean'`` (default) averages over all elements;
                   ``'sum'`` sums over per-frame (H*W)^2 pairs and T frames,
                   averages over batch — for combining image+video downstream.
    Returns:
        Scalar loss.
    """
    if z_pooled.ndim == 4:
        z_pooled = z_pooled.unsqueeze(1)
        f_target = f_target.unsqueeze(1)

    B, T, H, W, _ = z_pooled.shape

    z = z_pooled.reshape(B * T, H * W, -1)
    f = f_target.reshape(B * T, H * W, -1)

    z_norm = F.normalize(z, dim=-1)
    f_norm = F.normalize(f, dim=-1)

    sim_z = torch.bmm(z_norm, z_norm.transpose(1, 2))
    sim_f = torch.bmm(f_norm, f_norm.transpose(1, 2))

    diff = (sim_z - sim_f).abs()
    per_frame = F.relu(diff - margin)
    if reduction == "sum":
        # sum over (H*W, H*W) per frame, then sum over T, mean over B
        return per_frame.sum(dim=(1, 2)).reshape(B, T).sum(dim=1).mean()
    return per_frame.mean()


def d_axis_distill_loss(
    q_proj: torch.Tensor,
    semantic_feat: torch.Tensor,
) -> torch.Tensor:
    """D-axis distillation loss for audio.

    Computes cosine similarity along the time axis for each feature dimension,
    then applies sigmoid + log to encourage alignment.

    Args:
        q_proj: projected VAE audio latent, (B, D_s, T)
        semantic_feat: teacher audio feature, (B, D_s, T)
    Returns:
        Scalar loss.
    """
    cos_sim = F.cosine_similarity(q_proj, semantic_feat, dim=-1)  # (B, D_s)
    loss = -torch.log(torch.sigmoid(cos_sim) + 1e-8).mean()
    return loss
