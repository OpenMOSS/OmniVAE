"""
Async distillation client for cross-server encoder feature extraction.

Mode C: training server uploads pre-processed data (PIL frames / numpy audio)
to a remote encoder service, which returns semantic features via HTTP.

Two main classes:

  AsyncDistillClient  — thread-pool based HTTP client with per-rank GPU routing.
  DistillPrefetcher   — wraps the client to overlap encoder calls with training.
"""

from __future__ import annotations

import io
import logging
import pickle
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _parse_gpu_map(raw: str) -> Dict[int, int]:
    """Parse 'rank:gpu_id,rank:gpu_id,...' into {rank: gpu_id} dict."""
    if not raw or not raw.strip():
        return {}
    mapping = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid gpu_map entry '{pair}', expected 'rank:gpu_id'")
        mapping[int(parts[0])] = int(parts[1])
    return mapping


def _tensor_to_pil_frames(
    video: torch.Tensor,
    fps: float,
    target_fps: float,
) -> list:
    """Convert (C, T, H, W) float tensor [-1,1] to sampled PIL frames.

    Uses center-frame sampling identical to VisualEncoder._sample_center_frames.
    """
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


def _tensor_to_first_frame_pil(video: torch.Tensor):
    """Convert first frame of (C, T, H, W) tensor [-1,1] to a single PIL Image."""
    from PIL import Image as PILImage
    img_uint8 = ((video[:, 0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return PILImage.fromarray(img_uint8.permute(1, 2, 0).cpu().numpy())


def _audio_tensor_to_16k_numpy(
    audio: torch.Tensor,
    sample_rate: int,
) -> np.ndarray:
    """Convert (1, T_a) audio tensor to 16kHz mono numpy for Qwen audio encoder."""
    import librosa
    wav_np = audio[0].float().cpu().numpy()
    if sample_rate != 16000:
        wav_np = librosa.resample(wav_np, orig_sr=sample_rate, target_sr=16000)
    return wav_np.astype(np.float32)


def _tokenize_video_frames(
    frames: list,
    processor,
    target_height: int,
    target_width: int,
) -> dict:
    """Run Qwen processor tokenization on PIL frames → pixel_values_videos + grid_thw.

    Done on training side so the encoder service skips all CPU preprocessing.
    """
    from qwen_omni_utils import process_mm_info
    target_pixels = target_height * target_width
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "resized_height": target_height,
                    "resized_width": target_width,
                    "min_pixels": target_pixels,
                    "max_pixels": target_pixels,
                }
            ],
        },
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=False, image_patch_size=16,
    )
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
        videos_kwargs={"do_resize": False, "do_sample_frames": False},
    )
    return {
        "pixel_values_videos": inputs["pixel_values_videos"],
        "video_grid_thw": inputs["video_grid_thw"],
    }


def _tokenize_image(
    pil_image,
    processor,
) -> dict:
    """Run Qwen processor tokenization on a single PIL image."""
    from qwen_omni_utils import process_mm_info
    conversation = [
        {"role": "user", "content": [{"type": "image", "image": pil_image}]},
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    return {
        "pixel_values_videos": inputs["pixel_values"],
        "video_grid_thw": inputs["image_grid_thw"],
    }


class AsyncDistillClient:
    """HTTP client that uploads pre-tokenized tensors to encoder service with GPU routing.

    Preprocessing (Qwen processor tokenization) is done on the training side,
    so the encoder service receives ready-to-forward tensors via
    /extract/upload/visual/tensors — zero server-side CPU preprocessing.

    For each sample in a batch, sends three parallel requests:
      1. Video tensors  → /extract/upload/visual/tensors (gpu_id from video_gpu_map)
      2. Image tensors  → /extract/upload/visual/tensors (gpu_id = image_gpu_id)
      3. Audio waveform → /extract/upload/audio/tensor   (gpu_id = audio_gpu_id)
    """

    def __init__(
        self,
        base_url: str,
        video_gpu_map: Dict[int, int],
        image_gpu_id: int,
        audio_gpu_id: int,
        rank: int,
        encoder_fps: float = 4.0,
        encoder_resolution: int = 256,
        data_fps: float = 24.0,
        audio_sample_rate: int = 48000,
        timeout: float = 120.0,
        max_workers: int = 6,
    ):
        import requests
        self.base_url = base_url.rstrip("/")
        self.video_gpu_id = video_gpu_map.get(rank, 0)
        self.image_gpu_id = image_gpu_id
        self.audio_gpu_id = audio_gpu_id
        self.rank = rank
        self.encoder_fps = encoder_fps
        self.encoder_resolution = encoder_resolution
        self.data_fps = data_fps
        self.audio_sample_rate = audio_sample_rate
        self.timeout = timeout
        self._session = requests.Session()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._processor = None
        self._processor_path = None

        logger.info(
            "AsyncDistillClient: rank=%d, base_url=%s, "
            "video_gpu=%d, image_gpu=%d, audio_gpu=%d",
            rank, base_url, self.video_gpu_id, image_gpu_id, audio_gpu_id,
        )

    def set_processor_path(self, path: str):
        """Set path to Qwen processor for client-side tokenization."""
        self._processor_path = path

    def _get_processor(self):
        """Lazy-load Qwen processor on first use."""
        if self._processor is None:
            from transformers import AutoProcessor
            path = self._processor_path or "Qwen/Qwen3-Omni"
            self._processor = AutoProcessor.from_pretrained(
                path, trust_remote_code=True,
            )
            logger.info("AsyncDistillClient: Qwen processor loaded from %s", path)
        return self._processor

    @staticmethod
    def _serialize_compact(obj) -> bytes:
        """Serialize with pickle protocol 5 for zero-copy numpy/tensor buffers."""
        buf = io.BytesIO()
        pickle.dump(obj, buf, protocol=5)
        return buf.getvalue()

    @staticmethod
    def _parse_response(content: bytes) -> np.ndarray:
        """Parse pickle response, handling float16/float32/list formats."""
        data = pickle.loads(content)
        feat = data["features"]
        if isinstance(feat, np.ndarray):
            return feat.astype(np.float32) if feat.dtype != np.float32 else feat
        shape = data["shape"]
        return np.array(feat, dtype=np.float32).reshape(shape)

    def _upload_visual_tensors(
        self, preprocessed: dict, gpu_id: int,
    ) -> Dict[str, np.ndarray]:
        """Upload pre-tokenized tensors to /extract/upload/visual/tensors."""
        payload = self._serialize_compact(preprocessed)
        resp = self._session.post(
            f"{self.base_url}/extract/upload/visual/tensors",
            files={"file": ("t.pkl", io.BytesIO(payload), "application/octet-stream")},
            data={
                "feature_mode": "full",
                "response_format": "pickle",
                "gpu_id": str(gpu_id),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {"features": self._parse_response(resp.content)}

    def _upload_audio_tensor(
        self, wav_np: np.ndarray, gpu_id: int,
    ) -> Dict[str, np.ndarray]:
        """Upload numpy waveform to /extract/upload/audio/tensor."""
        payload = self._serialize_compact(wav_np)
        resp = self._session.post(
            f"{self.base_url}/extract/upload/audio/tensor",
            files={"file": ("a.pkl", io.BytesIO(payload), "application/octet-stream")},
            data={
                "feature_mode": "full",
                "response_format": "pickle",
                "gpu_id": str(gpu_id),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {"features": self._parse_response(resp.content)}

    def _upload_video_frames(
        self, frames: list, gpu_id: int,
    ) -> Dict[str, np.ndarray]:
        """Upload PIL frames to /extract/upload/visual/frames (server-side preprocess)."""
        payload = self._serialize_compact(frames)
        resp = self._session.post(
            f"{self.base_url}/extract/upload/visual/frames",
            files={"file": ("f.pkl", io.BytesIO(payload), "application/octet-stream")},
            data={
                "feature_mode": "full",
                "response_format": "pickle",
                "target_height": str(self.encoder_resolution),
                "target_width": str(self.encoder_resolution),
                "gpu_id": str(gpu_id),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {"features": self._parse_response(resp.content)}

    def _upload_image_file(
        self, pil_image, gpu_id: int,
    ) -> Dict[str, np.ndarray]:
        """Upload a single PIL image to /extract/upload/image (server-side preprocess)."""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        resp = self._session.post(
            f"{self.base_url}/extract/upload/image",
            files={"file": ("frame.png", buf, "image/png")},
            data={
                "feature_mode": "full",
                "response_format": "pickle",
                "gpu_id": str(gpu_id),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {"features": self._parse_response(resp.content)}

    def _process_and_upload_video(
        self, video_tensor: torch.Tensor, gpu_id: int,
    ) -> Dict[str, np.ndarray]:
        """Extract and upload video frames. Uses tensor mode if processor available."""
        remaining = video_tensor[:, 1:]
        frames = _tensor_to_pil_frames(remaining, self.data_fps, self.encoder_fps)
        if self._processor_path is not None:
            preprocessed = _tokenize_video_frames(
                frames, self._get_processor(),
                self.encoder_resolution, self.encoder_resolution,
            )
            return self._upload_visual_tensors(preprocessed, gpu_id)
        return self._upload_video_frames(frames, gpu_id)

    def _upload_image_tensors(
        self, preprocessed: dict, gpu_id: int,
    ) -> Dict[str, np.ndarray]:
        """Upload pre-tokenized image tensors to /extract/upload/image/tensors."""
        payload = self._serialize_compact(preprocessed)
        resp = self._session.post(
            f"{self.base_url}/extract/upload/image/tensors",
            files={"file": ("t.pkl", io.BytesIO(payload), "application/octet-stream")},
            data={
                "feature_mode": "full",
                "response_format": "pickle",
                "gpu_id": str(gpu_id),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {"features": self._parse_response(resp.content)}

    def _process_and_upload_image(
        self, video_tensor: torch.Tensor, gpu_id: int,
    ) -> Dict[str, np.ndarray]:
        """Extract and upload first frame. Uses tensor mode if processor available."""
        first_pil = _tensor_to_first_frame_pil(video_tensor)
        if self._processor_path is not None:
            preprocessed = _tokenize_image(first_pil, self._get_processor())
            return self._upload_image_tensors(preprocessed, gpu_id)
        return self._upload_image_file(first_pil, gpu_id)

    def extract_batch(
        self,
        video: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Extract semantic features for a full batch.

        Preprocessing (Qwen processor tokenization) runs on training-side threads,
        then pre-tokenized tensors are uploaded to the encoder service which
        skips all CPU preprocessing and goes straight to GPU forward.

        Args:
            video: (B, C, T, H, W), range [-1, 1]
            audio: (B, 1, T_a)

        Returns:
            dict with 'image_feat', 'video_feat', 'audio_feat' as torch tensors,
            plus '_time_video', '_time_image', '_time_audio' (seconds) for profiling,
            or None on failure.
        """
        if video is None and audio is None:
            return None

        B = video.shape[0] if video is not None else audio.shape[0]
        futures: Dict[str, List[Future]] = {
            "video": [], "image": [], "audio": [],
        }

        t_submit = time.perf_counter()

        try:
            for b in range(B):
                if video is not None:
                    v = video[b]  # (C, T, H, W)
                    futures["video"].append(
                        self._pool.submit(
                            self._process_and_upload_video, v, self.video_gpu_id,
                        )
                    )
                    futures["image"].append(
                        self._pool.submit(
                            self._process_and_upload_image, v, self.image_gpu_id,
                        )
                    )

                if audio is not None:
                    wav_np = _audio_tensor_to_16k_numpy(
                        audio[b], self.audio_sample_rate,
                    )
                    futures["audio"].append(
                        self._pool.submit(
                            self._upload_audio_tensor, wav_np, self.audio_gpu_id,
                        )
                    )

            out: Dict[str, torch.Tensor] = {}

            from concurrent.futures import wait as futures_wait

            t_video, t_image, t_audio = 0.0, 0.0, 0.0

            if futures["video"]:
                futures_wait(futures["video"])
                t_video = time.perf_counter() - t_submit
                video_feats = [f.result()["features"] for f in futures["video"]]
                out["video_feat"] = torch.from_numpy(np.stack(video_feats, axis=0))

            if futures["image"]:
                futures_wait(futures["image"])
                t_image = time.perf_counter() - t_submit
                image_feats = [f.result()["features"] for f in futures["image"]]
                out["image_feat"] = torch.from_numpy(np.stack(image_feats, axis=0))

            if futures["audio"]:
                futures_wait(futures["audio"])
                t_audio = time.perf_counter() - t_submit
                audio_feats = [f.result()["features"] for f in futures["audio"]]
                out["audio_feat"] = torch.from_numpy(np.stack(audio_feats, axis=0))

            t_total = time.perf_counter() - t_submit
            out["_time_video"] = t_video
            out["_time_image"] = t_image
            out["_time_audio"] = t_audio
            out["_time_total"] = t_total

            return out if out else None

        except Exception as e:
            logger.warning("AsyncDistillClient: extraction failed: %s", e, exc_info=True)
            return None

    def shutdown(self):
        self._pool.shutdown(wait=False)


class DistillPrefetcher:
    """Overlap encoder service calls with training by submitting requests
    as soon as data is loaded, then retrieving results when needed.

    Usage in training loop::

        batch = next(train_iter)
        prefetcher.prefetch(batch)         # non-blocking submit
        ...                                # VAE forward, etc.
        feats = prefetcher.get_features()  # blocking wait (may already be done)
    """

    def __init__(
        self,
        base_url: str,
        video_gpu_map: Dict[int, int],
        image_gpu_id: int,
        audio_gpu_id: int,
        rank: int,
        encoder_fps: float = 4.0,
        encoder_resolution: int = 256,
        data_fps: float = 24.0,
        audio_sample_rate: int = 48000,
        timeout: float = 120.0,
        max_workers: int = 6,
        needs_video: bool = True,
        needs_audio: bool = True,
        processor_path: Optional[str] = None,
    ):
        self.client = AsyncDistillClient(
            base_url=base_url,
            video_gpu_map=video_gpu_map,
            image_gpu_id=image_gpu_id,
            audio_gpu_id=audio_gpu_id,
            rank=rank,
            encoder_fps=encoder_fps,
            encoder_resolution=encoder_resolution,
            data_fps=data_fps,
            audio_sample_rate=audio_sample_rate,
            timeout=timeout,
            max_workers=max_workers,
        )
        if processor_path:
            self.client.set_processor_path(processor_path)
        self.needs_video = needs_video
        self.needs_audio = needs_audio
        self._submit_pool = ThreadPoolExecutor(max_workers=1)
        self._pending: Optional[Future] = None
        self._last_submit_time: float = 0.0
        self._last_wait_time: float = 0.0
        self._last_time_video: float = 0.0
        self._last_time_image: float = 0.0
        self._last_time_audio: float = 0.0
        self._last_time_total: float = 0.0

    @property
    def last_submit_ms(self) -> float:
        return self._last_submit_time * 1000

    @property
    def last_wait_ms(self) -> float:
        return self._last_wait_time * 1000

    @property
    def last_time_video_ms(self) -> float:
        return self._last_time_video * 1000

    @property
    def last_time_image_ms(self) -> float:
        return self._last_time_image * 1000

    @property
    def last_time_audio_ms(self) -> float:
        return self._last_time_audio * 1000

    @property
    def last_time_total_ms(self) -> float:
        return self._last_time_total * 1000

    def prefetch(self, batch: Dict) -> None:
        """Submit encoder requests for a batch (non-blocking)."""
        data = batch.get("data", batch)
        video = data.get("video") if self.needs_video else None
        audio = data.get("audio") if self.needs_audio else None

        if video is None and audio is None:
            self._pending = None
            return

        t0 = time.perf_counter()
        self._pending = self._submit_pool.submit(
            self.client.extract_batch, video, audio,
        )
        self._last_submit_time = time.perf_counter() - t0

    def get_features(
        self, device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Block until encoder results are ready, return tensors on device."""
        if self._pending is None:
            return None

        t0 = time.perf_counter()
        result = self._pending.result()
        self._last_wait_time = time.perf_counter() - t0
        self._pending = None

        if result is None:
            return None

        self._last_time_video = result.pop("_time_video", 0.0)
        self._last_time_image = result.pop("_time_image", 0.0)
        self._last_time_audio = result.pop("_time_audio", 0.0)
        self._last_time_total = result.pop("_time_total", 0.0)

        return {k: v.to(device) for k, v in result.items()}

    def shutdown(self):
        self._submit_pool.shutdown(wait=False)
        self.client.shutdown()
