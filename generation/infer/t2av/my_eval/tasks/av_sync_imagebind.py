"""AV-sync + ImageBind based metrics.

Produces four per-sample metrics that all share the same heavy backbone load
(Synchformer + ImageBind):

* DeSync     -- Synchformer offset prediction. MOVA-style 2-window average from
                ``eval_av_quality.py:380-390``.
* AV-Align   -- ffmpeg/optical-flow IoU from MOVA ``av_align_score.py`` (mux on
                the fly because that pipeline reads audio out of mp4). Optional.
* IB-AV      -- ImageBind cosine similarity between video and audio modalities.
* IB-TV      -- ImageBind cosine similarity between video and av_caption text.
* IB-TA      -- ImageBind cosine similarity between audio_prompt text and audio.

ImageBind preprocessing reuses the official vendored helpers from
``generation/evaluation/models/imagebind/imagebind/data.py``; Synchformer
preprocessing is done with a small pyav-based decoder (the upstream
``av_bench.data.video_dataset`` is not vendored locally).
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[5]
AV_QUALITY_DIR = REPO_ROOT / "generation" / "evaluation" / "metrics" / "av_quality"
IMAGEBIND_DIR = REPO_ROOT / "generation" / "evaluation" / "models" / "imagebind"
PYTORCHVIDEO_DIR = REPO_ROOT / "generation" / "evaluation" / "models" / "pytorchvideo"


def _shim_torchvision_functional_tensor() -> None:
    """torchvision >= 0.17 dropped ``transforms.functional_tensor``; pytorchvideo
    0.1.5 still imports it. Re-export the new functional module under the old
    name before pytorchvideo / imagebind are imported."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional as _tvF
        sys.modules["torchvision.transforms.functional_tensor"] = _tvF
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[av_sync_imagebind] could not shim torchvision.functional_tensor: {exc}", flush=True)


def _shim_imagebind_video_loader() -> None:
    """Vendored imagebind/data.py passes ``sample_rate=...`` to
    ``EncodedVideo.from_path``; that kwarg was removed in modern pytorchvideo
    so the call crashes with ``TypeError: ... unexpected keyword argument
    'sample_rate'``. Replace the helper with an equivalent version that drops
    the kwarg (decode_audio=False makes the audio sample rate irrelevant
    anyway).

    We grab every symbol we need from ``imagebind.data`` itself so we don't
    have to re-import any torchvision / pytorchvideo internals that the
    upstream module already successfully imported.
    """
    try:
        import imagebind.data as _ib_data  # type: ignore
    except Exception as exc:
        print(f"[av_sync_imagebind] could not import imagebind.data for shim: {exc}", flush=True)
        return

    EncodedVideo = _ib_data.EncodedVideo
    ConstantClipsPerVideoSampler = _ib_data.ConstantClipsPerVideoSampler
    pv_transforms = _ib_data.pv_transforms
    NormalizeVideo = _ib_data.NormalizeVideo
    transforms = _ib_data.transforms
    SpatialCrop = _ib_data.SpatialCrop
    get_clip_timepoints = _ib_data.get_clip_timepoints

    def _patched_load_and_transform_video_data(
        video_paths,
        device,
        clip_duration: float = 2,
        clips_per_video: int = 5,
        sample_rate: int = 16000,  # accepted for backward compat; unused
    ):
        if video_paths is None:
            return None
        video_outputs = []
        video_transform = transforms.Compose([
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
        frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)
        for video_path in video_paths:
            # Use pyav: it's already installed in the verse-bench env (see
            # common.sh's install_python_requirements) and produces identical
            # frame samples for our 0.5 fps / 5-clip setup. Avoids needing
            # an extra ``pip install decord``.
            video = EncodedVideo.from_path(
                video_path,
                decoder="pyav",
                decode_audio=False,
            )
            all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)
            all_video = []
            for clip_timepoints in all_clips_timepoints:
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                if clip is None:
                    raise ValueError(f"No clip found in {video_path}")
                video_clip = frame_sampler(clip["video"])
                video_clip = video_clip / 255.0
                all_video.append(video_clip)
            all_video = [video_transform(c) for c in all_video]
            all_video = SpatialCrop(224, num_crops=3)(all_video)
            all_video = torch.stack(all_video, dim=0)
            video_outputs.append(all_video)
        return torch.stack(video_outputs, dim=0).to(device)

    _ib_data.load_and_transform_video_data = _patched_load_and_transform_video_data
    print("[av_sync_imagebind] patched imagebind.data.load_and_transform_video_data (sample_rate kwarg dropped)", flush=True)


def _shim_imagebind_audio_loader() -> None:
    try:
        import imagebind.data as _ib_data  # type: ignore
    except Exception as exc:
        print(f"[av_sync_imagebind] could not import imagebind.data for audio shim: {exc}", flush=True)
        return

    ConstantClipsPerVideoSampler = _ib_data.ConstantClipsPerVideoSampler
    get_clip_timepoints = _ib_data.get_clip_timepoints
    waveform2melspec = _ib_data.waveform2melspec
    transforms = _ib_data.transforms

    def _patched_load_and_transform_audio_data(
        audio_paths,
        device,
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2,
        clips_per_video=3,
        mean=-4.268,
        std=9.138,
    ):
        if audio_paths is None:
            return None
        audio_outputs = []
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
        normalize = transforms.Normalize(mean=mean, std=std)
        for audio_path in audio_paths:
            audio, _ = load_wav_mono(audio_path, sample_rate)
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            all_clips_timepoints = get_clip_timepoints(
                clip_sampler, waveform.size(1) / sample_rate
            )
            all_clips = []
            for clip_timepoints in all_clips_timepoints:
                waveform_clip = waveform[
                    :,
                    int(clip_timepoints[0] * sample_rate): int(clip_timepoints[1] * sample_rate),
                ]
                waveform_melspec = waveform2melspec(
                    waveform_clip, sample_rate, num_mel_bins, target_length
                )
                all_clips.append(normalize(waveform_melspec).to(device))
            audio_outputs.append(torch.stack(all_clips, dim=0))
        return torch.stack(audio_outputs, dim=0)

    _ib_data.load_and_transform_audio_data = _patched_load_and_transform_audio_data
    print("[av_sync_imagebind] patched imagebind.data.load_and_transform_audio_data (shared audio cache)", flush=True)


def _ensure_pythonpath() -> None:
    # NOTE: do NOT prepend PYTORCHVIDEO_DIR. The vendored copy is missing the
    # ``pytorchvideo.data`` submodule that imagebind/data.py needs; prepending
    # it would shadow the pip-installed full version. We install pytorchvideo
    # via setup_my_eval_deps.sh and rely on the site-packages copy.
    for p in (str(AV_QUALITY_DIR), str(IMAGEBIND_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _shim_torchvision_functional_tensor()
    # Only fall back to vendored pytorchvideo if nothing else is installed; in
    # that case we append (not prepend) so a real pip install still wins.
    try:
        import pytorchvideo.data  # noqa: F401
    except ImportError:
        vendored = str(PYTORCHVIDEO_DIR)
        if vendored not in sys.path:
            sys.path.append(vendored)


_ensure_pythonpath()

from my_eval.utils.distributed import log, slice_for_rank
from my_eval.utils.io_utils import already_done, write_per_sample
from my_eval.utils.audio_video import mux_av, rank_tmp_dir, load_wav_mono


_SYNC_FPS = 25.0
_SYNC_SIZE = 224
_SYNC_MODEL_CACHE: Dict[str, tuple[Any, Any, Any]] = {}
_IMAGEBIND_MODEL_CACHE: Dict[str, Any] = {}
_IMAGEBIND_KEYS = {"IB-AV", "IB-TV", "IB-TA"}
_SYNCHFORMER_KEYS = {"DeSync"}


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _add_timing(timing: Dict[str, float], key: str, elapsed: float) -> None:
    timing[key] = float(timing.get(key, 0.0)) + float(elapsed)


def _decode_video_pyav(video_path: str, fps: float, expected_length: int) -> torch.Tensor:
    """Decode video into (T, C, H, W) uint8 RGB frames at the requested fps."""
    import av as pyav

    container = pyav.open(video_path)
    try:
        video_stream = container.streams.video[0]
        graph = pyav.filter.Graph()
        source = graph.add_buffer(template=video_stream)
        fps_filter = graph.add("fps", f"fps={fps}:round=near:start_time=0")
        fmt_filter = graph.add("format", "pix_fmts=rgb24")
        sink = graph.add("buffersink")
        source.link_to(fps_filter)
        fps_filter.link_to(fmt_filter)
        fmt_filter.link_to(sink)
        graph.configure()

        frames: List[torch.Tensor] = []
        for frame in container.decode(video_stream):
            source.push(frame)
            while True:
                try:
                    out_frame = sink.pull()
                except Exception as exc:
                    name = exc.__class__.__name__
                    if name in {"BlockingIOError", "EOFError", "FFmpegError"}:
                        break
                    raise
                frames.append(torch.from_numpy(out_frame.to_ndarray(format="rgb24")))
                if len(frames) >= expected_length:
                    break
            if len(frames) >= expected_length:
                break
        source.push(None)
        while len(frames) < expected_length:
            try:
                out_frame = sink.pull()
            except Exception as exc:
                name = exc.__class__.__name__
                if name in {"BlockingIOError", "EOFError", "FFmpegError"}:
                    break
                raise
            frames.append(torch.from_numpy(out_frame.to_ndarray(format="rgb24")))
    finally:
        container.close()

    if len(frames) == 0:
        raise RuntimeError(f"no frames decoded from {video_path}")
    while len(frames) < expected_length:
        frames.append(frames[-1])
    return torch.stack(frames[:expected_length]).permute(0, 3, 1, 2).contiguous()


def _sync_transform(video_tchw_uint8: torch.Tensor) -> torch.Tensor:
    from torchvision.transforms import v2 as T
    pipeline = T.Compose([
        T.Resize(_SYNC_SIZE, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(_SYNC_SIZE),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return pipeline(video_tchw_uint8)


def _load_audio_for_sync(audio_path: str, duration_sec: float) -> torch.Tensor:
    audio, _ = load_wav_mono(audio_path, 16000)
    waveform = torch.from_numpy(audio).float()
    expected = int(16000 * duration_sec)
    if waveform.size(0) < expected:
        waveform = torch.nn.functional.pad(waveform, (0, expected - waveform.size(0)))
    else:
        waveform = waveform[:expected]
    return waveform  # (T,)


def _pad_or_truncate_spec(audio: torch.Tensor, max_spec_t: int) -> torch.Tensor:
    """Inlined copy of av_bench.data.audio_dataset.pad_or_truncate (which is not
    locally vendored). Pads (or truncates) along the last dim with zeros."""
    diff = max_spec_t - audio.shape[-1]
    if diff > 0:
        audio = torch.nn.functional.pad(audio, (0, diff))
    elif diff < 0:
        audio = audio[..., :max_spec_t]
    return audio


def _encode_video_with_sync(synchformer, sync_video_tchw: torch.Tensor) -> torch.Tensor:
    """(T, C, H, W) -> (S, t, D)."""
    from einops import rearrange
    t = sync_video_tchw.shape[0]
    segment_size = 16
    step_size = 8
    num_segments = max((t - segment_size) // step_size + 1, 1)
    segments = []
    for i in range(num_segments):
        s = i * step_size
        e = s + segment_size
        if e > t:
            seg = sync_video_tchw[max(t - segment_size, 0):t]
        else:
            seg = sync_video_tchw[s:e]
        if seg.shape[0] < segment_size:
            pad = seg.shape[0]
            extra = seg[-1:].expand(segment_size - pad, -1, -1, -1).clone()
            seg = torch.cat([seg, extra], dim=0)
        segments.append(seg)
    x = torch.stack(segments, dim=0).unsqueeze(0)  # (1, S, T, C, H, W)
    x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
    feats = synchformer.extract_vfeats(x)
    feats = rearrange(feats, "(b s) 1 t d -> b s t d", b=1)
    return feats[0]


def _encode_audio_with_sync(synchformer, wav_t: torch.Tensor, mel) -> torch.Tensor:
    """1D waveform -> (S, t, D)."""
    x = wav_t.unsqueeze(0)
    b, t = x.shape
    segment_size = 10240
    step_size = 10240 // 2
    num_segments = max((t - segment_size) // step_size + 1, 1)
    segments = []
    for i in range(num_segments):
        s = i * step_size
        e = s + segment_size
        if e > t:
            seg = x[:, max(t - segment_size, 0):t]
        else:
            seg = x[:, s:e]
        if seg.shape[1] < segment_size:
            seg = torch.nn.functional.pad(seg, (0, segment_size - seg.shape[1]))
        segments.append(seg)
    x = torch.stack(segments, dim=1)
    x = mel(x)
    x = torch.log(x + 1e-6)
    x = _pad_or_truncate_spec(x, 66)
    mean_v = -4.2677393
    std_v = 4.5689974
    x = (x - mean_v) / (2 * std_v)
    x = synchformer.extract_afeats(x.unsqueeze(2))
    return x[0]


def _imagebind_video_embedding(
    imagebind,
    video_paths: List[str],
    device: torch.device,
    timing: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    from imagebind.data import load_and_transform_video_data
    from imagebind.models.imagebind_model import ModalityType
    started_at = time.time()
    x = load_and_transform_video_data(video_paths, device)
    _cuda_sync(device)
    if timing is not None:
        _add_timing(timing, "imagebind_preprocess_elapsed_sec", time.time() - started_at)
    started_at = time.time()
    emb = imagebind({ModalityType.VISION: x})[ModalityType.VISION]
    _cuda_sync(device)
    if timing is not None:
        _add_timing(timing, "imagebind_infer_elapsed_sec", time.time() - started_at)
    return emb


def _imagebind_audio_embedding(
    imagebind,
    audio_paths: List[str],
    device: torch.device,
    timing: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    from imagebind.data import load_and_transform_audio_data
    from imagebind.models.imagebind_model import ModalityType
    started_at = time.time()
    x = load_and_transform_audio_data(audio_paths, device)
    _cuda_sync(device)
    if timing is not None:
        _add_timing(timing, "imagebind_preprocess_elapsed_sec", time.time() - started_at)
    started_at = time.time()
    emb = imagebind({ModalityType.AUDIO: x})[ModalityType.AUDIO]
    _cuda_sync(device)
    if timing is not None:
        _add_timing(timing, "imagebind_infer_elapsed_sec", time.time() - started_at)
    return emb


def _imagebind_text_embedding(
    imagebind,
    prompts: List[str],
    device: torch.device,
    timing: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    from imagebind.data import load_and_transform_text
    from imagebind.models.imagebind_model import ModalityType
    started_at = time.time()
    tokens = load_and_transform_text(prompts, device)
    _cuda_sync(device)
    if timing is not None:
        _add_timing(timing, "imagebind_preprocess_elapsed_sec", time.time() - started_at)
    started_at = time.time()
    emb = imagebind({ModalityType.TEXT: tokens})[ModalityType.TEXT]
    _cuda_sync(device)
    if timing is not None:
        _add_timing(timing, "imagebind_infer_elapsed_sec", time.time() - started_at)
    return emb


def _try_av_align(video_with_audio_path: str) -> float:
    if str(AV_QUALITY_DIR) not in sys.path:
        sys.path.insert(0, str(AV_QUALITY_DIR))
    from av_align_score import compute_single_video_av  # type: ignore
    try:
        _, score = compute_single_video_av(video_with_audio_path)
        return float(score)
    except Exception as exc:
        print(f"[av_sync_imagebind] AV-Align failed for {video_with_audio_path}: {exc}", flush=True)
        return float("nan")


def _resolve_synchformer_ckpt() -> Path:
    explicit = os.environ.get("MY_EVAL_SYNCHFORMER_CKPT")
    if explicit:
        return Path(explicit).expanduser()
    candidates = [
        AV_QUALITY_DIR / "weights" / "synchformer_state_dict.pth",
        REPO_ROOT / "generation" / "evaluation" / "verse_bench" / "models" / "24-01-04T16-39-21.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Synchformer checkpoint not found. Set MY_EVAL_SYNCHFORMER_CKPT or place "
        "synchformer_state_dict.pth under av_quality/weights/."
    )


def _load_synchformer(device: torch.device):
    from av_bench.synchformer.synchformer import Synchformer  # type: ignore
    import torchaudio

    print(f"[av_sync_synchformer] loading Synchformer on {device}", flush=True)
    load_timing: Dict[str, float] = {}
    started_at = time.time()
    sync_model = Synchformer().to(device).eval()
    ckpt_path = _resolve_synchformer_ckpt()
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("vfeat_") for k in sd):
        sd = sd["model"]
    missing, unexpected = sync_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[av_sync_synchformer] synchformer missing keys: {len(missing)} (head={missing[:3]})", flush=True)
    if unexpected:
        print(f"[av_sync_synchformer] synchformer unexpected keys: {len(unexpected)} (head={unexpected[:3]})", flush=True)
    _cuda_sync(device)
    load_timing["synchformer_model_load_elapsed_sec"] = time.time() - started_at

    started_at = time.time()
    sync_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, win_length=400, hop_length=160, n_fft=1024, n_mels=128
    ).to(device)
    sync_grid = torch.from_numpy(np.linspace(-2, 2, 21)).float()
    _cuda_sync(device)
    load_timing["synchformer_aux_setup_elapsed_sec"] = time.time() - started_at
    return sync_model, sync_mel, sync_grid, load_timing


def _load_imagebind(device: torch.device):
    from imagebind.models import imagebind_model

    print(f"[av_sync_imagebind] loading ImageBind on {device}", flush=True)
    started_at = time.time()
    imagebind = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
    _cuda_sync(device)
    load_timing = {"imagebind_model_load_elapsed_sec": time.time() - started_at}
    _shim_imagebind_video_loader()
    _shim_imagebind_audio_loader()
    return imagebind, load_timing


def _resolve_device(local_rank: int) -> torch.device:
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return device


def _zero_load_timing() -> Dict[str, float]:
    return {
        "synchformer_model_load_elapsed_sec": 0.0,
        "imagebind_model_load_elapsed_sec": 0.0,
        "synchformer_aux_setup_elapsed_sec": 0.0,
    }


def _get_synchformer(device: torch.device, *, reuse_models: bool):
    key = str(device)
    if reuse_models and key in _SYNC_MODEL_CACHE:
        sync_model, sync_mel, sync_grid = _SYNC_MODEL_CACHE[key]
        return sync_model, sync_mel, sync_grid, _zero_load_timing(), 0.0
    started_at = time.time()
    sync_model, sync_mel, sync_grid, load_timing = _load_synchformer(device)
    elapsed = time.time() - started_at
    if reuse_models:
        _SYNC_MODEL_CACHE[key] = (sync_model, sync_mel, sync_grid)
    return sync_model, sync_mel, sync_grid, load_timing, elapsed


def _get_imagebind(device: torch.device, *, reuse_models: bool):
    key = str(device)
    if reuse_models and key in _IMAGEBIND_MODEL_CACHE:
        return _IMAGEBIND_MODEL_CACHE[key], _zero_load_timing(), 0.0
    started_at = time.time()
    imagebind, load_timing = _load_imagebind(device)
    elapsed = time.time() - started_at
    if reuse_models:
        _IMAGEBIND_MODEL_CACHE[key] = imagebind
    timing = _zero_load_timing()
    timing.update(load_timing)
    return imagebind, timing, elapsed


def clear_model_cache() -> List[str]:
    cleared: List[str] = []
    if _SYNC_MODEL_CACHE:
        _SYNC_MODEL_CACHE.clear()
        cleared.append("_SYNC_MODEL_CACHE")
    if _IMAGEBIND_MODEL_CACHE:
        _IMAGEBIND_MODEL_CACHE.clear()
        cleared.append("_IMAGEBIND_MODEL_CACHE")
    return cleared


def _needs_synchformer(metric_key_set: set[str]) -> bool:
    return bool(metric_key_set & _SYNCHFORMER_KEYS)


def _needs_imagebind(metric_key_set: set[str]) -> bool:
    return bool(metric_key_set & _IMAGEBIND_KEYS)


def preload_task(
    rank: int,
    local_rank: int,
    metric_keys: Optional[List[str]] = None,
    **_: Any,
) -> Dict[str, float]:
    metric_key_set = set(metric_keys or ["IB-AV", "IB-TV", "IB-TA"])
    device = _resolve_device(local_rank)
    timing = _zero_load_timing()
    elapsed = 0.0
    if _needs_synchformer(metric_key_set):
        _, _, _, load_timing, part = _get_synchformer(device, reuse_models=True)
        timing.update(load_timing)
        elapsed += part
    if _needs_imagebind(metric_key_set):
        _, load_timing, part = _get_imagebind(device, reuse_models=True)
        timing.update(load_timing)
        elapsed += part
    timing["model_load_elapsed_sec"] = elapsed
    log(rank, f"[av_sync_imagebind] preload complete keys={sorted(metric_key_set)} model_load={elapsed:.3f}s")
    return timing


def _ib_tv_text(rec: Dict[str, Any]) -> str:
    # New manifests store av_caption explicitly. Older manifests use "prompt"
    # for the same expanded joint caption, so keep that as the primary fallback.
    return (rec.get("av_caption") or rec.get("prompt") or rec.get("video_prompt") or "").strip()


def _ib_ta_text(rec: Dict[str, Any]) -> str:
    return (rec.get("audio_prompt") or rec.get("prompt") or rec.get("av_caption") or "").strip()


def _iter_batches(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _compute_imagebind_batch(
    imagebind,
    batch: List[Dict[str, Any]],
    device: torch.device,
    metric_key_set: set[str],
    timing: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {
        rec["file_stem"]: {"IB-AV": float("nan"), "IB-TV": float("nan"), "IB-TA": float("nan")}
        for rec in batch
    }
    need_video = "IB-AV" in metric_key_set or "IB-TV" in metric_key_set
    need_audio = "IB-AV" in metric_key_set or "IB-TA" in metric_key_set
    need_tv_text = "IB-TV" in metric_key_set
    need_ta_text = "IB-TA" in metric_key_set

    v_emb = (
        _imagebind_video_embedding(imagebind, [rec["video_path"] for rec in batch], device, timing)
        if need_video else None
    )
    a_emb = (
        _imagebind_audio_embedding(imagebind, [rec["audio_path"] for rec in batch], device, timing)
        if need_audio else None
    )

    if "IB-AV" in metric_key_set and v_emb is not None and a_emb is not None:
        vals = F.cosine_similarity(v_emb, a_emb, dim=-1).detach().float().cpu().tolist()
        for rec, val in zip(batch, vals):
            scores[rec["file_stem"]]["IB-AV"] = float(val)

    if need_tv_text and v_emb is not None:
        tv_items = [(i, rec, _ib_tv_text(rec)) for i, rec in enumerate(batch) if _ib_tv_text(rec)]
        if tv_items:
            tv_emb = _imagebind_text_embedding(imagebind, [text for _, _, text in tv_items], device, timing)
            tv_v_emb = v_emb[[i for i, _, _ in tv_items]]
            vals = F.cosine_similarity(tv_v_emb, tv_emb, dim=-1).detach().float().cpu().tolist()
            for (_, rec, _), val in zip(tv_items, vals):
                scores[rec["file_stem"]]["IB-TV"] = float(val)

    if need_ta_text and a_emb is not None:
        ta_items = [(i, rec, _ib_ta_text(rec)) for i, rec in enumerate(batch) if _ib_ta_text(rec)]
        if ta_items:
            ta_emb = _imagebind_text_embedding(imagebind, [text for _, _, text in ta_items], device, timing)
            ta_a_emb = a_emb[[i for i, _, _ in ta_items]]
            vals = F.cosine_similarity(ta_emb, ta_a_emb, dim=-1).detach().float().cpu().tolist()
            for (_, rec, _), val in zip(ta_items, vals):
                scores[rec["file_stem"]]["IB-TA"] = float(val)

    return scores


def _compute_imagebind_scores(
    imagebind,
    records: List[Dict[str, Any]],
    device: torch.device,
    metric_key_set: set[str],
    rank: int,
    batch_size: int,
    timing: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    if not ({"IB-AV", "IB-TV", "IB-TA"} & metric_key_set):
        return scores
    batch_size = max(1, int(batch_size))
    for batch in _iter_batches(records, batch_size):
        try:
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                scores.update(_compute_imagebind_batch(imagebind, batch, device, metric_key_set, timing))
        except Exception as exc:
            if len(batch) == 1:
                log(rank, f"ImageBind failed for {batch[0]['file_stem']}: {exc}")
                continue
            log(rank, f"ImageBind batch failed ({len(batch)} samples): {exc}; retry one-by-one")
            for rec in batch:
                try:
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        scores.update(_compute_imagebind_batch(imagebind, [rec], device, metric_key_set, timing))
                except Exception as one_exc:
                    log(rank, f"ImageBind failed for {rec['file_stem']}: {one_exc}")
    return scores


@torch.inference_mode()
def run_task(
    rank: int,
    local_rank: int,
    world_size: int,
    target_dir: Path,
    manifest: Dict[str, Any],
    skip_completed: bool = True,
    metric_keys: Optional[List[str]] = None,
    duration_sec: Optional[float] = None,
    reuse_models: bool = False,
    output_kind: str = "av_sync_imagebind",
    **_: Any,
) -> Dict[str, float]:
    model_load_elapsed_sec = 0.0
    metric_keys = metric_keys or ["IB-AV", "IB-TV", "IB-TA"]
    metric_key_set = set(metric_keys)
    records = list(manifest.get("records", []))
    my_records = slice_for_rank(records, rank, world_size)
    log(rank, f"[av_sync_imagebind] my_records={len(my_records)}/{len(records)}")
    if not my_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    pending_records = [
        rec for rec in my_records
        if not (skip_completed and already_done(target_dir, output_kind, rec["file_stem"], metric_keys))
    ]
    if not pending_records:
        return {"model_load_elapsed_sec": model_load_elapsed_sec}

    if duration_sec is None:
        duration_sec = float(os.environ.get("MY_EVAL_DURATION_SEC", 8.0))
    imagebind_batch_size = int(os.environ.get("MY_EVAL_IMAGEBIND_BATCH_SIZE", "4"))

    device = _resolve_device(local_rank)
    timing: Dict[str, float] = {
        "synchformer_model_load_elapsed_sec": 0.0,
        "imagebind_model_load_elapsed_sec": 0.0,
        "synchformer_aux_setup_elapsed_sec": 0.0,
        "imagebind_preprocess_elapsed_sec": 0.0,
        "imagebind_infer_elapsed_sec": 0.0,
        "synchformer_preprocess_elapsed_sec": 0.0,
        "synchformer_infer_elapsed_sec": 0.0,
        "av_align_elapsed_sec": 0.0,
    }
    sync_model = None
    sync_mel = None
    sync_grid = None
    imagebind = None
    if _needs_synchformer(metric_key_set):
        sync_model, sync_mel, sync_grid, load_timing, part = _get_synchformer(
            device, reuse_models=reuse_models
        )
        timing.update(load_timing)
        model_load_elapsed_sec += part
    if _needs_imagebind(metric_key_set):
        imagebind, load_timing, part = _get_imagebind(device, reuse_models=reuse_models)
        timing.update(load_timing)
        model_load_elapsed_sec += part

    rank_tmp = rank_tmp_dir(target_dir, output_kind, rank)
    sync_expected = max(int(round(_SYNC_FPS * duration_sec)), 16)
    ib_scores = (
        _compute_imagebind_scores(
            imagebind=imagebind,
            records=pending_records,
            device=device,
            metric_key_set=metric_key_set,
            rank=rank,
            batch_size=imagebind_batch_size,
            timing=timing,
        )
        if imagebind is not None else {}
    )
    if {"IB-AV", "IB-TV", "IB-TA"} & metric_key_set:
        log(rank, f"[av_sync_imagebind] ImageBind batched scoring done "
                  f"(batch_size={max(1, imagebind_batch_size)})")

    for idx, rec in enumerate(pending_records):
        stem = rec["file_stem"]
        video_path = rec["video_path"]
        audio_path = rec["audio_path"]

        payload: Dict[str, Any] = {
            "DeSync": float("nan"),
            "IB-AV": float("nan"),
            "IB-TV": float("nan"),
            "IB-TA": float("nan"),
            "video_path": video_path,
            "audio_path": audio_path,
        }
        if "AV-Align" in metric_key_set:
            payload["AV-Align"] = float("nan")
        payload.update(ib_scores.get(stem, {}))
        payload["ib_tv_text"] = _ib_tv_text(rec)
        payload["ib_ta_text"] = _ib_ta_text(rec)

        muxed_str = video_path

        # --- Synchformer DeSync ----------------------------------------------
        if "DeSync" in metric_key_set:
            try:
                assert sync_model is not None and sync_mel is not None and sync_grid is not None
                preprocess_started_at = time.time()
                sync_raw = _decode_video_pyav(video_path, fps=_SYNC_FPS, expected_length=sync_expected)
                sync_tensor = _sync_transform(sync_raw[:sync_expected]).to(device)
                sync_wav = _load_audio_for_sync(audio_path, duration_sec).to(device)
                _cuda_sync(device)
                _add_timing(timing, "synchformer_preprocess_elapsed_sec", time.time() - preprocess_started_at)
                infer_started_at = time.time()
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    sync_v_feat = _encode_video_with_sync(sync_model, sync_tensor)
                    sync_a_feat = _encode_audio_with_sync(sync_model, sync_wav, sync_mel)

                    off1 = float("nan")
                    off2 = float("nan")
                    try:
                        v_first = sync_v_feat[:14].unsqueeze(0)
                        a_first = sync_a_feat[:14].unsqueeze(0)
                        logits = sync_model.compare_v_a(v_first, a_first)
                        off1 = float(abs(sync_grid[int(torch.argmax(logits, dim=-1).item())].item()))
                    except Exception as exc:
                        log(rank, f"DeSync(first14) failed for {stem}: {exc}")
                    try:
                        v_last = sync_v_feat[-14:].unsqueeze(0)
                        a_last = sync_a_feat[-14:].unsqueeze(0)
                        logits = sync_model.compare_v_a(v_last, a_last)
                        off2 = float(abs(sync_grid[int(torch.argmax(logits, dim=-1).item())].item()))
                    except Exception as exc:
                        log(rank, f"DeSync(last14) failed for {stem}: {exc}")
                    if math.isfinite(off1) and math.isfinite(off2):
                        payload["DeSync"] = (off1 + off2) / 2.0
                    elif math.isfinite(off1):
                        payload["DeSync"] = off1
                    elif math.isfinite(off2):
                        payload["DeSync"] = off2
                _cuda_sync(device)
                _add_timing(timing, "synchformer_infer_elapsed_sec", time.time() - infer_started_at)
            except Exception as exc:
                log(rank, f"Synchformer failed for {stem}: {exc}")

        # --- AV-Align ---------------------------------------------------------
        if "AV-Align" in metric_key_set:
            try:
                av_align_started_at = time.time()
                muxed = rank_tmp / f"{stem}.av.mp4"
                muxed_str = mux_av(video_path, audio_path, str(muxed))
                payload["AV-Align"] = _try_av_align(muxed_str)
                _add_timing(timing, "av_align_elapsed_sec", time.time() - av_align_started_at)
            except Exception as exc:
                log(rank, f"AV-Align failed for {stem}: {exc}")

        write_per_sample(target_dir, output_kind, stem, payload)
        if (idx + 1) % 10 == 0:
            log(rank, f"  {output_kind} {idx + 1}/{len(pending_records)}")

    if not reuse_models:
        if sync_model is not None:
            del sync_model, sync_mel
        if imagebind is not None:
            del imagebind
    if torch.cuda.is_available() and not reuse_models:
        torch.cuda.empty_cache()
    timing["model_load_elapsed_sec"] = model_load_elapsed_sec
    return timing
