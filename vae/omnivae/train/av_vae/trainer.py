"""
AudioVideoVAETrainer — 音视频 VAE 联合训练器
"""

import math
import os
import gc
import json
import time
import yaml
import logging
import shutil
import random
import itertools
from pathlib import Path
from contextlib import nullcontext, ExitStack
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import soundfile as sf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torch.optim.lr_scheduler import CosineAnnealingLR  # kept for backward compat
from torchaudio.transforms import MelSpectrogram
from einops import rearrange
from tqdm import tqdm

import lpips
from omnivae.models.audio_video_vae import AudioVideoVAE
from omnivae.dataset.audio_video_streaming_dataset import (
    build_audio_video_streaming_dataset,
    build_video_only_dataset,
    build_audio_only_dataset,
    scan_jsonl_files,
    AudioVideoCollator,
)
from omnivae.dataset.video_utils import save_video_tensor_to_mp4
from omnivae.models.causalvideovae.eval.cal_ssim import calculate_ssim
from omnivae.models.causalvideovae.eval.cal_fvd import calculate_fvd
from omnivae.eval.audio.stoi import evaluate_stoi
from omnivae.eval.audio.pesq_local import evaluate_pesq
from omnivae.eval.audio.speaker_similarity import evaluate_sim

from .utils import (
    _project_root,
    accum_log,
    _resolve_cfg_reference,
    _parse_positive_int_list,
    _format_int_list_suffix,
    _parse_dtype,
    find_latest_checkpoint,
)
from .losses import (
    MultiResolutionMelSpectrogramLoss,
    WaveformLoss,
    compute_segment_intra_precision,
    compute_segment_sampled_precision,
    compute_global_sampled_precision,
)
from .distill_loss import (
    SemanticFeatureClient,
    LocalSemanticEncoder,
    marginal_cosine_similarity_loss,
    marginal_distance_matrix_loss,
    d_axis_distill_loss,
    spatial_normalize,
)
from .disc_loss import feature_loss, discriminator_loss, adversarial_loss
from .video_disc_loss import (
    hinge_d_loss as video_hinge_d_loss,
    vanilla_d_loss as video_vanilla_d_loss,
    generator_loss as video_generator_loss,
    calculate_adaptive_weight as video_calc_adaptive_weight,
)
from omnivae.models.audio_video_vae.discriminators import (
    build_audio_discriminators,
    build_video_discriminator,
)
from .state import StreamingTrainState, EMA


def _avg_loss_dicts(dicts):
    """Average a list of loss dicts by key.

    Returns a dict mapping each key to the arithmetic mean over the
    dicts that contain that key. Non-numeric values are taken from the
    last dict that contains them. Used to aggregate per-micro-batch
    loss scalars across a gradient-accumulation cycle.
    """
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dict(dicts[0])
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    passthrough: Dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
            else:
                passthrough[k] = v
    out: Dict[str, Any] = {k: sums[k] / counts[k] for k in sums}
    for k, v in passthrough.items():
        out.setdefault(k, v)
    return out


class _MultiGroupWarmupCosineScheduler:
    """Per-group warmup + cosine scheduler.

    Replaces the previous single `LambdaLR` to support a dedicated schedule
    for the `video_vae` param group (offset clock + its own warmup / total
    steps / min ratio). All other named groups follow the global clock.

    Robust to `optimizer.add_param_group(...)` calls made after construction
    (e.g. video_vae being added at the phase-freeze unfreeze step, or
    video_logvar being added inside `_build_loss_functions`): the group's
    current `lr` is captured lazily as its `base_lr` the first time it is
    seen by `step()`.

    API-compatible with `torch.optim.lr_scheduler.LambdaLR`: exposes
    `step()`, `state_dict()`, `load_state_dict()`. Also tolerates loading
    old `LambdaLR` state dicts (only reads `last_epoch` as the step counter;
    `base_lrs_by_name` fall back to the values captured at construction).
    """

    def __init__(
        self,
        optimizer,
        *,
        g_warmup: int,
        g_total: int,
        v_warmup: int,
        v_total: int,
        v_start: int,
        v_min_ratio: float,
        a_warmup: int = 0,
        a_total: int = 0,
        a_start: int = 0,
        a_min_ratio: float = 0.0,
    ):
        self.optimizer = optimizer
        self._g_warmup = max(1, int(g_warmup))
        self._g_total = max(self._g_warmup + 1, int(g_total))
        self._v_warmup = max(1, int(v_warmup))
        self._v_total = max(self._v_warmup + 1, int(v_total))
        self._v_start = max(0, int(v_start))
        self._v_min_ratio = float(v_min_ratio)
        self._a_warmup = max(1, int(a_warmup))
        self._a_total = max(self._a_warmup + 1, int(a_total))
        self._a_start = max(0, int(a_start))
        self._a_min_ratio = float(a_min_ratio)
        self._last_step = 0
        self._base_lrs: Dict[int, float] = {id(g): float(g['lr']) for g in optimizer.param_groups}

    def _factor_global(self, step: int) -> float:
        if step < self._g_warmup:
            return float(step) / float(self._g_warmup)
        progress = float(step - self._g_warmup) / float(max(1, self._g_total - self._g_warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _factor_video(self, step: int) -> float:
        if step < self._v_start:
            return 0.0
        local = step - self._v_start
        if local < self._v_warmup:
            return float(local) / float(self._v_warmup)
        progress = float(local - self._v_warmup) / float(max(1, self._v_total - self._v_warmup))
        cos = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return self._v_min_ratio + (1.0 - self._v_min_ratio) * cos

    def _factor_audio(self, step: int) -> float:
        if step < self._a_start:
            return 0.0
        local = step - self._a_start
        if local < self._a_warmup:
            return float(local) / float(self._a_warmup)
        progress = float(local - self._a_warmup) / float(max(1, self._a_total - self._a_warmup))
        cos = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return self._a_min_ratio + (1.0 - self._a_min_ratio) * cos

    def _group_factor(self, name: str, step: int) -> float:
        if name == 'video_vae':
            return self._factor_video(step)
        if name == 'audio_vae':
            return self._factor_audio(step)
        return self._factor_global(step)

    def step(self) -> None:
        self._last_step += 1
        for g in self.optimizer.param_groups:
            gid = id(g)
            if gid not in self._base_lrs:
                self._base_lrs[gid] = float(g['lr'])
            base = self._base_lrs[gid]
            name = g.get('name', '')
            factor = self._group_factor(name, self._last_step)
            g['lr'] = base * factor

    def get_last_lr(self) -> List[float]:
        out: List[float] = []
        for g in self.optimizer.param_groups:
            gid = id(g)
            if gid not in self._base_lrs:
                self._base_lrs[gid] = float(g['lr'])
            base = self._base_lrs[gid]
            factor = self._group_factor(g.get('name', ''), self._last_step)
            out.append(base * factor)
        return out

    # Legacy-compat alias: older fallback code in ckpt resume sets
    # `scheduler.last_epoch = N` to fast-forward the clock.
    @property
    def last_epoch(self) -> int:
        return self._last_step

    @last_epoch.setter
    def last_epoch(self, value: int) -> None:
        self._last_step = int(value)

    def state_dict(self) -> Dict[str, Any]:
        base_by_name: Dict[str, float] = {}
        for i, g in enumerate(self.optimizer.param_groups):
            name = g.get('name') or f'grp_{i}'
            base_by_name[name] = self._base_lrs.get(id(g), float(g['lr']))
        return {
            'last_step': int(self._last_step),
            'base_lrs_by_name': base_by_name,
            'g_warmup': self._g_warmup,
            'g_total': self._g_total,
            'v_warmup': self._v_warmup,
            'v_total': self._v_total,
            'v_start': self._v_start,
            'v_min_ratio': self._v_min_ratio,
            'a_warmup': self._a_warmup,
            'a_total': self._a_total,
            'a_start': self._a_start,
            'a_min_ratio': self._a_min_ratio,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        # Compatibility branch: old LambdaLR state dict has `last_epoch` /
        # `_last_lr` / `base_lrs` / `lr_lambdas` but no `base_lrs_by_name`.
        if 'base_lrs_by_name' in sd:
            self._last_step = int(sd.get('last_step', 0))
            base_by_name = sd.get('base_lrs_by_name', {})
            for i, g in enumerate(self.optimizer.param_groups):
                name = g.get('name') or f'grp_{i}'
                if name in base_by_name:
                    self._base_lrs[id(g)] = float(base_by_name[name])
            # Schedule shape fields are intentionally NOT restored from sd;
            # we always use the ones constructed from the current config so
            # that users can change total/warmup on resume. If strict
            # reproducibility is required, restart instead of resume.
        else:
            # Legacy LambdaLR state: read last_epoch if present, discard rest.
            last_epoch = int(sd.get('last_epoch', sd.get('_step_count', 0)))
            self._last_step = last_epoch
            logging.warning(
                "[scheduler] loaded legacy LambdaLR state_dict; only 'last_epoch' "
                f"={last_epoch} was preserved. base_lrs re-captured from current "
                "optimizer.param_groups (pre-step lrs)."
            )


class AudioVideoVAETrainer:
    """音视频 VAE 联合训练器"""

    def __init__(
        self,
        cfg: Dict[str, Any],
        tag: str = None,
        continue_train: bool = False,
        pretrained_checkpoint: Optional[str] = None,
        keep_audio_vae_pretrained: bool = False,
        pretrained_video_checkpoint: Optional[str] = None,
        pretrained_audio_checkpoint: Optional[str] = None,
        pretrained_contrastive_checkpoint: Optional[str] = None,
        pretrained_disc_checkpoint: Optional[str] = None,
        pretrained_disc_load_optim: bool = False,
    ):
        super().__init__()

        self.tag = tag
        self.continue_train = continue_train
        self.pretrained_checkpoint = pretrained_checkpoint
        self.keep_audio_vae_pretrained = keep_audio_vae_pretrained
        self.pretrained_video_checkpoint = pretrained_video_checkpoint
        self.pretrained_audio_checkpoint = pretrained_audio_checkpoint
        self.pretrained_contrastive_checkpoint = pretrained_contrastive_checkpoint
        # Discriminator-only warm-start. Independent of generator pretrained
        # checkpoints (audio/video/full); applied after them. Only loads disc
        # weights by default; set load_optim=True to also restore optim_d /
        # scheduler_d / scaler_d state.
        self.pretrained_disc_checkpoint = pretrained_disc_checkpoint
        self.pretrained_disc_load_optim = bool(pretrained_disc_load_optim)

        # ---- Distributed setup ----
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1

        if self.is_distributed:
            for _attempt in range(5):
                try:
                    torch.cuda.set_device(self.local_rank)
                    break
                except RuntimeError:
                    if _attempt < 4:
                        logging.warning(f"CUDA not ready (attempt {_attempt + 1}/5), retrying in 3s...")
                        time.sleep(3)
                    else:
                        raise
            dist.init_process_group(backend='nccl')
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.is_main = self.rank == 0

        if self.is_main:
            logging.info("=" * 60)
            logging.info("Initializing AudioVideoVAETrainer")
            logging.info("=" * 60)
            logging.info(f"[1/8] Distributed: rank={self.rank}, world_size={self.world_size}, device={self.device}")

        # ---- Random seed ----
        self.seed = cfg.get('seed', 42)
        torch.manual_seed(self.seed + self.rank)

        # ---- Config params ----
        self.cfg = cfg
        training_cfg = cfg.get('training', {})
        self.reset_scheduler_on_resume = bool(
            training_cfg.get('reset_scheduler_on_resume', False))
        self.log_steps = training_cfg.get('log_steps', 100)
        self.stdout_steps = training_cfg.get('stdout_steps', 10)
        self.save_model_steps = training_cfg.get('save_steps', 1000)
        self.eval_steps = training_cfg.get('eval_steps', 5000)
        self.tot_train_steps = training_cfg.get('max_steps', 100000)
        self.batch_size = training_cfg.get('batch_size', 1)
        self.grad_log_steps = training_cfg.get('grad_log_steps', 0)
        self.gradient_accumulation_steps = int(training_cfg.get('gradient_accumulation_steps', 1))
        assert self.gradient_accumulation_steps >= 1, \
            f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
        self.tb_num_fixed = training_cfg.get('tb_num_fixed_samples', 5)
        self.tb_num_random = training_cfg.get('tb_num_random_samples', 5)
        self.tb_train_media_steps = training_cfg.get('tb_train_media_steps', 2000)
        self.tb_train_media_count = training_cfg.get('tb_train_media_count', 10)

        data_cfg = cfg.get('data', {})
        train_cfg = data_cfg.get('train', {})
        self.audio_sample_rate = train_cfg.get('audio_sample_rate', 24000)

        if self.is_main:
            logging.info(f"[2/8] Training: max_steps={self.tot_train_steps}, "
                         f"batch_size={self.batch_size}, log_steps={self.log_steps}, "
                         f"gradient_accumulation_steps={self.gradient_accumulation_steps}")

        # ---- Output directories ----
        output_cfg = cfg.get('output', {})
        assert self.tag, "Tag is required. Please provide --tag to organize experiment outputs."
        exp_root = Path(output_cfg.get('exp_root', './exp'))
        self.exp_dir = exp_root / self.tag
        self.log_dir = self.exp_dir / 'log'
        self.results_folder = self.exp_dir / 'checkpoints'
        self.tensorboard_dir = self.exp_dir / 'tensorboard'

        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.results_folder.mkdir(parents=True, exist_ok=True)
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        if self.is_main:
            logging.info(f"[3/8] Exp dir: {self.exp_dir}")

        if self.is_distributed:
            dist.barrier()

        # Save config snapshot
        if self.is_main:
            with open(self.results_folder / 'config.yaml', 'w') as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        # Code snapshot
        if self.is_main:
            code_snapshot_dir = self.exp_dir / 'code'
            source_omnivae = Path(_project_root) / 'omnivae'
            if source_omnivae.exists():
                target_omnivae = code_snapshot_dir / 'omnivae'
                if target_omnivae.exists():
                    shutil.rmtree(target_omnivae)
                shutil.copytree(
                    source_omnivae, target_omnivae,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
                )

        # TensorBoard
        if self.is_main:
            self.writer = tensorboard.SummaryWriter(self.tensorboard_dir)

        # ---- dtype ----
        dtype_str = training_cfg.get('dtype', 'bfloat16')
        if dtype_str in ('bf16', 'bfloat16'):
            self.dtype = torch.bfloat16
        elif dtype_str in ('fp16', 'float16'):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype_str in ('fp16', 'float16')))
        self.use_autocast = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)

        self.video_vae_dtype = _parse_dtype(training_cfg.get('video_vae_dtype')) or self.dtype
        self.audio_vae_dtype = _parse_dtype(training_cfg.get('audio_vae_dtype')) or self.dtype
        self.contrastive_dtype = _parse_dtype(training_cfg.get('contrastive_dtype')) or self.dtype
        self.llm_dtype = _parse_dtype(training_cfg.get('llm_dtype')) or self.dtype

        if self.is_main:
            logging.info(f"[4/8] AMP dtype={self.dtype}, per-module: "
                         f"video={self.video_vae_dtype}, audio={self.audio_vae_dtype}, "
                         f"contrastive={self.contrastive_dtype}, llm={self.llm_dtype}")

        # ---- Build ----
        if self.is_main:
            logging.info("[5/8] Building Model...")
        self._build_model(cfg)

        if self.is_main:
            logging.info("[6/8] Building Datasets...")
        self._build_datasets(cfg)

        if self.is_main:
            logging.info("[7/8] Building Optimizer...")
        self._build_optimizer(cfg)

        if self.is_main:
            logging.info("[8/8] Building Loss Functions...")
        self._build_loss_functions(cfg)

        self.train_state = StreamingTrainState()

        # Modality RNG for image+video alterstep mixed training (SSVAE-style).
        # Seeded identically across ranks so all GPUs draw the same modality
        # at the same step (saves a torch.distributed.broadcast).
        # Set lazily here in case _build_datasets ran before this line.
        self._modality_rng = np.random.default_rng(int(self.seed) + 7919)
        if not hasattr(self, '_current_modality'):
            self._current_modality = None
        if not hasattr(self, 'image_video_weights'):
            self.image_video_weights = None
        if not hasattr(self, 'use_image_video_alter'):
            self.use_image_video_alter = False

        self._setup_audio_eval_dirs()

        if self.is_main:
            logging.info("-" * 60)
            logging.info("Active losses summary:")
            logging.info(f"  Video modality loaded : {self.needs_video}")
            logging.info(f"  Audio modality loaded : {self.needs_audio}")
            logging.info(f"  Video reconstruction  : {self.use_video_recon}")
            logging.info(f"  Audio reconstruction  : {self.use_audio_recon}")
            logging.info(f"  Segment contrastive   : {self.use_segment_contrastive}")
            logging.info(f"  Global contrastive    : {self.use_global_contrastive}")
            logging.info(f"  LLM caption           : {self.use_llm_caption}")
            logging.info(f"  Semantic distillation : {self.use_semantic_distill}")
            if self.use_semantic_distill:
                has_vid_proj = self.unwrapped_model.image_distill_proj is not None
                has_aud_proj = self.unwrapped_model.audio_distill_proj is not None
                logging.info(f"    Video distill proj  : {has_vid_proj}")
                logging.info(f"    Audio distill proj  : {has_aud_proj}")
                logging.info(f"    Adaptive balance    : {self.adaptive_distill_balance}")
            logging.info("-" * 60)
            logging.info("=" * 60)
            logging.info("AudioVideoVAETrainer Initialization Complete!")
            logging.info("=" * 60)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _setup_audio_eval_dirs(self):
        self.audio_gt_dir = self.exp_dir / 'audio_eval' / 'gt'
        self.audio_syn_dir = self.exp_dir / 'audio_eval' / 'syn'
        self.video_eval_dir = self.exp_dir / 'video_eval'
        if self.is_main:
            self.audio_gt_dir.mkdir(parents=True, exist_ok=True)
            self.audio_syn_dir.mkdir(parents=True, exist_ok=True)
            self.video_eval_dir.mkdir(parents=True, exist_ok=True)

    @property
    def unwrapped_model(self):
        if self.is_distributed:
            return self.model.module
        return self.model

    @staticmethod
    def _select_tb_sample_indices(
        total_count: int, num_fixed: int = 5, num_random: int = 5,
    ) -> List[int]:
        fixed = list(range(min(num_fixed, total_count)))
        remaining = list(range(num_fixed, total_count))
        rand_count = min(num_random, len(remaining))
        sampled = random.sample(remaining, rand_count) if rand_count > 0 else []
        return fixed + sorted(sampled)

    @staticmethod
    def _compute_log_mel_spectrogram(
        waveform: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel_transform = MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256,
            n_mels=80, power=2.0,
        )
        mel = mel_transform(waveform)
        log_mel = torch.log10(mel.clamp(min=1e-5))
        return log_mel.squeeze(0)

    def _autocast_context(self):
        if not self.use_autocast:
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=self.dtype)

    def _compute_kl_loss(self, posterior: Any, device: torch.device) -> torch.Tensor:
        if hasattr(posterior, 'kl'):
            kl = posterior.kl()
            if isinstance(kl, torch.Tensor) and kl.ndim > 0:
                kl = torch.sum(kl) / kl.shape[0]
            return kl
        return torch.tensor(0.0, device=device)

    def _compute_video_kl_loss(self, posterior: Any, device: torch.device) -> torch.Tensor:
        if hasattr(posterior, 'kl'):
            if hasattr(posterior, 'deterministic') and posterior.deterministic:
                return torch.tensor(0.0, device=device)

            if all(hasattr(posterior, attr) for attr in ('mean', 'var', 'logvar')):
                kl_per_elem = 0.5 * (
                    torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar
                )
                if kl_per_elem.ndim >= 5:
                    # Match the video reconstruction scaling: mean over time, sum over
                    # latent channel/spatial dims, then mean over the batch.
                    reduce_dims = tuple(dim for dim in range(1, kl_per_elem.ndim) if dim != 2)
                    kl = kl_per_elem.sum(dim=reduce_dims).mean(dim=1)
                    return kl.mean()

            kl = posterior.kl(reduction="sum")
            if isinstance(kl, torch.Tensor) and kl.ndim > 0:
                kl = torch.sum(kl) / kl.shape[0]
            return kl
        return torch.tensor(0.0, device=device)

    def _record_weighted_loss(
        self, losses: Dict[str, torch.Tensor],
        name: str, raw_loss: torch.Tensor, weight: float,
    ) -> torch.Tensor:
        weighted_loss = raw_loss * weight
        losses[f"{name}_raw"] = raw_loss
        losses[f"{name}_weighted"] = weighted_loss
        return weighted_loss

    def _all_reduce_eval_metrics(
        self, metric_sums: Dict[str, float], count: int,
    ) -> tuple:
        if not self.is_distributed:
            if count > 0:
                return {k: v / count for k, v in metric_sums.items()}, count
            return metric_sums, count
        keys = sorted(metric_sums.keys())
        tensor = torch.tensor(
            [metric_sums[k] for k in keys] + [float(count)], device=self.device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_count = int(tensor[-1].item())
        if total_count > 0:
            return {k: tensor[i].item() / total_count for i, k in enumerate(keys)}, total_count
        return {k: 0.0 for k in keys}, 0

    def _build_eval_video_name(self, source_identifier: str, sample_index: int) -> str:
        if source_identifier:
            if "#" in source_identifier:
                packed_path, packed_offsets = source_identifier.split("#", 1)
                base_name = f"{Path(packed_path).name}__{packed_offsets.replace('#', '__')}"
            else:
                base_name = Path(source_identifier).name
        else:
            base_name = "sample"
        return f"{base_name}.mp4"

    def _build_eval_audio_name(self, source_identifier: str, sample_index: int) -> str:
        if source_identifier:
            if "#" in source_identifier:
                packed_path, packed_offsets = source_identifier.split("#", 1)
                base_name = f"{Path(packed_path).stem}__{packed_offsets.replace('#', '__')}.wav"
            else:
                source_name = Path(source_identifier).name
                source_suffix = Path(source_name).suffix
                base_name = source_name if source_suffix.lower() == ".wav" else f"{Path(source_name).stem}.wav"
        else:
            base_name = "sample.wav"
        return base_name

    # ------------------------------------------------------------------
    # TensorBoard media writers
    # ------------------------------------------------------------------

    def _write_video_tb_samples(self, video_samples: List[tuple], ds_name: str, step: int):
        if not video_samples:
            return
        indices = self._select_tb_sample_indices(len(video_samples), self.tb_num_fixed, self.tb_num_random)
        for rank_in_selection, idx in enumerate(indices):
            gt, recon, fname = video_samples[idx]
            gt_01 = (gt.clamp(-1, 1) + 1) * 0.5
            recon_01 = (recon.clamp(-1, 1) + 1) * 0.5
            side_by_side = torch.cat([gt_01, recon_01], dim=-1)
            vid_uint8 = (side_by_side * 255).to(torch.uint8)
            vid_uint8 = vid_uint8.permute(1, 0, 2, 3).unsqueeze(0)
            tag = f"eval_video/{ds_name}/{rank_in_selection:02d}"
            if fname:
                tag += f"_{Path(fname).stem}"
            try:
                self.writer.add_video(tag, vid_uint8, global_step=step,
                                      fps=int(max(1, self.val_video_save_fps)))
            except Exception as e:
                logging.warning(f"TensorBoard add_video failed for {tag}: {e}")

    def _write_audio_tb_samples(self, audio_samples: List[tuple], ds_name: str, step: int):
        if not audio_samples:
            return
        indices = self._select_tb_sample_indices(len(audio_samples), self.tb_num_fixed, self.tb_num_random)
        for rank_in_selection, idx in enumerate(indices):
            gt, recon, fname = audio_samples[idx]
            suffix = f"_{Path(fname).stem}" if fname else ""
            tag_prefix = f"eval_audio/{ds_name}/{rank_in_selection:02d}{suffix}"
            try:
                self.writer.add_audio(f"{tag_prefix}/gt", gt.clamp(-1, 1).contiguous(),
                                      global_step=step, sample_rate=self.audio_sample_rate)
                self.writer.add_audio(f"{tag_prefix}/recon", recon.clamp(-1, 1).contiguous(),
                                      global_step=step, sample_rate=self.audio_sample_rate)
            except Exception as e:
                logging.warning(f"TensorBoard add_audio failed for {tag_prefix}: {e}")
            try:
                gt_mel = self._compute_log_mel_spectrogram(gt.squeeze(0), self.audio_sample_rate)
                recon_mel = self._compute_log_mel_spectrogram(recon.squeeze(0), self.audio_sample_rate)
                min_t = min(gt_mel.shape[-1], recon_mel.shape[-1])
                combined = torch.cat([gt_mel[:, :min_t], recon_mel[:, :min_t]], dim=0)
                vmin, vmax = combined.min(), combined.max()
                if vmax - vmin > 1e-8:
                    combined = (combined - vmin) / (vmax - vmin)
                else:
                    combined = torch.zeros_like(combined)
                combined = combined.flip(0)
                self.writer.add_image(f"{tag_prefix}/mel_gt_vs_recon",
                                      combined.unsqueeze(0), global_step=step)
            except Exception as e:
                logging.warning(f"TensorBoard mel spectrogram failed for {tag_prefix}: {e}")

    def _write_train_media_tb(self, media_data: Dict[str, torch.Tensor], step: int):
        if not media_data:
            return
        n_samples = self.tb_train_media_count

        video_gt = media_data.get('video_gt')
        video_recon = media_data.get('video_recon')
        if video_gt is not None and video_recon is not None:
            B = video_gt.shape[0]
            count = min(n_samples, B)
            indices = random.sample(range(B), count) if count < B else list(range(B))
            for rank_i, idx in enumerate(sorted(indices)):
                gt_01 = (video_gt[idx].clamp(-1, 1) + 1) * 0.5
                recon_01 = (video_recon[idx].clamp(-1, 1) + 1) * 0.5
                side_by_side = torch.cat([gt_01, recon_01], dim=-1)
                vid_uint8 = (side_by_side * 255).to(torch.uint8).permute(1, 0, 2, 3).unsqueeze(0)
                try:
                    self.writer.add_video(f"train_media/video/{rank_i:02d}", vid_uint8,
                                          global_step=step, fps=int(max(1, self.val_video_save_fps)))
                except Exception as e:
                    logging.warning(f"TensorBoard train add_video failed: {e}")

        audio_gt = media_data.get('audio_gt')
        audio_recon = media_data.get('audio_recon')
        if audio_gt is not None and audio_recon is not None:
            B = audio_gt.shape[0]
            count = min(n_samples, B)
            indices = random.sample(range(B), count) if count < B else list(range(B))
            for rank_i, idx in enumerate(sorted(indices)):
                gt = audio_gt[idx]
                recon = audio_recon[idx]
                tag_prefix = f"train_media/audio/{rank_i:02d}"
                try:
                    self.writer.add_audio(f"{tag_prefix}/gt", gt.clamp(-1, 1).contiguous(),
                                          global_step=step, sample_rate=self.audio_sample_rate)
                    self.writer.add_audio(f"{tag_prefix}/recon", recon.clamp(-1, 1).contiguous(),
                                          global_step=step, sample_rate=self.audio_sample_rate)
                except Exception as e:
                    logging.warning(f"TensorBoard train add_audio failed for {tag_prefix}: {e}")

    def _save_video_eval_batch(
        self, ds_name: str, step: int,
        video: torch.Tensor, recon: torch.Tensor,
        file_names: List[str], sample_offset: int,
    ):
        step_dir = self.video_eval_dir / f"step_{step:08d}" / ds_name
        gt_dir = step_dir / "gt"
        recon_dir = step_dir / "recon"
        gt_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = step_dir / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as manifest_f:
            for b in range(video.shape[0]):
                sample_index = sample_offset + b
                source_identifier = file_names[b] if b < len(file_names) else ""
                video_name = self._build_eval_video_name(source_identifier, sample_index)
                save_video_tensor_to_mp4(video[b], gt_dir / video_name, fps=self.val_video_save_fps)
                save_video_tensor_to_mp4(recon[b], recon_dir / video_name, fps=self.val_video_save_fps)
                manifest_f.write(json.dumps({
                    "index": sample_index, "source": source_identifier,
                    "gt": str((gt_dir / video_name).relative_to(self.exp_dir)),
                    "recon": str((recon_dir / video_name).relative_to(self.exp_dir)),
                }, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # _build_model
    # ------------------------------------------------------------------

    def _build_model(self, cfg: Dict):
        model_cfg = cfg.get('model', {})
        video_cfg = model_cfg.get('video', {})
        audio_cfg = model_cfg.get('audio', {})
        contrastive_cfg = model_cfg.get('contrastive', {})
        llm_cfg = model_cfg.get('llm', {})
        distill_cfg = model_cfg.get('distill', {})

        training_cfg = cfg.get('training', {})
        qk_norm = training_cfg.get('qk_norm', False)
        if qk_norm:
            video_cfg['qk_norm'] = True
            contrastive_cfg['qk_norm'] = True
            if self.is_main:
                logging.info("QK-Norm enabled for all attention layers")

        loss_cfg = cfg.get('loss', {})
        use_video_recon = loss_cfg.get('use_video_recon', True)
        use_audio_recon = loss_cfg.get('use_audio_recon', True)
        use_contrastive = (loss_cfg.get('use_segment_contrastive', True)
                           or loss_cfg.get('use_global_contrastive', True))
        use_llm_caption = loss_cfg.get('use_llm_caption', False)

        self.needs_video = use_video_recon or use_contrastive or use_llm_caption
        self.needs_audio = use_audio_recon or use_contrastive or use_llm_caption

        if self.is_main:
            logging.info(f"Modality needs: video={self.needs_video}, audio={self.needs_audio}")

        self.model = AudioVideoVAE(
            video_vae_kwargs=video_cfg,
            audio_vae_kwargs=audio_cfg,
            contrastive_kwargs=contrastive_cfg,
            llm_kwargs=llm_cfg,
            distill_kwargs=distill_cfg,
            skip_video_vae=not self.needs_video,
            skip_audio_vae=not self.needs_audio,
        ).to(self.device)

        self._pre_align_llm_vocab_size()

        # Gradient checkpointing
        training_cfg = cfg.get('training', {})
        if training_cfg.get('gradient_checkpointing', False):
            self.model.enable_video_gradient_checkpointing()
            if self.is_main:
                logging.info("Video gradient checkpointing enabled.")

        self.model.module_dtypes = {
            'video_vae': self.video_vae_dtype,
            'audio_vae': self.audio_vae_dtype,
            'contrastive': self.contrastive_dtype,
            'llm': self.llm_dtype,
            'distill': self.contrastive_dtype,
        }

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )

        self._freeze_unused_params(cfg)

        self.use_ema = training_cfg.get('use_ema', True)
        if self.use_ema:
            ema_decay = training_cfg.get('ema_decay', 0.999)
            self.ema = EMA(self.unwrapped_model, decay=ema_decay)
        else:
            self.ema = None

        if self.is_main:
            logging.info(f"EMA: {'enabled (decay=' + str(training_cfg.get('ema_decay', 0.999)) + ')' if self.use_ema else 'disabled'}")
            self.unwrapped_model.print_model_info()

        # ----------------------------------------------------------------
        # Audio discriminators (GAN training) - optional
        # ----------------------------------------------------------------
        # `loss_cfg` / `use_audio_recon` / `training_cfg` are already in scope
        # earlier in this function.
        self.use_audio_disc = bool(loss_cfg.get('use_audio_disc', False))
        self.audio_disc_start_step = int(loss_cfg.get('audio_disc_start_step', 0) or 0)
        self.lambda_audio_adv = float(loss_cfg.get('lambda_audio_adv', 1.0))
        self.lambda_audio_feature_matching = float(loss_cfg.get('lambda_audio_feature_matching', 1.0))

        _dmgn = training_cfg.get('disc_max_grad_norm', None)
        self.disc_max_grad_norm = float(_dmgn) if _dmgn is not None else None
        _disc_dtype_name = training_cfg.get('disc_dtype', 'fp32')
        self.disc_dtype = _parse_dtype(_disc_dtype_name) if _disc_dtype_name is not None else torch.float32

        self.discriminators: Dict[str, nn.Module] = {}
        if self.use_audio_disc:
            if not use_audio_recon:
                if self.is_main:
                    logging.warning(
                        "[audio-disc] use_audio_disc=True but use_audio_recon=False; "
                        "disabling discriminators (need audio decoder output)."
                    )
                self.use_audio_disc = False
            else:
                disc_params_cfg = loss_cfg.get('audio_disc_params', {}) or {}
                raw_discs = build_audio_discriminators(disc_params_cfg)
                for name, disc in raw_discs.items():
                    disc = disc.to(self.device)
                    if self.is_distributed:
                        disc = DDP(
                            disc,
                            device_ids=[self.local_rank],
                            find_unused_parameters=True,
                        )
                    self.discriminators[name] = disc
                if self.is_main:
                    total_disc = 0
                    for name, disc in self.discriminators.items():
                        d_unwrapped = disc.module if isinstance(disc, DDP) else disc
                        n_params = sum(p.numel() for p in d_unwrapped.parameters())
                        total_disc += n_params
                        logging.info(f"[audio-disc] {name}: {n_params:,} params")
                    logging.info(
                        f"[audio-disc] total={total_disc:,}, start_step={self.audio_disc_start_step}, "
                        f"lambda_adv={self.lambda_audio_adv}, lambda_fm={self.lambda_audio_feature_matching}, "
                        f"disc_dtype={self.disc_dtype}, max_grad_norm={self.disc_max_grad_norm}"
                    )

        # ----------------------------------------------------------------
        # Video discriminator (CausalVAE-style 3D PatchGAN) - optional.
        # Trains with strict G/D alternation (see `train_step`), so it
        # lives in `self.discriminators` alongside audio disc but flips
        # generator/disc updates on a per-optimizer-step basis.
        # ----------------------------------------------------------------
        self.use_video_disc = bool(loss_cfg.get('use_video_disc', False))
        self.video_disc_start_step = int(loss_cfg.get('video_disc_start_step', 0) or 0)
        self.lambda_video_adv = float(loss_cfg.get('lambda_video_adv', 0.5))
        self.video_disc_loss_type = str(loss_cfg.get('video_disc_loss_type', 'hinge'))
        if self.video_disc_loss_type not in ('hinge', 'vanilla'):
            raise ValueError(
                f"video_disc_loss_type must be 'hinge' or 'vanilla', got '{self.video_disc_loss_type}'"
            )
        self.video_disc_adaptive_weight = bool(loss_cfg.get('video_disc_adaptive_weight', False))
        # Upper clamp for the VQGAN adaptive d_weight. Default 1.0 (instead of
        # the upstream 1e4) to avoid adversarial blow-up when G is pretrained
        # and recon grad is tiny. Set larger (e.g. 1e4) to recover legacy
        # behavior.
        self.video_disc_adaptive_weight_max = float(
            loss_cfg.get('video_disc_adaptive_weight_max', 1.0)
        )
        # Skip D parameter update (but still compute d_loss / logits for TB)
        # once d_loss falls below this threshold. 0 disables the gate.
        # Typical: 0.2 ~ 0.4 for hinge loss.
        self.video_disc_lazy_threshold = float(
            loss_cfg.get('video_disc_lazy_threshold', 0.0)
        )
        self.distill_every_steps = bool(loss_cfg.get('distill_every_steps', False))

        if self.use_video_disc:
            use_video_recon = loss_cfg.get('use_video_recon', True)
            if not use_video_recon:
                if self.is_main:
                    logging.warning(
                        "[video-disc] use_video_disc=True but use_video_recon=False; "
                        "disabling (need video decoder output)."
                    )
                self.use_video_disc = False
            else:
                video_disc_cfg = loss_cfg.get('video_disc_params', {}) or {}
                v_disc = build_video_discriminator(video_disc_cfg)
                v_disc = v_disc.to(self.device)
                if self.is_distributed:
                    v_disc = DDP(
                        v_disc,
                        device_ids=[self.local_rank],
                        find_unused_parameters=False,
                    )
                self.discriminators['video'] = v_disc
                if self.is_main:
                    d_unwrapped = v_disc.module if isinstance(v_disc, DDP) else v_disc
                    n_params = sum(p.numel() for p in d_unwrapped.parameters())
                    logging.info(
                        f"[video-disc] NLayerDiscriminator3D: {n_params:,} params, "
                        f"start_step={self.video_disc_start_step}, "
                        f"lambda_adv={self.lambda_video_adv}, "
                        f"loss_type={self.video_disc_loss_type}, "
                        f"adaptive_weight={self.video_disc_adaptive_weight}, "
                        f"distill_every_steps={self.distill_every_steps}"
                    )

        # Alternating-update scheme (CausalVAE style): with use_video_disc,
        # after `train_state.step >= video_disc_start_step` we flip based on
        # step parity — even steps run the G update window (full forward +
        # losses + optim_g.step), odd steps run the D-only window (video
        # disc update only, optionally accumulating distill grad into G
        # params for the next G step). Uses `train_state.step` directly so
        # the parity is naturally persisted across checkpoints.
        self._current_is_d_window = False

    def _pre_align_llm_vocab_size(self):
        if self.model.llm_caption_head is None:
            return
        ckpt_dir = None
        if self.continue_train:
            ckpt_dir = find_latest_checkpoint(self.results_folder, prefix="Trainer_")
        if ckpt_dir is None and self.pretrained_checkpoint:
            ckpt_dir = self.pretrained_checkpoint
        if ckpt_dir is None:
            return
        ckpt_path = os.path.join(ckpt_dir, 'state_dict.pt')
        if not os.path.exists(ckpt_path):
            return
        embed_key = 'llm_caption_head.llm.model.embed_tokens.weight'
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model_state_dict', {})
            if embed_key not in state_dict:
                return
            ckpt_vocab_size = state_dict[embed_key].shape[0]
        finally:
            del ckpt
            gc.collect()
        current_vocab_size = self.model.llm_caption_head.llm.get_input_embeddings().weight.shape[0]
        if ckpt_vocab_size != current_vocab_size:
            if self.is_main:
                logging.info(f"Pre-aligning LLM vocab: {current_vocab_size} -> {ckpt_vocab_size}")
            self.model.llm_caption_head.llm.resize_token_embeddings(ckpt_vocab_size)

    def _freeze_unused_params(self, cfg: Dict):
        loss_cfg = cfg.get('loss', {})
        use_video_recon = loss_cfg.get('use_video_recon', True)
        use_audio_recon = loss_cfg.get('use_audio_recon', True)
        use_contrastive = (
            loss_cfg.get('use_segment_contrastive', True)
            or loss_cfg.get('use_global_contrastive', True)
        )
        use_llm_caption = loss_cfg.get('use_llm_caption', False)
        use_semantic_distill = loss_cfg.get('use_semantic_distill', False)
        freeze_vae_encoders = loss_cfg.get('freeze_vae_encoders', False)
        needs_encoder = (use_contrastive or use_llm_caption or use_semantic_distill) and not freeze_vae_encoders

        # Phase freezing: freeze entire video VAE (encoder + decoder) until a given step.
        self.freeze_video_vae = bool(loss_cfg.get('freeze_video_vae', False))
        self.freeze_video_vae_until_step = int(loss_cfg.get('freeze_video_vae_until_step', 0))
        self._video_vae_frozen = False

        # Phase freezing: freeze entire audio VAE (encoder + decoder) until a given step.
        self.freeze_audio_vae = bool(loss_cfg.get('freeze_audio_vae', False))
        self.freeze_audio_vae_until_step = int(loss_cfg.get('freeze_audio_vae_until_step', 0))
        self._audio_vae_frozen = False

        model = self.unwrapped_model
        frozen_count = 0

        def _freeze_modules(modules):
            nonlocal frozen_count
            for mod in modules:
                for p in mod.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_count += 1

        if not use_video_recon and model.video_vae is not None:
            if needs_encoder:
                decoder = model.get_video_decoder()
                if decoder:
                    _freeze_modules(decoder)
            else:
                _freeze_modules([model.video_vae])

        if not use_audio_recon and model.audio_vae is not None:
            if needs_encoder:
                _freeze_modules([model.audio_vae.post_quant_conv, model.audio_vae.decoder])
            else:
                _freeze_modules([model.audio_vae])

        if not use_llm_caption and model.llm_caption_head is not None:
            _freeze_modules([model.llm_caption_head])

        # Phase freezing: freeze the whole video VAE. Encoder still runs forward
        # (so that contrastive / audio reconstruction get latents), but its
        # parameters do not receive gradients until the unfreeze step.
        if self.freeze_video_vae and model.video_vae is not None:
            _freeze_modules([model.video_vae])
            self._video_vae_frozen = True
            if self.is_main:
                logging.info(
                    f"[phase-freeze] video_vae frozen; will unfreeze at step "
                    f"{self.freeze_video_vae_until_step} (0 = never)."
                )

        # Phase freezing: freeze the whole audio VAE (symmetric to video). The
        # encoder still runs forward so that contrastive / video reconstruction
        # paths have latents, but its parameters do not receive gradients until
        # the unfreeze step.
        if self.freeze_audio_vae and model.audio_vae is not None:
            _freeze_modules([model.audio_vae])
            self._audio_vae_frozen = True
            if self.is_main:
                logging.info(
                    f"[phase-freeze] audio_vae frozen; will unfreeze at step "
                    f"{self.freeze_audio_vae_until_step} (0 = never)."
                )

        # Encoder-only freeze for the audio VAE: freezes encoder + quant_conv,
        # leaves post_quant_conv + decoder trainable. Mutually exclusive with
        # freeze_audio_vae (which freezes the whole module).
        self.freeze_audio_encoder = bool(loss_cfg.get('freeze_audio_encoder', False))
        if self.freeze_audio_encoder and model.audio_vae is not None:
            if self.freeze_audio_vae:
                if self.is_main:
                    logging.warning(
                        "[freeze] freeze_audio_encoder ignored because freeze_audio_vae=True "
                        "already freezes the entire audio VAE."
                    )
            else:
                _freeze_modules([model.audio_vae.encoder, model.audio_vae.quant_conv])
                if self.is_main:
                    logging.info(
                        "[freeze] audio encoder frozen (encoder + quant_conv); "
                        "post_quant_conv + decoder remain trainable."
                    )

        # Encoder-only freeze for the video VAE: freezes encoder + conv1
        # (the equivalent of audio's quant_conv), leaves conv2 + decoder
        # trainable. Mutually exclusive with freeze_video_vae (which freezes
        # the whole module). Useful for GAN-only decoder finetuning where the
        # latent space established by a previous (e.g. distillation) stage
        # must be preserved bit-exact.
        self.freeze_video_encoder = bool(loss_cfg.get('freeze_video_encoder', False))
        if self.freeze_video_encoder and model.video_vae is not None:
            if self.freeze_video_vae:
                if self.is_main:
                    logging.warning(
                        "[freeze] freeze_video_encoder ignored because freeze_video_vae=True "
                        "already freezes the entire video VAE."
                    )
            else:
                enc_modules = model.get_video_encoder()
                if enc_modules:
                    _freeze_modules(enc_modules)
                    if self.is_main:
                        logging.info(
                            "[freeze] video encoder frozen (encoder + conv1); "
                            "conv2 + decoder remain trainable."
                        )
                elif self.is_main:
                    logging.warning(
                        "[freeze] freeze_video_encoder requested but "
                        "model.get_video_encoder() returned None; skipping."
                    )

        if self.is_main:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"Params: {trainable_params:,}/{total_params:,} trainable "
                         f"(video_recon={use_video_recon}, audio_recon={use_audio_recon}, "
                         f"contrastive={use_contrastive}, distill={use_semantic_distill}, "
                         f"freeze_video_vae={self.freeze_video_vae}, "
                         f"freeze_audio_vae={self.freeze_audio_vae}, "
                         f"freeze_video_encoder={self.freeze_video_encoder}, "
                         f"freeze_audio_encoder={self.freeze_audio_encoder})")

    def _unfreeze_video_vae_and_extend_optimizer(self):
        """Unfreeze video_vae params and register them as a new optimizer group.

        The AdamW state for the newly-unfrozen parameters starts from zero,
        so the first few updates may be larger than usual; callers should
        choose freeze_video_vae_until_step accordingly (ideally within the
        warmup window or paired with a small per-group lr multiplier).
        """
        model = self.unwrapped_model
        if model.video_vae is None:
            self._video_vae_frozen = False
            return

        newly_unfrozen = []
        for p in model.video_vae.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                newly_unfrozen.append(p)

        if not newly_unfrozen:
            self._video_vae_frozen = False
            return

        # Avoid duplicating parameters that might already live in an existing
        # group (should not happen, but guard anyway).
        existing_ids = set()
        for group in self.optimizer.param_groups:
            existing_ids.update(id(p) for p in group['params'])
        params_to_add = [p for p in newly_unfrozen if id(p) not in existing_ids]

        if params_to_add:
            lr_use = self._lr_video_vae_resolved if self._lr_video_vae_resolved is not None else self._lr_global
            self.optimizer.add_param_group({
                'params': params_to_add,
                'lr': lr_use,
                'name': 'video_vae',
            })

        # Resync EMA shadow for the newly trainable parameters. The EMA
        # class only tracks params it saw in __init__, so we explicitly seed
        # the shadow dict with the current values of the newly unfrozen
        # video_vae parameters.
        if self.ema is not None:
            with torch.no_grad():
                for name, param in self.ema.model.named_parameters():
                    if param.requires_grad and name not in self.ema.shadow:
                        self.ema.shadow[name] = param.data.clone()

        self._video_vae_frozen = False
        # stage2 transition happens when BOTH VAEs are unfrozen (or only video
        # was ever frozen). If audio is still phase-frozen, we stay in stage1.
        stage2_entered = not getattr(self, '_audio_vae_frozen', False)
        # Mark the step at which stage2 starts so that the v2-hybrid path can
        # linearly blend from anchor-scale to gradient-scale over the next
        # adaptive_v2_stage2_blend_steps updates.
        if stage2_entered and self.adaptive_v2_stage2_use_gradient:
            self._stage2_unfreeze_step = int(self.train_state.step)
            if self.is_main:
                logging.info(
                    f"[phase-freeze] adaptive_v2 stage2 gradient balance "
                    f"activated at step {self._stage2_unfreeze_step}; "
                    f"blend_steps={self.adaptive_v2_stage2_blend_steps}"
                )

        # Re-enable video decoder forward + video reconstruction loss / eval,
        # following the user-configured flags (these may legitimately be False
        # if the user opted out of video_recon independently of freeze).
        self.skip_video_decoder = not self.use_video_recon
        self.eval_video_recon = self._eval_video_recon_target

        if self.is_main:
            logging.info(
                f"[phase-freeze] video_vae unfrozen at step {self.train_state.step}; "
                f"added {len(params_to_add)} params to optimizer as a new group "
                f"(lr={self.optimizer.param_groups[-1]['lr']:.3e}); "
                f"resume skip_video_decoder={self.skip_video_decoder}, "
                f"eval_video_recon={self.eval_video_recon}."
            )
            if stage2_entered and self.adaptive_loss_balance_v2:
                _src, _rv, _ra, _rc = self._resolve_adaptive_v2_params()
                logging.info(
                    f"[phase-freeze] adaptive v2 switched to stage2: "
                    f"anchor={_src}, ratios v/a/c={_rv}/{_ra}/{_rc}"
                )

    def _unfreeze_audio_vae_and_extend_optimizer(self):
        """Unfreeze audio_vae params and register them as a new optimizer group.

        Symmetric to ``_unfreeze_video_vae_and_extend_optimizer``. The AdamW
        state for the newly-unfrozen parameters starts from zero, so the first
        few updates may be larger than usual; pair with a dedicated
        ``lr_audio_vae_warmup_steps`` / ``lr_audio_vae_min_ratio`` to smooth
        the transition.
        """
        model = self.unwrapped_model
        if model.audio_vae is None:
            self._audio_vae_frozen = False
            return

        newly_unfrozen = []
        for p in model.audio_vae.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                newly_unfrozen.append(p)

        if not newly_unfrozen:
            self._audio_vae_frozen = False
            return

        existing_ids = set()
        for group in self.optimizer.param_groups:
            existing_ids.update(id(p) for p in group['params'])
        params_to_add = [p for p in newly_unfrozen if id(p) not in existing_ids]

        if params_to_add:
            lr_use = (
                self._lr_audio_vae_resolved
                if getattr(self, '_lr_audio_vae_resolved', None) is not None
                else self._lr_global
            )
            self.optimizer.add_param_group({
                'params': params_to_add,
                'lr': lr_use,
                'name': 'audio_vae',
            })

        # Resync EMA shadow for the newly trainable parameters.
        if self.ema is not None:
            with torch.no_grad():
                for name, param in self.ema.model.named_parameters():
                    if param.requires_grad and name not in self.ema.shadow:
                        self.ema.shadow[name] = param.data.clone()

        self._audio_vae_frozen = False
        # stage2 only enters when BOTH VAEs are unfrozen. If video is still
        # phase-frozen we stay in stage1 until it also unfreezes.
        stage2_entered = not getattr(self, '_video_vae_frozen', False)
        if stage2_entered and self.adaptive_v2_stage2_use_gradient:
            self._stage2_unfreeze_step = int(self.train_state.step)
            if self.is_main:
                logging.info(
                    f"[phase-freeze] adaptive_v2 stage2 gradient balance "
                    f"activated at step {self._stage2_unfreeze_step}; "
                    f"blend_steps={self.adaptive_v2_stage2_blend_steps}"
                )

        # Re-enable audio decoder forward + audio reconstruction loss / eval.
        self.skip_audio_decoder = not self.use_audio_recon
        self.eval_audio_recon = self._eval_audio_recon_target

        if self.is_main:
            logging.info(
                f"[phase-freeze] audio_vae unfrozen at step {self.train_state.step}; "
                f"added {len(params_to_add)} params to optimizer as a new group "
                f"(lr={self.optimizer.param_groups[-1]['lr']:.3e}); "
                f"resume skip_audio_decoder={self.skip_audio_decoder}, "
                f"eval_audio_recon={self.eval_audio_recon}."
            )
            if stage2_entered and self.adaptive_loss_balance_v2:
                _src, _rv, _ra, _rc = self._resolve_adaptive_v2_params()
                logging.info(
                    f"[phase-freeze] adaptive v2 switched to stage2: "
                    f"anchor={_src}, ratios v/a/c={_rv}/{_ra}/{_rc}"
                )

    def _resolve_adaptive_v2_params(self):
        """Return (anchor_source, ratio_video, ratio_audio, ratio_contrastive)
        honoring stage1 overrides while either VAE is phase-frozen. Any
        stage1 field set to None falls back to the canonical stage2 value,
        so this reduces to the stage2 tuple when nothing is overridden.
        """
        if (getattr(self, '_video_vae_frozen', False)
                or getattr(self, '_audio_vae_frozen', False)):
            src = (self.adaptive_anchor_source_stage1
                   if self.adaptive_anchor_source_stage1 is not None
                   else self.adaptive_anchor_source)
            rv = (self.adaptive_ratio_video_stage1
                  if self.adaptive_ratio_video_stage1 is not None
                  else self.adaptive_ratio_video)
            ra = (self.adaptive_ratio_audio_stage1
                  if self.adaptive_ratio_audio_stage1 is not None
                  else self.adaptive_ratio_audio)
            rc = (self.adaptive_ratio_contrastive_stage1
                  if self.adaptive_ratio_contrastive_stage1 is not None
                  else self.adaptive_ratio_contrastive)
            return src, rv, ra, rc
        return (self.adaptive_anchor_source,
                self.adaptive_ratio_video,
                self.adaptive_ratio_audio,
                self.adaptive_ratio_contrastive)

    # ------------------------------------------------------------------
    # _build_datasets
    # ------------------------------------------------------------------

    def _build_datasets(self, cfg: Dict):
        data_cfg = cfg.get('data', {})
        training_cfg = cfg.get('training', {})
        loss_cfg = cfg.get('loss', {})
        train_cfg = data_cfg.get('train', {})
        build_val_video = bool(loss_cfg.get('eval_video_recon', loss_cfg.get('use_video_recon', True)))
        build_val_audio = bool(loss_cfg.get('eval_audio_recon', loss_cfg.get('use_audio_recon', True)))
        use_contrastive_eval_default = bool(
            loss_cfg.get('use_segment_contrastive', True)
            or loss_cfg.get('use_global_contrastive', True)
        )
        build_val_contrastive = bool(loss_cfg.get('eval_contrastive', use_contrastive_eval_default))
        build_val_caption = bool(loss_cfg.get('eval_llm_caption', loss_cfg.get('use_llm_caption', False)))

        # ---- Train dataset ----
        data_mixture_yaml = train_cfg.get('data_mixture_yaml')
        metadata_paths = train_cfg.get('metadata_paths', {})
        metadata_weights = train_cfg.get('metadata_weights', {})

        _distill_encoder_fps = None
        _distill_audio_target_sr = None

        # ---- Image+Video alterstep mode ----
        self.use_image_video_alter = bool(train_cfg.get('use_image_video_alter', False))
        if self.use_image_video_alter:
            raise NotImplementedError(
                "Image+video alterstep training is not part of the public "
                "OmniVAE training boundary. Remove data.train.use_image_video_alter "
                "or set it to false."
            )
        if self.use_image_video_alter:
            from omnivae.dataset.audio_video_streaming_dataset import (
                build_iv_alterstep_streaming_dataset,
                passthrough_collate_fn,
            )

            image_loader = (train_cfg.get('image_loader') or 'jsonl').lower()
            relaion_cfg = train_cfg.get('image_relaion') or {}
            if not isinstance(relaion_cfg, dict):
                raise ValueError(
                    f"data.train.image_relaion must be a dict, got {type(relaion_cfg).__name__}"
                )

            image_data_mixture_yaml = train_cfg.get('image_data_mixture_yaml')
            image_dataset_path = train_cfg.get('image_dataset_path')
            image_metadata_paths = None
            if image_dataset_path:
                image_metadata_paths = {Path(image_dataset_path).stem: image_dataset_path}

            if image_loader == 'jsonl':
                if not image_data_mixture_yaml and not image_metadata_paths:
                    raise ValueError(
                        "use_image_video_alter is enabled with image_loader='jsonl' "
                        "but neither image_data_mixture_yaml nor image_dataset_path is provided."
                    )
            elif image_loader == 'relaion':
                if not relaion_cfg.get('root') or not relaion_cfg.get('base_image_path'):
                    raise ValueError(
                        "use_image_video_alter + image_loader='relaion' requires "
                        "data.train.image_relaion.root and data.train.image_relaion.base_image_path "
                        "(or --image_relaion_root / --image_relaion_base_image_path)."
                    )
            else:
                raise ValueError(
                    f"unknown data.train.image_loader={image_loader!r}; "
                    f"expected 'jsonl' or 'relaion'."
                )

            image_bs = int(train_cfg.get('image_batch_size') or self.batch_size)
            video_bs = int(train_cfg.get('video_batch_size') or self.batch_size)
            iv_weights = train_cfg.get('image_video_weights') or [1.0, 1.0]
            if len(iv_weights) != 2:
                raise ValueError(
                    f"image_video_weights must be length 2 [image, video], "
                    f"got {iv_weights!r}"
                )
            self.image_video_weights = [float(w) for w in iv_weights]
            self.image_batch_size_eff = image_bs
            self.video_batch_size_eff = video_bs

            relaion_kwargs: Dict[str, Any] = {}
            if image_loader == 'relaion':
                relaion_kwargs = dict(
                    relaion_root=relaion_cfg['root'],
                    relaion_slave_path=relaion_cfg.get('slave_path'),
                    relaion_base_image_path=relaion_cfg['base_image_path'],
                    relaion_split=relaion_cfg.get('split', 'train'),
                    relaion_image_size=relaion_cfg.get('image_size'),
                    relaion_center_crop=bool(relaion_cfg.get('center_crop', False)),
                    relaion_random_flip=bool(relaion_cfg.get('random_flip', False)),
                    relaion_recaption_prob=float(relaion_cfg.get('recaption_prob', 0.0)),
                    relaion_cache_dir=relaion_cfg.get('cache_dir'),
                    relaion_max_samples=relaion_cfg.get('max_samples'),
                    relaion_repeat=int(relaion_cfg.get('repeat', 1)),
                )

            self.train_ds = build_iv_alterstep_streaming_dataset(
                image_loader=image_loader,
                image_dataset_dir_or_paths=(image_metadata_paths if image_loader == 'jsonl' else None),
                image_data_mixture_yaml=(image_data_mixture_yaml if image_loader == 'jsonl' else None),
                image_video_root=(train_cfg.get('image_file_root') or train_cfg.get('file_root')
                                  if image_loader == 'jsonl' else None),
                image_mixture_save_path=(self.exp_dir / 'image_data_mixture_triplets.tsv'
                                         if (image_loader == 'jsonl' and image_data_mixture_yaml
                                             and self.is_main) else None),
                image_batch_size=image_bs,
                **relaion_kwargs,
                video_dataset_dir_or_paths=metadata_paths if not data_mixture_yaml else None,
                video_data_mixture_yaml=data_mixture_yaml,
                video_weights=metadata_weights if (metadata_weights and not data_mixture_yaml) else None,
                video_root=train_cfg.get('file_root'),
                video_mixture_save_path=(self.exp_dir / 'data_mixture_triplets.tsv'
                                         if (data_mixture_yaml and self.is_main) else None),
                video_batch_size=video_bs,
                num_frames=train_cfg.get('num_frames', 25),
                resolution=train_cfg.get('resolution', 256),
                sample_rate=train_cfg.get('video_frame_sample_rate', train_cfg.get('sample_rate', 1)),
                target_fps=train_cfg.get('target_fps'),
                audio_sample_rate=train_cfg.get('audio_sample_rate', 24000),
                max_audio_duration=train_cfg.get('max_audio_duration'),
                use_torchcodec=train_cfg.get('use_torchcodec', True),
                data_rank=self.rank,
                data_world_size=self.world_size,
                seed=self.seed,
                random_start=training_cfg.get('random_start', False),
                spatial_transform_mode=train_cfg.get('spatial_transform_mode', 'resize_center_crop'),
                spatial_roundtrip_short_edge=train_cfg.get('spatial_roundtrip_short_edge'),
                distill_encoder_fps=_distill_encoder_fps,
                distill_audio_target_sr=_distill_audio_target_sr,
            )

            train_num_workers = train_cfg.get('num_workers', 4)
            train_dl_kwargs = dict(
                dataset=self.train_ds, batch_size=1,
                collate_fn=passthrough_collate_fn, num_workers=train_num_workers, pin_memory=True,
            )
            if train_num_workers > 0:
                train_dl_kwargs["prefetch_factor"] = train_cfg.get('prefetch_factor', 2)
            self.train_dl = DataLoader(**train_dl_kwargs)

            if self.is_main:
                logging.info(
                    f"[iv-alterstep] enabled: image_loader={image_loader}, "
                    f"image_batch_size={image_bs}, video_batch_size={video_bs}, "
                    f"weights={self.image_video_weights}"
                )
                if image_loader == 'relaion':
                    logging.info(
                        f"[iv-alterstep] relaion source: root={relaion_cfg.get('root')}, "
                        f"split={relaion_cfg.get('split', 'train')}, "
                        f"recaption_prob={relaion_cfg.get('recaption_prob', 0.0)}, "
                        f"max_samples={relaion_cfg.get('max_samples')}"
                    )
        else:
            self.image_video_weights = None
            self.image_batch_size_eff = None
            self.video_batch_size_eff = None

            self.train_ds = build_audio_video_streaming_dataset(
                dataset_dir_or_paths=metadata_paths if not data_mixture_yaml else None,
                num_frames=train_cfg.get('num_frames', 25),
                resolution=train_cfg.get('resolution', 256),
                sample_rate=train_cfg.get('video_frame_sample_rate', train_cfg.get('sample_rate', 1)),
                target_fps=train_cfg.get('target_fps'),
                audio_sample_rate=train_cfg.get('audio_sample_rate', 24000),
                max_audio_duration=train_cfg.get('max_audio_duration'),
                use_torchcodec=train_cfg.get('use_torchcodec', True),
                data_rank=self.rank,
                data_world_size=self.world_size,
                weights=metadata_weights if (metadata_weights and not data_mixture_yaml) else None,
                seed=self.seed,
                use_file_processor=train_cfg.get('use_file_processor', False),
                data_mixture_yaml=data_mixture_yaml,
                mixture_save_path=(self.exp_dir / 'data_mixture_triplets.tsv'
                                   if (data_mixture_yaml and self.is_main) else None),
                random_start=training_cfg.get('random_start', False),
                video_root=train_cfg.get('file_root'),
                spatial_transform_mode=train_cfg.get('spatial_transform_mode', 'resize_center_crop'),
                spatial_roundtrip_short_edge=train_cfg.get('spatial_roundtrip_short_edge'),
                distill_encoder_fps=_distill_encoder_fps,
                distill_audio_target_sr=_distill_audio_target_sr,
            )

            train_num_workers = train_cfg.get('num_workers', 4)
            train_dl_kwargs = dict(
                dataset=self.train_ds, batch_size=self.batch_size,
                collate_fn=AudioVideoCollator(), num_workers=train_num_workers, pin_memory=True,
            )
            if train_num_workers > 0:
                train_dl_kwargs["prefetch_factor"] = train_cfg.get('prefetch_factor', 2)
            self.train_dl = DataLoader(**train_dl_kwargs)

        # ---- Val video ----
        val_video_cfg = data_cfg.get('val_video', {})
        val_video_sample_rate = val_video_cfg.get('video_frame_sample_rate', val_video_cfg.get('sample_rate', 1))
        val_video_target_fps = val_video_cfg.get('target_fps')
        if val_video_target_fps is not None:
            self.val_video_save_fps = float(val_video_target_fps) / float(val_video_sample_rate)
        else:
            self.val_video_save_fps = float(max(1, val_video_sample_rate))
        self.val_video_max_samples = val_video_cfg.get('max_samples')
        # Two ways to declare validation video sources (jsonl_paths wins):
        #   - jsonl_paths: explicit {name: path} dict (or list of paths).
        #     Prefer this when the directory containing val jsonls is mixed
        #     with other jsonls you don't want to evaluate on.
        #   - video_dir: legacy. scan_jsonl_files() picks every *.jsonl in
        #     the directory, naming each ds by jsonl stem.
        jsonl_paths_cfg = val_video_cfg.get('jsonl_paths')
        video_dir = val_video_cfg.get('video_dir')
        self.val_video_datasets = {}
        self.val_video_dataloaders = {}
        if build_val_video and jsonl_paths_cfg:
            if isinstance(jsonl_paths_cfg, dict):
                video_dataset_paths = {str(name): str(path) for name, path in jsonl_paths_cfg.items()}
            elif isinstance(jsonl_paths_cfg, (list, tuple)):
                video_dataset_paths = {Path(p).stem: str(p) for p in jsonl_paths_cfg}
            else:
                raise ValueError(
                    f"data.val_video.jsonl_paths must be dict or list, "
                    f"got {type(jsonl_paths_cfg).__name__}."
                )
        elif build_val_video and video_dir:
            video_dataset_paths = scan_jsonl_files(video_dir)
        else:
            video_dataset_paths = {}
        if build_val_video and video_dataset_paths:
            if self.is_main:
                logging.info(f"Found {len(video_dataset_paths)} video val datasets")
            val_video_root = val_video_cfg.get('video_root')
            for ds_name, ds_path in video_dataset_paths.items():
                ds = build_video_only_dataset(
                    dataset_dir_or_paths={ds_name: ds_path},
                    num_frames=val_video_cfg.get('num_frames', 25),
                    resolution=val_video_cfg.get('resolution', 256),
                    sample_rate=val_video_sample_rate,
                    target_fps=val_video_target_fps,
                    data_rank=self.rank, data_world_size=self.world_size,
                    use_file_processor=True, video_root=val_video_root,
                    spatial_transform_mode=val_video_cfg.get(
                        'spatial_transform_mode',
                        train_cfg.get('spatial_transform_mode', 'resize_center_crop')),
                    spatial_roundtrip_short_edge=val_video_cfg.get(
                        'spatial_roundtrip_short_edge',
                        train_cfg.get('spatial_roundtrip_short_edge')),
                )
                self.val_video_datasets[ds_name] = ds
                self.val_video_dataloaders[ds_name] = DataLoader(
                    ds, batch_size=val_video_cfg.get('batch_size', self.batch_size),
                    collate_fn=AudioVideoCollator(),
                    num_workers=val_video_cfg.get('num_workers', 2),
                )

        # ---- Val audio ----
        val_audio_cfg = data_cfg.get('val_audio', {})
        self.val_audio_max_samples = val_audio_cfg.get('max_samples')
        # Same jsonl_paths-vs-audio_dir contract as val_video above.
        audio_jsonl_paths_cfg = val_audio_cfg.get('jsonl_paths')
        audio_dir = val_audio_cfg.get('audio_dir')
        self.val_audio_datasets = {}
        self.val_audio_dataloaders = {}
        if build_val_audio and audio_jsonl_paths_cfg:
            if isinstance(audio_jsonl_paths_cfg, dict):
                audio_dataset_paths = {str(name): str(path) for name, path in audio_jsonl_paths_cfg.items()}
            elif isinstance(audio_jsonl_paths_cfg, (list, tuple)):
                audio_dataset_paths = {Path(p).stem: str(p) for p in audio_jsonl_paths_cfg}
            else:
                raise ValueError(
                    f"data.val_audio.jsonl_paths must be dict or list, "
                    f"got {type(audio_jsonl_paths_cfg).__name__}."
                )
        elif build_val_audio and audio_dir:
            audio_dataset_paths = scan_jsonl_files(audio_dir)
        else:
            audio_dataset_paths = {}
        if build_val_audio and audio_dataset_paths:
            if self.is_main:
                logging.info(f"Found {len(audio_dataset_paths)} audio val datasets")
            for ds_name, ds_path in audio_dataset_paths.items():
                ds = build_audio_only_dataset(
                    dataset_dir_or_paths={ds_name: ds_path},
                    sample_rate=val_audio_cfg.get('audio_sample_rate',
                                                  val_audio_cfg.get('sample_rate', 24000)),
                    max_duration=val_audio_cfg.get('max_duration'),
                    data_rank=self.rank, data_world_size=self.world_size,
                    use_file_processor=True,
                )
                self.val_audio_datasets[ds_name] = ds
                self.val_audio_dataloaders[ds_name] = DataLoader(
                    ds, batch_size=val_audio_cfg.get('batch_size', self.batch_size),
                    collate_fn=AudioVideoCollator(),
                    num_workers=val_audio_cfg.get('num_workers', 2),
                )

        # ---- Val contrastive ----
        val_contrastive_cfg = data_cfg.get('val_contrastive', {})
        self.val_contrastive_max_samples = val_contrastive_cfg.get('max_samples')
        self.val_segment_num_negatives_list = _parse_positive_int_list(
            val_contrastive_cfg.get('val_segment_num_negatives'),
            default_value=64, field_name='data.val_contrastive.val_segment_num_negatives', cfg=cfg,
        )
        self.val_segment_num_negatives = self.val_segment_num_negatives_list[0]
        _raw_neg_videos = val_contrastive_cfg.get('val_segment_num_negative_videos', 64)
        _raw_neg_videos = _resolve_cfg_reference(_raw_neg_videos, cfg)
        self.val_segment_num_negative_videos = int(_raw_neg_videos) if _raw_neg_videos is not None else 64
        self.val_global_num_negatives_list = _parse_positive_int_list(
            val_contrastive_cfg.get('val_global_num_negatives'),
            default_value=32, field_name='data.val_contrastive.val_global_num_negatives', cfg=cfg,
        )
        self.val_global_num_negatives = self.val_global_num_negatives_list[0]
        contrastive_jsonl_paths = val_contrastive_cfg.get('jsonl_paths', {})
        self.val_contrastive_dataloaders = {}
        if build_val_contrastive and contrastive_jsonl_paths:
            for ds_name, ds_path in contrastive_jsonl_paths.items():
                ds = build_audio_video_streaming_dataset(
                    dataset_dir_or_paths={ds_name: ds_path},
                    num_frames=val_contrastive_cfg.get('num_frames', train_cfg.get('num_frames', 25)),
                    resolution=val_contrastive_cfg.get('resolution', train_cfg.get('resolution', 256)),
                    sample_rate=val_contrastive_cfg.get('video_frame_sample_rate',
                                                        train_cfg.get('video_frame_sample_rate', 1)),
                    target_fps=val_contrastive_cfg.get('target_fps', train_cfg.get('target_fps')),
                    audio_sample_rate=val_contrastive_cfg.get('audio_sample_rate',
                                                              train_cfg.get('audio_sample_rate', 24000)),
                    use_file_processor=True, raise_stop_iteration=True,
                    data_rank=self.rank, data_world_size=self.world_size,
                    spatial_transform_mode=val_contrastive_cfg.get(
                        'spatial_transform_mode',
                        train_cfg.get('spatial_transform_mode', 'resize_center_crop')),
                    spatial_roundtrip_short_edge=val_contrastive_cfg.get(
                        'spatial_roundtrip_short_edge',
                        train_cfg.get('spatial_roundtrip_short_edge')),
                )
                self.val_contrastive_dataloaders[ds_name] = DataLoader(
                    ds, batch_size=val_contrastive_cfg.get('batch_size', self.batch_size),
                    collate_fn=AudioVideoCollator(),
                    num_workers=val_contrastive_cfg.get('num_workers', 2),
                )

        # ---- Val caption ----
        val_caption_cfg = data_cfg.get('val_caption', {})
        self.val_caption_max_samples = val_caption_cfg.get('max_samples')
        self.val_caption_tb_generate_samples = val_caption_cfg.get('tb_generate_samples', 10)
        caption_jsonl_paths = val_caption_cfg.get('jsonl_paths', {})
        val_caption_file_root = val_caption_cfg.get('file_root', train_cfg.get('file_root'))
        self.val_caption_dataloaders = {}
        if build_val_caption and caption_jsonl_paths:
            for ds_name, ds_path in caption_jsonl_paths.items():
                ds = build_audio_video_streaming_dataset(
                    dataset_dir_or_paths={ds_name: ds_path},
                    num_frames=val_caption_cfg.get('num_frames', train_cfg.get('num_frames', 25)),
                    resolution=val_caption_cfg.get('resolution', train_cfg.get('resolution', 256)),
                    sample_rate=val_caption_cfg.get('video_frame_sample_rate',
                                                    train_cfg.get('video_frame_sample_rate', 1)),
                    target_fps=val_caption_cfg.get('target_fps', train_cfg.get('target_fps')),
                    audio_sample_rate=val_caption_cfg.get('audio_sample_rate',
                                                          train_cfg.get('audio_sample_rate', 24000)),
                    use_file_processor=True, raise_stop_iteration=True,
                    data_rank=self.rank, data_world_size=self.world_size,
                    video_root=val_caption_file_root,
                    spatial_transform_mode=val_caption_cfg.get(
                        'spatial_transform_mode',
                        train_cfg.get('spatial_transform_mode', 'resize_center_crop')),
                    spatial_roundtrip_short_edge=val_caption_cfg.get(
                        'spatial_roundtrip_short_edge',
                        train_cfg.get('spatial_roundtrip_short_edge')),
                )
                self.val_caption_dataloaders[ds_name] = DataLoader(
                    ds, batch_size=val_caption_cfg.get('batch_size', self.batch_size),
                    collate_fn=AudioVideoCollator(),
                    num_workers=val_caption_cfg.get('num_workers', 2),
                )

        # ---- Eval flags ----
        self.eval_ssim = training_cfg.get('eval_ssim', True)
        self.eval_fvd = training_cfg.get('eval_fvd', False)
        self.eval_fvd_method = training_cfg.get('eval_fvd_method', 'styleganv')

    # ------------------------------------------------------------------
    # _build_optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(self, cfg: Dict):
        training_cfg = cfg.get('training', {})
        lr = training_cfg.get('lr', 1e-4)
        self.max_grad_norm = training_cfg.get('max_grad_norm', 1.0)

        betas = tuple(training_cfg.get('betas', [0.9, 0.999]))
        weight_decay = training_cfg.get('weight_decay', 0.01)

        def _resolve_lr(key: str) -> float:
            val = training_cfg.get(key)
            return float(val) if val is not None else float(lr)

        lr_video_vae = _resolve_lr('lr_video_vae')
        lr_audio_vae = _resolve_lr('lr_audio_vae')
        lr_contrastive_head = _resolve_lr('lr_contrastive_head')
        lr_llm_caption_head = _resolve_lr('lr_llm_caption_head')
        lr_distill_proj = _resolve_lr('lr_distill_proj')

        # Cache raw (possibly None) values for use by phase-unfreeze logic.
        self._lr_global = float(lr)
        self._lr_video_vae_resolved = (
            float(training_cfg.get('lr_video_vae')) if training_cfg.get('lr_video_vae') is not None else None
        )
        self._lr_audio_vae_resolved = (
            float(training_cfg.get('lr_audio_vae')) if training_cfg.get('lr_audio_vae') is not None else None
        )
        self._lr_video_logvar_resolved = (
            float(training_cfg.get('lr_video_logvar')) if training_cfg.get('lr_video_logvar') is not None else None
        )

        model = self.unwrapped_model

        param_groups: List[Dict[str, Any]] = []
        used_ids = set()

        def _add_module_group(module, lr_val: float, name: str) -> None:
            if module is None:
                return
            params = [p for p in module.parameters()
                      if p.requires_grad and id(p) not in used_ids]
            if not params:
                return
            used_ids.update(id(p) for p in params)
            param_groups.append({'params': params, 'lr': lr_val, 'name': name})

        _add_module_group(model.video_vae, lr_video_vae, 'video_vae')
        _add_module_group(model.audio_vae, lr_audio_vae, 'audio_vae')
        _add_module_group(model.contrastive_head, lr_contrastive_head, 'contrastive_head')
        _add_module_group(model.llm_caption_head, lr_llm_caption_head, 'llm_caption_head')
        for _proj_name in ('image_distill_proj', 'video_distill_proj', 'audio_distill_proj'):
            _add_module_group(getattr(model, _proj_name, None), lr_distill_proj, _proj_name)

        rest_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in used_ids
        ]
        if rest_params:
            param_groups.append({'params': rest_params, 'lr': float(lr), 'name': 'rest'})

        if not param_groups:
            # Edge case: nothing is trainable yet (e.g., everything frozen).
            # AdamW requires at least one non-empty param group; fall back to
            # a dummy one so that later add_param_group calls work.
            param_groups.append({'params': [], 'lr': float(lr), 'name': 'rest'})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        if self.is_main:
            for grp in self.optimizer.param_groups:
                n = sum(p.numel() for p in grp['params'])
                logging.info(
                    f"Optimizer group '{grp.get('name','?')}': lr={grp['lr']:.3e}, params={n:,}"
                )

        warmup_steps = training_cfg.get('warmup_steps', 5000)
        self.warmup_steps = warmup_steps
        tot_steps = self.tot_train_steps

        # ---- Per-group (video_vae) scheduler overrides ----
        # All four fields may be null in cfg => fall back to sensible defaults
        # derived from the global scheduler + freeze_video_vae_until_step.
        loss_cfg_preview = cfg.get('loss', {})
        _freeze_until = int(loss_cfg_preview.get('freeze_video_vae_until_step', 0) or 0)
        _v_warmup_cfg = training_cfg.get('lr_video_vae_warmup_steps')
        _v_total_cfg = training_cfg.get('lr_video_vae_total_steps')
        _v_start_cfg = training_cfg.get('lr_video_vae_start_step')
        _v_min_cfg = training_cfg.get('lr_video_vae_min_ratio')
        v_warmup = int(_v_warmup_cfg) if _v_warmup_cfg is not None else int(warmup_steps)
        v_start = int(_v_start_cfg) if _v_start_cfg is not None else _freeze_until
        if _v_total_cfg is not None:
            v_total = int(_v_total_cfg)
        else:
            v_total = max(v_warmup + 1, int(tot_steps) - int(v_start))
        v_min_ratio = float(_v_min_cfg) if _v_min_cfg is not None else 0.0

        # ---- Per-group (audio_vae) scheduler overrides (symmetric to video) ----
        _freeze_audio_until = int(loss_cfg_preview.get('freeze_audio_vae_until_step', 0) or 0)
        _a_warmup_cfg = training_cfg.get('lr_audio_vae_warmup_steps')
        _a_total_cfg = training_cfg.get('lr_audio_vae_total_steps')
        _a_start_cfg = training_cfg.get('lr_audio_vae_start_step')
        _a_min_cfg = training_cfg.get('lr_audio_vae_min_ratio')
        a_warmup = int(_a_warmup_cfg) if _a_warmup_cfg is not None else int(warmup_steps)
        a_start = int(_a_start_cfg) if _a_start_cfg is not None else _freeze_audio_until
        if _a_total_cfg is not None:
            a_total = int(_a_total_cfg)
        else:
            a_total = max(a_warmup + 1, int(tot_steps) - int(a_start))
        a_min_ratio = float(_a_min_cfg) if _a_min_cfg is not None else 0.0

        self.scheduler = _MultiGroupWarmupCosineScheduler(
            self.optimizer,
            g_warmup=int(warmup_steps),
            g_total=int(tot_steps),
            v_warmup=v_warmup,
            v_total=v_total,
            v_start=v_start,
            v_min_ratio=v_min_ratio,
            a_warmup=a_warmup,
            a_total=a_total,
            a_start=a_start,
            a_min_ratio=a_min_ratio,
        )

        if self.is_main:
            _vv_override_set = any(
                training_cfg.get(k) is not None
                for k in (
                    'lr_video_vae_warmup_steps',
                    'lr_video_vae_total_steps',
                    'lr_video_vae_start_step',
                    'lr_video_vae_min_ratio',
                )
            )
            if _vv_override_set:
                logging.info(
                    "[scheduler] video_vae group has dedicated warmup+cosine: "
                    f"warmup={v_warmup}, total={v_total}, start_step={v_start}, "
                    f"min_ratio={v_min_ratio:g}; other groups use "
                    f"warmup={warmup_steps}, total={tot_steps}."
                )
            _aa_override_set = any(
                training_cfg.get(k) is not None
                for k in (
                    'lr_audio_vae_warmup_steps',
                    'lr_audio_vae_total_steps',
                    'lr_audio_vae_start_step',
                    'lr_audio_vae_min_ratio',
                )
            )
            if _aa_override_set:
                logging.info(
                    "[scheduler] audio_vae group has dedicated warmup+cosine: "
                    f"warmup={a_warmup}, total={a_total}, start_step={a_start}, "
                    f"min_ratio={a_min_ratio:g}; other groups use "
                    f"warmup={warmup_steps}, total={tot_steps}."
                )

        # ---------------------------------------------------------------
        # Audio discriminator optimizer (independent from generator)
        # ---------------------------------------------------------------
        self.optim_d = None
        self.scheduler_d = None
        self.scaler_d = None
        _any_disc_enabled = (
            getattr(self, 'use_audio_disc', False)
            or getattr(self, 'use_video_disc', False)
        )
        if _any_disc_enabled and self.discriminators:
            lr_disc = float(training_cfg.get('lr_disc', lr))
            disc_betas = tuple(training_cfg.get('disc_betas', [0.9, 0.99]))
            disc_wd = float(training_cfg.get('disc_weight_decay', 0.0))

            disc_params = list(itertools.chain(
                *[d.parameters() for d in self.discriminators.values()]
            ))
            self.optim_d = torch.optim.AdamW(
                [{'params': disc_params, 'lr': lr_disc, 'name': 'disc'}],
                lr=lr_disc, betas=disc_betas, weight_decay=disc_wd,
            )

            self.scheduler_d = _MultiGroupWarmupCosineScheduler(
                self.optim_d,
                g_warmup=int(warmup_steps),
                g_total=int(tot_steps),
                v_warmup=1, v_total=2, v_start=0, v_min_ratio=0.0,
            )

            # Dedicated GradScaler for D optimizer. Keeping G/D scales
            # independent avoids the scenario where a D-window inf check
            # downscales the shared scaler mid-way through a G-window's
            # accumulated gradients (which are already scaled by the
            # previous scale factor and would be unscaled with the wrong
            # factor on the next G boundary).
            self.scaler_d = torch.cuda.amp.GradScaler(enabled=self.scaler.is_enabled())

            if self.is_main:
                n_disc_params = sum(p.numel() for p in disc_params)
                logging.info(
                    f"[audio-disc] disc AdamW: lr={lr_disc:.3e}, betas={disc_betas}, "
                    f"weight_decay={disc_wd}, params={n_disc_params:,}"
                )

    # ------------------------------------------------------------------
    # _build_loss_functions
    # ------------------------------------------------------------------

    def _build_loss_functions(self, cfg: Dict):
        loss_cfg = cfg.get('loss', {})

        # ---- Loss switches ----
        self.use_video_recon = loss_cfg.get('use_video_recon', True)
        self.use_audio_recon = loss_cfg.get('use_audio_recon', True)
        self.use_segment_contrastive = loss_cfg.get('use_segment_contrastive', True)
        self.use_global_contrastive = loss_cfg.get('use_global_contrastive', True)

        # ---- Video loss weights ----
        self.lambda_video_recon = loss_cfg.get('lambda_video_recon', 1.0)
        self.lambda_video_kl = loss_cfg.get('lambda_video_kl', 1.65e-2)
        self.lambda_video_lpips = loss_cfg.get('lambda_video_lpips', 0.0)
        self.video_logvar_init = loss_cfg.get('video_logvar_init', 0.0)
        self.video_learn_logvar = loss_cfg.get('video_learn_logvar', False)
        self.video_loss_type = loss_cfg.get('video_loss_type', 'l1')
        self.video_loss_reduction = loss_cfg.get('video_loss_reduction', 'sum')
        assert self.video_loss_reduction in ('sum', 'mean')
        self.video_logvar = nn.Parameter(
            torch.full((), self.video_logvar_init, device=self.device),
            requires_grad=self.video_learn_logvar,
        )

        # ---- Audio loss weights ----
        self.lambda_audio_mel = loss_cfg.get('lambda_audio_mel', 1.0)
        self.lambda_audio_recon = loss_cfg.get('lambda_audio_recon', 1.0)
        self.lambda_audio_kl = loss_cfg.get('lambda_audio_kl', 1e-6)
        self.lambda_segment_contrastive = loss_cfg.get('lambda_segment_contrastive', 1.0)
        self.lambda_global_contrastive = loss_cfg.get('lambda_global_contrastive', 1.0)
        self.global_contrastive_start_steps = loss_cfg.get('global_contrastive_start_steps', 0)

        # Multi-granularity: per-granularity start steps and weights
        contrastive_head = self.unwrapped_model.contrastive_head
        n_gran = contrastive_head.n_granularities if contrastive_head is not None else 1

        _raw_start = loss_cfg.get('segment_avclip_start_steps', 0)
        if isinstance(_raw_start, (list, tuple)):
            self.segment_avclip_start_steps_list = [int(x) for x in _raw_start]
        else:
            self.segment_avclip_start_steps_list = [int(_raw_start)] * n_gran
        if len(self.segment_avclip_start_steps_list) != n_gran:
            raise ValueError(
                f"segment_avclip_start_steps length {len(self.segment_avclip_start_steps_list)} "
                f"!= segment_count_list length {n_gran}"
            )

        _raw_weights = loss_cfg.get('segment_count_weights', 1.0)
        if isinstance(_raw_weights, (list, tuple)):
            self.segment_count_weights_list = [float(x) for x in _raw_weights]
        else:
            self.segment_count_weights_list = [float(_raw_weights)] * n_gran
        if len(self.segment_count_weights_list) != n_gran:
            raise ValueError(
                f"segment_count_weights length {len(self.segment_count_weights_list)} "
                f"!= segment_count_list length {n_gran}"
            )

        # ---- LLM Caption ----
        self.use_llm_caption = loss_cfg.get('use_llm_caption', False)
        self.lambda_llm_caption = loss_cfg.get('lambda_llm_caption', 1.0)
        self.lambda_group_llm = loss_cfg.get('lambda_group_llm', 1.0)

        # ---- Group weights ----
        self.lambda_group_video = loss_cfg.get('lambda_group_video', 1.0)
        self.lambda_group_audio = loss_cfg.get('lambda_group_audio', 1.0)
        self.lambda_group_contrastive = loss_cfg.get('lambda_group_contrastive', 1.0)

        # ---- Video loss clamp ----
        self.video_loss_clamp = loss_cfg.get('video_loss_clamp', False)
        self.video_recon_clamp_max = loss_cfg.get('video_recon_clamp_max', 8000.0)
        self.video_lpips_clamp_max = loss_cfg.get('video_lpips_clamp_max', 0.25)
        self.video_kl_clamp_max = loss_cfg.get('video_kl_clamp_max', 6000.0)
        self._ema_video_recon = 0.0
        self._ema_video_lpips = 0.0
        self._ema_video_kl = 0.0
        self._ema_initialized = False
        self._ema_decay = 0.99
        self._spike_multiplier = 5.0

        # ---- Adaptive balance ----
        self.adaptive_loss_balance = loss_cfg.get('adaptive_loss_balance', False)
        self.adaptive_balance_audio_ratio = loss_cfg.get('adaptive_balance_audio_ratio', 0.5)
        self.adaptive_balance_contrastive_ratio = loss_cfg.get('adaptive_balance_contrastive_ratio', 0.5)

        # Uncertainty balance
        self.use_uncertainty_balance = loss_cfg.get('adaptive_loss_balance_by_uncertainty', False)
        self.uncertainty_warmup_steps = loss_cfg.get('uncertainty_warmup_steps', 100)

        # Gradient balance
        self.use_gradient_balance = loss_cfg.get('adaptive_loss_balance_by_gradient', False)
        self.gradient_balance_video_ratio = loss_cfg.get('gradient_balance_video_ratio', 0.5)
        self.gradient_balance_audio_ratio = loss_cfg.get('gradient_balance_audio_ratio', 0.5)
        self.gradient_balance_clamp_max = loss_cfg.get('gradient_balance_clamp_max', 100000.0)
        self.gradient_balance_interval = loss_cfg.get('gradient_balance_interval', 5)
        self._gbal_cached_w_c = 1.0
        self._gbal_cached_w_a = 1.0

        # Adaptive v2 hybrid: switch stage2 to gradient balance with optional
        # linear blend from anchor-scale over the first N steps after unfreeze.
        # stage2 ratios fall back to gradient_balance_*_ratio when None.
        self.adaptive_v2_stage2_use_gradient = bool(
            loss_cfg.get('adaptive_v2_stage2_use_gradient', False))
        self.adaptive_v2_stage2_blend_steps = int(
            loss_cfg.get('adaptive_v2_stage2_blend_steps', 0))
        _s2_rv = loss_cfg.get('gradient_ratio_video_stage2', None)
        _s2_ra = loss_cfg.get('gradient_ratio_audio_stage2', None)
        self.gradient_ratio_video_stage2 = (
            float(_s2_rv) if _s2_rv is not None else None)
        self.gradient_ratio_audio_stage2 = (
            float(_s2_ra) if _s2_ra is not None else None)
        # -1 = stage2 was not entered in this run (cold-start or resume-past-unfreeze).
        self._stage2_unfreeze_step: int = -1

        # Adaptive balance v2: EMA of a chosen anchor group; video/audio/contrastive
        # group totals are scaled so that each matches (EMA(anchor) * ratio_*).
        self.adaptive_loss_balance_v2 = loss_cfg.get('adaptive_loss_balance_v2', False)
        self.adaptive_anchor_source = str(
            loss_cfg.get('adaptive_anchor_source', 'video_vae')
        ).lower()
        if self.adaptive_anchor_source not in ('video_vae', 'audio_vae', 'contrastive'):
            raise ValueError(
                f"Unknown adaptive_anchor_source={self.adaptive_anchor_source!r}; "
                "expected one of video_vae|audio_vae|contrastive"
            )
        self.adaptive_anchor_ema_decay = float(loss_cfg.get('adaptive_anchor_ema_decay', 0.99))
        self.adaptive_anchor_warmup_steps = int(loss_cfg.get('adaptive_anchor_warmup_steps', 200))
        self.adaptive_scale_clamp_min = float(loss_cfg.get('adaptive_scale_clamp_min', 1e-3))
        self.adaptive_scale_clamp_max = float(loss_cfg.get('adaptive_scale_clamp_max', 100.0))
        self.adaptive_ratio_video = float(loss_cfg.get('adaptive_ratio_video', 1.0))
        self.adaptive_ratio_audio = float(loss_cfg.get('adaptive_ratio_audio', 1.0))
        self.adaptive_ratio_contrastive = float(loss_cfg.get('adaptive_ratio_contrastive', 1.0))
        # Stage1 (phase-freeze) overrides. None = fall back to stage2 values.
        _s1_src = loss_cfg.get('adaptive_anchor_source_stage1', None)
        if _s1_src is not None:
            _s1_src = str(_s1_src).lower()
            if _s1_src not in ('video_vae', 'audio_vae', 'contrastive'):
                raise ValueError(
                    f"Unknown adaptive_anchor_source_stage1={_s1_src!r}; "
                    "expected one of video_vae|audio_vae|contrastive"
                )
        self.adaptive_anchor_source_stage1 = _s1_src
        _s1_rv = loss_cfg.get('adaptive_ratio_video_stage1', None)
        _s1_ra = loss_cfg.get('adaptive_ratio_audio_stage1', None)
        _s1_rc = loss_cfg.get('adaptive_ratio_contrastive_stage1', None)
        self.adaptive_ratio_video_stage1 = float(_s1_rv) if _s1_rv is not None else None
        self.adaptive_ratio_audio_stage1 = float(_s1_ra) if _s1_ra is not None else None
        self.adaptive_ratio_contrastive_stage1 = float(_s1_rc) if _s1_rc is not None else None
        self._ema_video_vae: Optional[float] = None
        self._ema_audio_vae: Optional[float] = None
        self._ema_contrastive: Optional[float] = None

        active_balance = sum([self.adaptive_loss_balance,
                              self.use_uncertainty_balance,
                              self.use_gradient_balance,
                              self.adaptive_loss_balance_v2])
        if active_balance > 1:
            raise ValueError("Only one balance method can be enabled at a time.")

        if self.is_main and self.adaptive_loss_balance_v2:
            logging.info(
                f"Adaptive balance v2 enabled: anchor={self.adaptive_anchor_source}, "
                f"decay={self.adaptive_anchor_ema_decay}, "
                f"warmup={self.adaptive_anchor_warmup_steps}, "
                f"ratios v/a/c={self.adaptive_ratio_video}/{self.adaptive_ratio_audio}/"
                f"{self.adaptive_ratio_contrastive}, "
                f"scale_clamp=[{self.adaptive_scale_clamp_min}, {self.adaptive_scale_clamp_max}]"
            )
            _s1_any = (self.adaptive_anchor_source_stage1 is not None
                       or self.adaptive_ratio_video_stage1 is not None
                       or self.adaptive_ratio_audio_stage1 is not None
                       or self.adaptive_ratio_contrastive_stage1 is not None)
            if _s1_any:
                logging.info(
                    f"Adaptive balance v2 stage1 overrides: "
                    f"anchor={self.adaptive_anchor_source_stage1}, "
                    f"ratios v/a/c={self.adaptive_ratio_video_stage1}/"
                    f"{self.adaptive_ratio_audio_stage1}/"
                    f"{self.adaptive_ratio_contrastive_stage1} "
                    f"(None = fallback to stage2 value)"
                )

        self.skip_video_decoder = not self.use_video_recon
        self.skip_audio_decoder = not self.use_audio_recon

        # ---- Eval switches ----
        self._eval_video_recon_target = loss_cfg.get('eval_video_recon', self.use_video_recon)
        self.eval_video_recon = self._eval_video_recon_target
        self._eval_audio_recon_target = loss_cfg.get('eval_audio_recon', self.use_audio_recon)
        self.eval_audio_recon = self._eval_audio_recon_target

        # While video_vae is phase-frozen we also shut down the video decoder
        # forward and the entire video reconstruction loss path (recon / lpips /
        # video_kl), plus skip video eval. The state is flipped back inside
        # _unfreeze_video_vae_and_extend_optimizer() once the unfreeze step is
        # reached. NOTE: self.freeze_video_vae is assigned later in
        # _freeze_unused_params(), so we peek at loss_cfg directly here.
        _freeze_video_vae_init = bool(loss_cfg.get('freeze_video_vae', False))
        if _freeze_video_vae_init:
            self.skip_video_decoder = True
            self.eval_video_recon = False
        # Symmetric audio phase-freeze init: while audio_vae is phase-frozen
        # we skip the audio decoder forward and audio recon loss / eval.
        _freeze_audio_vae_init = bool(loss_cfg.get('freeze_audio_vae', False))
        if _freeze_audio_vae_init:
            self.skip_audio_decoder = True
            self.eval_audio_recon = False
        use_contrastive = self.use_segment_contrastive or self.use_global_contrastive
        self.eval_contrastive = loss_cfg.get('eval_contrastive', use_contrastive)
        self.eval_contrastive_in_all = loss_cfg.get('eval_contrastive_in_all', False)
        self.eval_llm_caption = loss_cfg.get('eval_llm_caption', self.use_llm_caption)

        if self.is_main:
            logging.info(f"Loss flags: video_recon={self.use_video_recon}, audio_recon={self.use_audio_recon}, "
                         f"segment_contrastive={self.use_segment_contrastive}, global_contrastive={self.use_global_contrastive}, "
                         f"llm_caption={self.use_llm_caption}, "
                         f"global_contrastive_start_steps={self.global_contrastive_start_steps}")
            logging.info(f"Multi-granularity: n_gran={n_gran}, "
                         f"segment_avclip_start_steps={self.segment_avclip_start_steps_list}, "
                         f"segment_count_weights={self.segment_count_weights_list}")
            logging.info(f"Decoder skip: video={self.skip_video_decoder}, audio={self.skip_audio_decoder}")
            logging.info(f"Eval flags: video={self.eval_video_recon}, audio={self.eval_audio_recon}, "
                         f"contrastive={self.eval_contrastive}, contrastive_in_all={self.eval_contrastive_in_all}, "
                         f"llm_caption={self.eval_llm_caption}")
            logging.info(f"Group weights: video={self.lambda_group_video}, audio={self.lambda_group_audio}, "
                         f"contrastive={self.lambda_group_contrastive}, llm={self.lambda_group_llm}")
            if self.video_loss_clamp:
                logging.info(f"Video loss clamp: recon_max={self.video_recon_clamp_max}, "
                             f"lpips_max={self.video_lpips_clamp_max}, kl_max={self.video_kl_clamp_max}")

        if self.video_loss_clamp and self.is_main:
            self.spike_sample_dir = self.exp_dir / 'spike_samples'
            self.spike_sample_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.spike_sample_dir = None

        # Logvar param group
        if self.video_learn_logvar:
            _lv_lr = (self._lr_video_logvar_resolved
                      if getattr(self, '_lr_video_logvar_resolved', None) is not None
                      else getattr(self, '_lr_global', 1e-4))
            self.optimizer.add_param_group({'params': [self.video_logvar], 'lr': _lv_lr,
                                            'name': 'video_logvar'})

        # Uncertainty params
        self._ub_task_keys = []
        _use_distill = loss_cfg.get('use_semantic_distill', False)
        if self.use_uncertainty_balance:
            if self.use_video_recon:
                self._ub_task_keys.append('video')
            if self.use_audio_recon:
                self._ub_task_keys.append('audio')
            if use_contrastive:
                self._ub_task_keys.append('contrastive')
            if self.use_llm_caption:
                self._ub_task_keys.append('llm')
            if _use_distill:
                self._ub_task_keys.append('distill')
            ub_params = []
            for key in self._ub_task_keys:
                param = nn.Parameter(torch.zeros(1, device=self.device))
                setattr(self, f'ub_log_var_{key}', param)
                ub_params.append(param)
            if ub_params:
                self.optimizer.add_param_group({'params': ub_params})
            self._ub_warmup_sums: Dict[str, float] = {k: 0.0 for k in self._ub_task_keys}
            self._ub_warmup_counts: Dict[str, int] = {k: 0 for k in self._ub_task_keys}
            self._ub_initialized = False

        # ---- LPIPS ----
        if self.use_video_recon:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            for p in self.lpips_model.parameters():
                p.requires_grad = False
        else:
            self.lpips_model = None

        # ---- Mel / Waveform ----
        if self.use_audio_recon:
            self.mel_loss = MultiResolutionMelSpectrogramLoss(**loss_cfg['mel_loss_kwargs']).to(self.device)
            self.waveform_loss = WaveformLoss()
        else:
            self.mel_loss = None
            self.waveform_loss = None

        # ---- Semantic Distillation ----
        self.use_semantic_distill = loss_cfg.get('use_semantic_distill', False)
        self.semantic_encoder: Optional[LocalSemanticEncoder] = None
        self.semantic_client: Optional[SemanticFeatureClient] = None
        self.distill_prefetcher = None  # Mode C: remote upload
        self.adaptive_distill_balance = False
        self.adaptive_distill_video_ratio = 0.1
        self.adaptive_distill_audio_ratio = 0.1

        if self.use_semantic_distill:
            self.lambda_distill_image_cosine = loss_cfg.get('lambda_distill_image_cosine', 1.0)
            self.lambda_distill_image_distance = loss_cfg.get('lambda_distill_image_distance', 1.0)
            self.lambda_distill_video_cosine = loss_cfg.get('lambda_distill_video_cosine', 1.0)
            self.lambda_distill_video_distance = loss_cfg.get('lambda_distill_video_distance', 1.0)
            self.lambda_distill_audio_d_axis = loss_cfg.get('lambda_distill_audio_d_axis', 120.0)
            self.lambda_distill_audio_t_axis = loss_cfg.get('lambda_distill_audio_t_axis', 1.0)
            self.lambda_group_distill = loss_cfg.get('lambda_group_distill', 1.0)
            self.distill_margin_cosine = loss_cfg.get('distill_margin_cosine', 0.0)
            self.distill_margin_distance = loss_cfg.get('distill_margin_distance', 0.25)
            self.distill_w_hyper = loss_cfg.get('distill_w_hyper', 0.1)
            self.distill_audio_type = loss_cfg.get('distill_audio_type', 'd_axis')

            # Phased activation: at which training step each modality's
            # distillation loss (and associated teacher-feature extraction)
            # begins. Default 0 preserves the legacy behaviour.
            self.video_distill_start_step = int(loss_cfg.get('video_distill_start_step', 0) or 0)
            self.audio_distill_start_step = int(loss_cfg.get('audio_distill_start_step', 0) or 0)

            # iREPA options
            self.distill_spatial_norm = loss_cfg.get('distill_spatial_norm', True)
            self.distill_spatial_norm_gamma = loss_cfg.get('distill_spatial_norm_gamma', 0.7)
            self.distill_use_dist_matrix = loss_cfg.get('distill_use_dist_matrix', False)

            # Adaptive distill balance (align distill with recon per modality)
            self.adaptive_distill_balance = loss_cfg.get('adaptive_distill_balance', False)
            self.adaptive_distill_video_ratio = loss_cfg.get('adaptive_distill_video_ratio', 0.1)
            self.adaptive_distill_audio_ratio = loss_cfg.get('adaptive_distill_audio_ratio', 0.1)

            # Switch the distill-balance mode from loss-value ratio (default)
            # to gradient-norm ratio. Only meaningful when
            # adaptive_distill_balance=True; reuses gradient_balance_interval
            # + gradient_balance_clamp_max for throttling and upper bound.
            self.adaptive_distill_use_gradient = bool(
                loss_cfg.get('adaptive_distill_use_gradient', False))
            self._distill_gbal_cached_w_vd = 1.0
            self._distill_gbal_cached_w_ad = 1.0

            self.distill_encoder_fps = float(loss_cfg.get('encoder_fps', 4.0))
            self.distill_encoder_resolution = int(loss_cfg.get('encoder_resolution', 128))
            _data_cfg = cfg.get('data', {})
            self.distill_data_fps = float(_data_cfg.get('train', {}).get('target_fps', 24))

            semantic_model_path = loss_cfg.get('semantic_model_path')
            semantic_api_url = loss_cfg.get('semantic_api_url')
            distill_upload_mode = loss_cfg.get('distill_upload_mode', False)

            distill_vision_layer = loss_cfg.get('distill_vision_layer', 18)
            distill_audio_layer = loss_cfg.get('distill_audio_layer', 24)

            if semantic_model_path:
                # Mode A: local inference
                self.semantic_encoder = LocalSemanticEncoder(
                    model_path=semantic_model_path,
                    device=self.device,
                    encoder_fps=self.distill_encoder_fps,
                    encoder_resolution=self.distill_encoder_resolution,
                    vision_layer=distill_vision_layer,
                    audio_layer=distill_audio_layer,
                )
                if self.is_main:
                    logging.info(
                        f"LocalSemanticEncoder loaded from {semantic_model_path} "
                        f"(vision_layer={distill_vision_layer}, audio_layer={distill_audio_layer})"
                    )
            elif semantic_api_url and distill_upload_mode:
                if self.is_main and (
                    (distill_vision_layer is not None and distill_vision_layer != 18)
                    or (distill_audio_layer is not None and distill_audio_layer != 24)
                ):
                    logging.warning(
                        "distill_vision_layer/distill_audio_layer are only honoured in "
                        "Mode A (semantic_model_path). Ignored under Mode C (upload); "
                        "remote service always returns its last-layer features."
                    )
                # Mode C: remote upload (cross-server, no shared storage)
                from .distill_client import DistillPrefetcher, _parse_gpu_map
                gpu_map = _parse_gpu_map(loss_cfg.get('distill_video_gpu_map', ''))
                self.distill_prefetcher = DistillPrefetcher(
                    base_url=semantic_api_url,
                    video_gpu_map=gpu_map,
                    image_gpu_id=int(loss_cfg.get('distill_image_gpu_id', 0)),
                    audio_gpu_id=int(loss_cfg.get('distill_audio_gpu_id', 0)),
                    rank=self.rank,
                    encoder_fps=self.distill_encoder_fps,
                    encoder_resolution=self.distill_encoder_resolution,
                    data_fps=self.distill_data_fps,
                    audio_sample_rate=self.audio_sample_rate,
                    max_workers=int(loss_cfg.get('distill_num_upload_workers', 6)),
                    needs_video=self.needs_video,
                    needs_audio=self.needs_audio,
                    processor_path=loss_cfg.get('distill_processor_path'),
                )
                if self.is_main:
                    logging.info(f"DistillPrefetcher (upload mode): api={semantic_api_url}, "
                                 f"gpu_map={gpu_map}, image_gpu={loss_cfg.get('distill_image_gpu_id', 0)}, "
                                 f"audio_gpu={loss_cfg.get('distill_audio_gpu_id', 0)}")
            elif semantic_api_url:
                if self.is_main and (
                    (distill_vision_layer is not None and distill_vision_layer != 18)
                    or (distill_audio_layer is not None and distill_audio_layer != 24)
                ):
                    logging.warning(
                        "distill_vision_layer/distill_audio_layer are only honoured in "
                        "Mode A (semantic_model_path). Ignored under Mode B (remote shared "
                        "path); remote service always returns its last-layer features."
                    )
                # Mode B: remote shared path
                self.semantic_client = SemanticFeatureClient(api_url=semantic_api_url)
                if self.is_main:
                    logging.info(f"SemanticFeatureClient using API: {semantic_api_url}")
            else:
                if self.is_main:
                    logging.warning("use_semantic_distill=True but no semantic_model_path or semantic_api_url set")
                self.use_semantic_distill = False

            if self.is_main and self.use_semantic_distill:
                logging.info(f"Distillation: spatial_norm={self.distill_spatial_norm}, "
                             f"gamma={self.distill_spatial_norm_gamma}, "
                             f"use_dist_matrix={self.distill_use_dist_matrix}, "
                             f"margin_cosine={self.distill_margin_cosine}, "
                             f"audio_type={self.distill_audio_type}, "
                             f"adaptive_balance={self.adaptive_distill_balance}"
                             + (f", video_ratio={self.adaptive_distill_video_ratio}, "
                                f"audio_ratio={self.adaptive_distill_audio_ratio}"
                                if self.adaptive_distill_balance else ""))
                logging.info(f"Distillation start steps: "
                             f"video_distill_start_step={self.video_distill_start_step}, "
                             f"audio_distill_start_step={self.audio_distill_start_step}")
                logging.info(
                    f"Adaptive distill mode: "
                    f"{'gradient-norm' if (self.adaptive_distill_balance and self.adaptive_distill_use_gradient) else ('loss-value' if self.adaptive_distill_balance else 'static')}"
                    + (f", interval={self.gradient_balance_interval}, clamp_max={self.gradient_balance_clamp_max}"
                       if (self.adaptive_distill_balance and self.adaptive_distill_use_gradient) else "")
                )

    # ------------------------------------------------------------------
    # Contrastive loss/metrics
    # ------------------------------------------------------------------

    def compute_contrastive_loss(
        self,
        contrastive_out: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=self.device)

        # Per-granularity segment losses
        granularities = contrastive_out.get("granularities", [])
        segment_sum = torch.tensor(0.0, device=self.device)

        for i, g in enumerate(granularities):
            g_losses = g.get("losses", {})
            sc_label = g.get("segment_count")
            sc_tag = str(sc_label) if sc_label is not None else "null"
            raw_loss = g_losses.get("segment_contrastive_loss")
            if raw_loss is None:
                continue

            losses[f"segment_contrastive_sc{sc_tag}_raw"] = raw_loss.float().detach()

            if self.train_state.step >= self.segment_avclip_start_steps_list[i]:
                w_i = self.segment_count_weights_list[i]
                segment_sum = segment_sum + w_i * raw_loss

        if segment_sum.requires_grad or segment_sum.item() != 0.0:
            losses["segment_contrastive_weighted"] = segment_sum.float().detach()
            total = total + self.lambda_segment_contrastive * segment_sum

        # Global loss
        global_losses = contrastive_out.get("losses", {})
        if "global_contrastive_loss" in global_losses:
            if self.train_state.step < self.global_contrastive_start_steps:
                losses["global_contrastive_raw"] = global_losses["global_contrastive_loss"].float().detach()
            else:
                total = total + self._record_weighted_loss(
                    losses,
                    "global_contrastive",
                    global_losses["global_contrastive_loss"],
                    self.lambda_global_contrastive,
                )

        logit_scale, global_logit_scale = contrastive_out.get("logit_scales", (None, None))
        if logit_scale is not None:
            losses["segment_logit_scale"] = logit_scale.float()
        if global_logit_scale is not None:
            losses["global_logit_scale"] = global_logit_scale.float()

        losses["contrastive_total"] = total
        return losses

    @torch.no_grad()
    def compute_contrastive_metrics(
        self,
        contrastive_out: Dict[str, Any],
    ) -> Dict[str, float]:
        metrics = {}
        contrastive_head = self.unwrapped_model.contrastive_head
        if contrastive_head is None:
            return metrics

        logit_scale, global_logit_scale = contrastive_out.get("logit_scales", (None, None))

        # Per-granularity segment metrics
        granularities = contrastive_out.get("granularities", [])
        for i, g in enumerate(granularities):
            segment_vfeat = g.get("segment_vfeat")
            segment_afeat = g.get("segment_afeat")
            B = g.get("B", 0)
            S = g.get("S", 0)
            B_eff = g.get("B_eff", B)
            rank_offset = g.get("rank_offset", 0)
            sc_label = g.get("segment_count")
            sc_tag = str(sc_label) if sc_label is not None else "null"
            suffix = f"_sc{sc_tag}" if len(granularities) > 1 else ""

            if segment_vfeat is not None and segment_afeat is not None and B > 0 and S > 0:
                vfeat_pool = g.get("segment_vfeat_pool", segment_vfeat)
                afeat_pool = g.get("segment_afeat_pool", segment_afeat)
                scale = logit_scale if logit_scale is not None else torch.tensor(1.0, device=self.device)

                sim_v2a, sim_a2v, _, num_intra = contrastive_head.sample_negatives_for_loss(
                    vfeat_local=segment_vfeat,
                    afeat_local=segment_afeat,
                    vfeat_pool=vfeat_pool,
                    afeat_pool=afeat_pool,
                    B=B, B_eff=B_eff, S=S,
                    scale=scale,
                    rank_offset=rank_offset,
                    num_negatives=contrastive_head.num_negatives_list[i],
                    num_negative_videos=contrastive_head.num_negative_videos_list[i],
                )
                seg_prec = compute_segment_sampled_precision(sim_v2a, sim_a2v, num_intra)
                for k, v in seg_prec.items():
                    metrics[f"{k}{suffix}"] = v
                if S > 1:
                    intra_prec = compute_segment_intra_precision(segment_vfeat, segment_afeat, B, S)
                    for k, v in intra_prec.items():
                        metrics[f"{k}{suffix}"] = v

        # Global metrics
        global_vfeat = contrastive_out.get("global_vfeat")
        global_afeat = contrastive_out.get("global_afeat")
        B_first = granularities[0].get("B", 0) if granularities else 0
        if global_vfeat is not None and global_afeat is not None and B_first > 1:
            global_vfeat_pool = contrastive_out.get("global_vfeat_pool", global_vfeat)
            global_afeat_pool = contrastive_out.get("global_afeat_pool", global_afeat)
            metrics.update(compute_global_sampled_precision(
                global_vfeat_pool, global_afeat_pool,
                num_negatives=min(32, global_vfeat_pool.shape[0] - 1),
            ))

        return metrics

    # ------------------------------------------------------------------
    # Video / Audio loss
    # ------------------------------------------------------------------

    def compute_video_loss(
        self, video: torch.Tensor, recon: torch.Tensor, posterior: Any,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        video_2d = rearrange(video, "b c t h w -> (b t) c h w").contiguous()
        recon_2d = rearrange(recon, "b c t h w -> (b t) c h w").contiguous()

        if self.video_loss_type == 'l2':
            pixel_loss = self.lambda_video_recon * torch.pow(video_2d - recon_2d, 2)
        else:
            pixel_loss = self.lambda_video_recon * torch.abs(video_2d - recon_2d)

        video_lpips_scalar = torch.tensor(0.0, device=video.device)
        if self.lambda_video_lpips > 0 and self.lpips_model is not None:
            video_lpips_scalar = self.lpips_model(
                video_2d.float().clamp(-1, 1),
                recon_2d.float().clamp(-1, 1),
            ).mean()

        if self.video_loss_reduction == "mean":
            pixel_loss_scalar = pixel_loss.mean()
        else:
            pixel_loss_scalar = torch.sum(pixel_loss) / pixel_loss.shape[0]

        video_kl = self._compute_video_kl_loss(posterior, video.device)

        losses['video_recon_raw'] = pixel_loss_scalar.detach()
        losses['video_kl_raw'] = video_kl.detach()
        if self.lambda_video_lpips > 0:
            losses['video_lpips_raw'] = video_lpips_scalar.detach()

        if self.video_loss_clamp:
            pre_recon = pixel_loss_scalar.detach().item()
            pre_lpips = video_lpips_scalar.detach().item()
            pre_kl = video_kl.detach().item()

            if pre_recon > self.video_recon_clamp_max:
                cap_ratio = self.video_recon_clamp_max / max(pre_recon, 1e-8)
                pixel_loss = pixel_loss.detach() * cap_ratio
            pixel_loss_scalar = torch.clamp(pixel_loss_scalar, max=self.video_recon_clamp_max)

            video_lpips_scalar = torch.clamp(video_lpips_scalar, max=self.video_lpips_clamp_max)
            video_kl = torch.clamp(video_kl, max=self.video_kl_clamp_max)

            clamped_parts = []
            if pre_recon > self.video_recon_clamp_max:
                clamped_parts.append(f"recon {pre_recon:.1f}->{self.video_recon_clamp_max:.1f}")
            if pre_lpips > self.video_lpips_clamp_max:
                clamped_parts.append(f"lpips {pre_lpips:.4f}->{self.video_lpips_clamp_max:.4f}")
            if pre_kl > self.video_kl_clamp_max:
                clamped_parts.append(f"kl {pre_kl:.1f}->{self.video_kl_clamp_max:.1f}")
            if clamped_parts:
                logging.warning(f"Step {self.train_state.step}: Video loss clamped: {', '.join(clamped_parts)}")

            losses['video_recon_clamped'] = pixel_loss_scalar.detach()
            losses['video_kl_clamped'] = video_kl.detach()
            if self.lambda_video_lpips > 0:
                losses['video_lpips_clamped'] = video_lpips_scalar.detach()

        rec_loss = pixel_loss
        if self.lambda_video_lpips > 0 and self.lpips_model is not None:
            rec_loss = rec_loss + self.lambda_video_lpips * video_lpips_scalar.to(rec_loss.dtype)

        logvar = self.video_logvar.to(device=rec_loss.device, dtype=rec_loss.dtype)
        nll_loss = rec_loss / torch.exp(logvar) + logvar

        if self.video_loss_reduction == "mean":
            video_nll = nll_loss.mean()
        else:
            video_nll = torch.sum(nll_loss) / nll_loss.shape[0]

        total = video_nll + self.lambda_video_kl * video_kl

        losses['video_nll_raw'] = video_nll.detach()
        losses['video_logvar'] = logvar.detach()
        losses['video_total'] = total
        if (self.use_gradient_balance or self.adaptive_v2_stage2_use_gradient
                or (self.adaptive_distill_balance and self.adaptive_distill_use_gradient)):
            losses['_video_recon_raw_live'] = pixel_loss_scalar
        return losses

    def _update_video_ema_and_detect_spike(
        self,
        video_losses: Dict[str, torch.Tensor],
        data: Dict[str, Any],
        losses: Dict[str, torch.Tensor],
    ) -> None:
        recon_val = video_losses['video_recon_raw'].item() if isinstance(video_losses['video_recon_raw'], torch.Tensor) else video_losses['video_recon_raw']
        lpips_val = video_losses.get('video_lpips_raw', torch.tensor(0.0))
        lpips_val = lpips_val.item() if isinstance(lpips_val, torch.Tensor) else lpips_val
        kl_val = video_losses['video_kl_raw'].item() if isinstance(video_losses['video_kl_raw'], torch.Tensor) else video_losses['video_kl_raw']

        if not self._ema_initialized:
            self._ema_video_recon = recon_val
            self._ema_video_lpips = lpips_val
            self._ema_video_kl = kl_val
            self._ema_initialized = True
        else:
            d = self._ema_decay
            self._ema_video_recon = d * self._ema_video_recon + (1 - d) * recon_val
            self._ema_video_lpips = d * self._ema_video_lpips + (1 - d) * lpips_val
            self._ema_video_kl = d * self._ema_video_kl + (1 - d) * kl_val

        ema_recon = max(self._ema_video_recon, 1e-8)
        ema_lpips = max(self._ema_video_lpips, 1e-8)
        ema_kl = max(self._ema_video_kl, 1e-8)

        losses['video_recon_volatility'] = recon_val / ema_recon
        losses['video_lpips_volatility'] = lpips_val / ema_lpips
        losses['video_kl_volatility'] = kl_val / ema_kl
        losses['video_recon_ema'] = self._ema_video_recon
        losses['video_kl_ema'] = self._ema_video_kl

        if not self.is_main:
            return

        is_spike = False
        spike_details = []
        k = self._spike_multiplier
        if recon_val > k * ema_recon:
            is_spike = True
            spike_details.append(f"recon={recon_val:.1f} vs EMA={ema_recon:.1f}")
        if lpips_val > k * ema_lpips and lpips_val > 1e-6:
            is_spike = True
            spike_details.append(f"lpips={lpips_val:.4f} vs EMA={ema_lpips:.4f}")
        if kl_val > k * ema_kl:
            is_spike = True
            spike_details.append(f"kl={kl_val:.1f} vs EMA={ema_kl:.1f}")

        if is_spike:
            step = self.train_state.step
            logging.warning(f"Step {step}: Video loss spike detected! {', '.join(spike_details)}")
            losses['video_spike'] = 1.0
            self._save_spike_samples(data, step, spike_details)

    def _save_spike_samples(
        self, data: Dict[str, Any], step: int, spike_details: list,
    ) -> None:
        if self.spike_sample_dir is None:
            return
        source_paths = data.get('source_paths', [])
        file_names = data.get('file_names', [])
        if not source_paths and not file_names:
            return

        import json
        manifest_path = self.spike_sample_dir / 'spike_manifest.jsonl'
        step_dir = self.spike_sample_dir / f'step_{step:08d}'

        for i, src in enumerate(source_paths):
            if not src:
                continue
            fname = file_names[i] if i < len(file_names) else f"sample_{i}"
            entry = {
                'step': step,
                'source_path': src,
                'file_name': fname,
                'spike_details': spike_details,
            }
            try:
                with manifest_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except OSError as e:
                logging.warning(f"Failed to write spike manifest: {e}")

            src_path = Path(src)
            if src_path.exists() and src_path.is_file():
                try:
                    step_dir.mkdir(parents=True, exist_ok=True)
                    dst = step_dir / src_path.name
                    if not dst.exists():
                        import shutil
                        shutil.copy2(src_path, dst)
                except OSError as e:
                    logging.warning(f"Failed to copy spike sample {src}: {e}")

    def compute_audio_loss(
        self, audio: torch.Tensor, recon: torch.Tensor, posterior: Any,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        min_len = min(audio.shape[-1], recon.shape[-1])
        audio = audio[..., :min_len].float()
        recon = recon[..., :min_len].float()

        audio_mel = self.mel_loss(recon, audio)
        audio_recon = self.waveform_loss(recon, audio)
        audio_kl = self._compute_kl_loss(posterior, audio.device)

        losses['audio_total'] = (
            self._record_weighted_loss(losses, 'audio_mel', audio_mel, self.lambda_audio_mel)
            + self._record_weighted_loss(losses, 'audio_recon', audio_recon, self.lambda_audio_recon)
            + self._record_weighted_loss(losses, 'audio_kl', audio_kl, self.lambda_audio_kl)
        )
        if (self.use_gradient_balance or self.adaptive_v2_stage2_use_gradient
                or (self.adaptive_distill_balance and self.adaptive_distill_use_gradient)):
            losses['_audio_mel_live'] = audio_mel
            losses['_audio_recon_live'] = audio_recon
        return losses

    # ------------------------------------------------------------------
    # Gradient balance
    # ------------------------------------------------------------------

    def _compute_per_loss_grad_norms(
        self, named_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        grad_norms: Dict[str, float] = {}
        params = [p for p in self.model.parameters() if p.requires_grad]
        for name, loss_val in named_losses.items():
            if not isinstance(loss_val, torch.Tensor) or not loss_val.requires_grad:
                continue
            grads = torch.autograd.grad(loss_val, params, retain_graph=True, allow_unused=True)
            total_norm_sq = sum(g.detach().float().pow(2).sum() for g in grads if g is not None)
            grad_norms[f'grad_norm/{name}'] = total_norm_sq.sqrt().item()
        return grad_norms

    def _compute_gradient_balance_weights(
        self, video_recon_ref, audio_recon_ref, contrastive_total, audio_total,
    ) -> tuple:
        v_enc = self.unwrapped_model.get_video_encoder_last_layer()
        a_enc = self.unwrapped_model.get_audio_encoder_last_layer()
        eps = 1e-6
        clamp_max = self.gradient_balance_clamp_max

        def _grad_norm(loss, param):
            g = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
            return g.detach().float().norm() if g is not None else torch.tensor(eps, device=loss.device)

        norm_video_v = _grad_norm(video_recon_ref, v_enc)
        norm_contra_v = _grad_norm(contrastive_total, v_enc)
        w_contrastive = (self.gradient_balance_video_ratio * norm_video_v / (norm_contra_v + eps))
        w_contrastive = w_contrastive.clamp(max=clamp_max).item()

        norm_contra_a = _grad_norm(contrastive_total, a_enc)
        norm_audio_a = _grad_norm(audio_total, a_enc)
        scaled_contra_a = w_contrastive * norm_contra_a
        w_audio = ((1.0 / self.gradient_balance_audio_ratio) * scaled_contra_a / (norm_audio_a + eps))
        w_audio = w_audio.clamp(max=clamp_max).item()

        diag = {
            'gbal/grad_norm_video_recon_on_venc': norm_video_v.item(),
            'gbal/grad_norm_contrastive_on_venc': norm_contra_v.item(),
            'gbal/grad_norm_contrastive_on_aenc': norm_contra_a.item(),
            'gbal/grad_norm_audio_total_on_aenc': norm_audio_a.item(),
        }
        return w_contrastive, w_audio, diag

    def _compute_distill_gradient_balance_weights(
        self,
        video_recon_live,
        video_distill_total,
        audio_recon_live,
        audio_distill_total,
    ) -> tuple:
        """Per-modality grad-norm balance for semantic distillation.

        Returns (w_vd, w_ad, diag). A weight is None if the corresponding
        modality is inactive (distill_total missing) or the relevant
        encoder produced no gradient (e.g. frozen video VAE); callers
        should keep the cached value instead of overwriting.
        """
        eps = 1e-6
        clamp_max = self.gradient_balance_clamp_max

        def _grad_norm(loss, param):
            if loss is None or param is None:
                return None
            g = torch.autograd.grad(
                loss, param, retain_graph=True, allow_unused=True,
            )[0]
            return g.detach().float().norm() if g is not None else None

        diag: Dict[str, float] = {}
        w_vd = None
        w_ad = None

        if video_distill_total is not None and video_recon_live is not None:
            v_enc = self.unwrapped_model.get_video_encoder_last_layer()
            n_rec = _grad_norm(video_recon_live, v_enc)
            n_dst = _grad_norm(video_distill_total, v_enc)
            if n_rec is not None and n_dst is not None and n_dst.item() > eps:
                w_vd = (self.adaptive_distill_video_ratio * n_rec
                        / (n_dst + eps)).clamp(max=clamp_max).item()
                diag['gbal/grad_norm_video_recon_on_venc_distill'] = n_rec.item()
                diag['gbal/grad_norm_video_distill_on_venc'] = n_dst.item()

        if audio_distill_total is not None and audio_recon_live is not None:
            a_enc = self.unwrapped_model.get_audio_encoder_last_layer()
            n_rec = _grad_norm(audio_recon_live, a_enc)
            n_dst = _grad_norm(audio_distill_total, a_enc)
            if n_rec is not None and n_dst is not None and n_dst.item() > eps:
                w_ad = (self.adaptive_distill_audio_ratio * n_rec
                        / (n_dst + eps)).clamp(max=clamp_max).item()
                diag['gbal/grad_norm_audio_recon_on_aenc_distill'] = n_rec.item()
                diag['gbal/grad_norm_audio_distill_on_aenc'] = n_dst.item()

        return w_vd, w_ad, diag

    # ------------------------------------------------------------------
    # Image+Video alterstep modality selection
    # ------------------------------------------------------------------

    def _select_modality_batch(
        self,
        batch: Dict[str, Any],
        step_is_d_only: bool = False,
    ) -> Dict[str, Any]:
        """从 IVAlterstep batch 中根据权重抽取一份 modality batch。

        若 ``batch`` 不是 alterstep 格式（即不含 ``image_batch`` / ``video_batch``
        两键），原样返回 — 此时 ``self._current_modality`` 保持为 ``'video'``。

        D-only window 强制选 video，因为现行 3D PatchGAN 判别器
        (NLayerDiscriminator3D, kernel=4 stride=2) 无法处理 T=1 输入。
        """
        if not isinstance(batch, dict):
            return batch
        if 'image_batch' not in batch or 'video_batch' not in batch:
            # 单流 batch，沿用旧路径
            self._current_modality = 'video'
            return batch

        if step_is_d_only:
            self._current_modality = 'video'
            return batch['video_batch']

        weights = self.image_video_weights or [1.0, 1.0]
        if sum(weights) <= 0:
            raise ValueError(f"image_video_weights must sum > 0, got {weights}")
        norm_w = np.asarray(weights, dtype=np.float64)
        norm_w = norm_w / norm_w.sum()
        idx = int(self._modality_rng.choice(2, p=norm_w))

        if idx == 0:
            self._current_modality = 'image'
            return batch['image_batch']
        else:
            self._current_modality = 'video'
            return batch['video_batch']

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any], collect_media: bool = False,
                   is_accum_boundary: bool = True, accum_steps: int = 1):
        # ---- G/D alternating dispatch (CausalVAE-style) ----
        # `train_state.step` is constant across the N micro-steps of one
        # accumulation cycle (only updated after the cycle ends), so the
        # parity check here naturally holds for the whole window.
        step_is_d_only = (
            self.use_video_disc
            and self.discriminators.get('video') is not None
            and not self.skip_video_decoder
            and self.train_state.step >= self.video_disc_start_step
            and (self.train_state.step % 2 == 1)
        )
        self._current_is_d_window = step_is_d_only

        # ---- Image+Video alterstep modality sampling ----
        # When iv-alterstep is enabled, the dataloader produces a dict with
        # both ``image_batch`` and ``video_batch``. Pick exactly one of them
        # for this step using a deterministic RNG (seeded the same on every
        # rank) so all GPUs agree without needing a torch.distributed sync.
        # During a D-only window we always pick the video branch, because the
        # 3D PatchGAN discriminator (kernel=4, stride=2 on T) cannot handle
        # T=1 inputs.
        batch = self._select_modality_batch(batch, step_is_d_only=step_is_d_only)

        if step_is_d_only:
            return self._train_step_video_disc_only(
                batch=batch,
                is_accum_boundary=is_accum_boundary,
                accum_steps=accum_steps,
                collect_media=collect_media,
            )

        data = batch['data']
        is_image_step = bool(data.get('is_image', False))
        self._current_modality = 'image' if is_image_step else 'video'
        video = data.get('video') if self.needs_video else None
        audio = data.get('audio') if self.needs_audio else None
        audio_lengths = data.get('audio_lengths') if self.needs_audio else None
        captions = data.get('captions') if self.use_llm_caption else None
        video_descriptions = data.get('video_descriptions') if self.use_llm_caption else None
        audio_descriptions = data.get('audio_descriptions') if self.use_llm_caption else None
        long_video_ids = data.get('long_video_ids')

        # Image step: image samples have no audio/captions/long-video metadata.
        # Suppress the corresponding pipelines so contrastive / LLM caption /
        # audio recon / audio distill / sibling-aware contrastive code paths
        # are all skipped for this micro-step.
        if is_image_step:
            audio = None
            audio_lengths = None
            captions = None
            video_descriptions = None
            audio_descriptions = None
            long_video_ids = None

        if video is not None:
            video = video.to(self.device)
        if audio is not None:
            audio = audio.to(self.device)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)

        contrastive_out = None
        group_losses = {}

        # ---- Semantic feature extraction (before forward for target shapes) ----
        semantic_feats = None
        distill_target_shapes = None

        if self.use_semantic_distill:
            _cur_step = self.train_state.step
            video_distill_active = self.needs_video and _cur_step >= self.video_distill_start_step
            audio_distill_active = self.needs_audio and _cur_step >= self.audio_distill_start_step

            distill_video = video if video_distill_active else None
            # Image step: never run audio distill (no audio).
            distill_audio = audio if (audio_distill_active and not is_image_step) else None

            any_active = (distill_video is not None) or (distill_audio is not None)

            if any_active and self.semantic_encoder is not None:
                with torch.no_grad():
                    semantic_feats = self.semantic_encoder.extract_from_tensors(
                        video=distill_video, audio=distill_audio,
                        video_fps=self.distill_data_fps,
                        audio_sample_rate=self.audio_sample_rate,
                    )
            elif any_active and self.distill_prefetcher is not None:
                # Mode C: retrieve pre-fetched results (submitted in train loop)
                semantic_feats = self.distill_prefetcher.get_features(self.device)
            elif any_active and self.semantic_client is not None:
                # Mode B: remote shared path
                file_paths = data.get('file_paths', [])
                if file_paths:
                    semantic_feats = self.semantic_client.extract(
                        file_paths=file_paths,
                        target_fps=self.distill_encoder_fps,
                        resolution=self.distill_encoder_resolution,
                        audio_sample_rate=self.audio_sample_rate,
                        device=self.device,
                    )

            if semantic_feats is not None:
                # Filter by modality need AND phased activation. Mode B/C may
                # return all modalities regardless of our inputs, so this pop
                # also acts as a safety net for those paths.
                if not video_distill_active:
                    semantic_feats.pop("image_feat", None)
                    semantic_feats.pop("video_feat", None)
                if not audio_distill_active or is_image_step:
                    semantic_feats.pop("audio_feat", None)
                # Image step: T=1, no temporal video latent — drop video_feat
                # so model.forward doesn't try to invoke video_distill_proj
                # on an empty (B, C, 0, H, W) slice.
                if is_image_step:
                    semantic_feats.pop("video_feat", None)

                distill_target_shapes = {}
                img_f = semantic_feats.get("image_feat")
                if img_f is not None:
                    distill_target_shapes["image"] = tuple(img_f.shape[1:])
                vf = semantic_feats.get("video_feat")
                if vf is not None:
                    distill_target_shapes["video"] = tuple(vf.shape[1:])
                af = semantic_feats.get("audio_feat")
                if af is not None:
                    distill_target_shapes["audio"] = tuple(af.shape[1:])

        # ---- Model forward ----
        outputs = self.model(
            video, audio,
            audio_lengths=audio_lengths,
            captions=captions,
            video_descriptions=video_descriptions,
            audio_descriptions=audio_descriptions,
            skip_video_decoder=self.skip_video_decoder,
            skip_audio_decoder=self.skip_audio_decoder,
            distill_target_shapes=distill_target_shapes,
            long_video_ids=long_video_ids,
        )

        total_loss = torch.tensor(0.0, device=self.device)
        losses = {}

        # Latent stats
        if video is not None and 'video' in outputs:
            losses['video_latent_mean'] = outputs['video']['latent'].float().mean()
            losses['video_latent_std'] = outputs['video']['latent'].float().std(unbiased=False)
        if audio is not None and 'audio' in outputs:
            losses['audio_latent_mean'] = outputs['audio']['latent'].float().mean()
            losses['audio_latent_std'] = outputs['audio']['latent'].float().std(unbiased=False)

        # ---- Contrastive ----
        contrastive_raw_total = None
        if 'contrastive' in outputs:
            contrastive_losses = self.compute_contrastive_loss(outputs['contrastive'])
            losses.update(contrastive_losses)
            contrastive_raw_total = contrastive_losses['contrastive_total']
            contrastive_out = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in outputs['contrastive'].items() if k != 'losses'
            }

        # ---- LLM Caption ----
        llm_raw_total = None
        if 'llm' in outputs and self.use_llm_caption:
            llm_loss_raw = outputs['llm']['loss']
            llm_loss_weighted = llm_loss_raw * self.lambda_llm_caption
            losses['llm_caption_raw'] = llm_loss_raw
            losses['llm_caption_weighted'] = llm_loss_weighted
            llm_raw_total = llm_loss_weighted

        # ---- Video recon ----
        video_raw_total = None
        video_losses = {}
        if video is not None and 'video' in outputs and not self.skip_video_decoder:
            video_out = outputs['video']
            video_losses = self.compute_video_loss(video, video_out['recon'], video_out['posterior'])
            losses.update(video_losses)
            video_raw_total = video_losses['video_total']

            if self.video_loss_clamp:
                self._update_video_ema_and_detect_spike(video_losses, data, losses)

        # ---- Video generator adversarial (CausalVAE-style 3D PatchGAN) ----
        # In the alternating scheme, G-window steps add -E[D(fake)] to the
        # generator's reconstruction loss; D-only steps are handled elsewhere
        # (see `_train_step_video_disc_only`). Use either a static lambda or
        # the VQGAN-style adaptive weight ||∇L_rec|| / ||∇L_g||.
        if (
            self.use_video_disc
            and self.discriminators.get('video') is not None
            and video_raw_total is not None
            and 'video' in outputs
            and outputs['video'].get('recon') is not None
            and self.train_state.step >= self.video_disc_start_step
            # Image step (T=1): NLayerDiscriminator3D uses kernel=4 stride=2
            # on the temporal axis, which can't accept T=1 inputs. Skip the
            # G-side adversarial loss entirely on image steps.
            and not is_image_step
        ):
            def _video_disc_autocast():
                # For fp32 disc: explicitly disable any outer autocast so the
                # disc runs at full precision (its weights were built in fp32).
                # Otherwise the outer bf16 autocast would cast disc weights to
                # bf16 while the input is also bf16 — the conv would "succeed"
                # but with much lower precision than intended.
                if self.disc_dtype != torch.float32:
                    return torch.cuda.amp.autocast(dtype=self.disc_dtype)
                return torch.cuda.amp.autocast(enabled=False)

            v_recon = outputs['video']['recon']
            with _video_disc_autocast():
                _fake_in = v_recon.to(dtype=self.disc_dtype)
                logits_fake = self.discriminators['video'](_fake_in)
            g_loss = video_generator_loss(logits_fake)

            if self.video_disc_adaptive_weight:
                # Use the raw video reconstruction loss (pre-group weighting)
                # as the "nll" anchor. `video_losses['video_total']` already
                # includes KL; that's acceptable — CausalVAE does similar.
                nll_ref = video_losses.get('video_total', video_raw_total)
                try:
                    last_layer = self.unwrapped_model.get_video_last_layer()
                except (AttributeError, RuntimeError):
                    last_layer = None
                if last_layer is not None:
                    d_weight = video_calc_adaptive_weight(
                        nll_loss=nll_ref,
                        g_loss=g_loss,
                        last_layer=last_layer,
                        discriminator_weight=self.lambda_video_adv,
                        clamp_max=self.video_disc_adaptive_weight_max,
                    )
                else:
                    d_weight = torch.tensor(self.lambda_video_adv, device=g_loss.device)
            else:
                d_weight = torch.tensor(self.lambda_video_adv, device=g_loss.device)

            video_raw_total = video_raw_total + d_weight * g_loss
            losses['video_g_loss'] = g_loss.detach()
            losses['video_d_weight'] = (
                d_weight.detach() if isinstance(d_weight, torch.Tensor) else torch.tensor(float(d_weight))
            )

        # ---- Audio recon ----
        audio_raw_total = None
        audio_losses = {}
        if audio is not None and 'audio' in outputs and not self.skip_audio_decoder:
            audio_out = outputs['audio']
            audio_losses = self.compute_audio_loss(audio, audio_out['recon'], audio_out['posterior'])
            losses.update(audio_losses)
            audio_raw_total = audio_losses['audio_total']

        # ---- Audio discriminator (LSGAN) + generator adversarial/feature matching ----
        # Decoupled gen/disc optimizers. Disc update runs only on accumulation
        # boundaries (mirroring the reference trainer_with_disc.py). Generator
        # adversarial / feature-matching losses are accumulated into
        # `audio_raw_total` on every micro-step, so they flow back through the
        # normal backward + grad-accum machinery below.
        if (
            self.use_audio_disc
            and self.discriminators
            and self.train_state.step >= self.audio_disc_start_step
            and audio is not None
            and 'audio' in outputs
            and audio_raw_total is not None
        ):
            audio_real = audio
            audio_fake = outputs['audio']['recon']
            T_clip = min(audio_real.shape[-1], audio_fake.shape[-1])
            audio_real_t = audio_real[..., :T_clip]
            audio_fake_t = audio_fake[..., :T_clip]

            def _disc_autocast():
                if self.disc_dtype != torch.float32:
                    return torch.cuda.amp.autocast(dtype=self.disc_dtype)
                return nullcontext()

            # --- Disc update (real vs detached fake) ---
            if is_accum_boundary:
                self.optim_d.zero_grad(set_to_none=True)
                audio_fake_det = audio_fake_t.detach()
                with _disc_autocast():
                    loss_disc_all = audio_real_t.new_zeros(())
                    for _name, disc in self.discriminators.items():
                        if _name == 'video':
                            continue
                        y_d_rs, y_d_gs, _, _ = disc(audio_real_t, audio_fake_det)
                        loss_disc_all = loss_disc_all + discriminator_loss(y_d_rs, y_d_gs)
                loss_disc_all.backward()
                _dmax = self.disc_max_grad_norm if self.disc_max_grad_norm is not None else self.max_grad_norm
                if _dmax is not None and _dmax > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(itertools.chain(
                            *[d.parameters() for _n, d in self.discriminators.items() if _n != 'video']
                        )),
                        _dmax,
                    )
                self.optim_d.step()
                self.scheduler_d.step()
                losses['disc_audio_total'] = loss_disc_all.detach()

            # --- Generator adversarial path (fake without detach) ---
            with _disc_autocast():
                loss_adv_sum = audio_real_t.new_zeros(())
                loss_fm_sum = audio_real_t.new_zeros(())
                for _name, disc in self.discriminators.items():
                    if _name == 'video':
                        continue
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(audio_real_t, audio_fake_t)
                    loss_fm_sum = loss_fm_sum + feature_loss(fmap_rs, fmap_gs)
                    loss_adv_sum = loss_adv_sum + adversarial_loss(y_d_gs)
            gen_adv_part = (
                self.lambda_audio_adv * loss_adv_sum
                + self.lambda_audio_feature_matching * loss_fm_sum
            )
            losses['audio_adv'] = loss_adv_sum.detach()
            losses['audio_feature_matching'] = loss_fm_sum.detach()
            losses['audio_adv_weighted'] = (self.lambda_audio_adv * loss_adv_sum).detach()
            losses['audio_fm_weighted'] = (self.lambda_audio_feature_matching * loss_fm_sum).detach()
            audio_raw_total = audio_raw_total + gen_adv_part

        # ---- Semantic Distillation Loss ----
        distill_raw_total = None
        video_distill_total = None
        audio_distill_total = None
        if self.use_semantic_distill and semantic_feats is not None and ('video' in outputs or 'audio' in outputs):
            _cur_step_loss = self.train_state.step
            video_distill_active = _cur_step_loss >= self.video_distill_start_step
            audio_distill_active = _cur_step_loss >= self.audio_distill_start_step

            image_feat = semantic_feats.get("image_feat")
            video_feat = semantic_feats.get("video_feat")
            audio_feat = semantic_feats.get("audio_feat")

            # Apply spatial normalization to teacher features (iREPA)
            if self.distill_spatial_norm:
                if image_feat is not None:
                    B_i = image_feat.shape[0]
                    trailing = image_feat.shape[1:]
                    H_i, W_i, D_i = trailing[0], trailing[1], trailing[2]
                    image_feat = spatial_normalize(
                        image_feat.reshape(B_i, H_i * W_i, D_i),
                        self.distill_spatial_norm_gamma,
                    ).reshape(B_i, H_i, W_i, D_i)
                if video_feat is not None:
                    B_v, T_v, H_v, W_v, D_v = video_feat.shape
                    video_feat = spatial_normalize(
                        video_feat.reshape(B_v * T_v, H_v * W_v, D_v),
                        self.distill_spatial_norm_gamma,
                    ).reshape(B_v, T_v, H_v, W_v, D_v)

            # Combined Image + Video VF Loss (frame-averaged)
            video_out = outputs.get("video", {})
            has_img = (image_feat is not None
                       and "image_distill_proj" in video_out
                       and video_distill_active)
            has_vid = (video_feat is not None
                       and "distill_proj" in video_out
                       and video_distill_active)

            if has_img or has_vid:
                visual_distill = torch.tensor(0.0, device=self.device)

                if has_img and has_vid:
                    z_proj_i = video_out["image_distill_proj"]   # (B, H_i, W_i, D)
                    z_proj_v = video_out["distill_proj"]         # (B, T, H_v, W_v, D)
                    img_f = image_feat.to(dtype=z_proj_i.dtype, device=z_proj_i.device)
                    vid_f = video_feat.to(dtype=z_proj_v.dtype, device=z_proj_v.device)

                    n_img = img_f.shape[-3] * img_f.shape[-2]          # H_i * W_i
                    n_vid = vid_f.shape[-4] * vid_f.shape[-3] * vid_f.shape[-2]  # T * H_v * W_v

                    img_cos_sum, img_cos_sim = marginal_cosine_similarity_loss(
                        z_proj_i, img_f, margin=self.distill_margin_cosine,
                        reduction="sum", nonneg=self.adaptive_distill_balance,
                    )
                    vid_cos_sum, vid_cos_sim = marginal_cosine_similarity_loss(
                        z_proj_v, vid_f, margin=self.distill_margin_cosine,
                        reduction="sum", nonneg=self.adaptive_distill_balance,
                    )
                    d_cos_combined = (img_cos_sum + vid_cos_sum) / (n_img + n_vid)
                    losses['distill_visual_cosine_raw'] = d_cos_combined.detach()
                    losses['distill_visual_cosine_sim_avg'] = (img_cos_sim * n_img + vid_cos_sim * n_vid) / (n_img + n_vid)
                    losses['distill_zproj_image_spatial_std'] = z_proj_i.reshape(z_proj_i.shape[0], -1, z_proj_i.shape[-1]).std(dim=-2).mean().detach()
                    losses['distill_zproj_video_spatial_std'] = z_proj_v.reshape(-1, z_proj_v.shape[-3] * z_proj_v.shape[-2], z_proj_v.shape[-1]).std(dim=-2).mean().detach()
                    losses['distill_teacher_image_spatial_std'] = img_f.reshape(img_f.shape[0], -1, img_f.shape[-1]).std(dim=-2).mean().detach()
                    losses['distill_teacher_video_spatial_std'] = vid_f.reshape(-1, vid_f.shape[-3] * vid_f.shape[-2], vid_f.shape[-1]).std(dim=-2).mean().detach()
                    visual_distill = visual_distill + self.lambda_distill_video_cosine * d_cos_combined

                    if self.distill_use_dist_matrix:
                        z_pooled_i = video_out["image_distill_pooled"]
                        z_pooled_v = video_out["distill_pooled"]
                        img_dist_sum = marginal_distance_matrix_loss(
                            z_pooled_i, img_f, margin=self.distill_margin_distance, reduction="sum",
                        )
                        vid_dist_sum = marginal_distance_matrix_loss(
                            z_pooled_v, vid_f, margin=self.distill_margin_distance, reduction="sum",
                        )
                        n_img_pairs = 1 * n_img ** 2
                        n_vid_pairs = vid_f.shape[-4] * (vid_f.shape[-3] * vid_f.shape[-2]) ** 2
                        d_dist_combined = (img_dist_sum + vid_dist_sum) / (n_img_pairs + n_vid_pairs)
                        losses['distill_visual_distance_raw'] = d_dist_combined.detach()
                        visual_distill = visual_distill + self.lambda_distill_video_distance * d_dist_combined

                elif has_img:
                    z_proj_i = video_out["image_distill_proj"]
                    img_f = image_feat.to(dtype=z_proj_i.dtype, device=z_proj_i.device)
                    d_i_cos, d_i_cos_sim = marginal_cosine_similarity_loss(
                        z_proj_i, img_f, margin=self.distill_margin_cosine,
                        nonneg=self.adaptive_distill_balance,
                    )
                    losses['distill_visual_cosine_raw'] = d_i_cos.detach()
                    losses['distill_visual_cosine_sim_avg'] = d_i_cos_sim
                    losses['distill_zproj_image_spatial_std'] = z_proj_i.reshape(z_proj_i.shape[0], -1, z_proj_i.shape[-1]).std(dim=-2).mean().detach()
                    losses['distill_teacher_image_spatial_std'] = img_f.reshape(img_f.shape[0], -1, img_f.shape[-1]).std(dim=-2).mean().detach()
                    visual_distill = visual_distill + self.lambda_distill_video_cosine * d_i_cos
                    if self.distill_use_dist_matrix:
                        z_pooled_i = video_out["image_distill_pooled"]
                        d_i_dist = marginal_distance_matrix_loss(
                            z_pooled_i, img_f, margin=self.distill_margin_distance,
                        )
                        losses['distill_visual_distance_raw'] = d_i_dist.detach()
                        visual_distill = visual_distill + self.lambda_distill_video_distance * d_i_dist

                else:  # has_vid only
                    z_proj_v = video_out["distill_proj"]
                    vid_f = video_feat.to(dtype=z_proj_v.dtype, device=z_proj_v.device)
                    d_v_cos, d_v_cos_sim = marginal_cosine_similarity_loss(
                        z_proj_v, vid_f, margin=self.distill_margin_cosine,
                        nonneg=self.adaptive_distill_balance,
                    )
                    losses['distill_visual_cosine_raw'] = d_v_cos.detach()
                    losses['distill_visual_cosine_sim_avg'] = d_v_cos_sim
                    losses['distill_zproj_video_spatial_std'] = z_proj_v.reshape(-1, z_proj_v.shape[-3] * z_proj_v.shape[-2], z_proj_v.shape[-1]).std(dim=-2).mean().detach()
                    losses['distill_teacher_video_spatial_std'] = vid_f.reshape(-1, vid_f.shape[-3] * vid_f.shape[-2], vid_f.shape[-1]).std(dim=-2).mean().detach()
                    visual_distill = visual_distill + self.lambda_distill_video_cosine * d_v_cos
                    if self.distill_use_dist_matrix:
                        z_pooled_v = video_out["distill_pooled"]
                        d_v_dist = marginal_distance_matrix_loss(
                            z_pooled_v, vid_f, margin=self.distill_margin_distance,
                        )
                        losses['distill_visual_distance_raw'] = d_v_dist.detach()
                        visual_distill = visual_distill + self.lambda_distill_video_distance * d_v_dist

                video_distill_total = self.distill_w_hyper * visual_distill
                losses['distill_video_total'] = video_distill_total.detach()

            # Audio distillation loss
            if (audio_feat is not None and 'audio' in outputs
                    and "distill_proj" in outputs["audio"]
                    and audio_distill_active):
                z_proj_a = outputs["audio"]["distill_proj"]
                aud_f = audio_feat.to(dtype=z_proj_a.dtype, device=z_proj_a.device)
                if self.distill_audio_type == 'd_axis':
                    aud_f_t = aud_f.permute(0, 2, 1)
                    d_a = d_axis_distill_loss(z_proj_a, aud_f_t)
                    losses['distill_audio_d_axis_raw'] = d_a.detach()
                    audio_distill_total = self.lambda_distill_audio_d_axis * d_a
                else:
                    d_a_cos, d_a_cos_sim = marginal_cosine_similarity_loss(
                        z_proj_a, aud_f, margin=self.distill_margin_cosine,
                        nonneg=self.adaptive_distill_balance,
                    )
                    losses['distill_audio_t_axis_raw'] = d_a_cos.detach()
                    losses['distill_audio_cosine_sim_avg'] = d_a_cos_sim
                    audio_distill_total = self.lambda_distill_audio_t_axis * d_a_cos
                losses['distill_audio_total'] = audio_distill_total.detach()

            _vd = video_distill_total if video_distill_total is not None else torch.tensor(0.0, device=self.device)
            _ad = audio_distill_total if audio_distill_total is not None else torch.tensor(0.0, device=self.device)
            losses['distill_total'] = (_vd + _ad).detach()
            distill_raw_total = True  # sentinel: distillation was computed

        # ---- Group weighting ----
        raw_totals: Dict[str, Optional[torch.Tensor]] = {
            'video': video_raw_total,
            'audio': audio_raw_total,
            'contrastive': contrastive_raw_total,
            'llm': llm_raw_total,
        }

        if self.adaptive_loss_balance_v2:
            current_step = self.train_state.step
            decay = self.adaptive_anchor_ema_decay
            in_warmup = current_step < self.adaptive_anchor_warmup_steps

            # Update EMAs of all three group raw totals (Python floats).
            def _update_ema(name: str, raw):
                if raw is None:
                    return
                val = raw.detach().item() if isinstance(raw, torch.Tensor) else float(raw)
                if not math.isfinite(val):
                    return
                attr = f'_ema_{name}'
                cur = getattr(self, attr)
                if cur is None:
                    setattr(self, attr, val)
                else:
                    setattr(self, attr, decay * cur + (1.0 - decay) * val)

            # Image step has T=1 — its recon loss has a very different scale
            # from video recon and would drift the EMA used as adaptive
            # anchor. Only update EMAs on video steps. Audio/contrastive
            # raw_totals are already None on image steps, but we explicitly
            # gate by modality for clarity.
            if not is_image_step:
                _update_ema('video_vae', video_raw_total)
                _update_ema('audio_vae', audio_raw_total)
                _update_ema('contrastive', contrastive_raw_total)

            # Expose EMAs for logging.
            if self._ema_video_vae is not None:
                losses['adaptive_v2_ema_video_vae'] = self._ema_video_vae
            if self._ema_audio_vae is not None:
                losses['adaptive_v2_ema_audio_vae'] = self._ema_audio_vae
            if self._ema_contrastive is not None:
                losses['adaptive_v2_ema_contrastive'] = self._ema_contrastive

            # Stage-aware anchor/ratios: stage1 (video_vae frozen) may use a
            # separate anchor & ratios; stage2 (unfrozen, or freeze disabled)
            # uses the canonical adaptive_* values.
            _src, _rv, _ra, _rc = self._resolve_adaptive_v2_params()
            anchor_val = getattr(self, f'_ema_{_src}')

            # ---- Hybrid mode: stage2 switches to gradient balance ----
            # blend_alpha = 0 → pure anchor-scale (stage1 behavior)
            # blend_alpha = 1 → pure gradient-scale (stage2 full gradient balance)
            # Only non-zero when stage2 was entered in this run (or already in
            # stage2 via resume, where alpha defaults to 1.0).
            use_hybrid = self.adaptive_v2_stage2_use_gradient
            in_stage2 = use_hybrid and (not getattr(self, '_video_vae_frozen', False))
            if in_stage2:
                if self._stage2_unfreeze_step < 0:
                    # Resumed directly into stage2 (no unfreeze event this run):
                    # treat blending as already completed.
                    blend_alpha = 1.0
                else:
                    bs = max(1, self.adaptive_v2_stage2_blend_steps)
                    elapsed = max(0, current_step - self._stage2_unfreeze_step)
                    blend_alpha = 1.0 if self.adaptive_v2_stage2_blend_steps <= 0 else \
                        min(1.0, elapsed / bs)
            else:
                blend_alpha = 0.0

            # Compute gradient-balance weights for contrastive/audio (only in
            # stage2 with hybrid enabled; reuses the existing boundary+interval
            # gate so that autograd.grad retains the graph only when needed).
            grad_w_c = 1.0
            grad_w_a = 1.0
            if (in_stage2
                    and contrastive_raw_total is not None
                    and video_raw_total is not None
                    and audio_raw_total is not None
                    and video_losses is not None and audio_losses is not None
                    and '_video_recon_raw_live' in video_losses
                    and '_audio_mel_live' in audio_losses
                    and '_audio_recon_live' in audio_losses):
                should_compute = (is_accum_boundary
                                  and current_step % self.gradient_balance_interval == 0)
                # Override ratios with stage2-specific values if provided.
                _saved_vr = self.gradient_balance_video_ratio
                _saved_ar = self.gradient_balance_audio_ratio
                if self.gradient_ratio_video_stage2 is not None:
                    self.gradient_balance_video_ratio = self.gradient_ratio_video_stage2
                if self.gradient_ratio_audio_stage2 is not None:
                    self.gradient_balance_audio_ratio = self.gradient_ratio_audio_stage2
                try:
                    if should_compute:
                        video_recon_ref = video_losses['_video_recon_raw_live']
                        audio_recon_ref = audio_losses['_audio_mel_live'] + audio_losses['_audio_recon_live']
                        w_c, w_a, gbal_diag = self._compute_gradient_balance_weights(
                            video_recon_ref, audio_recon_ref,
                            contrastive_raw_total, audio_raw_total,
                        )
                        self._gbal_cached_w_c = w_c
                        self._gbal_cached_w_a = w_a
                        losses.update(gbal_diag)
                    grad_w_c = self._gbal_cached_w_c
                    grad_w_a = self._gbal_cached_w_a
                finally:
                    self.gradient_balance_video_ratio = _saved_vr
                    self.gradient_balance_audio_ratio = _saved_ar

                losses['adaptive_v2_gbal_w_contrastive'] = grad_w_c
                losses['adaptive_v2_gbal_w_audio'] = grad_w_a
                losses['adaptive_v2_blend_alpha'] = blend_alpha

            # Per-group gradient-scale targets (what "scale_t" would be in a
            # pure-gradient run). Video is the reference, so its scale is 1.0.
            grad_scales = {
                'video': 1.0,
                'audio': grad_w_a,
                'contrastive': grad_w_c,
            }

            group_raws = {
                'video': (video_raw_total, _rv, self.lambda_group_video),
                'audio': (audio_raw_total, _ra, self.lambda_group_audio),
                'contrastive': (contrastive_raw_total, _rc,
                                self.lambda_group_contrastive),
            }

            for gkey, (raw, ratio, lambda_static) in group_raws.items():
                if raw is None:
                    continue
                # Anchor-based scale (stage1 formula). During anchor-warmup or
                # when the EMA is not yet valid, fall back to lambda_static so
                # the blend base-case degrades to static weighting.
                if in_warmup or anchor_val is None or anchor_val <= 0.0:
                    anchor_scale = 1.0  # -> gw = raw * 1.0 * lambda_static
                    anchor_scale_is_static = True
                else:
                    denom = raw.detach().clamp(min=1e-8)
                    scale_t = (anchor_val * ratio) / denom
                    scale_t = scale_t.clamp(
                        min=self.adaptive_scale_clamp_min,
                        max=self.adaptive_scale_clamp_max,
                    )
                    anchor_scale = scale_t
                    anchor_scale_is_static = False

                if in_stage2:
                    # Blend anchor-scale with gradient-scale.
                    g_scale = float(grad_scales[gkey])
                    if anchor_scale_is_static:
                        # anchor_scale == 1.0 (static weighting as base)
                        final_scale = (1.0 - blend_alpha) * 1.0 + blend_alpha * g_scale
                        final_scale = max(
                            self.adaptive_scale_clamp_min,
                            min(self.adaptive_scale_clamp_max, final_scale),
                        )
                        gw = raw * final_scale * lambda_static
                        losses[f'adaptive_v2_scale_{gkey}'] = float(final_scale)
                    else:
                        final_scale = (1.0 - blend_alpha) * anchor_scale + blend_alpha * g_scale
                        final_scale = final_scale.clamp(
                            min=self.adaptive_scale_clamp_min,
                            max=self.adaptive_scale_clamp_max,
                        )
                        gw = raw * final_scale * lambda_static
                        losses[f'adaptive_v2_scale_{gkey}'] = final_scale.detach()
                else:
                    # Pure anchor-scale (stage1 original behavior).
                    if anchor_scale_is_static:
                        gw = raw * lambda_static
                    else:
                        gw = raw * anchor_scale * lambda_static
                        losses[f'adaptive_v2_scale_{gkey}'] = anchor_scale.detach()

                losses[f'{gkey}_group_weighted'] = gw
                total_loss = total_loss + gw
                group_losses[gkey] = gw

            if llm_raw_total is not None:
                lgw = llm_raw_total * self.lambda_group_llm
                losses['llm_group_weighted'] = lgw
                total_loss = total_loss + lgw
                group_losses['llm'] = lgw

        elif self.use_uncertainty_balance:
            current_step = self.train_state.step
            in_warmup = current_step < self.uncertainty_warmup_steps
            if in_warmup:
                for key in self._ub_task_keys:
                    raw = raw_totals[key]
                    if raw is not None:
                        self._ub_warmup_sums[key] += raw.item()
                        self._ub_warmup_counts[key] += 1
            elif not self._ub_initialized:
                for key in self._ub_task_keys:
                    if self._ub_warmup_counts[key] > 0:
                        mean_loss = self._ub_warmup_sums[key] / self._ub_warmup_counts[key]
                        getattr(self, f'ub_log_var_{key}').data.fill_(math.log(max(mean_loss, 1e-8)))
                self._ub_initialized = True
            for key in self._ub_task_keys:
                raw = raw_totals[key]
                if raw is None:
                    continue
                if in_warmup:
                    static_weight = getattr(self, f'lambda_group_{key}', 1.0)
                    weighted = raw * static_weight
                else:
                    log_var = getattr(self, f'ub_log_var_{key}')
                    weighted = torch.exp(-log_var) * raw + log_var
                    losses[f'ub_log_var_{key}'] = log_var.detach()
                    losses[f'ub_weight_{key}'] = torch.exp(-log_var).detach()
                losses[f'{key}_group_weighted'] = weighted
                total_loss = total_loss + weighted
                group_losses[key] = weighted
        elif self.use_gradient_balance:
            if video_raw_total is not None:
                video_gw = video_raw_total * self.lambda_group_video
                losses['video_group_weighted'] = video_gw
                total_loss = total_loss + video_gw
                group_losses['video'] = video_gw
            if (contrastive_raw_total is not None and video_raw_total is not None
                    and audio_raw_total is not None):
                should_compute = (is_accum_boundary
                                  and self.train_state.step % self.gradient_balance_interval == 0)
                if should_compute:
                    video_recon_ref = video_losses['_video_recon_raw_live']
                    audio_recon_ref = audio_losses['_audio_mel_live'] + audio_losses['_audio_recon_live']
                    w_c, w_a, gbal_diag = self._compute_gradient_balance_weights(
                        video_recon_ref, audio_recon_ref, contrastive_raw_total, audio_raw_total,
                    )
                    self._gbal_cached_w_c = w_c
                    self._gbal_cached_w_a = w_a
                    losses.update(gbal_diag)
                else:
                    w_c, w_a = self._gbal_cached_w_c, self._gbal_cached_w_a
                losses['gbal_w_contrastive'] = w_c
                losses['gbal_w_audio'] = w_a
                contra_gw = contrastive_raw_total * w_c * self.lambda_group_contrastive
                losses['contrastive_group_weighted'] = contra_gw
                total_loss = total_loss + contra_gw
                group_losses['contrastive'] = contra_gw
                audio_gw = audio_raw_total * w_a * self.lambda_group_audio
                losses['audio_group_weighted'] = audio_gw
                total_loss = total_loss + audio_gw
                group_losses['audio'] = audio_gw
            else:
                if audio_raw_total is not None:
                    audio_gw = audio_raw_total * self.lambda_group_audio
                    losses['audio_group_weighted'] = audio_gw
                    total_loss = total_loss + audio_gw
                    group_losses['audio'] = audio_gw
                if contrastive_raw_total is not None:
                    contra_gw = contrastive_raw_total * self.lambda_group_contrastive
                    losses['contrastive_group_weighted'] = contra_gw
                    total_loss = total_loss + contra_gw
                    group_losses['contrastive'] = contra_gw
            if llm_raw_total is not None:
                llm_gw = llm_raw_total * self.lambda_group_llm
                losses['llm_group_weighted'] = llm_gw
                total_loss = total_loss + llm_gw
                group_losses['llm'] = llm_gw
        else:
            if video_raw_total is not None:
                vgw = video_raw_total * self.lambda_group_video
                losses['video_group_weighted'] = vgw
                total_loss = total_loss + vgw
                group_losses['video'] = vgw
            if llm_raw_total is not None:
                lgw = llm_raw_total * self.lambda_group_llm
                losses['llm_group_weighted'] = lgw
                total_loss = total_loss + lgw
                group_losses['llm'] = lgw
            _use_adaptive = (self.adaptive_loss_balance
                             and video_raw_total is not None and video_raw_total > 0)
            if _use_adaptive:
                video_ref = video_raw_total.detach()
            if audio_raw_total is not None:
                if _use_adaptive and audio_raw_total > 0:
                    adaptive_audio_scale = (video_ref * self.adaptive_balance_audio_ratio) / audio_raw_total.detach()
                    agw = audio_raw_total * adaptive_audio_scale
                    losses['adaptive_audio_scale'] = adaptive_audio_scale
                else:
                    agw = audio_raw_total * self.lambda_group_audio
                losses['audio_group_weighted'] = agw
                total_loss = total_loss + agw
                group_losses['audio'] = agw
            if contrastive_raw_total is not None:
                if _use_adaptive and contrastive_raw_total > 0:
                    adaptive_contrastive_scale = (video_ref * self.adaptive_balance_contrastive_ratio) / contrastive_raw_total.detach()
                    cgw = contrastive_raw_total * adaptive_contrastive_scale
                    losses['adaptive_contrastive_scale'] = adaptive_contrastive_scale
                else:
                    cgw = contrastive_raw_total * self.lambda_group_contrastive
                losses['contrastive_group_weighted'] = cgw
                total_loss = total_loss + cgw
                group_losses['contrastive'] = cgw

        # Distillation group weight
        if distill_raw_total is not None:
            _vd = video_distill_total if video_distill_total is not None else torch.tensor(0.0, device=self.device)
            _ad = audio_distill_total if audio_distill_total is not None else torch.tensor(0.0, device=self.device)

            if self.adaptive_distill_balance and self.adaptive_distill_use_gradient:
                # -- gradient-norm mode --
                # Align |grad(distill)| with |grad(recon)| per modality via
                # encoder-last-layer grad norms. Throttled by
                # gradient_balance_interval; result cached between refreshes.
                vd_weighted = torch.tensor(0.0, device=self.device)
                ad_weighted = torch.tensor(0.0, device=self.device)

                v_recon_live = video_losses.get('_video_recon_raw_live') if video_losses else None
                a_mel_live = audio_losses.get('_audio_mel_live') if audio_losses else None
                a_wav_live = audio_losses.get('_audio_recon_live') if audio_losses else None
                if a_mel_live is not None and a_wav_live is not None:
                    a_recon_live = a_mel_live + a_wav_live
                else:
                    a_recon_live = None

                should_compute = (
                    is_accum_boundary
                    and self.train_state.step % self.gradient_balance_interval == 0
                )
                if should_compute:
                    w_vd, w_ad, diag = self._compute_distill_gradient_balance_weights(
                        v_recon_live, video_distill_total,
                        a_recon_live, audio_distill_total,
                    )
                    if w_vd is not None:
                        self._distill_gbal_cached_w_vd = w_vd
                    if w_ad is not None:
                        self._distill_gbal_cached_w_ad = w_ad
                    if diag:
                        losses.update(diag)

                scale_v = self._distill_gbal_cached_w_vd
                scale_a = self._distill_gbal_cached_w_ad
                if video_distill_total is not None:
                    vd_weighted = _vd * scale_v
                    losses['adaptive_distill_video_scale'] = float(scale_v)
                if audio_distill_total is not None:
                    ad_weighted = _ad * scale_a
                    losses['adaptive_distill_audio_scale'] = float(scale_a)

                losses['distill_video_weighted'] = vd_weighted
                losses['distill_audio_weighted'] = ad_weighted
                dgw = vd_weighted + ad_weighted
            elif self.adaptive_distill_balance:
                # -- loss-value mode (legacy) --
                vd_weighted = torch.tensor(0.0, device=self.device)
                ad_weighted = torch.tensor(0.0, device=self.device)
                # Use recon-only (excluding KL) as anchor for adaptive scaling.
                # When adaptive_loss_balance_v2 is active, lift the recon anchor
                # to the "post v2 group-scaling" equivalent so the distill loss
                # tracks the *effective* recon contribution that actually enters
                # total_loss (i.e. raw * v2_scale * lambda_group).
                video_recon_ref = video_losses.get('video_nll_raw') if video_losses else None
                audio_recon_ref = None
                if audio_losses:
                    _mel = audio_losses.get('audio_mel_weighted')
                    _wav = audio_losses.get('audio_recon_weighted')
                    if _mel is not None and _wav is not None:
                        audio_recon_ref = (_mel + _wav).detach()

                def _post_v2_factor(gkey: str, lambda_group: float) -> float:
                    """Effective group-level multiplier that v2 applies to `raw`.

                    Returns v2_scale * lambda_group when v2 is active (falling
                    back to 1.0 * lambda_group during v2 warmup when the scale
                    is not yet published). Returns 1.0 when v2 is disabled so
                    the anchor stays at its pre-v2 magnitude.
                    """
                    if not self.adaptive_loss_balance_v2:
                        return 1.0
                    val = losses.get(f'adaptive_v2_scale_{gkey}', 1.0)
                    if isinstance(val, torch.Tensor):
                        val = float(val.detach().item())
                    else:
                        val = float(val)
                    return val * float(lambda_group)

                if self.adaptive_loss_balance_v2:
                    v_factor = _post_v2_factor('video', self.lambda_group_video)
                    a_factor = _post_v2_factor('audio', self.lambda_group_audio)
                    if video_recon_ref is not None:
                        video_recon_ref = video_recon_ref * v_factor
                    if audio_recon_ref is not None:
                        audio_recon_ref = audio_recon_ref * a_factor
                    losses['adaptive_distill_video_anchor_post_v2'] = v_factor
                    losses['adaptive_distill_audio_anchor_post_v2'] = a_factor

                if video_distill_total is not None and video_recon_ref is not None and video_recon_ref > 0 and _vd > 0:
                    scale_v = (video_recon_ref * self.adaptive_distill_video_ratio) / _vd.detach()
                    vd_weighted = _vd * scale_v
                    losses['adaptive_distill_video_scale'] = scale_v
                elif video_distill_total is not None:
                    vd_weighted = _vd * self.lambda_group_distill
                if audio_distill_total is not None and audio_recon_ref is not None and audio_recon_ref > 0 and _ad > 0:
                    scale_a = (audio_recon_ref * self.adaptive_distill_audio_ratio) / _ad.detach()
                    ad_weighted = _ad * scale_a
                    losses['adaptive_distill_audio_scale'] = scale_a
                elif audio_distill_total is not None:
                    ad_weighted = _ad * self.lambda_group_distill
                losses['distill_video_weighted'] = vd_weighted
                losses['distill_audio_weighted'] = ad_weighted
                dgw = vd_weighted + ad_weighted
            else:
                vd_w = _vd * self.lambda_group_distill
                ad_w = _ad * self.lambda_group_distill
                losses['distill_video_weighted'] = vd_w
                losses['distill_audio_weighted'] = ad_w
                dgw = vd_w + ad_w

            losses['distill_group_weighted'] = dgw
            total_loss = total_loss + dgw
            group_losses['distill'] = dgw

        losses['total'] = total_loss

        if not total_loss.requires_grad:
            logging.warning("total_loss has no grad_fn. Skipping backward.")
            return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}, contrastive_out, None

        # NaN/Inf loss detection (do NOT touch optimizer here; outer loop handles zero_grad)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.warning(f"Step {self.train_state.step}: NaN/Inf loss detected ({total_loss.item():.4f}), skipping update")
            losses['skipped_nan'] = 1.0
            return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}, contrastive_out, None

        # Grad diagnostics: only on accumulation boundary (final micro-step)
        do_grad_diag = (
            self.grad_log_steps > 0 and self.is_main
            and is_accum_boundary
            and (self.train_state.step + 1) % self.grad_log_steps == 0
            and len(group_losses) >= 1
        )
        if do_grad_diag:
            grad_norms = self._compute_per_loss_grad_norms(group_losses)
            losses.update(grad_norms)

        # ---- Backward (scale loss by 1/N under gradient accumulation) ----
        if accum_steps > 1:
            backward_loss = total_loss / accum_steps
        else:
            backward_loss = total_loss

        # Skip DDP all-reduce on non-boundary micro-steps for efficiency.
        # Also skip all-reduce on discriminator params during the generator
        # backward: their gradients here are just a byproduct of the adv +
        # feature-matching path (fake not detached), and will be zeroed at
        # the next disc-update boundary anyway.
        with ExitStack() as _stack:
            if self.is_distributed and not is_accum_boundary:
                _stack.enter_context(self.model.no_sync())
            if (
                self.is_distributed
                and (self.use_audio_disc or self.use_video_disc)
                and self.discriminators
            ):
                for _d in self.discriminators.values():
                    if isinstance(_d, DDP):
                        _stack.enter_context(_d.no_sync())
            self.scaler.scale(backward_loss).backward()

        # Non-boundary micro-steps: just accumulate gradients, no clip/step/zero
        if not is_accum_boundary:
            losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            return losses_scalar, contrastive_out, None

        if self.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            losses['grad_norm/total_before_clip'] = total_grad_norm

            # NaN/Inf gradient detection
            if torch.isnan(total_grad_norm) or torch.isinf(total_grad_norm):
                logging.warning(f"Step {self.train_state.step}: NaN/Inf grad norm detected, skipping update")
                self.optimizer.zero_grad(set_to_none=True)
                losses['skipped_nan_grad'] = 1.0
                return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}, contrastive_out, None

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Clear any disc-param grads that leaked from G's adversarial forward
        # (both the audio adv path and the new video adv path produce grads on
        # disc params via `disc(fake)` without detach, which must not carry
        # over into the next disc-window backward).
        if self.optim_d is not None:
            self.optim_d.zero_grad(set_to_none=True)

        if self.ema is not None:
            self.ema.update()

        losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

        media_data = None
        if collect_media:
            media_data = {}
            if video is not None and 'video' in outputs and not self.skip_video_decoder:
                media_data['video_gt'] = video.detach().float().cpu()
                media_data['video_recon'] = outputs['video']['recon'].detach().float().cpu()
            if audio is not None and 'audio' in outputs and not self.skip_audio_decoder:
                media_data['audio_gt'] = audio.detach().float().cpu()
                media_data['audio_recon'] = outputs['audio']['recon'].detach().float().cpu()

        return losses_scalar, contrastive_out, media_data

    # ------------------------------------------------------------------
    # Video discriminator alternating training helpers
    # ------------------------------------------------------------------

    def _video_disc_autocast(self):
        """Autocast context for the video discriminator forward.

        For fp32 disc: explicitly disable any outer autocast so the disc runs
        at full precision (its weights were built in fp32). Otherwise an
        enclosing bf16 autocast would transparently cast disc weights to
        bf16, costing precision on hinge loss + adaptive-weight gradient
        ratio computations.
        """
        if self.disc_dtype != torch.float32:
            return torch.cuda.amp.autocast(dtype=self.disc_dtype)
        return torch.cuda.amp.autocast(enabled=False)

    def _generator_params(self):
        """Iterate G-side (model) parameters that require grad.

        Used by `_check_generator_grad_finite` / `_zero_generator_grads`
        to scope NaN/Inf checks and zeroing to the generator only,
        leaving disc params (held in `self.discriminators`) untouched.
        """
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def _check_generator_grad_finite(self) -> bool:
        """Return True iff every populated G param gradient is finite."""
        for p in self._generator_params():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
        return True

    def _zero_generator_grads(self):
        """Zero out grads on G params (keep disc grads intact)."""
        for p in self._generator_params():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def _compute_d_only_distill(
        self,
        outputs: Dict[str, Any],
        semantic_feats: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Distill-loss for D-only window (static weighting, no adaptive balance).

        Returns
        -------
        (total, logs): ``total`` is the scalar loss ready for backward
            (includes ``lambda_group_distill``); ``None`` if no active
            distill component fired. ``logs`` are detached scalars for
            the losses dict.
        """
        _cur_step = self.train_state.step
        video_distill_active = _cur_step >= self.video_distill_start_step
        audio_distill_active = _cur_step >= self.audio_distill_start_step

        logs: Dict[str, torch.Tensor] = {}

        image_feat = semantic_feats.get("image_feat")
        video_feat = semantic_feats.get("video_feat")
        audio_feat = semantic_feats.get("audio_feat")

        if self.distill_spatial_norm:
            if image_feat is not None:
                B_i = image_feat.shape[0]
                H_i, W_i, D_i = image_feat.shape[1], image_feat.shape[2], image_feat.shape[3]
                image_feat = spatial_normalize(
                    image_feat.reshape(B_i, H_i * W_i, D_i),
                    self.distill_spatial_norm_gamma,
                ).reshape(B_i, H_i, W_i, D_i)
            if video_feat is not None:
                B_v, T_v, H_v, W_v, D_v = video_feat.shape
                video_feat = spatial_normalize(
                    video_feat.reshape(B_v * T_v, H_v * W_v, D_v),
                    self.distill_spatial_norm_gamma,
                ).reshape(B_v, T_v, H_v, W_v, D_v)

        video_out = outputs.get("video", {}) or {}
        has_img = (image_feat is not None
                   and "image_distill_proj" in video_out
                   and video_distill_active)
        has_vid = (video_feat is not None
                   and "distill_proj" in video_out
                   and video_distill_active)

        video_distill_total: Optional[torch.Tensor] = None
        if has_img or has_vid:
            visual_distill = torch.tensor(0.0, device=self.device)
            if has_img and has_vid:
                z_proj_i = video_out["image_distill_proj"]
                z_proj_v = video_out["distill_proj"]
                img_f = image_feat.to(dtype=z_proj_i.dtype, device=z_proj_i.device)
                vid_f = video_feat.to(dtype=z_proj_v.dtype, device=z_proj_v.device)
                n_img = img_f.shape[-3] * img_f.shape[-2]
                n_vid = vid_f.shape[-4] * vid_f.shape[-3] * vid_f.shape[-2]
                img_cos_sum, _ = marginal_cosine_similarity_loss(
                    z_proj_i, img_f, margin=self.distill_margin_cosine,
                    reduction="sum", nonneg=self.adaptive_distill_balance,
                )
                vid_cos_sum, _ = marginal_cosine_similarity_loss(
                    z_proj_v, vid_f, margin=self.distill_margin_cosine,
                    reduction="sum", nonneg=self.adaptive_distill_balance,
                )
                d_cos = (img_cos_sum + vid_cos_sum) / (n_img + n_vid)
                visual_distill = visual_distill + self.lambda_distill_video_cosine * d_cos
                logs['distill_visual_cosine_raw'] = d_cos.detach()
                if self.distill_use_dist_matrix:
                    z_pooled_i = video_out["image_distill_pooled"]
                    z_pooled_v = video_out["distill_pooled"]
                    img_dist_sum = marginal_distance_matrix_loss(
                        z_pooled_i, img_f, margin=self.distill_margin_distance, reduction="sum",
                    )
                    vid_dist_sum = marginal_distance_matrix_loss(
                        z_pooled_v, vid_f, margin=self.distill_margin_distance, reduction="sum",
                    )
                    n_img_pairs = 1 * n_img ** 2
                    n_vid_pairs = vid_f.shape[-4] * (vid_f.shape[-3] * vid_f.shape[-2]) ** 2
                    d_dist = (img_dist_sum + vid_dist_sum) / (n_img_pairs + n_vid_pairs)
                    visual_distill = visual_distill + self.lambda_distill_video_distance * d_dist
                    logs['distill_visual_distance_raw'] = d_dist.detach()
            elif has_img:
                z_proj_i = video_out["image_distill_proj"]
                img_f = image_feat.to(dtype=z_proj_i.dtype, device=z_proj_i.device)
                d_i_cos, _ = marginal_cosine_similarity_loss(
                    z_proj_i, img_f, margin=self.distill_margin_cosine,
                    nonneg=self.adaptive_distill_balance,
                )
                visual_distill = visual_distill + self.lambda_distill_video_cosine * d_i_cos
                logs['distill_visual_cosine_raw'] = d_i_cos.detach()
                if self.distill_use_dist_matrix:
                    z_pooled_i = video_out["image_distill_pooled"]
                    d_i_dist = marginal_distance_matrix_loss(
                        z_pooled_i, img_f, margin=self.distill_margin_distance,
                    )
                    visual_distill = visual_distill + self.lambda_distill_video_distance * d_i_dist
                    logs['distill_visual_distance_raw'] = d_i_dist.detach()
            else:  # has_vid only
                z_proj_v = video_out["distill_proj"]
                vid_f = video_feat.to(dtype=z_proj_v.dtype, device=z_proj_v.device)
                d_v_cos, _ = marginal_cosine_similarity_loss(
                    z_proj_v, vid_f, margin=self.distill_margin_cosine,
                    nonneg=self.adaptive_distill_balance,
                )
                visual_distill = visual_distill + self.lambda_distill_video_cosine * d_v_cos
                logs['distill_visual_cosine_raw'] = d_v_cos.detach()
                if self.distill_use_dist_matrix:
                    z_pooled_v = video_out["distill_pooled"]
                    d_v_dist = marginal_distance_matrix_loss(
                        z_pooled_v, vid_f, margin=self.distill_margin_distance,
                    )
                    visual_distill = visual_distill + self.lambda_distill_video_distance * d_v_dist
                    logs['distill_visual_distance_raw'] = d_v_dist.detach()

            video_distill_total = self.distill_w_hyper * visual_distill
            logs['distill_video_total'] = video_distill_total.detach()

        audio_out = outputs.get("audio", {}) or {}
        audio_distill_total: Optional[torch.Tensor] = None
        if (audio_feat is not None
                and "distill_proj" in audio_out
                and audio_distill_active):
            z_proj_a = audio_out["distill_proj"]
            aud_f = audio_feat.to(dtype=z_proj_a.dtype, device=z_proj_a.device)
            if self.distill_audio_type == 'd_axis':
                aud_f_t = aud_f.permute(0, 2, 1)
                d_a = d_axis_distill_loss(z_proj_a, aud_f_t)
                audio_distill_total = self.lambda_distill_audio_d_axis * d_a
                logs['distill_audio_d_axis_raw'] = d_a.detach()
            else:
                d_a_cos, _ = marginal_cosine_similarity_loss(
                    z_proj_a, aud_f, margin=self.distill_margin_cosine,
                    nonneg=self.adaptive_distill_balance,
                )
                audio_distill_total = self.lambda_distill_audio_t_axis * d_a_cos
                logs['distill_audio_t_axis_raw'] = d_a_cos.detach()
            logs['distill_audio_total'] = audio_distill_total.detach()

        if video_distill_total is None and audio_distill_total is None:
            return None, logs

        _vd = video_distill_total if video_distill_total is not None else torch.tensor(0.0, device=self.device)
        _ad = audio_distill_total if audio_distill_total is not None else torch.tensor(0.0, device=self.device)
        total = self.lambda_group_distill * (_vd + _ad)
        logs['distill_total'] = (_vd + _ad).detach()
        logs['distill_total_weighted_d_only'] = total.detach()
        return total, logs

    def _train_step_video_disc_only(
        self,
        batch: Dict[str, Any],
        is_accum_boundary: bool,
        accum_steps: int,
        collect_media: bool,
    ):
        """D-only training window for the video discriminator.

        Runs G forward (under ``torch.no_grad()`` unless
        ``distill_every_steps`` is enabled and a distill component is
        active on this step), computes real/fake logits, backward's the
        hinge/vanilla D loss, optionally backward's the distill loss
        (grads accumulate into G params for the *next* G step), then
        steps ``optim_d`` only.
        """
        # IVAlterstep guard: if for some reason an image_batch slipped into
        # this code path (e.g. mis-configured upstream), accept the
        # video_batch directly. This duplicates the logic in
        # _select_modality_batch as a defensive fallback.
        if isinstance(batch, dict) and 'image_batch' in batch and 'video_batch' in batch:
            self._current_modality = 'video'
            batch = batch['video_batch']

        data = batch['data']
        is_image_step = bool(data.get('is_image', False))
        video = data.get('video') if self.needs_video else None
        audio = data.get('audio') if self.needs_audio else None
        audio_lengths = data.get('audio_lengths') if self.needs_audio else None

        losses: Dict[str, Any] = {}

        if video is None or is_image_step:
            # Video missing or image-only batch on this D-only step — can't
            # drive a video-disc update (3D PatchGAN can't accept T=1).
            # Emit skip signal so the outer loop doesn't count progress.
            losses['video_disc_skipped'] = 1.0
            return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}, None, None

        video = video.to(self.device)
        if audio is not None:
            audio = audio.to(self.device)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(self.device)

        # ---- Decide whether this D-only step also runs distill ----
        want_distill = bool(self.distill_every_steps and self.use_semantic_distill)
        semantic_feats = None
        distill_target_shapes = None
        if want_distill:
            _cur_step = self.train_state.step
            video_distill_active = self.needs_video and _cur_step >= self.video_distill_start_step
            audio_distill_active = self.needs_audio and _cur_step >= self.audio_distill_start_step
            distill_video = video if video_distill_active else None
            distill_audio = audio if audio_distill_active else None
            any_active = (distill_video is not None) or (distill_audio is not None)

            if any_active and self.semantic_encoder is not None:
                with torch.no_grad():
                    semantic_feats = self.semantic_encoder.extract_from_tensors(
                        video=distill_video, audio=distill_audio,
                        video_fps=self.distill_data_fps,
                        audio_sample_rate=self.audio_sample_rate,
                    )
            elif any_active and self.distill_prefetcher is not None:
                semantic_feats = self.distill_prefetcher.get_features(self.device)
            elif any_active and self.semantic_client is not None:
                file_paths = data.get('file_paths', [])
                if file_paths:
                    semantic_feats = self.semantic_client.extract(
                        file_paths=file_paths,
                        target_fps=self.distill_encoder_fps,
                        resolution=self.distill_encoder_resolution,
                        audio_sample_rate=self.audio_sample_rate,
                        device=self.device,
                    )

            if semantic_feats is not None:
                if not video_distill_active:
                    semantic_feats.pop("image_feat", None)
                    semantic_feats.pop("video_feat", None)
                if not audio_distill_active:
                    semantic_feats.pop("audio_feat", None)
                distill_target_shapes = {}
                img_f = semantic_feats.get("image_feat")
                if img_f is not None:
                    distill_target_shapes["image"] = tuple(img_f.shape[1:])
                vf = semantic_feats.get("video_feat")
                if vf is not None:
                    distill_target_shapes["video"] = tuple(vf.shape[1:])
                af = semantic_feats.get("audio_feat")
                if af is not None:
                    distill_target_shapes["audio"] = tuple(af.shape[1:])
                if not distill_target_shapes:
                    distill_target_shapes = None
                    semantic_feats = None

        run_with_grad = want_distill and semantic_feats is not None

        # ---- Model forward ----
        # Skip audio decoder (not needed for D) and the LLM head (no captions
        # passed). We still need the video recon for the D input.
        forward_audio = audio if (run_with_grad and audio is not None
                                  and semantic_feats.get('audio_feat') is not None) else None
        forward_audio_lengths = audio_lengths if forward_audio is not None else None

        if run_with_grad:
            outputs = self.model(
                video, forward_audio,
                audio_lengths=forward_audio_lengths,
                captions=None,
                video_descriptions=None,
                audio_descriptions=None,
                skip_video_decoder=False,
                skip_audio_decoder=True,
                distill_target_shapes=distill_target_shapes,
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    video, None,
                    audio_lengths=None,
                    captions=None,
                    video_descriptions=None,
                    audio_descriptions=None,
                    skip_video_decoder=False,
                    skip_audio_decoder=True,
                    distill_target_shapes=None,
                )

        video_out = outputs.get('video', {}) if outputs else {}
        v_recon = video_out.get('recon')
        if v_recon is None:
            logging.warning(
                f"[video-disc] step {self.train_state.step}: video recon missing, skipping D update"
            )
            losses['video_disc_skipped'] = 1.0
            return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}, None, None

        # Latent stats (parity with G window)
        latent = video_out.get('latent')
        if latent is not None:
            losses['video_latent_mean'] = latent.float().mean().detach()
            losses['video_latent_std'] = latent.float().std(unbiased=False).detach()

        # ---- D forward + loss ----
        # NOTE: We concatenate real and fake along the batch axis and run a
        # single forward. Running two separate forwards would cause a version-
        # mismatch error when the discriminator has BatchNorm with
        # ``track_running_stats=True``: the second forward would bump
        # ``running_mean`` / ``running_var`` in-place while the first forward's
        # autograd graph still holds a saved reference at the previous version.
        v_disc = self.discriminators['video']
        with self._video_disc_autocast():
            _real_in = video.to(dtype=self.disc_dtype)
            _fake_in = v_recon.detach().to(dtype=self.disc_dtype)
            bs_real = _real_in.shape[0]
            _concat_in = torch.cat([_real_in, _fake_in], dim=0)
            _concat_logits = v_disc(_concat_in)
            logits_real = _concat_logits[:bs_real]
            logits_fake = _concat_logits[bs_real:]
            d_loss_fn = video_hinge_d_loss if self.video_disc_loss_type == 'hinge' else video_vanilla_d_loss
            d_loss = d_loss_fn(logits_real, logits_fake)

        losses['video_d_loss'] = d_loss.detach()
        losses['video_logits_real_mean'] = logits_real.detach().float().mean()
        losses['video_logits_fake_mean'] = logits_fake.detach().float().mean()

        # ---- Lazy-D gate: skip D backward+step when D is already "winning" ----
        # Rationale: when G is pretrained and D is fresh, D often overshoots
        # and d_loss collapses toward 0, producing a runaway adv signal that
        # damages reconstruction. Stopping D updates below a threshold keeps D
        # from getting arbitrarily sharp. We all-reduce the decision so every
        # DDP rank takes the same branch (otherwise ring all-reduce hangs).
        lazy_skip_d = False
        if self.video_disc_lazy_threshold > 0:
            _d_val = d_loss.detach().float()
            if self.is_distributed:
                _d_val = _d_val.clone()
                torch.distributed.all_reduce(_d_val, op=torch.distributed.ReduceOp.AVG)
            lazy_skip_d = _d_val.item() < self.video_disc_lazy_threshold
        losses['video_d_lazy_skipped'] = 1.0 if lazy_skip_d else 0.0

        # ---- Optional distill loss (→ grad accumulates into G params) ----
        distill_total_for_backward: Optional[torch.Tensor] = None
        if run_with_grad:
            distill_total_for_backward, distill_logs = self._compute_d_only_distill(
                outputs, semantic_feats,
            )
            for k, v in distill_logs.items():
                losses[k] = v

        # ---- Backward: D loss (skipped when lazy-gated) ----
        # `v_recon.detach()` guarantees no G grad flows from the D loss graph,
        # so we can unconditionally no_sync the model during D's backward.
        if not lazy_skip_d:
            scaled_d = d_loss / accum_steps if accum_steps > 1 else d_loss
            with ExitStack() as _stack:
                if self.is_distributed:
                    _stack.enter_context(self.model.no_sync())
                    if not is_accum_boundary:
                        for _d in self.discriminators.values():
                            if isinstance(_d, DDP):
                                _stack.enter_context(_d.no_sync())
                self.scaler_d.scale(scaled_d).backward()

        # ---- Backward: distill loss (G grad accumulates; no D grad from this) ----
        # Always no_sync G here so the all-reduce is deferred to the next G
        # boundary; this avoids a double-sync when the next G-step backward
        # adds new grads on top of the already-synced distill grads.
        if distill_total_for_backward is not None:
            scaled_distill = (
                distill_total_for_backward / accum_steps
                if accum_steps > 1 else distill_total_for_backward
            )
            with ExitStack() as _stack:
                if self.is_distributed:
                    _stack.enter_context(self.model.no_sync())
                    for _d in self.discriminators.values():
                        if isinstance(_d, DDP):
                            _stack.enter_context(_d.no_sync())
                self.scaler.scale(scaled_distill).backward()

        # ---- Media collection (only on boundary; cheap since recon is live) ----
        media_data = None
        if collect_media and is_accum_boundary:
            media_data = {
                'video_gt': video.detach().float().cpu(),
                'video_recon': v_recon.detach().float().cpu(),
            }

        if not is_accum_boundary:
            losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            return losses_scalar, None, media_data

        # ---- Boundary: clip + step D only (G params grads preserved for next G) ----
        if distill_total_for_backward is not None:
            # Guard against NaN/Inf in the accumulated distill grad on G params.
            # If corrupted, zero them so the next G step doesn't get poisoned.
            if not self._check_generator_grad_finite():
                logging.warning(
                    f"Step {self.train_state.step}: non-finite distill grad on G params "
                    f"during D-only window, zeroing G grads"
                )
                self._zero_generator_grads()
                losses['distill_skipped_nan_grad'] = 1.0

        _disc_params = list(itertools.chain(
            *[d.parameters() for d in self.discriminators.values()]
        ))
        _dmax = self.disc_max_grad_norm if self.disc_max_grad_norm is not None else self.max_grad_norm

        # Lazy-D: skip the whole D optimizer update path on this boundary.
        # We intentionally do NOT call scaler_d.update(); the scale factor
        # stays constant for this iteration, which is safe (next iteration's
        # scale() / step() / update() cycle proceeds normally).
        if lazy_skip_d:
            self.optim_d.zero_grad(set_to_none=True)
            losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            return losses_scalar, None, media_data

        self.scaler_d.unscale_(self.optim_d)
        if _dmax is not None and _dmax > 0:
            d_grad_norm = torch.nn.utils.clip_grad_norm_(_disc_params, _dmax)
            losses['grad_norm/video_disc_before_clip'] = d_grad_norm
            if torch.isnan(d_grad_norm) or torch.isinf(d_grad_norm):
                logging.warning(
                    f"Step {self.train_state.step}: NaN/Inf D grad norm, skipping D update"
                )
                self.optim_d.zero_grad(set_to_none=True)
                losses['video_disc_skipped_nan_grad'] = 1.0
                self.scaler_d.update()
                losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
                return losses_scalar, None, media_data

        self.scaler_d.step(self.optim_d)
        self.scaler_d.update()
        if self.scheduler_d is not None:
            self.scheduler_d.step()
        self.optim_d.zero_grad(set_to_none=True)

        # NOTE: deliberately no `optim_g.step` / `optim_g.zero_grad` / EMA
        # update here. Accumulated distill grads on G params are preserved
        # for the next G-window step.

        losses_scalar = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        return losses_scalar, None, media_data

    # ------------------------------------------------------------------
    # evaluate_video
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_video(self, step: Optional[int] = None) -> Dict[str, float]:
        if not self.val_video_dataloaders:
            return {}

        if self.ema is not None:
            self.ema.apply_shadow()
        self.unwrapped_model.eval()

        local_max_samples = (
            math.ceil(self.val_video_max_samples / self.world_size)
            if self.val_video_max_samples is not None else None
        )

        all_metrics = {}

        for ds_name, dataloader in self.val_video_dataloaders.items():
            metrics = {'video_recon_loss': 0.0, 'video_psnr': 0.0, 'video_lpips': 0.0}
            if self.eval_ssim:
                metrics['video_ssim'] = 0.0
            if self.eval_fvd:
                metrics['video_fvd'] = 0.0

            count = 0
            tb_video_samples: List[tuple] = []
            step_id = step if step is not None else self.train_state.step
            if self.is_main:
                step_dir = self.video_eval_dir / f"step_{step_id:08d}" / ds_name
                if step_dir.exists():
                    shutil.rmtree(step_dir)

            pbar = tqdm(dataloader, desc=f"Eval Video [{ds_name}]", disable=not self.is_main)
            for batch in pbar:
                data = batch['data']
                video = data['video'].to(self.device)
                file_names = data.get('file_names', [])

                outputs = self.unwrapped_model(video, None)
                recon = outputs['video']['recon']

                if self.is_main:
                    self._save_video_eval_batch(
                        ds_name=ds_name, step=step_id,
                        video=video.detach().cpu(), recon=recon.detach().cpu(),
                        file_names=file_names, sample_offset=count,
                    )
                    for b in range(video.shape[0]):
                        fname = file_names[b] if b < len(file_names) else ""
                        tb_video_samples.append((video[b].detach().cpu(), recon[b].detach().cpu(), fname))

                recon_loss = nn.functional.l1_loss(recon, video)
                metrics['video_recon_loss'] += recon_loss.item()

                video_01 = (video.clamp(-1.0, 1.0) + 1.0) * 0.5
                recon_01 = (recon.clamp(-1.0, 1.0) + 1.0) * 0.5

                B, C, T, H, W = video.shape
                video_frames = rearrange(video_01, "b c t h w -> (b t) c h w")
                recon_frames = rearrange(recon_01, "b c t h w -> (b t) c h w")
                mse = torch.mean(torch.square(video_frames - recon_frames), dim=(1, 2, 3))
                psnr = 20 * torch.log10(1 / torch.sqrt(mse + 1e-8))
                metrics['video_psnr'] += psnr.mean().item()

                if self.lpips_model is not None:
                    vid_2d = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                    rec_2d = recon.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                    lpips_val = self.lpips_model(vid_2d.clamp(-1, 1), rec_2d.clamp(-1, 1)).mean()
                    metrics['video_lpips'] += lpips_val.item()

                if self.eval_ssim:
                    try:
                        video_btchw = rearrange(video_01, "b c t h w -> b t c h w")
                        recon_btchw = rearrange(recon_01, "b c t h w -> b t c h w")
                        ssim_res = calculate_ssim(
                            video_btchw.detach().float().cpu(),
                            recon_btchw.detach().float().cpu(),
                        )
                        metrics['video_ssim'] += float(np.mean(list(ssim_res.get("value", {}).values())))
                    except Exception as e:
                        logging.warning(f"SSIM failed: {e}")

                if self.eval_fvd:
                    try:
                        video_btchw = rearrange(video_01, "b c t h w -> b t c h w")
                        recon_btchw = rearrange(recon_01, "b c t h w -> b t c h w")
                        fvd_device = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
                        fvd_res = calculate_fvd(
                            video_btchw.detach().to(device=fvd_device, dtype=torch.float32),
                            recon_btchw.detach().to(device=fvd_device, dtype=torch.float32),
                            fvd_device, method=self.eval_fvd_method,
                        )
                        if isinstance(fvd_res, dict) and "value" in fvd_res:
                            metrics['video_fvd'] += float(np.mean(list(fvd_res["value"].values())))
                        elif isinstance(fvd_res, (float, int)):
                            metrics['video_fvd'] += float(fvd_res)
                    except Exception as e:
                        logging.warning(f"FVD failed: {e}")

                count += 1
                if self.is_main:
                    pbar.set_postfix({k: f"{v / count:.4f}" for k, v in metrics.items()})
                if local_max_samples is not None and count >= local_max_samples:
                    break
                torch.cuda.empty_cache()

            metrics, count = self._all_reduce_eval_metrics(metrics, count)

            if self.is_main:
                for k, v in metrics.items():
                    all_metrics[f"{ds_name}/{k}"] = v
                if hasattr(self, 'writer') and tb_video_samples:
                    self._write_video_tb_samples(tb_video_samples, ds_name, step_id)

        if self.ema is not None:
            self.ema.restore()
        self.unwrapped_model.train()

        if self.is_main and all_metrics:
            metric_keys = ['video_recon_loss', 'video_psnr', 'video_lpips']
            if self.eval_ssim:
                metric_keys.append('video_ssim')
            if self.eval_fvd:
                metric_keys.append('video_fvd')
            for key in metric_keys:
                values = [v for k, v in all_metrics.items() if k.endswith(f'/{key}')]
                if values:
                    all_metrics[f'avg/{key}'] = sum(values) / len(values)

        if self.is_distributed:
            dist.barrier()

        return all_metrics

    # ------------------------------------------------------------------
    # evaluate_audio
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_audio(self) -> Dict[str, float]:
        if not self.val_audio_dataloaders:
            return {}

        if self.ema is not None:
            self.ema.apply_shadow()
        self.unwrapped_model.eval()

        local_max_samples = (
            math.ceil(self.val_audio_max_samples / self.world_size)
            if self.val_audio_max_samples is not None else None
        )

        all_metrics = {}

        for ds_name, dataloader in self.val_audio_dataloaders.items():
            ds_gt_dir = self.audio_gt_dir / ds_name
            ds_syn_dir = self.audio_syn_dir / ds_name

            if self.is_main:
                ds_gt_dir.mkdir(parents=True, exist_ok=True)
                ds_syn_dir.mkdir(parents=True, exist_ok=True)
                for f in ds_gt_dir.glob("*"):
                    f.unlink()
                for f in ds_syn_dir.glob("*"):
                    f.unlink()

            if self.is_distributed:
                dist.barrier()

            if not self.is_main:
                ds_gt_dir.mkdir(parents=True, exist_ok=True)
                ds_syn_dir.mkdir(parents=True, exist_ok=True)

            metrics = {'audio_mel_loss': 0.0, 'audio_recon_loss': 0.0}
            count = 0
            tb_audio_samples: List[tuple] = []

            pbar = tqdm(dataloader, desc=f"Eval Audio [{ds_name}]", disable=not self.is_main)
            for idx, batch in enumerate(pbar):
                data = batch['data']
                audio = data['audio'].to(self.device)
                file_names = data.get('file_names', [])

                outputs = self.unwrapped_model(None, audio)
                recon = outputs['audio']['recon']

                min_len = min(audio.shape[-1], recon.shape[-1])
                audio_trimmed = audio[..., :min_len]
                recon_trimmed = recon[..., :min_len]

                if self.mel_loss is not None:
                    metrics['audio_mel_loss'] += self.mel_loss(recon_trimmed, audio_trimmed).item()
                if self.waveform_loss is not None:
                    metrics['audio_recon_loss'] += self.waveform_loss(recon_trimmed, audio_trimmed).item()

                for b in range(audio.shape[0]):
                    sample_index = idx * audio.shape[0] + b
                    source_identifier = file_names[b] if b < len(file_names) else ""
                    file_name = f"r{self.rank}_{self._build_eval_audio_name(source_identifier, sample_index)}"
                    gt_wav = audio_trimmed[b, 0].cpu().numpy()
                    syn_wav = recon_trimmed[b, 0].float().cpu().numpy()
                    sf.write(str(ds_gt_dir / file_name), gt_wav, self.audio_sample_rate, format='WAV')
                    sf.write(str(ds_syn_dir / file_name), syn_wav, self.audio_sample_rate, format='WAV')

                    if self.is_main:
                        tb_audio_samples.append((
                            audio_trimmed[b].detach().float().cpu(),
                            recon_trimmed[b].detach().float().cpu(),
                            source_identifier,
                        ))

                count += 1
                if self.is_main:
                    pbar.set_postfix({k: f"{v / count:.4f}" for k, v in metrics.items()})
                if local_max_samples is not None and count >= local_max_samples:
                    break

            metrics, total_count = self._all_reduce_eval_metrics(metrics, count)

            if self.is_distributed:
                dist.barrier()

            if self.is_main:
                try:
                    gt_dir = str(ds_gt_dir)
                    syn_dir = str(ds_syn_dir)
                    _, mean_stoi = evaluate_stoi(gt_dir, syn_dir)
                    metrics['audio_stoi'] = mean_stoi
                    _, mean_pesq_nb = evaluate_pesq(gt_dir, syn_dir, mode='nb')
                    metrics['audio_pesq_nb'] = mean_pesq_nb
                    _, mean_pesq_wb = evaluate_pesq(gt_dir, syn_dir, mode='wb')
                    metrics['audio_pesq_wb'] = mean_pesq_wb
                    _, mean_sim = evaluate_sim(gt_dir, syn_dir)
                    metrics['audio_sim'] = mean_sim
                except Exception as e:
                    logging.warning(f"Audio evaluation failed for {ds_name}: {e}")

                for k, v in metrics.items():
                    all_metrics[f"{ds_name}/{k}"] = v
                if hasattr(self, 'writer') and tb_audio_samples:
                    step_id = self.train_state.step
                    self._write_audio_tb_samples(tb_audio_samples, ds_name, step_id)

        if self.ema is not None:
            self.ema.restore()
        self.unwrapped_model.train()

        if self.is_main and all_metrics:
            for key in ['audio_mel_loss', 'audio_recon_loss', 'audio_stoi',
                        'audio_pesq_nb', 'audio_pesq_wb', 'audio_sim']:
                values = [v for k, v in all_metrics.items() if k.endswith(f'/{key}')]
                if values:
                    all_metrics[f'avg/{key}'] = sum(values) / len(values)

        if self.is_distributed:
            dist.barrier()

        return all_metrics

    # ------------------------------------------------------------------
    # evaluate_contrastive
    # ------------------------------------------------------------------

    def _gather_contrastive_feats(
        self,
        local_feats: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Gather variable-length feature tensors from all ranks onto rank 0."""
        if not local_feats:
            local_cat = torch.empty(0)
        else:
            local_cat = torch.cat(local_feats, dim=0).contiguous()

        local_size = local_cat.shape[0]
        feat_dim = local_cat.shape[1] if local_cat.ndim >= 2 else 0

        sizes_list: List[Optional[int]] = [None] * self.world_size
        dist.all_gather_object(sizes_list, local_size)

        max_size = max(sizes_list)
        if max_size == 0:
            return []

        if local_size < max_size:
            pad_shape = list(local_cat.shape)
            pad_shape[0] = max_size - local_size
            if feat_dim == 0:
                pad_shape = [max_size - local_size]
            local_cat = torch.cat([local_cat, torch.zeros(pad_shape, dtype=local_cat.dtype)], dim=0)

        local_cat = local_cat.to(self.device)
        gathered = [torch.zeros_like(local_cat) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_cat)

        if self.is_main:
            trimmed = []
            for r in range(self.world_size):
                s = sizes_list[r]
                if s > 0:
                    trimmed.append(gathered[r][:s].cpu())
            return [torch.cat(trimmed, dim=0)] if trimmed else []
        return []

    @torch.no_grad()
    def evaluate_contrastive(self, step: Optional[int] = None) -> Dict[str, float]:
        """Evaluate contrastive learning quality on dedicated validation sets."""
        if not self.val_contrastive_dataloaders:
            return {}
        if not self.unwrapped_model.use_contrastive or self.unwrapped_model.contrastive_head is None:
            return {}

        if self.eval_contrastive_in_all:
            return self._evaluate_contrastive_merged(step=step)

        if self.ema is not None:
            self.ema.apply_shadow()
        self.unwrapped_model.eval()
        contrastive_head = self.unwrapped_model.contrastive_head
        model = self.unwrapped_model
        n_gran = contrastive_head.n_granularities

        local_max_samples = (
            math.ceil(self.val_contrastive_max_samples / self.world_size)
            if self.val_contrastive_max_samples is not None else None
        )

        all_metrics = {}

        for ds_name, dataloader in self.val_contrastive_dataloaders.items():
            per_gran_seg_v: List[List[torch.Tensor]] = [[] for _ in range(n_gran)]
            per_gran_seg_a: List[List[torch.Tensor]] = [[] for _ in range(n_gran)]
            per_gran_S: List[Optional[int]] = [None] * n_gran
            glob_vfeats: List[torch.Tensor] = []
            glob_afeats: List[torch.Tensor] = []
            count = 0
            _rt_interval = 50
            _eval_max_b = 500
            _last_rt: Dict[str, float] = {}

            pbar = tqdm(dataloader, desc=f"Eval Contrastive [{ds_name}]", disable=not self.is_main)
            for batch in pbar:
                data = batch['data']
                video = data.get('video')
                audio = data.get('audio')
                if video is None or audio is None:
                    continue
                video = video.to(self.device)
                audio = audio.to(self.device)
                audio_lengths = data.get('audio_lengths')
                if audio_lengths is not None:
                    audio_lengths = audio_lengths.to(self.device)

                with model._module_autocast('video_vae'):
                    video_posterior = model.video_vae.encode(video, streaming_inference=True)
                    video_latent = video_posterior.mode()
                with model._module_autocast('audio_vae'):
                    audio_padded = model.audio_vae.preprocess(audio)
                    audio_posterior, _, _, _, _ = model.audio_vae.encode(audio_padded)
                    audio_latent = audio_posterior.mode()

                audio_latent_lengths = model._compute_audio_latent_lengths(
                    audio_lengths=audio_lengths, max_latent_length=audio_latent.shape[-1],
                )

                with model._module_autocast('contrastive'):
                    c_out = contrastive_head(
                        video_latent=video_latent, audio_latent=audio_latent,
                        audio_latent_lengths=audio_latent_lengths, world_size=1,
                    )

                for gi, g in enumerate(c_out.get("granularities", [])):
                    if g.get("segment_vfeat") is not None:
                        per_gran_seg_v[gi].append(g["segment_vfeat"].float().cpu())
                        per_gran_seg_a[gi].append(g["segment_afeat"].float().cpu())
                        if per_gran_S[gi] is None:
                            per_gran_S[gi] = g["S"]

                if c_out.get('global_vfeat') is not None:
                    glob_vfeats.append(c_out['global_vfeat'].float().cpu())
                    glob_afeats.append(c_out['global_afeat'].float().cpu())

                count += 1
                if self.is_main:
                    postfix: Dict[str, Any] = {'n': count}
                    if count > 0 and count % _rt_interval == 0:
                      try:
                        torch.cuda.empty_cache()
                        if glob_vfeats:
                            _gv = torch.cat(glob_vfeats, dim=0)
                            if _gv.shape[0] > _eval_max_b:
                                _gv = _gv[:_eval_max_b]
                                _ga_cat = torch.cat(glob_afeats, dim=0)[:_eval_max_b]
                            else:
                                _ga_cat = torch.cat(glob_afeats, dim=0)
                            _gv = _gv.to(self.device)
                            _ga = _ga_cat.to(self.device)
                            if _gv.shape[0] > 1:
                                for _k_global in self.val_global_num_negatives_list:
                                    _k_used = min(_k_global, _gv.shape[0] - 1)
                                    if _k_used <= 0:
                                        continue
                                    _gm = compute_global_sampled_precision(_gv, _ga, num_negatives=_k_used)
                                    _last_rt[f'g_avg@{_k_global}'] = _gm['global_precision_avg']
                            del _gv, _ga
                        for gi in range(n_gran):
                            if not per_gran_seg_v[gi]:
                                continue
                            S_cur = per_gran_S[gi] or 1
                            sc_label = contrastive_head.segment_count_list[gi]
                            sc_tag = str(sc_label) if sc_label is not None else "null"
                            _sv_all = torch.cat(per_gran_seg_v[gi], dim=0)
                            _sa_all = torch.cat(per_gran_seg_a[gi], dim=0)
                            B_cur = _sv_all.shape[0] // max(S_cur, 1)
                            if B_cur > _eval_max_b:
                                _n_cap = _eval_max_b * S_cur
                                _sv_all = _sv_all[:_n_cap]
                                _sa_all = _sa_all[:_n_cap]
                                B_cur = _eval_max_b
                            _sv = _sv_all.to(self.device)
                            _sa = _sa_all.to(self.device)
                            if B_cur > 1:
                                _scale, _ = contrastive_head.clamp_logit_scales()
                                _scale = _scale if _scale is not None else torch.tensor(1.0, device=self.device)
                                for _k_seg in self.val_segment_num_negatives_list:
                                    _sv2a, _sa2v, _, _ni = contrastive_head.sample_negatives_for_loss(
                                        _sv, _sa, _sv, _sa,
                                        B=B_cur, B_eff=B_cur, S=S_cur,
                                        scale=_scale, rank_offset=0,
                                        num_negatives=_k_seg,
                                        num_negative_videos=self.val_segment_num_negative_videos,
                                    )
                                    _sp = compute_segment_sampled_precision(_sv2a, _sa2v, _ni)
                                    _last_rt[f's{sc_tag}_ovrl@{_k_seg}'] = _sp['seg_overall_prec_avg']
                            del _sv, _sa
                        torch.cuda.empty_cache()
                      except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        logging.warning(f"OOM in real-time contrastive monitoring (count={count}), skipping")
                    postfix.update({k: f'{v:.4f}' for k, v in _last_rt.items()})
                    pbar.set_postfix(postfix)
                if local_max_samples is not None and count >= local_max_samples:
                    break
                torch.cuda.empty_cache()

            # Gather features from all ranks
            if self.is_distributed:
                for gi in range(n_gran):
                    s_vals_gi: List[Optional[int]] = [None] * self.world_size
                    dist.all_gather_object(s_vals_gi, per_gran_S[gi])
                    if per_gran_S[gi] is None:
                        per_gran_S[gi] = next((s for s in s_vals_gi if s is not None), None)
                    per_gran_seg_v[gi] = self._gather_contrastive_feats(per_gran_seg_v[gi])
                    per_gran_seg_a[gi] = self._gather_contrastive_feats(per_gran_seg_a[gi])
                glob_vfeats = self._gather_contrastive_feats(glob_vfeats)
                glob_afeats = self._gather_contrastive_feats(glob_afeats)

            has_any = any(per_gran_seg_v[gi] for gi in range(n_gran)) or glob_vfeats
            if not has_any:
                continue

            if self.is_main:
                metrics: Dict[str, float] = {}

                seg_logit_scale, global_logit_scale_val = contrastive_head.clamp_logit_scales()
                if seg_logit_scale is not None:
                    metrics['segment_logit_scale'] = seg_logit_scale.item()
                if global_logit_scale_val is not None:
                    metrics['global_logit_scale'] = global_logit_scale_val.item()

                for gi in range(n_gran):
                    if not per_gran_seg_v[gi]:
                        continue
                    S = per_gran_S[gi] if per_gran_S[gi] is not None else 1
                    sc_label = contrastive_head.segment_count_list[gi]
                    sc_tag = str(sc_label) if sc_label is not None else "null"
                    gran_suffix = f"_sc{sc_tag}" if n_gran > 1 else ""

                    pool_v_cpu = torch.cat(per_gran_seg_v[gi], dim=0)
                    pool_a_cpu = torch.cat(per_gran_seg_a[gi], dim=0)
                    B_total = pool_v_cpu.shape[0] // S

                    B_use = min(B_total, _eval_max_b)
                    if B_use < B_total:
                        logging.info(f"  [{ds_name}] gi={gi}: subsampling {B_total} -> {B_use} videos for segment metrics")
                        n_cap = B_use * S
                        pool_v_cpu = pool_v_cpu[:n_cap]
                        pool_a_cpu = pool_a_cpu[:n_cap]
                        B_total = B_use

                    pool_v = pool_v_cpu.to(self.device)
                    pool_a = pool_a_cpu.to(self.device)
                    del pool_v_cpu, pool_a_cpu

                    if B_total > 0 and S > 0:
                        scale = seg_logit_scale if seg_logit_scale is not None else torch.tensor(1.0, device=self.device)
                        for k_seg in self.val_segment_num_negatives_list:
                            sim_v2a, sim_a2v, targets, num_intra = contrastive_head.sample_negatives_for_loss(
                                vfeat_local=pool_v, afeat_local=pool_a,
                                vfeat_pool=pool_v, afeat_pool=pool_a,
                                B=B_total, B_eff=B_total, S=S,
                                scale=scale, rank_offset=0,
                                num_negatives=k_seg,
                                num_negative_videos=self.val_segment_num_negative_videos,
                            )
                            seg_loss = (
                                nn.functional.cross_entropy(sim_v2a, targets) +
                                nn.functional.cross_entropy(sim_a2v, targets)
                            ) / 2
                            seg_suffix = f"_neg{k_seg}{gran_suffix}"
                            metrics[f"segment_contrastive_loss{seg_suffix}"] = seg_loss.item()
                            seg_prec = compute_segment_sampled_precision(sim_v2a, sim_a2v, num_intra)
                            for mk, mv in seg_prec.items():
                                metrics[f"{mk}{seg_suffix}"] = mv
                            if len(self.val_segment_num_negatives_list) == 1 and n_gran == 1:
                                metrics["segment_contrastive_loss"] = seg_loss.item()
                                metrics.update(seg_prec)
                        if S > 1:
                            intra = compute_segment_intra_precision(pool_v, pool_a, B_total, S)
                            for mk, mv in intra.items():
                                metrics[f"{mk}{gran_suffix}"] = mv

                    del pool_v, pool_a

                if glob_vfeats:
                    gpool_v_cpu = torch.cat(glob_vfeats, dim=0)
                    gpool_a_cpu = torch.cat(glob_afeats, dim=0)
                    if gpool_v_cpu.shape[0] > _eval_max_b:
                        gpool_v_cpu = gpool_v_cpu[:_eval_max_b]
                        gpool_a_cpu = gpool_a_cpu[:_eval_max_b]
                    gpool_v = gpool_v_cpu.to(self.device)
                    gpool_a = gpool_a_cpu.to(self.device)
                    del gpool_v_cpu, gpool_a_cpu

                    if gpool_v.shape[0] > 1:
                        g_scale = global_logit_scale_val if global_logit_scale_val is not None else torch.tensor(1.0, device=self.device)
                        B_g = gpool_v.shape[0]
                        for k_global in self.val_global_num_negatives_list:
                            K_g = min(k_global, B_g - 1)
                            if K_g <= 0:
                                continue
                            all_idx_g = torch.arange(B_g, device=self.device)
                            mask_g = all_idx_g.unsqueeze(0) != all_idx_g.unsqueeze(1)
                            neg_pool_g = all_idx_g.unsqueeze(0).expand(B_g, -1)[mask_g].view(B_g, B_g - 1)
                            perm_g = torch.rand(B_g, B_g - 1, device=self.device).argsort(dim=1)[:, :K_g]
                            neg_idx_g = neg_pool_g.gather(1, perm_g)
                            sample_idx_g = torch.cat([all_idx_g.unsqueeze(1), neg_idx_g], dim=1)

                            afeat_g_sampled = gpool_a[sample_idx_g]
                            vfeat_g_sampled = gpool_v[sample_idx_g]
                            sim_g_v2a = torch.einsum("bd,bkd->bk", gpool_v, afeat_g_sampled) / g_scale
                            sim_g_a2v = torch.einsum("bd,bkd->bk", gpool_a, vfeat_g_sampled) / g_scale
                            gt_sampled = torch.zeros(B_g, dtype=torch.long, device=self.device)
                            global_loss = (
                                nn.functional.cross_entropy(sim_g_v2a, gt_sampled) +
                                nn.functional.cross_entropy(sim_g_a2v, gt_sampled)
                            ) / 2
                            global_suffix = f"_gneg{k_global}"
                            metrics[f"global_contrastive_loss{global_suffix}"] = global_loss.item()
                            global_prec = compute_global_sampled_precision(
                                gpool_v, gpool_a, num_negatives=k_global
                            )
                            for mk, mv in global_prec.items():
                                metrics[f"{mk}{global_suffix}"] = mv
                            if len(self.val_global_num_negatives_list) == 1:
                                metrics["global_contrastive_loss"] = global_loss.item()
                                metrics.update(global_prec)

                    del gpool_v, gpool_a

                if self.is_main:
                    logging.info(f"  [{ds_name}] Contrastive eval metrics:")
                    for k, v in sorted(metrics.items()):
                        logging.info(f"    {k}: {v:.4f}")

                for k, v in metrics.items():
                    all_metrics[f"{ds_name}/{k}"] = v

            torch.cuda.empty_cache()

        if self.ema is not None:
            self.ema.restore()
        self.unwrapped_model.train()

        if all_metrics:
            metric_suffixes = set()
            for k in all_metrics:
                parts = k.split('/', 1)
                if len(parts) == 2:
                    metric_suffixes.add(parts[1])
            for suffix in metric_suffixes:
                values = [v for k, v in all_metrics.items() if k.endswith(f'/{suffix}')]
                if values:
                    all_metrics[f'avg/{suffix}'] = sum(values) / len(values)

        if self.is_distributed:
            dist.barrier()

        return all_metrics

    @torch.no_grad()
    def _evaluate_contrastive_merged(self, step: Optional[int] = None) -> Dict[str, float]:
        """Merge all contrastive val datasets, shuffle, then evaluate."""
        if self.ema is not None:
            self.ema.apply_shadow()
        self.unwrapped_model.eval()
        contrastive_head = self.unwrapped_model.contrastive_head
        model = self.unwrapped_model
        n_gran = contrastive_head.n_granularities

        local_max_samples = (
            math.ceil(self.val_contrastive_max_samples / self.world_size)
            if self.val_contrastive_max_samples is not None else None
        )

        _eval_max_b = 500
        per_gran_seg_v: List[List[torch.Tensor]] = [[] for _ in range(n_gran)]
        per_gran_seg_a: List[List[torch.Tensor]] = [[] for _ in range(n_gran)]
        per_gran_S: List[Optional[int]] = [None] * n_gran
        all_glob_vfeats: List[torch.Tensor] = []
        all_glob_afeats: List[torch.Tensor] = []
        total_count = 0

        for ds_name, dataloader in self.val_contrastive_dataloaders.items():
            pbar = tqdm(dataloader, desc=f"Collect Contrastive [{ds_name}]", disable=not self.is_main)
            for batch in pbar:
                data = batch['data']
                video = data.get('video')
                audio = data.get('audio')
                if video is None or audio is None:
                    continue
                video = video.to(self.device)
                audio = audio.to(self.device)
                audio_lengths = data.get('audio_lengths')
                if audio_lengths is not None:
                    audio_lengths = audio_lengths.to(self.device)

                with model._module_autocast('video_vae'):
                    video_posterior = model.video_vae.encode(video, streaming_inference=True)
                    video_latent = video_posterior.mode()
                with model._module_autocast('audio_vae'):
                    audio_padded = model.audio_vae.preprocess(audio)
                    audio_posterior, _, _, _, _ = model.audio_vae.encode(audio_padded)
                    audio_latent = audio_posterior.mode()
                audio_latent_lengths = model._compute_audio_latent_lengths(
                    audio_lengths=audio_lengths, max_latent_length=audio_latent.shape[-1],
                )
                with model._module_autocast('contrastive'):
                    c_out = contrastive_head(
                        video_latent=video_latent, audio_latent=audio_latent,
                        audio_latent_lengths=audio_latent_lengths, world_size=1,
                    )

                for gi, g in enumerate(c_out.get("granularities", [])):
                    if g.get("segment_vfeat") is not None:
                        per_gran_seg_v[gi].append(g["segment_vfeat"].float().cpu())
                        per_gran_seg_a[gi].append(g["segment_afeat"].float().cpu())
                        if per_gran_S[gi] is None:
                            per_gran_S[gi] = g["S"]

                if c_out.get('global_vfeat') is not None:
                    all_glob_vfeats.append(c_out['global_vfeat'].float().cpu())
                    all_glob_afeats.append(c_out['global_afeat'].float().cpu())

                total_count += 1
                if self.is_main:
                    pbar.set_postfix({'total': total_count})
                if local_max_samples is not None and total_count >= local_max_samples:
                    break
                torch.cuda.empty_cache()

            if local_max_samples is not None and total_count >= local_max_samples:
                break

        if self.is_main:
            logging.info(f"Collected {total_count} batches from "
                         f"{len(self.val_contrastive_dataloaders)} datasets, merging & shuffling...")

        # Shuffle per granularity
        for gi in range(n_gran):
            S_gi = per_gran_S[gi] if per_gran_S[gi] is not None else 1
            if per_gran_seg_v[gi]:
                merged_v = torch.cat(per_gran_seg_v[gi], dim=0)
                merged_a = torch.cat(per_gran_seg_a[gi], dim=0)
                total_B = merged_v.shape[0] // S_gi
                D = merged_v.shape[-1]
                perm = torch.randperm(total_B)
                per_gran_seg_v[gi] = [merged_v.view(total_B, S_gi, D)[perm].reshape(-1, D)]
                per_gran_seg_a[gi] = [merged_a.view(total_B, S_gi, D)[perm].reshape(-1, D)]

        if all_glob_vfeats:
            merged_glob_v = torch.cat(all_glob_vfeats, dim=0)
            merged_glob_a = torch.cat(all_glob_afeats, dim=0)
            perm_g = torch.randperm(merged_glob_v.shape[0])
            all_glob_vfeats = [merged_glob_v[perm_g]]
            all_glob_afeats = [merged_glob_a[perm_g]]

        if self.is_distributed:
            for gi in range(n_gran):
                s_vals_gi: List[Optional[int]] = [None] * self.world_size
                dist.all_gather_object(s_vals_gi, per_gran_S[gi])
                if per_gran_S[gi] is None:
                    per_gran_S[gi] = next((s for s in s_vals_gi if s is not None), None)
                per_gran_seg_v[gi] = self._gather_contrastive_feats(per_gran_seg_v[gi])
                per_gran_seg_a[gi] = self._gather_contrastive_feats(per_gran_seg_a[gi])
            all_glob_vfeats = self._gather_contrastive_feats(all_glob_vfeats)
            all_glob_afeats = self._gather_contrastive_feats(all_glob_afeats)

        all_metrics: Dict[str, float] = {}

        has_any = any(per_gran_seg_v[gi] for gi in range(n_gran)) or all_glob_vfeats
        if not has_any:
            if self.ema is not None:
                self.ema.restore()
            self.unwrapped_model.train()
            if self.is_distributed:
                dist.barrier()
            return all_metrics

        if self.is_main:
            metrics: Dict[str, float] = {}

            seg_logit_scale, global_logit_scale_val = contrastive_head.clamp_logit_scales()
            if seg_logit_scale is not None:
                metrics['segment_logit_scale'] = seg_logit_scale.item()
            if global_logit_scale_val is not None:
                metrics['global_logit_scale'] = global_logit_scale_val.item()

            for gi in range(n_gran):
                if not per_gran_seg_v[gi]:
                    continue
                S = per_gran_S[gi] if per_gran_S[gi] is not None else 1
                sc_label = contrastive_head.segment_count_list[gi]
                sc_tag = str(sc_label) if sc_label is not None else "null"
                gran_suffix = f"_sc{sc_tag}" if n_gran > 1 else ""

                pool_v_cpu = torch.cat(per_gran_seg_v[gi], dim=0)
                pool_a_cpu = torch.cat(per_gran_seg_a[gi], dim=0)
                B_total = pool_v_cpu.shape[0] // S

                B_use = min(B_total, _eval_max_b)
                if B_use < B_total:
                    logging.info(f"  [merged_all] gi={gi}: subsampling {B_total} -> {B_use} videos for segment metrics")
                    n_cap = B_use * S
                    pool_v_cpu = pool_v_cpu[:n_cap]
                    pool_a_cpu = pool_a_cpu[:n_cap]
                    B_total = B_use

                pool_v = pool_v_cpu.to(self.device)
                pool_a = pool_a_cpu.to(self.device)
                del pool_v_cpu, pool_a_cpu

                if B_total > 0 and S > 0:
                    scale = seg_logit_scale if seg_logit_scale is not None else torch.tensor(1.0, device=self.device)
                    for k_seg in self.val_segment_num_negatives_list:
                        sim_v2a, sim_a2v, targets, num_intra = contrastive_head.sample_negatives_for_loss(
                            vfeat_local=pool_v, afeat_local=pool_a,
                            vfeat_pool=pool_v, afeat_pool=pool_a,
                            B=B_total, B_eff=B_total, S=S,
                            scale=scale, rank_offset=0,
                            num_negatives=k_seg,
                            num_negative_videos=self.val_segment_num_negative_videos,
                        )
                        seg_loss = (
                            nn.functional.cross_entropy(sim_v2a, targets) +
                            nn.functional.cross_entropy(sim_a2v, targets)
                        ) / 2
                        seg_suffix = f"_neg{k_seg}{gran_suffix}"
                        metrics[f"segment_contrastive_loss{seg_suffix}"] = seg_loss.item()
                        seg_prec = compute_segment_sampled_precision(sim_v2a, sim_a2v, num_intra)
                        for mk, mv in seg_prec.items():
                            metrics[f"{mk}{seg_suffix}"] = mv
                        if len(self.val_segment_num_negatives_list) == 1 and n_gran == 1:
                            metrics["segment_contrastive_loss"] = seg_loss.item()
                            metrics.update(seg_prec)
                    if S > 1:
                        intra = compute_segment_intra_precision(pool_v, pool_a, B_total, S)
                        for mk, mv in intra.items():
                            metrics[f"{mk}{gran_suffix}"] = mv

                del pool_v, pool_a

            if all_glob_vfeats:
                gpool_v_cpu = torch.cat(all_glob_vfeats, dim=0)
                gpool_a_cpu = torch.cat(all_glob_afeats, dim=0)
                if gpool_v_cpu.shape[0] > _eval_max_b:
                    gpool_v_cpu = gpool_v_cpu[:_eval_max_b]
                    gpool_a_cpu = gpool_a_cpu[:_eval_max_b]
                gpool_v = gpool_v_cpu.to(self.device)
                gpool_a = gpool_a_cpu.to(self.device)
                del gpool_v_cpu, gpool_a_cpu

                if gpool_v.shape[0] > 1:
                    g_scale = global_logit_scale_val if global_logit_scale_val is not None else torch.tensor(1.0, device=self.device)
                    B_g = gpool_v.shape[0]
                    for k_global in self.val_global_num_negatives_list:
                        K_g = min(k_global, B_g - 1)
                        if K_g <= 0:
                            continue
                        all_idx_g = torch.arange(B_g, device=self.device)
                        mask_g = all_idx_g.unsqueeze(0) != all_idx_g.unsqueeze(1)
                        neg_pool_g = all_idx_g.unsqueeze(0).expand(B_g, -1)[mask_g].view(B_g, B_g - 1)
                        perm_g = torch.rand(B_g, B_g - 1, device=self.device).argsort(dim=1)[:, :K_g]
                        neg_idx_g = neg_pool_g.gather(1, perm_g)
                        sample_idx_g = torch.cat([all_idx_g.unsqueeze(1), neg_idx_g], dim=1)

                        afeat_g_sampled = gpool_a[sample_idx_g]
                        vfeat_g_sampled = gpool_v[sample_idx_g]
                        sim_g_v2a = torch.einsum("bd,bkd->bk", gpool_v, afeat_g_sampled) / g_scale
                        sim_g_a2v = torch.einsum("bd,bkd->bk", gpool_a, vfeat_g_sampled) / g_scale
                        gt_sampled = torch.zeros(B_g, dtype=torch.long, device=self.device)
                        global_loss = (
                            nn.functional.cross_entropy(sim_g_v2a, gt_sampled) +
                            nn.functional.cross_entropy(sim_g_a2v, gt_sampled)
                        ) / 2
                        global_suffix = f"_gneg{k_global}"
                        metrics[f"global_contrastive_loss{global_suffix}"] = global_loss.item()
                        global_prec = compute_global_sampled_precision(
                            gpool_v, gpool_a, num_negatives=k_global
                        )
                        for mk, mv in global_prec.items():
                            metrics[f"{mk}{global_suffix}"] = mv
                        if len(self.val_global_num_negatives_list) == 1:
                            metrics["global_contrastive_loss"] = global_loss.item()
                            metrics.update(global_prec)

                del gpool_v, gpool_a

            logging.info(f"  [merged_all] Contrastive eval metrics "
                         f"({len(self.val_contrastive_dataloaders)} datasets merged):")
            for k, v in sorted(metrics.items()):
                logging.info(f"    {k}: {v:.4f}")

            for k, v in metrics.items():
                all_metrics[f"merged_all/{k}"] = v

        torch.cuda.empty_cache()

        if self.ema is not None:
            self.ema.restore()
        self.unwrapped_model.train()

        if self.is_distributed:
            dist.barrier()

        return all_metrics

    # ------------------------------------------------------------------
    # evaluate_llm_caption
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_llm_caption(self, step: Optional[int] = None) -> Dict[str, float]:
        if not self.val_caption_dataloaders:
            return {}
        model = self.unwrapped_model
        llm_head = model.llm_caption_head
        if llm_head is None:
            return {}

        if self.ema is not None:
            self.ema.apply_shadow()
        model.eval()

        active_modes = [m for m, p in llm_head.caption_mode_probs.items() if p > 0]
        if not active_modes:
            if self.ema is not None:
                self.ema.restore()
            model.train()
            return {}

        latent_key = "latent_mean" if model.llm_use_mean else "latent"
        local_max_samples = (
            math.ceil(self.val_caption_max_samples / self.world_size)
            if self.val_caption_max_samples is not None else None
        )

        all_metrics: Dict[str, float] = {}
        for ds_name, dataloader in self.val_caption_dataloaders.items():
            mode_loss_sums = {m: 0.0 for m in active_modes}
            mode_counts = {m: 0 for m in active_modes}
            count = 0

            pbar = tqdm(dataloader, desc=f"Eval Caption [{ds_name}]", disable=not self.is_main)
            for batch in pbar:
                data = batch['data']
                video = data.get('video')
                audio = data.get('audio')
                audio_lengths = data.get('audio_lengths')
                captions = data.get('captions', [])
                video_descriptions = data.get('video_descriptions')
                audio_descriptions = data.get('audio_descriptions')

                if video is not None:
                    video = video.to(self.device)
                if audio is not None:
                    audio = audio.to(self.device)
                if audio_lengths is not None:
                    audio_lengths = audio_lengths.to(self.device)

                outputs = model(video, audio, captions=None,
                                skip_video_decoder=True, skip_audio_decoder=True)
                video_lat = outputs["video"][latent_key] if "video" in outputs else None
                audio_lat = outputs["audio"][latent_key] if "audio" in outputs else None
                audio_latent_lengths = None
                if audio_lat is not None and audio_lengths is not None:
                    audio_latent_lengths = model._compute_audio_latent_lengths(
                        audio_lengths, audio_lat.shape[-1],
                    )
                B = (video_lat if video_lat is not None else audio_lat).shape[0]

                for mode in active_modes:
                    if mode == "avcap" and not (video is not None and audio is not None):
                        continue
                    if mode == "vcap" and video is None:
                        continue
                    if mode == "acap" and audio is None:
                        continue
                    try:
                        with model._module_autocast('llm'):
                            result = llm_head(
                                video_latent=video_lat, audio_latent=audio_lat,
                                captions=captions if captions else [""] * B,
                                video_descriptions=video_descriptions,
                                audio_descriptions=audio_descriptions,
                                audio_latent_lengths=audio_latent_lengths,
                                caption_mode=mode,
                            )
                        mode_loss_sums[mode] += result["loss"].item()
                        mode_counts[mode] += 1
                    except Exception as e:
                        logging.warning(f"Caption eval failed for mode={mode}: {e}")

                count += 1
                if local_max_samples is not None and count >= local_max_samples:
                    break
                torch.cuda.empty_cache()

            for mode in active_modes:
                metric_key = f"llm_caption_loss_{mode}"
                metrics_for_reduce = {metric_key: mode_loss_sums[mode]}
                reduced, _ = self._all_reduce_eval_metrics(metrics_for_reduce, mode_counts[mode])
                if self.is_main and mode_counts[mode] > 0:
                    all_metrics[f"{ds_name}/{metric_key}"] = reduced[metric_key]

        if self.ema is not None:
            self.ema.restore()
        model.train()

        if self.is_main and all_metrics:
            for mode in active_modes:
                key = f"llm_caption_loss_{mode}"
                values = [v for k, v in all_metrics.items() if k.endswith(f"/{key}")]
                if values:
                    all_metrics[f"avg/{key}"] = sum(values) / len(values)

        if self.is_distributed:
            dist.barrier()
        return all_metrics

    # ------------------------------------------------------------------
    # save / load checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int):
        if self.is_distributed:
            dist.barrier()

        train_state_dict = self.train_state.state_dict().copy()
        train_state_gathered = [None] * self.world_size
        if self.is_distributed:
            dist.all_gather_object(train_state_gathered, train_state_dict)
        else:
            train_state_gathered[0] = train_state_dict

        steps_all_rank = [s["step"] for s in train_state_gathered]
        assert min(steps_all_rank) == max(steps_all_rank), \
            f"Step mismatch: min={min(steps_all_rank)}, max={max(steps_all_rank)}"

        if self.is_main:
            output_dir = self.results_folder / f'Trainer_{step:08d}'
            output_dir.mkdir(parents=True, exist_ok=True)

            ckpt = {
                'step': step,
                'model_state_dict': self.unwrapped_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'ema_state_dict': self.ema.state_dict() if self.ema is not None else None,
                'scaler_state_dict': self.scaler.state_dict(),
                'train_state': train_state_gathered,
                'config': self.cfg,
                'video_logvar': self.video_logvar.detach().cpu(),
            }

            if self.use_uncertainty_balance:
                ckpt['uncertainty_balance'] = {
                    'initialized': self._ub_initialized,
                    'warmup_sums': self._ub_warmup_sums,
                    'warmup_counts': self._ub_warmup_counts,
                    'log_vars': {
                        k: getattr(self, f'ub_log_var_{k}').detach().cpu()
                        for k in self._ub_task_keys
                    },
                }

            # Discriminator state (audio LSGAN and/or video PatchGAN)
            if (self.use_audio_disc or self.use_video_disc) and self.discriminators:
                ckpt['discriminators'] = {
                    name: (d.module if isinstance(d, DDP) else d).state_dict()
                    for name, d in self.discriminators.items()
                }
                ckpt['optim_d'] = self.optim_d.state_dict() if self.optim_d is not None else None
                ckpt['scheduler_d'] = self.scheduler_d.state_dict() if self.scheduler_d is not None else None
                ckpt['scaler_d'] = self.scaler_d.state_dict() if self.scaler_d is not None else None
            else:
                ckpt['discriminators'] = {}
                ckpt['optim_d'] = None
                ckpt['scheduler_d'] = None
                ckpt['scaler_d'] = None

            torch.save(ckpt, output_dir / 'state_dict.pt')

            if (self.unwrapped_model.llm_caption_head is not None
                    and hasattr(self.unwrapped_model.llm_caption_head, 'llm_tokenizer')):
                tokenizer_dir = output_dir / 'llm_tokenizer'
                self.unwrapped_model.llm_caption_head.llm_tokenizer.save_pretrained(str(tokenizer_dir))

            logging.info(f"Saved checkpoint to {output_dir}")

        if self.is_distributed:
            dist.barrier()

    def _align_llm_embeddings(self, state_dict: Dict[str, torch.Tensor]):
        model = self.unwrapped_model
        if model.llm_caption_head is None:
            return
        embed_key = 'llm_caption_head.llm.model.embed_tokens.weight'
        if embed_key not in state_dict:
            return
        ckpt_vocab_size = state_dict[embed_key].shape[0]
        current_vocab_size = model.llm_caption_head.llm.get_input_embeddings().weight.shape[0]
        if ckpt_vocab_size != current_vocab_size:
            logging.info(f"LLM embedding resize: {current_vocab_size} -> {ckpt_vocab_size}")
            model.llm_caption_head.llm.resize_token_embeddings(ckpt_vocab_size)

    def load_checkpoint(self, ckpt_dir: str):
        if self.is_distributed:
            dist.barrier()

        ckpt_path = os.path.join(ckpt_dir, 'state_dict.pt')
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self._align_llm_embeddings(ckpt['model_state_dict'])
        self.unwrapped_model.load_state_dict(ckpt['model_state_dict'])

        # Snapshot freshly-built per-group lrs BEFORE optimizer.load_state_dict
        # overwrites them with the ckpt-time values. Used by
        # reset_scheduler_on_resume to keep the new CLI/config lrs while still
        # restoring optimizer momentum / Adam moments from ckpt.
        fresh_lrs_by_name: Dict[str, float] = {}
        if self.reset_scheduler_on_resume:
            for i, g in enumerate(self.optimizer.param_groups):
                _name = g.get('name') or f'grp_{i}'
                fresh_lrs_by_name[_name] = float(g['lr'])

        try:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except (ValueError, RuntimeError) as e:
            logging.warning(f"Optimizer state loading failed: {e}")

        if self.reset_scheduler_on_resume and fresh_lrs_by_name:
            for i, g in enumerate(self.optimizer.param_groups):
                _name = g.get('name') or f'grp_{i}'
                if _name in fresh_lrs_by_name:
                    g['lr'] = fresh_lrs_by_name[_name]
            if self.is_main:
                logging.info(
                    f"[reset_scheduler_on_resume] optimizer.param_groups['lr'] "
                    f"restored to fresh values (momentum/Adam moments kept): "
                    f"{fresh_lrs_by_name}"
                )

        try:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except (KeyError, ValueError, RuntimeError, TypeError) as e:
            logging.warning(f"Scheduler state loading failed ({e}), re-initializing from checkpoint step")
            resumed_step = ckpt.get('scheduler_state_dict', {}).get('last_epoch', 0)
            if resumed_step > 0:
                self.scheduler.last_epoch = resumed_step
                for group, lr in zip(self.optimizer.param_groups, self.scheduler.get_last_lr()):
                    group['lr'] = lr
                logging.info(f"Scheduler reset to step {resumed_step}, lr={self.scheduler.get_last_lr()[0]:.2e}")

        if self.reset_scheduler_on_resume:
            # Re-anchor scheduler.base_lrs to the fresh optimizer lrs we just
            # restored. last_step is preserved from the ckpt-loaded scheduler
            # state so step counting continues where we left off; schedule
            # shape (warmup/total/min_ratio) is already taken from current cfg
            # by _MultiGroupWarmupCosineScheduler.load_state_dict.
            self.scheduler._base_lrs = {
                id(g): float(g['lr']) for g in self.optimizer.param_groups
            }
            if self.is_main:
                _last = getattr(self.scheduler, '_last_step',
                                getattr(self.scheduler, 'last_epoch', 0))
                logging.info(
                    f"[reset_scheduler_on_resume] scheduler base_lrs reset to "
                    f"current optimizer lrs; last_step={_last} preserved; "
                    f"schedule shape uses current cfg."
                )
        if self.ema is not None and ckpt.get('ema_state_dict') is not None:
            self.ema.load_state_dict(ckpt['ema_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])

        if 'video_logvar' in ckpt:
            self.video_logvar.data.copy_(ckpt['video_logvar'].to(self.device))

        if self.use_uncertainty_balance and 'uncertainty_balance' in ckpt:
            ub_state = ckpt['uncertainty_balance']
            self._ub_initialized = ub_state['initialized']
            self._ub_warmup_sums = ub_state['warmup_sums']
            self._ub_warmup_counts = ub_state['warmup_counts']
            for key, val in ub_state.get('log_vars', {}).items():
                param = getattr(self, f'ub_log_var_{key}', None)
                if param is not None:
                    param.data.copy_(val.to(self.device))

        # ---- Discriminators (audio and/or video, optional) ----
        # Old checkpoints (pre-GAN phase) won't have these keys; we load them
        # only if both the checkpoint and the current config enable disc.
        if (self.use_audio_disc or self.use_video_disc) and self.discriminators:
            ckpt_discs = ckpt.get('discriminators', {}) or {}
            for name, d in self.discriminators.items():
                if name in ckpt_discs:
                    target = d.module if isinstance(d, DDP) else d
                    try:
                        target.load_state_dict(ckpt_discs[name], strict=False)
                        if self.is_main:
                            logging.info(f"[disc] loaded {name} state from checkpoint.")
                    except (KeyError, ValueError, RuntimeError) as e:
                        logging.warning(f"[disc] loading {name} failed: {e}")
                elif self.is_main:
                    logging.info(
                        f"[disc] checkpoint has no '{name}' state; initializing from scratch."
                    )
            if self.optim_d is not None and ckpt.get('optim_d') is not None:
                try:
                    self.optim_d.load_state_dict(ckpt['optim_d'])
                except (ValueError, RuntimeError) as e:
                    logging.warning(f"[disc] optim_d state loading failed: {e}")
            if self.scheduler_d is not None and ckpt.get('scheduler_d') is not None:
                try:
                    self.scheduler_d.load_state_dict(ckpt['scheduler_d'])
                except (KeyError, ValueError, RuntimeError, TypeError) as e:
                    logging.warning(f"[disc] scheduler_d state loading failed: {e}")
            if self.scaler_d is not None and ckpt.get('scaler_d') is not None:
                try:
                    self.scaler_d.load_state_dict(ckpt['scaler_d'])
                except (KeyError, ValueError, RuntimeError, TypeError) as e:
                    logging.warning(f"[disc] scaler_d state loading failed: {e}")

        # train_state 兼容：当保存 ckpt 时的 world_size 与当前 world_size 不一致时，
        # 没有对应条目的 rank 不能停留在 step=0（否则后续会触发 save_checkpoint 中的
        # Step mismatch 断言）。规则：
        #   1) 至少有一个条目存在的话，所有 rank 的 train_state.step 都同步成
        #      第一个有效条目的 step（save 端断言保证所有有效条目 step 一致）。
        #   2) 当前 rank 在保存列表内时，正常加载该 rank 的 dataset_state_dict；
        #      不在列表内的 rank 仅保持 step 同步，dataset 从默认状态开始。
        train_state_list = ckpt['train_state'] or []
        canonical_step: Optional[int] = None
        for entry in train_state_list:
            if isinstance(entry, dict) and 'step' in entry:
                canonical_step = int(entry['step'])
                break

        if self.rank < len(train_state_list) and isinstance(
            train_state_list[self.rank], dict
        ):
            self.train_state.load_state_dict(train_state_list[self.rank])
            if self.train_state.dataset_state_dict.get("consumed_samples") is not None:
                self.train_ds.load_state_dict(self.train_state.dataset_state_dict)
        else:
            if canonical_step is not None:
                self.train_state.step = canonical_step
                logging.warning(
                    f"Rank {self.rank} not found in saved train_state "
                    f"(saved world_size={len(train_state_list)}, current world_size={self.world_size}). "
                    f"Synchronizing step={canonical_step}; dataset will start from default state."
                )
            else:
                logging.warning(
                    f"Rank {self.rank}: ckpt['train_state'] is empty or invalid; "
                    f"keeping default train_state."
                )

        # 兜底：即便 list 内某个 rank 条目 step 与其它不一致，强制对齐到 canonical_step。
        if canonical_step is not None and self.train_state.step != canonical_step:
            logging.warning(
                f"Rank {self.rank}: per-rank train_state.step={self.train_state.step} "
                f"differs from canonical_step={canonical_step}; overriding to canonical."
            )
            self.train_state.step = canonical_step

        if self.is_distributed:
            dist.barrier()

        logging.info(f"Loaded checkpoint from {ckpt_dir}, step={self.train_state.step}")

    def load_pretrained_checkpoint(self, ckpt_dir: str, keep_audio_vae: bool = False):
        if self.is_distributed:
            dist.barrier()

        ckpt_path = os.path.join(ckpt_dir, 'state_dict.pt')
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        state_dict = ckpt['model_state_dict']

        if keep_audio_vae:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('audio_vae.')}
            if self.is_main:
                logging.info("Pretrained load: skipping audio_vae.* keys (keep_audio_vae_pretrained=True)")

        self._align_llm_embeddings(state_dict)
        result = self.unwrapped_model.load_state_dict(state_dict, strict=False)
        if self.is_main:
            if result.missing_keys:
                logging.info(f"Pretrained load - missing_keys ({len(result.missing_keys)})")
            if result.unexpected_keys:
                logging.info(f"Pretrained load - unexpected_keys ({len(result.unexpected_keys)})")
            logging.info(f"Loaded pretrained checkpoint from {ckpt_dir} (strict=False)")

        if self.is_distributed:
            dist.barrier()

    def load_pretrained_split_checkpoints(
        self,
        video_ckpt_dir: Optional[str] = None,
        audio_ckpt_dir: Optional[str] = None,
        contrastive_ckpt_dir: Optional[str] = None,
    ):
        """Load weights from separate OmniVAE checkpoints by module prefix.

        - video_ckpt_dir: loads video_vae.*, image_distill_proj.*, video_distill_proj.*
        - audio_ckpt_dir: loads audio_vae.*, audio_distill_proj.*
        - contrastive_ckpt_dir: loads contrastive_head.*
        """
        if video_ckpt_dir is None and audio_ckpt_dir is None and contrastive_ckpt_dir is None:
            return

        if self.is_distributed:
            dist.barrier()

        merged_state_dict = {}

        if video_ckpt_dir is not None:
            ckpt_path = os.path.join(video_ckpt_dir, 'state_dict.pt')
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            full_sd = ckpt['model_state_dict']
            video_keys = {
                k: v for k, v in full_sd.items()
                if k.startswith(('video_vae.', 'image_distill_proj.', 'video_distill_proj.'))
            }
            merged_state_dict.update(video_keys)
            del ckpt, full_sd
            if self.is_main:
                logging.info(f"Split load: extracted {len(video_keys)} video_vae/distill keys from {video_ckpt_dir}")

        if audio_ckpt_dir is not None:
            ckpt_path = os.path.join(audio_ckpt_dir, 'state_dict.pt')
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            full_sd = ckpt['model_state_dict']
            audio_keys = {
                k: v for k, v in full_sd.items()
                if k.startswith(('audio_vae.', 'audio_distill_proj.'))
            }
            merged_state_dict.update(audio_keys)
            del ckpt, full_sd
            if self.is_main:
                logging.info(f"Split load: extracted {len(audio_keys)} audio_vae/distill keys from {audio_ckpt_dir}")

        if contrastive_ckpt_dir is not None:
            ckpt_path = os.path.join(contrastive_ckpt_dir, 'state_dict.pt')
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            full_sd = ckpt['model_state_dict']
            contrastive_keys = {
                k: v for k, v in full_sd.items()
                if k.startswith('contrastive_head.')
            }
            merged_state_dict.update(contrastive_keys)
            del ckpt, full_sd
            if self.is_main:
                logging.info(f"Split load: extracted {len(contrastive_keys)} contrastive_head.* keys from {contrastive_ckpt_dir}")

        if not merged_state_dict:
            return

        # Filter out shape-mismatched keys before load_state_dict:
        # `strict=False` tolerates missing / unexpected keys but still raises
        # RuntimeError on shape mismatch. This typically happens when the
        # current CLI flips an architectural switch (e.g. spatial_pool_mode
        # `transformer` -> `mean`, `spatial_merge_factor` change) so a param
        # with the same dotted name has a different shape. We silently drop
        # those keys so the affected submodules fall back to fresh init,
        # and emit a warning listing what was skipped.
        model_shapes = {
            k: tuple(v.shape) for k, v in self.unwrapped_model.state_dict().items()
        }
        skipped_shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        skipped_not_in_model: List[str] = []
        cleaned_state_dict: Dict[str, torch.Tensor] = {}
        for k, v in merged_state_dict.items():
            if k not in model_shapes:
                skipped_not_in_model.append(k)
                continue
            if tuple(v.shape) != model_shapes[k]:
                skipped_shape_mismatch.append((k, tuple(v.shape), model_shapes[k]))
                continue
            cleaned_state_dict[k] = v

        if self.is_main and skipped_shape_mismatch:
            logging.warning(
                f"Split load: {len(skipped_shape_mismatch)} keys dropped due to "
                "shape mismatch (the current model's architecture differs from "
                "the pretrained ckpt; these submodules will be re-initialized). "
                f"Examples: {skipped_shape_mismatch[:5]}"
            )

        result = self.unwrapped_model.load_state_dict(cleaned_state_dict, strict=False)
        if self.is_main:
            loaded_count = len(cleaned_state_dict) - len(result.unexpected_keys)
            logging.info(
                f"Split load: loaded {loaded_count} keys "
                f"(dropped shape-mismatch={len(skipped_shape_mismatch)}, "
                f"dropped not-in-model={len(skipped_not_in_model)}, "
                f"unexpected={len(result.unexpected_keys)})."
            )

        if self.is_distributed:
            dist.barrier()

    def load_pretrained_disc_checkpoint(
        self,
        ckpt_dir: str,
        load_optim: bool = False,
    ):
        """Warm-start discriminator(s) from a separate checkpoint.

        Applied independently from generator pretrained checkpoints
        (``pretrained_checkpoint`` / ``pretrained_video_checkpoint`` /
        ``pretrained_audio_checkpoint``); intended for cases like:

        - Stage-2.5: gen weights from Stage-1, disc state from Stage-2.
        - Plugging in a public/external discriminator warm-start.
        - Continuing GAN training after an exp_dir rename without
          ``--continue_train``.

        Loads weights for whatever names exist in both
        ``ckpt['discriminators']`` and ``self.discriminators``. Shape-mismatched
        params are silently dropped (with warning) so changing
        ``audio_disc_params`` between runs is non-fatal.

        Args:
            ckpt_dir: directory containing ``state_dict.pt``.
            load_optim: if True, also restore ``optim_d`` / ``scheduler_d`` /
                ``scaler_d``. Default False to keep a fresh disc warmup
                schedule with the new ``--lr_disc``.
        """
        if not (self.use_audio_disc or self.use_video_disc) or not self.discriminators:
            if self.is_main:
                logging.warning(
                    f"[disc-pretrained] no discriminator enabled in current run; "
                    f"ignoring pretrained_disc_checkpoint={ckpt_dir}"
                )
            return

        ckpt_path = os.path.join(ckpt_dir, 'state_dict.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[disc-pretrained] state_dict.pt not found under {ckpt_dir}"
            )

        if self.is_distributed:
            dist.barrier()

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        ckpt_discs = ckpt.get('discriminators', {}) or {}
        if not ckpt_discs:
            if self.is_main:
                logging.warning(
                    f"[disc-pretrained] checkpoint at {ckpt_dir} has no "
                    "'discriminators' field (possibly a pre-GAN checkpoint); "
                    "skipping."
                )
            del ckpt
            if self.is_distributed:
                dist.barrier()
            return

        loaded_names: List[str] = []
        skipped_missing: List[str] = []
        for name, d in self.discriminators.items():
            if name not in ckpt_discs:
                skipped_missing.append(name)
                continue
            target = d.module if isinstance(d, DDP) else d
            target_shapes = {
                k: tuple(v.shape) for k, v in target.state_dict().items()
            }
            cleaned: Dict[str, torch.Tensor] = {}
            shape_skipped: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
            not_in_model: List[str] = []
            for k, v in ckpt_discs[name].items():
                if k not in target_shapes:
                    not_in_model.append(k)
                    continue
                if tuple(v.shape) != target_shapes[k]:
                    shape_skipped.append((k, tuple(v.shape), target_shapes[k]))
                    continue
                cleaned[k] = v
            try:
                result = target.load_state_dict(cleaned, strict=False)
            except (KeyError, ValueError, RuntimeError) as e:
                if self.is_main:
                    logging.warning(f"[disc-pretrained] {name} load failed: {e}")
                continue
            loaded_names.append(name)
            if self.is_main:
                logging.info(
                    f"[disc-pretrained] loaded '{name}': "
                    f"matched={len(cleaned)} keys "
                    f"(shape-mismatch={len(shape_skipped)}, "
                    f"not-in-model={len(not_in_model)}, "
                    f"missing-after-load={len(result.missing_keys)})"
                )
                if shape_skipped:
                    logging.warning(
                        f"[disc-pretrained] '{name}' shape mismatches (first 5): "
                        f"{shape_skipped[:5]}"
                    )

        if self.is_main and skipped_missing:
            logging.info(
                f"[disc-pretrained] checkpoint missing disc(s): {skipped_missing} "
                "(these will start from fresh init)."
            )

        if load_optim:
            if self.optim_d is not None and ckpt.get('optim_d') is not None:
                try:
                    self.optim_d.load_state_dict(ckpt['optim_d'])
                    if self.is_main:
                        logging.info("[disc-pretrained] optim_d state restored.")
                except (ValueError, RuntimeError) as e:
                    if self.is_main:
                        logging.warning(
                            f"[disc-pretrained] optim_d load failed: {e}"
                        )
            if self.scheduler_d is not None and ckpt.get('scheduler_d') is not None:
                try:
                    self.scheduler_d.load_state_dict(ckpt['scheduler_d'])
                    if self.is_main:
                        logging.info("[disc-pretrained] scheduler_d state restored.")
                except (KeyError, ValueError, RuntimeError, TypeError) as e:
                    if self.is_main:
                        logging.warning(
                            f"[disc-pretrained] scheduler_d load failed: {e}"
                        )
            if self.scaler_d is not None and ckpt.get('scaler_d') is not None:
                try:
                    self.scaler_d.load_state_dict(ckpt['scaler_d'])
                    if self.is_main:
                        logging.info("[disc-pretrained] scaler_d state restored.")
                except (KeyError, ValueError, RuntimeError, TypeError) as e:
                    if self.is_main:
                        logging.warning(
                            f"[disc-pretrained] scaler_d load failed: {e}"
                        )
        elif self.is_main:
            logging.info(
                "[disc-pretrained] disc weights only (optim_d / scheduler_d / "
                "scaler_d kept fresh; pass --pretrained_disc_load_optim to also "
                "restore them)."
            )

        if self.is_main:
            logging.info(
                f"[disc-pretrained] done loading {len(loaded_names)} disc(s) "
                f"from {ckpt_dir}: {loaded_names}"
            )

        del ckpt
        gc.collect()
        if self.is_distributed:
            dist.barrier()

    # ------------------------------------------------------------------
    # validate_only
    # ------------------------------------------------------------------

    def _infer_exp_dir_from_checkpoint(self, checkpoint_path: str) -> Optional[Path]:
        ckpt_p = Path(checkpoint_path).resolve()
        candidate = ckpt_p.parent.parent
        if candidate.name == 'checkpoints' or (candidate / 'checkpoints').exists():
            if candidate.name == 'checkpoints':
                return candidate.parent
            return candidate
        if ckpt_p.parent.name == 'checkpoints':
            return ckpt_p.parent.parent
        return None

    def validate_only(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(self.results_folder, prefix="Trainer_")
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in {self.results_folder}.")

        inferred_exp_dir = self._infer_exp_dir_from_checkpoint(checkpoint_path)
        if inferred_exp_dir is not None and inferred_exp_dir.resolve() != self.exp_dir.resolve():
            if self.is_main:
                logging.info(f"Overriding exp_dir: {self.exp_dir} -> {inferred_exp_dir}")
            self.exp_dir = inferred_exp_dir
            self.results_folder = self.exp_dir / 'checkpoints'
            self._setup_audio_eval_dirs()

        self.load_checkpoint(checkpoint_path)
        ckpt_step = self.train_state.step

        if self.is_main:
            logging.info("=" * 60)
            logging.info(f"Validation-only mode  |  step = {ckpt_step}")
            logging.info("=" * 60)

        all_results: Dict[str, Any] = {"checkpoint": str(checkpoint_path), "step": ckpt_step}

        if self.eval_video_recon:
            if self.is_main:
                logging.info("[Valid] Video reconstruction ...")
            video_metrics = self.evaluate_video(step=ckpt_step)
            if self.is_main:
                all_results["video"] = video_metrics
                for k, v in video_metrics.items():
                    logging.info(f"  {k}: {v:.4f}")

        if self.eval_audio_recon:
            if self.is_main:
                logging.info("[Valid] Audio reconstruction ...")
            audio_metrics = self.evaluate_audio()
            if self.is_main:
                all_results["audio"] = audio_metrics
                for k, v in audio_metrics.items():
                    logging.info(f"  {k}: {v:.4f}")

        if self.eval_contrastive:
            if self.is_main:
                logging.info("[Valid] Contrastive ...")
            contrastive_metrics = self.evaluate_contrastive(step=ckpt_step)
            if self.is_main:
                all_results["contrastive"] = contrastive_metrics
                for k, v in contrastive_metrics.items():
                    logging.info(f"  {k}: {v:.4f}")

        if self.eval_llm_caption:
            if self.is_main:
                logging.info("[Valid] LLM caption ...")
            llm_metrics = self.evaluate_llm_caption(step=ckpt_step)
            if self.is_main:
                all_results["llm_caption"] = llm_metrics
                for k, v in llm_metrics.items():
                    logging.info(f"  {k}: {v:.4f}")

        if self.is_main:
            segment_neg_suffix = _format_int_list_suffix(self.val_segment_num_negatives_list)
            global_neg_suffix = _format_int_list_suffix(self.val_global_num_negatives_list)
            merged_suffix = "_merged" if self.eval_contrastive_in_all else ""
            result_path = self.exp_dir / (
                f"valid_results_step{ckpt_step:08d}"
                f"_neg{segment_neg_suffix}"
                f"_negv{self.val_segment_num_negative_videos}"
                f"_gneg{global_neg_suffix}"
                f"{merged_suffix}.json"
            )
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logging.info(f"Validation results saved to: {result_path}")

        if self.is_distributed:
            dist.barrier()

    # ------------------------------------------------------------------
    # train — main training loop
    # ------------------------------------------------------------------

    def train(self):
        self.model.train()
        for _d in self.discriminators.values():
            _d.train()

        latest_ckpt = None
        if self.continue_train:
            latest_ckpt = find_latest_checkpoint(self.results_folder, prefix="Trainer_")
            if latest_ckpt:
                self.load_checkpoint(latest_ckpt)
            elif self.is_main:
                logging.info("`--continue_train` enabled but no checkpoint found.")

        if self.pretrained_checkpoint and latest_ckpt is None:
            self.load_pretrained_checkpoint(
                self.pretrained_checkpoint,
                keep_audio_vae=self.keep_audio_vae_pretrained,
            )
        elif latest_ckpt is None and self.pretrained_checkpoint is None:
            if self.is_main:
                logging.info("Starting training from scratch.")

        if latest_ckpt is None and (
            self.pretrained_video_checkpoint
            or self.pretrained_audio_checkpoint
            or self.pretrained_contrastive_checkpoint
        ):
            self.load_pretrained_split_checkpoints(
                video_ckpt_dir=self.pretrained_video_checkpoint,
                audio_ckpt_dir=self.pretrained_audio_checkpoint,
                contrastive_ckpt_dir=self.pretrained_contrastive_checkpoint,
            )

        # Discriminator-only warm-start. Applied after generator pretrained
        # loaders (so their disc fields, if any, are not consulted) and only
        # when not resuming an existing run (continue_train resume already
        # restores disc state via load_checkpoint).
        if latest_ckpt is None and self.pretrained_disc_checkpoint:
            self.load_pretrained_disc_checkpoint(
                self.pretrained_disc_checkpoint,
                load_optim=self.pretrained_disc_load_optim,
            )

        start_step = self.train_state.step
        train_iter = iter(self.train_dl)
        log_dict = {}
        time_accum = {
            't_data': 0.0, 't_prefetch_submit': 0.0,
            't_forward': 0.0, 't_encoder_wait': 0.0,
            't_enc_video': 0.0, 't_enc_image': 0.0, 't_enc_audio': 0.0,
            't_enc_total': 0.0,
        }

        if self.is_main:
            pbar = tqdm(
                range(start_step, self.tot_train_steps),
                desc="Training", initial=start_step, total=self.tot_train_steps,
            )
        else:
            pbar = range(start_step, self.tot_train_steps)

        N = self.gradient_accumulation_steps
        for step in pbar:
            if (self.global_contrastive_start_steps > 0
                    and step == self.global_contrastive_start_steps
                    and self.is_main):
                logging.info(f"Step {step}: global contrastive loss activated")

            if (self._video_vae_frozen
                    and self.freeze_video_vae_until_step > 0
                    and step >= self.freeze_video_vae_until_step):
                self._unfreeze_video_vae_and_extend_optimizer()

            if (getattr(self, '_audio_vae_frozen', False)
                    and self.freeze_audio_vae_until_step > 0
                    and step >= self.freeze_audio_vae_until_step):
                self._unfreeze_audio_vae_and_extend_optimizer()

            should_collect_media = (
                self.tb_train_media_steps > 0
                and (step + 1) % self.tb_train_media_steps == 0
                and self.is_main
            )

            micro_losses = []
            micro_losses_image: List[Dict[str, Any]] = []
            micro_losses_video: List[Dict[str, Any]] = []
            contrastive_out = None
            media_data = None
            last_state_dict = {}

            # Per-update timing accumulators (summed across N micro-steps)
            _upd_t_data = 0.0
            _upd_t_prefetch = 0.0
            _upd_t_forward = 0.0
            _upd_t_encoder_wait = 0.0
            _upd_t_enc_video = 0.0
            _upd_t_enc_image = 0.0
            _upd_t_enc_audio = 0.0
            _upd_t_enc_total = 0.0

            for i in range(N):
                is_boundary = (i == N - 1)

                _t0 = time.perf_counter()

                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_dl)
                    batch = next(train_iter)

                _t_data = time.perf_counter()

                # Submit encoder requests immediately after data load (Mode C)
                if self.distill_prefetcher is not None:
                    self.distill_prefetcher.prefetch(batch)

                _t_prefetch = time.perf_counter()

                want_media = should_collect_media and is_boundary
                micro_out, micro_contrastive, micro_media = self.train_step(
                    batch, collect_media=want_media,
                    is_accum_boundary=is_boundary, accum_steps=N,
                )

                _t_step_done = time.perf_counter()

                last_state_dict = batch.get('state_dict', {})
                micro_losses.append(micro_out)
                # train_step sets self._current_modality each call; capture it
                # here (rather than inside train_step) so we don't have to edit
                # every return path. D-only micro-steps always operate on video.
                _micro_modality = getattr(self, '_current_modality', None) or 'video'
                if _micro_modality == 'image':
                    micro_losses_image.append(micro_out)
                else:
                    micro_losses_video.append(micro_out)
                if is_boundary:
                    contrastive_out = micro_contrastive
                    media_data = micro_media

                _upd_t_data += _t_data - _t0
                _upd_t_prefetch += _t_prefetch - _t_data
                step_total = _t_step_done - _t_prefetch
                if self.distill_prefetcher is not None:
                    pf = self.distill_prefetcher
                    _upd_t_encoder_wait += pf.last_wait_ms / 1000.0
                    _upd_t_forward += step_total - (pf.last_wait_ms / 1000.0)
                    _upd_t_enc_video += pf.last_time_video_ms / 1000.0
                    _upd_t_enc_image += pf.last_time_image_ms / 1000.0
                    _upd_t_enc_audio += pf.last_time_audio_ms / 1000.0
                    _upd_t_enc_total += pf.last_time_total_ms / 1000.0
                else:
                    _upd_t_forward += step_total

                if micro_out.get('skipped_nan') or micro_out.get('skipped_nan_grad'):
                    # Discard any partially-accumulated grads for this update cycle.
                    self.optimizer.zero_grad(set_to_none=True)
                    break

            self.train_state.update(last_state_dict)

            # Aggregate per-update timing into the rolling window accumulator
            time_accum['t_data'] += _upd_t_data
            time_accum['t_prefetch_submit'] += _upd_t_prefetch
            time_accum['t_forward'] += _upd_t_forward
            time_accum['t_encoder_wait'] += _upd_t_encoder_wait
            time_accum['t_enc_video'] += _upd_t_enc_video
            time_accum['t_enc_image'] += _upd_t_enc_image
            time_accum['t_enc_audio'] += _upd_t_enc_audio
            time_accum['t_enc_total'] += _upd_t_enc_total

            losses = _avg_loss_dicts(micro_losses)
            # Per-modality averages for tensorboard split (image/* vs video/*).
            # Empty buckets simply produce {} and are skipped at log time.
            losses_image = _avg_loss_dicts(micro_losses_image)
            losses_video = _avg_loss_dicts(micro_losses_video)

            # grad_norm/* is only produced on the boundary micro-step; keep the
            # latest value directly (averaging with missing keys is meaningless).
            grad_norm_metrics = {k: v for k, v in losses.items() if k.startswith('grad_norm/')}
            loss_metrics = {k: v for k, v in losses.items() if not k.startswith('grad_norm/')}
            loss_metrics_image = {
                k: v for k, v in losses_image.items() if not k.startswith('grad_norm/')
            }
            loss_metrics_video = {
                k: v for k, v in losses_video.items() if not k.startswith('grad_norm/')
            }

            log_dict = accum_log(log_dict, loss_metrics)

            if (step + 1) % self.stdout_steps == 0 and self.is_main:
                avg_losses = {k: v / self.stdout_steps for k, v in log_dict.items()}
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                n = self.stdout_steps
                exp_dir_str = str(self.exp_dir)
                time_str = (
                    f"data={time_accum['t_data']/n:.3f}s "
                    f"prefetch={time_accum['t_prefetch_submit']/n:.3f}s "
                    f"fwd+bwd={time_accum['t_forward']/n:.3f}s "
                    f"enc_wait={time_accum['t_encoder_wait']/n:.3f}s"
                )
                if time_accum['t_enc_total'] > 0:
                    time_str += (
                        f" (vid={time_accum['t_enc_video']/n:.3f}s"
                        f" img={time_accum['t_enc_image']/n:.3f}s"
                        f" aud={time_accum['t_enc_audio']/n:.3f}s"
                        f" total={time_accum['t_enc_total']/n:.3f}s)"
                )
                tqdm.write(f"Step {step + 1}: {loss_str}")
                tqdm.write(f"  [exp] {exp_dir_str}")
                tqdm.write(f"  [time] {time_str}")
                log_dict = {}
                time_accum = {k: 0.0 for k in time_accum}

            if grad_norm_metrics and self.is_main:
                short = {k.replace('grad_norm/', 'gn/'): f"{v:.2f}" for k, v in grad_norm_metrics.items()}
                pbar.set_postfix(short)

            if (step + 1) % self.log_steps == 0 and self.is_main:
                for k, v in loss_metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, step + 1)
                # Modality-split scalars: only emit when iv-alterstep is on, so
                # pure-video runs don't double-log under both train/* and
                # train/video/*.
                if getattr(self, 'use_image_video_alter', False):
                    for k, v in loss_metrics_image.items():
                        self.writer.add_scalar(f'train/image/{k}', v, step + 1)
                    for k, v in loss_metrics_video.items():
                        self.writer.add_scalar(f'train/video/{k}', v, step + 1)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], step + 1)
                if self.use_audio_disc and self.scheduler_d is not None:
                    self.writer.add_scalar(
                        'train/lr_disc', self.scheduler_d.get_last_lr()[0], step + 1,
                    )
                if contrastive_out is not None:
                    contrastive_metrics = self.compute_contrastive_metrics(contrastive_out)
                    for k, v in contrastive_metrics.items():
                        self.writer.add_scalar(f'train/{k}', v, step + 1)

            if grad_norm_metrics and self.is_main:
                for k, v in grad_norm_metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, step + 1)

            if media_data is not None and self.is_main and self.writer:
                self._write_train_media_tb(media_data, step + 1)

            if (step + 1) % self.save_model_steps == 0:
                self.save_checkpoint(step + 1)

            if (step + 1) % self.eval_steps == 0:
                if self.is_distributed:
                    dist.barrier()

                if self.eval_video_recon:
                    video_metrics = self.evaluate_video(step=step + 1)
                    if self.is_main:
                        for k, v in video_metrics.items():
                            self.writer.add_scalar(f'eval/{k}', v, step + 1)
                            logging.info(f"Eval {k}: {v:.4f}")

                if self.eval_audio_recon:
                    audio_metrics = self.evaluate_audio()
                    if self.is_main:
                        for k, v in audio_metrics.items():
                            self.writer.add_scalar(f'eval/{k}', v, step + 1)
                            logging.info(f"Eval {k}: {v:.4f}")

                if self.eval_contrastive:
                    contrastive_eval_metrics = self.evaluate_contrastive(step=step + 1)
                    if self.is_main:
                        for k, v in contrastive_eval_metrics.items():
                            self.writer.add_scalar(f'eval/{k}', v, step + 1)
                            logging.info(f"Eval {k}: {v:.4f}")

                if self.eval_llm_caption:
                    llm_metrics = self.evaluate_llm_caption(step=step + 1)
                    if self.is_main:
                        for k, v in llm_metrics.items():
                            self.writer.add_scalar(f'eval/{k}', v, step + 1)
                            logging.info(f"Eval {k}: {v:.4f}")

        self.save_checkpoint(self.tot_train_steps)

        if self.is_main:
            self.writer.close()
