"""
SynchformerVAE: Audio-Video synchronization model using VAE latent features.

Reads the AudioVideoVAE config.yaml to build WanVAE + DAC encoders and the
LatentAVContrastiveHead for segment aggregation (spatial pooling + temporal
chunking).  The contrastive head's projection/loss modules are unused; only
its pooling pipeline is called.

Data flow (default, skip_temporal_pool=False):
    video (B, C, T, H, W) -> WanVAE encoder -> (B, D_v, T', H', W')
    audio (B, 1, T_a)      -> DAC encoder    -> (B, D_a, T_l)
    -> optionally trim the right-side tail context
    -> contrastive_head._pool_video_segments  -> (B, S, D_v_seg)
    -> contrastive_head._pool_audio_segments  -> (B, S, D_a_seg)
    -> vproj / aproj -> GlobalTransformer -> offset logits

Data flow (skip_temporal_pool=True):
    video (B, C, T, H, W) -> WanVAE encoder -> (B, D_v, T', H', W')
    -> contrastive_head._spatial_pool         -> (B, D_sp, T')
    audio (B, 1, T_a)      -> DAC encoder    -> (B, D_a, T_l)
    -> optionally trim the right-side tail context
    -> optional adaptive_avg_pool1d           -> (B, D_a, n_audio_tokens)
    -> vproj / aproj -> GlobalTransformer -> offset logits
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from omnivae_sync.utils.utils import instantiate_from_config
from omnivae_sync.model.sync_model import GlobalTransformer, GlobalTransformerWithSyncabilityHead, init_weights


def _release_root() -> Path:
    explicit = os.environ.get("OMNIVAE_RELEASE_ROOT") or os.environ.get("OPEN_SOURCE_ROOT")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend([
        repo_root / "open_source",
        repo_root.parent / "open_source",
        repo_root / "open_source" / "open_source",
        repo_root.parent / "open_source" / "open_source",
    ])
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "models").is_dir() and (candidate / "eval").is_dir():
            return candidate.resolve()
    return (repo_root / "open_source").resolve()


def _resolve_release_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    if text.startswith(("models/", "eval/")):
        return str((_release_root() / text).resolve())
    return text


def _load_av_vae_config(config_path: str) -> dict:
    """Load AudioVideoVAE config.yaml and return as a plain dict."""
    config_path = _resolve_release_path(config_path) or config_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"AudioVideoVAE config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dir = str(Path(config_path).resolve().parent)
    cfg_dict['_config_path'] = str(Path(config_path).resolve())
    cfg_dict['_config_dir'] = cfg_dir
    if isinstance(cfg_dict.get('model'), dict):
        for section in ('video', 'audio', 'contrastive'):
            if isinstance(cfg_dict['model'].get(section), dict):
                cfg_dict['model'][section].setdefault('_config_dir', cfg_dir)
    return cfg_dict


def _resolve_config_path(path: str, config_dir: Optional[str] = None) -> str:
    """Resolve model_config paths saved in training configs from old repo layouts."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    candidates = [Path.cwd() / candidate]
    if config_dir:
        base = Path(config_dir)
        candidates.extend([base / candidate, base.parent / candidate])

    try:
        import omnivae

        omnivae_root = Path(omnivae.__file__).resolve().parent.parent
        candidates.extend([
            omnivae_root / candidate,
            omnivae_root / 'configs' / 'model' / candidate.name,
        ])
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / 'configs' / 'model' / candidate.name)

    for p in candidates:
        if p.exists():
            return str(p)

    tried = '\n'.join(f'  - {p}' for p in candidates)
    raise FileNotFoundError(f"Config file not found: {path}\nTried:\n{tried}")


def _build_video_vae(video_cfg: dict):
    """Build video VAE from the 'model.video' section of AudioVideoVAE config.

    Supports WanVAE (Wan2.1, z_dim=16) and WanVAE22 (Wan2.2, z_dim=48)
    via the ``model_name`` field in video_cfg.
    """
    from omnivae.models.causalvideovae.model import WanVAEModel, WanVAE22Model

    _VIDEO_VAE_CLASSES = {
        'WanVAE': WanVAEModel,
        'WanVAE22': WanVAE22Model,
    }

    model_name = video_cfg.get('model_name', 'WanVAE')
    VAEClass = _VIDEO_VAE_CLASSES.get(model_name)
    if VAEClass is None:
        raise ValueError(
            f"Unknown video model_name '{model_name}'. "
            f"Available: {list(_VIDEO_VAE_CLASSES.keys())}"
        )

    pretrained_path = video_cfg.get('pretrained_model_name_or_path')
    model_config = video_cfg.get('model_config')
    config_dir = video_cfg.get('_config_dir')

    # `from_pretrained` from diffusers expects an HF-style *directory* (with
    # config.json + weights) or an HF repo id; passing a single .pth/.pt file
    # makes it try to parse the binary as JSON and crash. So we only take the
    # `from_pretrained` path for directories / non-existent strings (treated
    # as repo ids); for single weight files we build the architecture from
    # `model_config` and rely on the caller's `vae_pretrained` (the combined
    # av_vae ckpt, which carries `video_vae.*` keys) to load weights later.
    def _is_single_weight_file(p: str) -> bool:
        return bool(p) and os.path.isfile(p) and p.lower().endswith(
            ('.pth', '.pt', '.bin', '.ckpt', '.safetensors')
        )

    use_from_pretrained = (
        pretrained_path
        and pretrained_path != '~'
        and not _is_single_weight_file(pretrained_path)
    )

    if use_from_pretrained:
        logging.info(f'Loading Video VAE ({model_name}) from pretrained: {pretrained_path}')
        vae = VAEClass.from_pretrained(pretrained_path)
    elif model_config:
        if pretrained_path and pretrained_path != '~':
            logging.info(
                f'pretrained_model_name_or_path={pretrained_path!r} is a single weight '
                f'file, not an HF directory. Building Video VAE ({model_name}) from '
                f'model_config={model_config!r}; weights will be loaded later from '
                f'vae_pretrained / video_vae_pretrained.'
            )
        if isinstance(model_config, dict):
            vae = VAEClass.from_config(model_config)
        elif isinstance(model_config, str):
            model_config = _resolve_config_path(model_config, config_dir)
            config_dict = VAEClass.load_config(model_config)
            vae = VAEClass.from_config(config_dict)
        else:
            raise ValueError(f"model_config must be a path or dict, got: {type(model_config)}")
    elif pretrained_path and pretrained_path != '~':
        raise FileNotFoundError(
            f"pretrained_model_name_or_path={pretrained_path!r} is a single weight file "
            f"(not an HF directory) and `model_config` is not set in the av_vae config. "
            f"Cannot build {model_name} architecture."
        )
    else:
        vae = VAEClass()

    z_dim = (
        getattr(vae, 'z_dim', None)
        or getattr(vae, 'embed_dim', None)
        or video_cfg.get('z_dim')
        or video_cfg.get('embed_dim')
    )
    logging.info(f'Video VAE built: model_name={model_name}, z_dim={z_dim}, '
                 f'temporal_compress={getattr(vae, "temporal_compress_factor", "?")}, '
                 f'spatial_compress={getattr(vae, "spatial_compress_factor", "?")}')
    return vae, z_dim


def _build_audio_vae(audio_cfg: dict):
    """Build DAC from the 'model.audio' section of AudioVideoVAE config."""
    from omnivae.models.audio_vae_dac.dac import DAC as DACModel

    vae = DACModel(
        encoder_dim=audio_cfg.get('encoder_dim', 64),
        encoder_rates=audio_cfg.get('encoder_rates', [2, 4, 8, 8]),
        latent_dim=audio_cfg.get('latent_dim'),
        decoder_dim=audio_cfg.get('decoder_dim', 1536),
        decoder_rates=audio_cfg.get('decoder_rates', [8, 8, 4, 2]),
        n_codebooks=audio_cfg.get('n_codebooks', 9),
        codebook_size=audio_cfg.get('codebook_size', 1024),
        codebook_dim=audio_cfg.get('codebook_dim', 8),
        quantizer_dropout=audio_cfg.get('quantizer_dropout', False),
        sample_rate=audio_cfg.get('sample_rate',
                                  audio_cfg.get('audio_sample_rate', 24000)),
        continuous=audio_cfg.get('continuous', True),
    )
    logging.info(f'Audio VAE built: latent_dim={vae.latent_dim}, hop_length={vae.hop_length}')
    return vae


def _build_contrastive_head(contrastive_cfg: dict, video_latent_dim: int, audio_latent_dim: int,
                            video_temporal_compress_factor: int):
    """Build LatentAVContrastiveHead for segment pooling only."""
    import inspect
    from omnivae.models.audio_video_vae.contrastive import LatentAVContrastiveHead

    cfg = dict(contrastive_cfg)
    for pop_key in ("enabled", "contrastive_type", "contrastive_use_mean",
                    "contrastive_grad_scale_video", "contrastive_grad_scale_audio",
                    "val_segment_num_negatives",
                    "val_segment_num_negative_videos", "val_global_num_negatives",
                    "video_latent_dim", "audio_latent_dim"):
        cfg.pop(pop_key, None)

    cfg['use_segment_loss'] = True
    cfg['skip_first_video_latent_frame'] = True
    cfg['video_temporal_compress_factor'] = video_temporal_compress_factor

    # Filter cfg against LatentAVContrastiveHead.__init__ signature to be robust
    # against extra fields in VAE checkpoint's contrastive config.
    try:
        sig = inspect.signature(LatentAVContrastiveHead.__init__)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values())
        if not accepts_var_kw:
            valid_keys = set(sig.parameters.keys()) - {'self'}
            dropped = {k: cfg[k] for k in list(cfg.keys()) if k not in valid_keys}
            for k in dropped:
                cfg.pop(k)
            if dropped:
                logging.info(f'Dropping unsupported contrastive kwargs: {list(dropped.keys())}')
    except (TypeError, ValueError):
        pass

    head = LatentAVContrastiveHead(
        video_latent_dim=video_latent_dim,
        audio_latent_dim=audio_latent_dim,
        **cfg,
    )
    logging.info(f'Contrastive head built for segment pooling '
                 f'(spatial={cfg.get("spatial_pool_mode", "mean")}, '
                 f'segment_temporal={cfg.get("segment_temporal_pool_mode", "mean")})')
    return head


class SynchformerVAE(nn.Module):
    """
    Synchronization model backed by WanVAE + DAC latent features.

    Uses LatentAVContrastiveHead for spatial aggregation and temporal segment
    pooling (frozen).  Only vproj, aproj, and GlobalTransformer are trained.
    """

    def __init__(
        self,
        av_vae_config: str,
        vproj,
        aproj,
        transformer,
        vae_pretrained: Optional[str] = None,
        source_vfps: int = 24,
        skip_temporal_pool: bool = False,
        n_audio_tokens: Optional[int] = None,
        model_target_vfps: Optional[int] = None,
        audio_merge_factor: int = 1,
        tail_extra_sec: float = 0.0,
        trim_tail_for_sync: bool = False,
        video_vae_pretrained: Optional[str] = None,
        audio_vae_pretrained: Optional[str] = None,
        freeze_contrastive_head: bool = True,
        init_contrastive_head_from_ckpt: bool = True,
        load_contrastive_head_from_external: bool = True,
        video_av_vae_config: Optional[str] = None,
        audio_av_vae_config: Optional[str] = None,
        contrastive_head_config: Optional[str] = None,
    ):
        super().__init__()
        self.skip_temporal_pool = skip_temporal_pool
        self.n_audio_tokens = n_audio_tokens
        self.audio_merge_factor = audio_merge_factor
        self.tail_extra_sec = float(tail_extra_sec)
        self.trim_tail_for_sync = bool(trim_tail_for_sync)
        self._freeze_contrastive_head = bool(freeze_contrastive_head)
        # When ``contrastive_head_config`` is provided, the contrastive head
        # comes from a custom structural spec and its weights must stay at
        # constructor-time random init.  Force-disable both ckpt-driven init
        # paths up front (before the rest of __init__ uses these flags).
        _norm = lambda p: None if (p is None or p == '' or p == '~') else _resolve_release_path(p)
        av_vae_config = _norm(av_vae_config)
        video_av_vae_config = _norm(video_av_vae_config)
        audio_av_vae_config = _norm(audio_av_vae_config)
        contrastive_head_config = _norm(contrastive_head_config)
        if contrastive_head_config is not None:
            if init_contrastive_head_from_ckpt:
                logging.info(
                    f'contrastive_head_config={contrastive_head_config!r} '
                    f'provided -> forcing init_contrastive_head_from_ckpt=False '
                    f'(contrastive head weights will be randomly initialized).'
                )
                init_contrastive_head_from_ckpt = False
            if load_contrastive_head_from_external:
                logging.info(
                    f'contrastive_head_config={contrastive_head_config!r} '
                    f'provided -> forcing load_contrastive_head_from_external=False '
                    f'(contrastive head weights will be randomly initialized).'
                )
                load_contrastive_head_from_external = False
        self._init_ct_from_ckpt = bool(init_contrastive_head_from_ckpt)
        self._load_ct_from_external = bool(load_contrastive_head_from_external)
        self._has_custom_ct_cfg = contrastive_head_config is not None
        self._diag_stats: Dict[str, float] = {}
        self._diag_log_interval = 50
        self._diag_step = 0

        # ---- Read AudioVideoVAE config(s) ----
        # Three (mutually compatible) ways to specify VAE / contrastive-head
        # structure:
        #   1) Unified ``av_vae_config``: one yaml with model.video / model.audio
        #      / model.contrastive (legacy path, fully backward compatible).
        #   2) Per-modality ``video_av_vae_config`` / ``audio_av_vae_config``:
        #      override model.video / model.audio respectively; missing
        #      sections fall back to the unified ``av_vae_config``.
        #   3) ``contrastive_head_config``: override model.contrastive only.
        #      May be either a full av_vae-style yaml (read ``model.contrastive``)
        #      or a yaml whose top-level keys are already the contrastive params.
        #      When set, contrastive-head weights are forced to stay random
        #      (init_contrastive_head_from_ckpt + load_contrastive_head_from_external
        #       were disabled above).
        unified_cfg = _load_av_vae_config(av_vae_config) if av_vae_config else None
        video_only_cfg = _load_av_vae_config(video_av_vae_config) if video_av_vae_config else None
        audio_only_cfg = _load_av_vae_config(audio_av_vae_config) if audio_av_vae_config else None

        if unified_cfg is not None:
            logging.info(f'Loaded unified av_vae_config: {av_vae_config}')
        if video_only_cfg is not None:
            logging.info(f'Loaded video_av_vae_config: {video_av_vae_config}')
        if audio_only_cfg is not None:
            logging.info(f'Loaded audio_av_vae_config: {audio_av_vae_config}')

        video_src = video_only_cfg or unified_cfg
        audio_src = audio_only_cfg or unified_cfg
        if video_src is None:
            raise ValueError(
                'No video VAE config available: pass either `av_vae_config` '
                '(unified) or `video_av_vae_config` (per-modality).'
            )
        if audio_src is None:
            raise ValueError(
                'No audio VAE config available: pass either `av_vae_config` '
                '(unified) or `audio_av_vae_config` (per-modality).'
            )
        video_cfg = video_src['model']['video']
        audio_cfg = audio_src['model']['audio']

        # Contrastive-head structure source:
        #   contrastive_head_config > av_vae_config > video_av_vae_config > audio_av_vae_config
        if contrastive_head_config is not None:
            ct_full = _load_av_vae_config(contrastive_head_config)
            if (isinstance(ct_full, dict) and 'model' in ct_full
                    and isinstance(ct_full['model'], dict)
                    and 'contrastive' in ct_full['model']):
                contrastive_cfg = dict(ct_full['model']['contrastive'])
                _ct_origin = f'{contrastive_head_config} (model.contrastive)'
            elif isinstance(ct_full, dict) and 'contrastive' in ct_full:
                contrastive_cfg = dict(ct_full['contrastive'])
                _ct_origin = f'{contrastive_head_config} (top-level contrastive)'
            else:
                contrastive_cfg = dict(ct_full) if isinstance(ct_full, dict) else {}
                _ct_origin = f'{contrastive_head_config} (raw top-level)'
            logging.info(f'Loaded contrastive_head_config from: {_ct_origin}')
        else:
            ct_src = unified_cfg or video_only_cfg or audio_only_cfg
            contrastive_cfg = dict(ct_src['model'].get('contrastive', {}))
            if video_only_cfg is not None and audio_only_cfg is not None and unified_cfg is None:
                _v_ct = video_only_cfg['model'].get('contrastive', {})
                _a_ct = audio_only_cfg['model'].get('contrastive', {})
                _skeys = (
                    'segment_temporal_pool_mode', 'global_temporal_pool_mode',
                    'spatial_pool_mode', 'segment_count', 'spatial_merge_factor',
                    'transformer_dim', 'transformer_nhead',
                    'spatial_transformer_layers', 'segment_transformer_layers',
                    'global_transformer_layers',
                )
                _vs = {k: _v_ct.get(k) for k in _skeys if k in _v_ct or k in _a_ct}
                _as_ = {k: _a_ct.get(k) for k in _skeys if k in _v_ct or k in _a_ct}
                if _vs != _as_:
                    logging.warning(
                        f'video_av_vae_config and audio_av_vae_config disagree on '
                        f'contrastive structure (video={_vs} vs audio={_as_}); '
                        f'using video_av_vae_config. Pass `contrastive_head_config` '
                        f'or `av_vae_config` to pin it explicitly.'
                    )

        # ``av_cfg`` is used for top-level fallback keys (target_fps, num_frames, ...).
        # Prefer unified, then video, then audio.
        av_cfg = unified_cfg or video_only_cfg or audio_only_cfg

        # ---- Merge structural contrastive params from checkpoint config ----
        # Skipped when ``contrastive_head_config`` is the authoritative source.
        _ckpt_data = None
        if vae_pretrained and vae_pretrained != '~':
            logging.info(f'Pre-loading checkpoint for config merge: {vae_pretrained}')
            _ckpt_data = torch.load(vae_pretrained, map_location='cpu', weights_only=False)
            if self._has_custom_ct_cfg:
                logging.info(
                    'Skipping contrastive structural-param merge from '
                    'vae_pretrained config (contrastive_head_config is '
                    'authoritative).'
                )
            else:
                _ckpt_contrastive = _ckpt_data.get('config', {}).get('model', {}).get('contrastive', {})
                _structural_keys = [
                    'segment_temporal_pool_mode', 'global_temporal_pool_mode',
                    'spatial_pool_mode', 'segment_count', 'spatial_merge_factor',
                    'transformer_dim', 'transformer_nhead',
                    'spatial_transformer_layers', 'segment_transformer_layers',
                    'global_transformer_layers', 'use_sdpa',
                    'segment_module_size', 'global_module_size', 'spatial_module_size',
                    'cnn_num_blocks_per_stage', 'cnn_kernel_size',
                ]
                _merged = {}
                for k in _structural_keys:
                    if k in _ckpt_contrastive and k not in contrastive_cfg:
                        contrastive_cfg[k] = _ckpt_contrastive[k]
                        _merged[k] = _ckpt_contrastive[k]
                if _merged:
                    logging.info(f'  Merged contrastive params from checkpoint: {_merged}')

        # ---- Video fps subsampling (source_vfps -> target_fps) ----
        target_fps = model_target_vfps or av_cfg.get('target_fps', 8)
        self.video_subsample_step = max(1, round(source_vfps / target_fps))
        logging.info(f'Video fps: source={source_vfps}, target={target_fps}, '
                     f'subsample_step={self.video_subsample_step}')

        # ---- Build encoders ----
        logging.info('Building Video VAE (WanVAE)...')
        self.video_vae, self.video_latent_dim = _build_video_vae(video_cfg)
        logging.info('Building Audio VAE (DAC)...')
        self.audio_vae = _build_audio_vae(audio_cfg)
        self.audio_latent_dim = self.audio_vae.latent_dim

        # ---- Tail-extra: precompute how many latent tokens to trim ----
        video_temporal_compress = getattr(self.video_vae, 'temporal_compress_factor', 4)
        audio_hop_length = getattr(self.audio_vae, 'hop_length', 512)
        audio_sample_rate = getattr(self.audio_vae, 'sample_rate', 48000)
        if self.tail_extra_sec > 0:
            self._tail_v_tokens = int(self.tail_extra_sec * target_fps / video_temporal_compress)
            self._tail_a_tokens = int(self.tail_extra_sec * audio_sample_rate / audio_hop_length)
            logging.info(f'Tail-extra: {self.tail_extra_sec}s -> '
                         f'{self._tail_v_tokens} video latent frames, '
                         f'{self._tail_a_tokens} audio latent frames '
                         f'(trim_tail_for_sync={self.trim_tail_for_sync})')
        else:
            self._tail_v_tokens = 0
            self._tail_a_tokens = 0

        # ---- Build contrastive head (for spatial pooling) ----
        logging.info('Building LatentAVContrastiveHead...')
        self.contrastive_head = _build_contrastive_head(
            contrastive_cfg,
            video_latent_dim=self.video_latent_dim,
            audio_latent_dim=self.audio_latent_dim,
            video_temporal_compress_factor=video_temporal_compress,
        )

        # ---- Compute latent_per_segment for proportional segment scaling (pool mode) ----
        self._latent_per_segment = 0.0
        self._tail_segments = 0
        if not skip_temporal_pool:
            _sc_list = getattr(self.contrastive_head, 'segment_count_list', None)
            if _sc_list:
                _valid = [sc for sc in _sc_list if sc is not None]
                _head_sc = max(_valid) if _valid else None
            else:
                _head_sc = getattr(self.contrastive_head, 'segment_count', None) or \
                           contrastive_cfg.get('segment_count', None)
                if isinstance(_head_sc, (list, tuple)):
                    _valid = [sc for sc in _head_sc if sc is not None]
                    _head_sc = max(_valid) if _valid else None
            _ckpt_cfg = (_ckpt_data or {}).get('config', {})
            _vae_nf = (
                _ckpt_cfg.get('data', {}).get('num_frames', None)
                or _ckpt_cfg.get('num_frames', None)
                or _ckpt_cfg.get('data', {}).get('train', {}).get('num_frames', None)
                or av_cfg.get('num_frames', None)
                or av_cfg.get('data', {}).get('num_frames', None)
                or av_cfg.get('data', {}).get('train', {}).get('num_frames', None)
            )
            if _head_sc and _head_sc > 0 and _vae_nf:
                _training_latent = (_vae_nf - 1) / video_temporal_compress
                self._latent_per_segment = _training_latent / _head_sc
                logging.info(
                    f'Pool mode segment scaling: '
                    f'vae_training=[nf={_vae_nf}, latent={_training_latent:.0f}, seg={_head_sc}], '
                    f'latent_per_segment={self._latent_per_segment:.2f}, '
                    f'tail_v_tokens={self._tail_v_tokens} '
                    f'(trim_tail_for_sync={self.trim_tail_for_sync}; '
                    f'runtime: total_seg=round(n_latent/{self._latent_per_segment:.1f}))'
                )
            else:
                logging.warning(
                    f'Cannot compute latent_per_segment (segment_count={_head_sc}, '
                    f'vae_num_frames={_vae_nf}). Pool mode will use _resolve_segment_count fallback.'
                )

        # ---- Load combined checkpoint weights ----
        if _ckpt_data is not None:
            self._load_vae_checkpoint_from_data(
                _ckpt_data,
                load_contrastive=self._init_ct_from_ckpt,
            )
            del _ckpt_data
        if not self._init_ct_from_ckpt:
            logging.info('contrastive_head will be RANDOMLY initialized '
                         '(init_contrastive_head_from_ckpt=False); '
                         'av_vae checkpoint contrastive_head.* keys are skipped')

        # ---- Optionally override VAE weights from external (independent) ckpts ----
        # When a user wants to keep only the contrastive_head from av_vae_pretrained
        # but use stand-alone video / audio VAE weights, supply these paths.
        # If BOTH external ckpts are given and ``load_contrastive_head_from_external``
        # is True, also load the contrastive_head's video-only submodules from the
        # video ckpt and audio-only submodules from the audio ckpt (shared params
        # like logit_scale stay as whatever ``vae_pretrained`` / random init left).
        _has_ext_video = bool(video_vae_pretrained and video_vae_pretrained != '~')
        _has_ext_audio = bool(audio_vae_pretrained and audio_vae_pretrained != '~')
        _ext_ct_both = self._load_ct_from_external and _has_ext_video and _has_ext_audio
        if _has_ext_video:
            logging.info(f'Overriding video VAE weights from external ckpt: {video_vae_pretrained}')
            self._load_external_video_vae(
                video_vae_pretrained,
                load_contrastive_head=_ext_ct_both,
            )
        if _has_ext_audio:
            logging.info(f'Overriding audio VAE weights from external ckpt: {audio_vae_pretrained}')
            self._load_external_audio_vae(
                audio_vae_pretrained,
                load_contrastive_head=_ext_ct_both,
            )
        if (_has_ext_video ^ _has_ext_audio) and self._load_ct_from_external:
            logging.info(
                'load_contrastive_head_from_external=True but only ONE of '
                'video_vae_pretrained / audio_vae_pretrained is set; skipping '
                'per-modality contrastive_head load to avoid half-mixed state.'
            )

        # ---- Freeze encoders (always); contrastive head freeze is configurable ----
        self.video_vae.requires_grad_(False)
        self.video_vae.eval()
        self.audio_vae.requires_grad_(False)
        self.audio_vae.eval()
        if self._freeze_contrastive_head:
            self.contrastive_head.requires_grad_(False)
            self.contrastive_head.eval()
        else:
            # Selectively unfreeze ONLY the submodules that are actually
            # exercised by SynchformerVAE.forward.  LatentAVContrastiveHead
            # also carries contrastive-loss-only modules (global pooling,
            # projection heads, logit_scale, ...) which never receive grad
            # here; leaving them with requires_grad=True triggers DDP's
            # "unused parameters" RuntimeError.
            self.contrastive_head.requires_grad_(False)
            _ct_trainable_attrs = (
                'spatial_transformer', 'spatial_attn_pool',
                'video_segment_conv', 'audio_segment_conv',
                'video_segment_temporal_transformer',
                'audio_segment_temporal_transformer',
            )
            _unfrozen: list[str] = []
            for attr in _ct_trainable_attrs:
                sub = getattr(self.contrastive_head, attr, None)
                if sub is not None and any(True for _ in sub.parameters()):
                    sub.requires_grad_(True)
                    _unfrozen.append(attr)
            n_train = sum(p.numel() for p in self.contrastive_head.parameters()
                          if p.requires_grad)
            n_total = sum(p.numel() for p in self.contrastive_head.parameters())
            logging.info(
                f'contrastive_head will be PARTIALLY TRAINED '
                f'(freeze_contrastive_head=False): '
                f'unfrozen submodules={_unfrozen}, '
                f'trainable params={n_train}/{n_total}'
            )

        # ---- Log segment_count info ----
        _head_sc = getattr(self.contrastive_head, 'segment_count', None)
        _head_sc_list = getattr(self.contrastive_head, 'segment_count_list', [])
        logging.info(f'Contrastive head segment_count={_head_sc}, '
                     f'segment_count_list={_head_sc_list}, '
                     f'latent_per_segment={self._latent_per_segment:.2f}')

        # ---- Infer projection input dims ----
        if skip_temporal_pool:
            v_proj_dim = self._infer_spatial_pool_dim()
            a_proj_dim = self.audio_latent_dim * self.audio_merge_factor
            logging.info(f'SynchformerVAE (no-pool): v_spatial_dim={v_proj_dim}, '
                         f'a_proj_dim={a_proj_dim} (latent_dim={self.audio_latent_dim} x merge={self.audio_merge_factor}), '
                         f'n_audio_tokens={n_audio_tokens or "all"}')
        else:
            v_proj_dim = self._infer_video_segment_dim()
            a_proj_dim = self._infer_audio_segment_dim()
            logging.info(f'SynchformerVAE: video_latent_dim={self.video_latent_dim}, '
                         f'audio_latent_dim={self.audio_latent_dim}, '
                         f'v_seg_dim={v_proj_dim}, a_seg_dim={a_proj_dim}')

        # ---- projection & transformer (override in_features dynamically) ----
        if OmegaConf.is_config(vproj):
            vproj = OmegaConf.to_container(vproj, resolve=True)
        if OmegaConf.is_config(aproj):
            aproj = OmegaConf.to_container(aproj, resolve=True)
        vproj['params']['in_features'] = v_proj_dim
        aproj['params']['in_features'] = a_proj_dim
        self.vproj = instantiate_from_config(vproj)
        self.aproj = instantiate_from_config(aproj)
        self.transformer = instantiate_from_config(transformer)

    # ------------------------------------------------------------------
    # dimension inference
    # ------------------------------------------------------------------
    def _infer_spatial_pool_dim(self) -> int:
        """Infer the output dim of contrastive head's spatial pooling only."""
        head = self.contrastive_head
        if head.spatial_pool_mode == "transformer":
            return head.spatial_transformer.d_model
        return self.video_latent_dim * (head.spatial_merge_factor ** 2)

    def _infer_video_segment_dim(self) -> int:
        """Infer the per-segment feature dim produced by contrastive head for video."""
        head = self.contrastive_head
        if getattr(head, 'video_segment_conv', None) is not None:
            return head.video_segment_conv.output_dim
        if getattr(head, 'video_segment_temporal_transformer', None) is not None:
            return head.video_segment_temporal_transformer.d_model
        if getattr(head, 'spatial_transformer', None) is not None:
            return head.spatial_transformer.d_model
        return self.video_latent_dim * (head.spatial_merge_factor ** 2)

    def _infer_audio_segment_dim(self) -> int:
        """Infer the per-segment feature dim produced by contrastive head for audio."""
        head = self.contrastive_head
        if getattr(head, 'audio_segment_conv', None) is not None:
            return head.audio_segment_conv.output_dim
        if getattr(head, 'audio_segment_temporal_transformer', None) is not None:
            return head.audio_segment_temporal_transformer.d_model
        return self.audio_latent_dim

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        vis: torch.Tensor,
        aud: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[str] = None,
        **kwargs,
    ):
        with torch.no_grad():
            video_latent = self._encode_video(vis)
            audio_latent = self._encode_audio(aud)

        # Drop the causal first latent frame (it encodes only 1 input frame
        # and is misaligned with the rest).  The data pipeline already skips
        # the corresponding audio, so no audio adjustment is needed here.
        video_latent = video_latent[:, :, 1:].detach()
        audio_latent = audio_latent.detach()

        # If contrastive_head is trainable, it MUST participate in autograd;
        # otherwise we keep the original no_grad+detach fast-path to save memory.
        ct_grad = not self._freeze_contrastive_head

        if self.skip_temporal_pool:
            # npool: each latent frame becomes one token. Keep tail trimming
            # opt-in so old tail1s checkpoints match their validation path.
            raw_v_tokens = video_latent.shape[2]
            raw_a_tokens = audio_latent.shape[2]
            if self.trim_tail_for_sync and self._tail_v_tokens > 0:
                if video_latent.shape[2] <= self._tail_v_tokens:
                    raise ValueError(
                        f'tail_extra_sec={self.tail_extra_sec} trims {self._tail_v_tokens} '
                        f'video latent frames, but only {video_latent.shape[2]} are available.'
                    )
                video_latent = video_latent[:, :, :-self._tail_v_tokens]
            if self.trim_tail_for_sync and self._tail_a_tokens > 0:
                if audio_latent.shape[2] <= self._tail_a_tokens:
                    raise ValueError(
                        f'tail_extra_sec={self.tail_extra_sec} trims {self._tail_a_tokens} '
                        f'audio latent frames, but only {audio_latent.shape[2]} are available.'
                    )
                audio_latent = audio_latent[:, :, :-self._tail_a_tokens]

            with torch.set_grad_enabled(ct_grad):
                vis_seq = self.contrastive_head._spatial_pool(video_latent)  # (B, D_sp, T')
            vis_seq = vis_seq.transpose(1, 2)  # (B, T', D_sp)
            if not ct_grad:
                vis_seq = vis_seq.detach()

            n_aud = self.n_audio_tokens or audio_latent.shape[-1]
            if audio_latent.shape[-1] != n_aud:
                aud_seq = F.adaptive_avg_pool1d(audio_latent, n_aud)
            else:
                aud_seq = audio_latent
            aud_seq = aud_seq.transpose(1, 2)  # (B, n_aud, D_a)
            # audio_latent is already detached from the audio VAE, so this op
            # has no trainable predecessors regardless of ct_grad.
            if not ct_grad:
                aud_seq = aud_seq.detach()

            if self.audio_merge_factor > 1:
                B, T_a, D_a = aud_seq.shape
                trim = T_a - T_a % self.audio_merge_factor
                aud_seq = aud_seq[:, :trim].reshape(
                    B, trim // self.audio_merge_factor, self.audio_merge_factor * D_a
                )
        else:
            # pool: pool the full latent sequence first. If trim_tail_for_sync is
            # enabled, temporal CNNs still see the right-side tail and we then
            # drop the corresponding output segments. The default is the legacy
            # validation behavior for existing tail1s checkpoints.
            raw_v_tokens = video_latent.shape[2]
            raw_a_tokens = audio_latent.shape[2]
            with torch.set_grad_enabled(ct_grad):
                if self._latent_per_segment > 0:
                    total_seg = max(1, round(video_latent.shape[2] / self._latent_per_segment))
                else:
                    total_seg = self.contrastive_head._resolve_segment_count(video_latent)
                vis_seq = self.contrastive_head._pool_video_segments(video_latent, total_seg)
                aud_seq = self.contrastive_head._pool_audio_segments(audio_latent, total_seg)
            if not ct_grad:
                vis_seq = vis_seq.detach()
                aud_seq = aud_seq.detach()
            tail_segments = 0
            if self.trim_tail_for_sync and self._tail_v_tokens > 0:
                if raw_v_tokens <= self._tail_v_tokens:
                    raise ValueError(
                        f'tail_extra_sec={self.tail_extra_sec} trims {self._tail_v_tokens} '
                        f'video latent frames, but only {raw_v_tokens} are available.'
                    )
                if self._latent_per_segment > 0:
                    tail_segments = round(self._tail_v_tokens / self._latent_per_segment)
                else:
                    tail_segments = round(total_seg * self._tail_v_tokens / raw_v_tokens)
                tail_segments = max(0, min(tail_segments, total_seg - 1))
            if tail_segments > 0:
                vis_seq = vis_seq[:, :-tail_segments]
                aud_seq = aud_seq[:, :-tail_segments]
            if self._diag_step == 0:
                logging.info(
                    f'Pool forward: video_latent={video_latent.shape[2]}, '
                    f'audio_latent={audio_latent.shape[2]}, '
                    f'total_seg={total_seg}, tail_seg={tail_segments}, '
                    f'final_seg={vis_seq.shape[1]}, ct_grad={ct_grad}, '
                    f'trim_tail_for_sync={self.trim_tail_for_sync}'
                )

        self._compute_diag_stats(vis_seq, aud_seq)

        vis_seq = self.vproj(vis_seq)
        aud_seq = self.aproj(aud_seq)

        logits = self.transformer(vis_seq, aud_seq)
        loss = self.compute_loss(logits, targets, loss_fn)
        return loss, logits

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_diag_stats(self, vis_seq: torch.Tensor, aud_seq: torch.Tensor):
        """Compute feature diagnostics every N steps (lightweight)."""
        self._diag_step += 1
        if self._diag_step % self._diag_log_interval != 1:
            return

        stats: Dict[str, float] = {}
        stats['seg_count'] = float(vis_seq.shape[1])

        # per-segment norms
        v_norms = vis_seq.float().norm(dim=-1)
        a_norms = aud_seq.float().norm(dim=-1)
        stats['vis/norm_mean'] = v_norms.mean().item()
        stats['vis/norm_std'] = v_norms.std().item()
        stats['aud/norm_mean'] = a_norms.mean().item()
        stats['aud/norm_std'] = a_norms.std().item()

        # inter-segment cosine similarity (measures diversity across segments)
        v_normed = F.normalize(vis_seq.float(), dim=-1)
        v_sim = torch.bmm(v_normed, v_normed.transpose(1, 2))
        S = v_sim.shape[1]
        mask = ~torch.eye(S, dtype=torch.bool, device=v_sim.device).unsqueeze(0).expand_as(v_sim)
        stats['vis/inter_seg_cos_sim'] = v_sim[mask].mean().item()

        a_normed = F.normalize(aud_seq.float(), dim=-1)
        a_sim = torch.bmm(a_normed, a_normed.transpose(1, 2))
        S_a = a_sim.shape[1]
        a_mask = ~torch.eye(S_a, dtype=torch.bool, device=a_sim.device).unsqueeze(0).expand_as(a_sim)
        stats['aud/inter_seg_cos_sim'] = a_sim[a_mask].mean().item()

        # feature magnitude
        stats['vis/feat_mean'] = vis_seq.float().mean().item()
        stats['vis/feat_std'] = vis_seq.float().std().item()
        stats['aud/feat_mean'] = aud_seq.float().mean().item()
        stats['aud/feat_std'] = aud_seq.float().std().item()

        self._diag_stats = stats

    # ------------------------------------------------------------------
    # encoding helpers
    # ------------------------------------------------------------------
    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, D_v, T', H', W')
        Uses streaming inference so WanVAE performs temporal compression.
        For T=41 input frames: 1 + (41-1)//4 = 11 latent frames.
        """
        if self.video_subsample_step > 1:
            video = video[:, :, ::self.video_subsample_step, :, :]
        with torch.cuda.amp.autocast(enabled=False):
            posterior = self.video_vae.encode(video.float(), streaming_inference=False)
            return posterior.mode()

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, 1, T_a) -> (B, D_a, T_l)"""
        with torch.cuda.amp.autocast(enabled=False):
            audio_padded = self.audio_vae.preprocess(audio.float())
            posterior, _, _, _, _ = self.audio_vae.encode(audio_padded)
            return posterior.mode()

    def compute_loss(self, logits, targets, loss_fn=None):
        if targets is None:
            return None
        if loss_fn is None or loss_fn == 'cross_entropy':
            logits = logits.float().clamp(-1e4, 1e4)
            return F.cross_entropy(logits, targets)
        raise NotImplementedError(f'Loss {loss_fn} not implemented')

    # ------------------------------------------------------------------
    # checkpoint loading
    # ------------------------------------------------------------------
    def _load_vae_checkpoint(self, ckpt_path: str):
        """Load a combined AudioVideoVAE checkpoint from file path."""
        logging.info(f'Loading VAE checkpoint from: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self._load_vae_checkpoint_from_data(ckpt)

    def _load_vae_checkpoint_from_data(self, ckpt: dict, load_contrastive: bool = True):
        """Load weights from an already-loaded checkpoint dict.

        Splits keys by prefix: video_vae.*, audio_vae.*, contrastive_head.*

        Args:
            ckpt: loaded checkpoint dict (may be wrapped in
                ``model_state_dict`` / ``state_dict`` / ``model``).
            load_contrastive: when False, contrastive_head.* keys in the ckpt
                are skipped, leaving the head with its constructor-time
                random initialization.
        """
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        video_sd, audio_sd, ct_sd = {}, {}, {}
        for k, v in state_dict.items():
            k = k.replace('module.', '', 1)
            if k.startswith('video_vae.'):
                video_sd[k[len('video_vae.'):]] = v
            elif k.startswith('audio_vae.'):
                audio_sd[k[len('audio_vae.'):]] = v
            elif k.startswith('contrastive_head.'):
                ct_sd[k[len('contrastive_head.'):]] = v

        if video_sd:
            r = self.video_vae.load_state_dict(video_sd, strict=False)
            logging.info(f'Video VAE load: missing={len(r.missing_keys)}, '
                         f'unexpected={len(r.unexpected_keys)}')
        else:
            logging.warning('No video_vae.* keys found in checkpoint')

        if audio_sd:
            r = self.audio_vae.load_state_dict(audio_sd, strict=False)
            logging.info(f'Audio VAE load: missing={len(r.missing_keys)}, '
                         f'unexpected={len(r.unexpected_keys)}')
        else:
            logging.warning('No audio_vae.* keys found in checkpoint')

        if not load_contrastive:
            logging.info(f'Skipping contrastive_head.* load from av_vae ckpt '
                         f'({len(ct_sd)} keys ignored)')
        elif ct_sd:
            r = self.contrastive_head.load_state_dict(ct_sd, strict=False)
            logging.info(f'Contrastive head load: missing={len(r.missing_keys)}, '
                         f'unexpected={len(r.unexpected_keys)}')
        else:
            logging.warning('No contrastive_head.* keys found in checkpoint '
                            '(pooling modules may be parameter-free)')

    @staticmethod
    def _extract_state_dict(ckpt_obj, sub_prefix: Optional[str] = None) -> Dict[str, Any]:
        """Pick the actual state_dict from common ckpt container layouts and
        optionally keep only keys under ``<sub_prefix>.`` (after stripping
        ``module.``).  Returns a dict ready to feed into ``load_state_dict``.
        """
        if isinstance(ckpt_obj, dict):
            if 'model_state_dict' in ckpt_obj:
                state_dict = ckpt_obj['model_state_dict']
            elif 'state_dict' in ckpt_obj:
                state_dict = ckpt_obj['state_dict']
            elif 'model' in ckpt_obj:
                state_dict = ckpt_obj['model']
            else:
                state_dict = ckpt_obj
        else:
            state_dict = ckpt_obj

        out: Dict[str, Any] = {}
        prefix = f'{sub_prefix}.' if sub_prefix else None
        # If a sub_prefix is requested, first check whether *any* key starts
        # with it; if not, treat the dict as already-stripped (e.g. raw
        # video VAE state_dict without the ``video_vae.`` prefix).
        has_prefixed = False
        if prefix is not None:
            for k in state_dict.keys():
                kk = k.replace('module.', '', 1)
                if kk.startswith(prefix):
                    has_prefixed = True
                    break

        for k, v in state_dict.items():
            kk = k.replace('module.', '', 1)
            if prefix is not None and has_prefixed:
                if kk.startswith(prefix):
                    out[kk[len(prefix):]] = v
            else:
                out[kk] = v
        return out

    # Prefixes of LatentAVContrastiveHead submodules that operate on a single
    # modality.  Used when loading contrastive_head weights from per-modality
    # external UniVAE ckpts.  Shared params (``logit_scale``,
    # ``global_logit_scale``) are intentionally NOT included on either side.
    _CT_VIDEO_PREFIXES = (
        'spatial_attn_pool.',
        'spatial_transformer.',
        'video_segment_conv.',
        'video_segment_temporal_transformer.',
        'video_global_temporal_transformer.',
        'video_segment_proj.',
        'video_global_proj.',
    )
    _CT_AUDIO_PREFIXES = (
        'audio_segment_conv.',
        'audio_segment_temporal_transformer.',
        'audio_global_temporal_transformer.',
        'audio_segment_proj.',
        'audio_global_proj.',
    )

    def _load_modality_contrastive_head(self, ckpt: dict, modality: str, src_path: str):
        """Load only one modality's submodules of ``self.contrastive_head`` from
        an arbitrary UniVAE-style ckpt.

        ``modality`` is ``'video'`` or ``'audio'``.  Keys that don't match the
        modality-specific prefix list are silently dropped, so this is safe to
        run against the video ckpt and the audio ckpt independently without
        them stepping on each other's weights.
        """
        ct_sd = self._extract_state_dict(ckpt, sub_prefix='contrastive_head')
        if not ct_sd:
            logging.warning(
                f'No contrastive_head.* keys found in external {modality} ckpt '
                f'({src_path}); skipping per-modality contrastive_head load.'
            )
            return

        prefixes = (self._CT_VIDEO_PREFIXES if modality == 'video'
                    else self._CT_AUDIO_PREFIXES)
        filtered = {k: v for k, v in ct_sd.items() if k.startswith(prefixes)}
        if not filtered:
            logging.warning(
                f'External {modality} ckpt has contrastive_head.* keys but none '
                f'match the {modality} submodule prefixes; nothing loaded.'
            )
            return

        r = self.contrastive_head.load_state_dict(filtered, strict=False)
        # ``unexpected`` here means: keys present in ``filtered`` but absent
        # from ``self.contrastive_head`` (e.g. submodules disabled at build
        # time).  ``missing`` is the full list of keys *not* in ``filtered``,
        # which is expected (we only loaded one modality), so don't log it.
        logging.info(
            f'External {modality} contrastive_head load: loaded={len(filtered)} keys '
            f'(unexpected={len(r.unexpected_keys)})'
        )

    def _load_external_video_vae(self, path: str, load_contrastive_head: bool = False):
        """Load video VAE weights from an external location.

        Supported formats (auto-detected):
        - directory with a HuggingFace-style ``from_pretrained`` layout
        - a ``.pt`` / ``.pth`` checkpoint file (raw state_dict or wrapped
          inside ``model_state_dict`` / ``state_dict`` / ``model`` and
          optionally prefixed with ``video_vae.``)

        When ``load_contrastive_head`` is True and ``path`` is a file, also
        loads the video-only submodules of ``self.contrastive_head`` from the
        ckpt's ``contrastive_head.*`` keys.  HF-style directories don't carry
        contrastive_head weights, so the flag is silently ignored there.
        """
        if os.path.isdir(path):
            logging.info(f'Loading external video VAE (HF dir): {path}')
            try:
                external = type(self.video_vae).from_pretrained(path)
            except Exception as e:
                raise RuntimeError(
                    f'Failed to load external video VAE from dir {path}: {e}'
                )
            ext_sd = external.state_dict()
            r = self.video_vae.load_state_dict(ext_sd, strict=False)
            del external
            if load_contrastive_head:
                logging.info(
                    'load_contrastive_head=True but external video VAE is an HF '
                    'directory (no contrastive_head weights); skipping.'
                )
        elif os.path.isfile(path):
            logging.info(f'Loading external video VAE (.pt): {path}')
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            ext_sd = self._extract_state_dict(ckpt, sub_prefix='video_vae')
            if not ext_sd:
                raise RuntimeError(
                    f'External video VAE ckpt is empty after key filtering: {path}'
                )
            r = self.video_vae.load_state_dict(ext_sd, strict=False)
            if load_contrastive_head:
                self._load_modality_contrastive_head(ckpt, 'video', path)
        else:
            raise FileNotFoundError(f'External video VAE path not found: {path}')

        logging.info(f'External Video VAE load: missing={len(r.missing_keys)}, '
                     f'unexpected={len(r.unexpected_keys)}')

    def _load_external_audio_vae(self, path: str, load_contrastive_head: bool = False):
        """Load audio VAE weights from an external ``.pt`` file.

        DAC has no HuggingFace-style ``from_pretrained`` layout in this code
        base, so a directory path is rejected.

        When ``load_contrastive_head`` is True, also loads the audio-only
        submodules of ``self.contrastive_head`` from the ckpt's
        ``contrastive_head.*`` keys.
        """
        if os.path.isdir(path):
            raise NotImplementedError(
                f'External audio VAE from a directory is not supported '
                f'(DAC uses raw .pt). Got: {path}'
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f'External audio VAE path not found: {path}')

        logging.info(f'Loading external audio VAE (.pt): {path}')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        ext_sd = self._extract_state_dict(ckpt, sub_prefix='audio_vae')
        if not ext_sd:
            raise RuntimeError(
                f'External audio VAE ckpt is empty after key filtering: {path}'
            )
        r = self.audio_vae.load_state_dict(ext_sd, strict=False)
        if load_contrastive_head:
            self._load_modality_contrastive_head(ckpt, 'audio', path)
        logging.info(f'External Audio VAE load: missing={len(r.missing_keys)}, '
                     f'unexpected={len(r.unexpected_keys)}')

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        key = 'transformer.pos_emb_cfg.pos_emb'
        if key in sd:
            weight_len = sd[key].shape[1]
            self_len = self.transformer.pos_emb_cfg.pos_emb.shape[1]
            if weight_len > self_len:
                sd[key] = sd[key][:, :self_len, :]
                logging.warning(f'Trimming pos_emb from {weight_len} to {self_len}')
            elif weight_len < self_len:
                raise ValueError(
                    f'Cannot load state_dict with shorter pos_emb ({weight_len} vs {self_len})')
        return super().load_state_dict(sd, strict)
