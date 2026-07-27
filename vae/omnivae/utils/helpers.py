import logging
import torchaudio
import os
import sys
import glob
import torch
import numpy as np
import re

try:
    import debugpy
except ImportError:
    debugpy = None

try:
    import lpips
except ImportError:
    lpips = None

from pathlib import Path

# from pydub import AudioSegment

def count_params_by_module(model_name, model):
    logging.info(f"Counting num_parameters of {model_name}:")
    
    param_stats = {}
    total_params = 0  # 统计总参数量
    total_requires_grad_params = 0  # 统计 requires_grad=True 的参数量
    total_no_grad_params = 0  # 统计 requires_grad=False 的参数量
    
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in param_stats:
            param_stats[module_name] = {'total': 0, 'requires_grad': 0, 'no_grad': 0}
        
        param_num = param.numel()
        param_stats[module_name]['total'] += param_num
        total_params += param_num
        
        if param.requires_grad:
            param_stats[module_name]['requires_grad'] += param_num
            total_requires_grad_params += param_num
        else:
            param_stats[module_name]['no_grad'] += param_num
            total_no_grad_params += param_num
    
    # 计算每列的最大宽度
    max_module_name_length = max(len(module) for module in param_stats)
    max_param_length = max(len(f"{stats['total'] / 1e6:.2f}M") for stats in param_stats.values())
    
    # 输出每个模块的参数统计信息
    for module, stats in param_stats.items():
        logging.info(f"\t{module:<{max_module_name_length}}: "
                     f"Total: {stats['total'] / 1e6:<{max_param_length}.2f}M, "
                     f"Requires Grad: {stats['requires_grad'] / 1e6:<{max_param_length}.2f}M, "
                     f"No Grad: {stats['no_grad'] / 1e6:<{max_param_length}.2f}M")
    
    # 输出总参数统计信息
    logging.info(f"\tTotal parameters: {total_params / 1e6:.2f}M parameters")
    logging.info(f"\tRequires Grad parameters: {total_requires_grad_params / 1e6:.2f}M parameters")
    logging.info(f"\tNo Grad parameters: {total_no_grad_params / 1e6:.2f}M parameters")
    logging.info(f"################################################################")

# 有 BUG, 用了这个，会向磁盘中写很多东西
# def load_audio_format_same_as_torchaudio_load(audio_path):
#     # 获取文件后缀
#     ext = os.path.splitext(audio_path)[1].lower()
    
#     if ext != '.m4a':
#         # 使用 torchaudio 读取非 .m4a 格式音频
#         waveform, sample_rate = torchaudio.load(audio_path)
#     else:
#         # 使用 pydub 读取 .m4a 格式音频
#         audio = AudioSegment.from_file(audio_path, format='m4a')
        
#         # 获取音频参数
#         sample_rate = audio.frame_rate
#         channels = audio.channels
#         assert channels in [1, 2]
        
#         # 转换为 numpy 数组
#         samples = np.array(audio.get_array_of_samples())

#         # 获取音频的位深度，16-bit 或 32-bit 等
#         sample_width = audio.sample_width  # 1 byte = 8 bits, so sample_width = 2 for 16-bit
#         max_int_value = 2 ** (sample_width * 8 - 1)  # 最大值

#         # 将样本数据归一化到 [-1, 1]
#         samples = samples.astype(np.float32) / max_int_value
        
#         # 如果是双通道音频，将 samples 重新排列为两个通道
#         if channels == 2:
#             samples = samples.reshape((-1, 2))  # 将双通道数据分开

#         # 转换为 PyTorch Tensor
#         waveform = torch.from_numpy(samples).float()
        
#         # 如果是单通道的情况下，需要调整为 (1, num_frames)
#         if channels == 1:
#             waveform = waveform.unsqueeze(0)  # 添加一个维度，变成 (1, num_frames)
#         else:
#             waveform = waveform.t()  # 如果是双通道，保证维度是 (channels, num_frames)

#     return waveform, sample_rate

def load_and_resample_audio(audio_path, target_sample_rate):
    wav, raw_sample_rate = torchaudio.load(audio_path) # (1, T)   tensor 
    if raw_sample_rate != target_sample_rate:   
        wav = torchaudio.functional.resample(wav, raw_sample_rate, target_sample_rate) # tensor 
    return wav.squeeze()

def set_logging():
    rank = os.environ.get("RANK", 0)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format=f"%(asctime)s [RANK {rank}] (%(module)s:%(lineno)d) %(levelname)s : %(message)s",
        force=True
    )
    
def waiting_for_debug(ip, port):
    if debugpy is None:
        raise ImportError("debugpy is required for waiting_for_debug(); install debugpy or disable debug attach.")
    rank = os.environ.get("RANK", "0")
    debugpy.listen((ip, port)) # 把这边的 localhost 改成集群节点 ip
    print(f"[rank = {rank}] Waiting for debugger attach...")
    debugpy.wait_for_client()
    print(f"[rank = {rank}] Debugger attached")
    
def load_audio(audio_path, target_sample_rate):
    # Load audio file, wav shape: (channels, time)
    wav, raw_sample_rate = torchaudio.load(audio_path)
    
    # If multi-channel, convert to mono by averaging across channels
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # Average across channels, keep channel dim
    
    # Resample if necessary
    if raw_sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, raw_sample_rate, target_sample_rate)
    
    # Convert to numpy, add channel dimension, then back to tensor with desired shape
    wav = np.expand_dims(wav.squeeze(0).numpy(), axis=1)  # Shape: (time, 1)
    wav = torch.tensor(wav).reshape(1, 1, -1)  # Shape: (1, 1, time)
    
    return wav

def save_audio(audio_outpath, audio_out, sample_rate):
    torchaudio.save(
        audio_outpath, 
        audio_out, 
        sample_rate=sample_rate, 
        encoding='PCM_S', 
        bits_per_sample=16
    )
    logging.info(f"success save audio at {audio_outpath}")
    
def find_audio_files(input_dir):
    audio_extensions = ['*.flac', '*.mp3', '*.wav']
    audios_input = []
    for ext in audio_extensions:
        audios_input.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    logging.info(f"Find {len(audios_input)} audios at {input_dir}")
    return sorted(audios_input)

def normalize_text(text):
    # 移除所有标点符号（包括英文和中文标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 转换为小写（对英文有效，中文无影响）
    text = text.lower()
    # 去除多余空格
    text = ' '.join(text.split())
    return text

def resample_on_gpu(audio, orig_freq, target_freq, device="cuda"):
    """
    使用 GPU 加速重采样音频。
    
    参数:
        audio (ndarray): 输入音频（numpy 数组）
        orig_freq (int): 原始采样率
        target_freq (int): 目标采样率
        device (str): 设备类型，默认为 "cuda"
    
    返回:
        ndarray: 重采样后的音频
    """
    # 转换为 PyTorch 张量并移到 GPU
    audio_tensor = torch.from_numpy(audio).float().to(device)
    
    # 如果是多声道，添加通道维度
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # [1, T]
    else:
        audio_tensor = audio_tensor.T  # [C, T]
    
    # GPU 上重采样
    resampled_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=orig_freq, new_freq=target_freq
    )
    
    # 转换回 numpy 数组
    resampled_audio = resampled_tensor.cpu().numpy()
    
    # 如果是单声道，去掉通道维度
    if resampled_audio.shape[0] == 1:
        resampled_audio = resampled_audio[0]
    else:
        resampled_audio = resampled_audio.T  # 恢复为 [T, C]
    
    return resampled_audio


# ==================== Training Utilities ====================

def suppress_useless_warnings():
    """Ignore deprecation warnings coming from torchvision model helpers.

    The torchvision models module logs UserWarning messages when callers use
    the deprecated ``pretrained`` argument or pass non-enum `weights` values.
    Calling this function early (e.g. in ``train``) ensures the warnings are
    filtered and do not clutter logs.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torchvision\.models\._utils",
    )

    # ignore warnings from the amp.autocast API deprecation
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
    )
    # also silence GradScaler deprecation
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*torch\.cuda\.amp\.GradScaler.*deprecated.*",
    )

    # suppress scipy.ndimage.filters deprecation from moviepy
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*sobel.*scipy\.ndimage.*deprecated.*",
    )

    # suppress dateutil.tz utcfromtimestamp deprecation
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*utcfromtimestamp.*deprecated.*",
    )

    # suppress torchvision video deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*video decoding and encoding.*deprecated.*torchcodec.*",
    )

    # suppress multiprocessing fork() warning in multi-threaded context
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*fork\(\).*multi-threaded.*deadlocks.*",
    )

    # suppress pynvml deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*pynvml.*deprecated.*nvidia-ml-py.*",
    )

    # suppress google protobuf PyType_Spec deprecation
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*PyType_Spec.*metaclass.*custom tp_new.*",
    )

    # suppress pkg_resources deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*pkg_resources.*deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*pkg_resources\.declare_namespace.*",
    )

    # suppress sentry_sdk Hub deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*sentry_sdk\.Hub.*deprecated.*",
    )
    # SentryHubDeprecationWarning 可能是特殊的 Warning 类
    warnings.filterwarnings(
        "ignore",
        message=r".*sentry_sdk\.Hub.*deprecated.*",
    )


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(rank, log_dir=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    class _TeeStream:
        def __init__(self, primary, secondary):
            self._primary = primary
            self._secondary = secondary

        def write(self, data):
            try:
                self._primary.write(data)
            except Exception:
                pass
            try:
                self._secondary.write(data)
            except Exception:
                pass
            if "\n" in data or "\r" in data:
                self.flush()
            return len(data)

        def flush(self):
            try:
                self._primary.flush()
            except Exception:
                pass
            try:
                self._secondary.flush()
            except Exception:
                pass

        def isatty(self):
            return bool(getattr(self._primary, "isatty", lambda: False)())

        def fileno(self):
            return self._primary.fileno()

        def __getattr__(self, name):
            return getattr(self._primary, name)
    
    # use plain formatter without colors
    formatter = logging.Formatter(
        f"[RANK {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 控制台处理器：将日志写到 stdout (进程间用 stderr 记录速度等)
    stream_handler = logging.StreamHandler(stream=sys.__stdout__)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(stream_handler)
    
    # 文件处理器：为每个 rank 创建两个日志文件，分别记录 stdout/stderr
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = log_dir / f"rank_{rank}_stdout.log"
        stderr_file = log_dir / f"rank_{rank}_stderr.log"

        # stdout file handler receives the same records as the console handler
        out_handler = logging.FileHandler(stdout_file, mode="a", encoding="utf-8")
        out_handler.setLevel(logging.DEBUG)
        out_formatter = logging.Formatter(
            f"[RANK {rank}] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        out_handler.setFormatter(out_formatter)
        logger.addHandler(out_handler)

        # redirect raw streams so that prints/progress bars go to appropriate files
        try:
            stdout_fh = open(stdout_file, "a", encoding="utf-8", buffering=1)
            stderr_fh = open(stderr_file, "a", encoding="utf-8", buffering=1)
            if not isinstance(sys.stdout, _TeeStream):
                sys.stdout = _TeeStream(sys.stdout, stdout_fh)
            if not isinstance(sys.stderr, _TeeStream):
                sys.stderr = _TeeStream(sys.stderr, stderr_fh)
        except Exception:
            pass
    
    return logger


def merge_config_args(args, parser):
    """Merge YAML config into argparse Namespace; CLI overrides config."""
    import yaml
    from pathlib import Path
    
    if getattr(args, 'config', None) is None:
        return args
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    for key, val in cfg.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        default = parser.get_default(key)
        if current == default or current is None:
            setattr(args, key, val)
    return args


def _normalize_dataset_paths(dataset_dir_or_paths):
    """Normalize dataset paths to dict format."""
    import json
    from pathlib import Path
    
    if dataset_dir_or_paths is None:
        return None
    if isinstance(dataset_dir_or_paths, str):
        try:
            parsed = json.loads(dataset_dir_or_paths)
            dataset_dir_or_paths = parsed
        except Exception:
            pass
    if isinstance(dataset_dir_or_paths, list):
        return {Path(p).stem: p for p in dataset_dir_or_paths}
    if isinstance(dataset_dir_or_paths, dict):
        return dataset_dir_or_paths
    return {Path(dataset_dir_or_paths).stem: dataset_dir_or_paths}


def _normalize_weights(raw_weights, dataset_names):
    """Normalize weights to dict format."""
    import json
    
    if raw_weights is None:
        return None
    if isinstance(raw_weights, str):
        try:
            raw_weights = json.loads(raw_weights)
        except Exception:
            pass
    if isinstance(raw_weights, list):
        if len(raw_weights) != len(dataset_names):
            raise ValueError("Length of weights must match number of datasets.")
        return {name: w for name, w in zip(dataset_names, raw_weights)}
    if isinstance(raw_weights, dict):
        return raw_weights
    raise TypeError(f"Unsupported weights type: {type(raw_weights)}")


def _parse_video_configs(raw, default_frames: int, default_resolution: int, logger=None, name: str = "eval"):
    """
    Parse video configs from list/tuple/dict or string.
    Supported examples:
      - [[33, 256, 256], [45, 512, 512]]
      - (33, 256, 256)
      - "(33, 256, 256) (45, 512, 512)"
    Returns list of dicts: {frames, resolution, tag}
    """
    import ast
    
    def _default():
        return [
            {
                "frames": int(default_frames),
                "resolution": int(default_resolution),
                "tag": f"f{int(default_frames)}_r{int(default_resolution)}",
            }
        ]

    if raw is None:
        return _default()

    cfgs = None
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return _default()
        if len(raw) == 3 and all(isinstance(v, (int, float)) for v in raw):
            cfgs = [raw]
        else:
            cfgs = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return _default()
        if not s.startswith("["):
            s = s.replace(") (", "), (")
            s = f"[{s}]"
        try:
            cfgs = ast.literal_eval(s)
        except Exception as e:
            if logger is not None:
                logger.warning(f"Failed to parse {name}_video_configs from string: {e}")
            return _default()
    elif isinstance(raw, dict):
        if "num_frames" in raw and "resolution" in raw:
            cfgs = [raw]
        else:
            if logger is not None:
                logger.warning(f"Unsupported {name}_video_configs dict schema: {raw}")
            return _default()
    else:
        if logger is not None:
            logger.warning(f"Unsupported {name}_video_configs type: {type(raw)}")
        return _default()

    if isinstance(cfgs, tuple):
        cfgs = [cfgs]

    results = []
    for cfg in cfgs:
        frames = None
        resolution = None
        if isinstance(cfg, (list, tuple)) and len(cfg) >= 3:
            frames = int(cfg[0])
            h = int(cfg[1])
            w = int(cfg[2])
            resolution = h
            if w != h and logger is not None:
                logger.warning(
                    f"{name}_video_configs uses non-square size ({h}, {w}); using resolution={h}."
                )
        elif isinstance(cfg, dict) and "num_frames" in cfg and "resolution" in cfg:
            frames = int(cfg["num_frames"])
            resolution = int(cfg["resolution"])
        else:
            if logger is not None:
                logger.warning(f"Skip invalid {name}_video_configs entry: {cfg}")
            continue

        results.append(
            {
                "frames": frames,
                "resolution": resolution,
                "tag": f"f{frames}_r{resolution}",
            }
        )

    return results if results else _default()


def _to_thwc_uint8(video):
    """Convert CTHW/TCHW/THWC video to uint8 THWC for saving."""
    from typing import Union
    from pathlib import Path
    
    v = torch.as_tensor(video).detach().cpu()
    if v.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tuple(v.shape)}")
    if v.shape[0] in (1, 3):
        v = v.permute(1, 2, 3, 0)  # CTHW -> THWC
    elif v.shape[1] in (1, 3):
        v = v.permute(0, 2, 3, 1)  # TCHW -> THWC
    # otherwise assume already THWC
    if v.dtype.is_floating_point:
        if float(v.min()) < 0:
            v = (v.clamp(-1.0, 1.0) + 1.0) * 0.5
        else:
            v = v.clamp(0.0, 1.0)
        v = (v * 255.0).round().to(torch.uint8)
    elif v.dtype != torch.uint8:
        v = v.to(torch.uint8)
    return v.cpu()


def save_video_tensor(video, path, fps: int = 10):
    """Save torch tensor as video file."""
    from torchvision.io import write_video
    from pathlib import Path
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vid_uint8 = _to_thwc_uint8(video)
    write_video(str(path), vid_uint8, fps=fps)


def save_numpy_video(video_array, path, fps: int = 10):
    """Save numpy array as video file."""
    from torchvision.io import write_video
    from pathlib import Path
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vid_uint8 = _to_thwc_uint8(video_array)
    write_video(str(path), vid_uint8, fps=fps)


def save_video_list_to_dir(videos: list, output_dir, prefix: str, fps: int = 10):
    """Save list of videos to directory."""
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, video_np in enumerate(videos):
        save_numpy_video(video_np, output_dir / f"{prefix}_{idx:03d}.mp4", fps=fps)


def dump_eval_groundtruth(dataset, output_dir, logger, fps: int = 10, max_samples: int = None):
    """Dump evaluation ground truth videos."""
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx in range(len(dataset)):
        if max_samples is not None and saved >= max_samples:
            break
        sample = dataset[idx]
        if isinstance(sample, dict):
            video = sample.get('video')
            file_name = sample.get('file_name') or f"sample_{idx}.mp4"
        else:
            continue
        if video is None:
            continue
        try:
            save_video_tensor(video, output_dir / file_name, fps=fps)
            saved += 1
        except Exception as e:
            logger.warning(f"Failed to save eval sample {idx}: {e}")
    logger.info(f"Saved {saved} eval samples to {output_dir}")


def ddp_setup(args=None):
    """Setup DDP (Distributed Data Parallel)."""
    import torch.distributed as dist
    from datetime import timedelta
    
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    timeout_minutes = None
    if args is not None:
        timeout_minutes = getattr(args, "ddp_timeout_minutes", None)
    if timeout_minutes is None:
        timeout_minutes = int(os.environ.get("TORCH_DDP_TIMEOUT_MIN", "10"))

    # Important: init_process_group timeout controls the ProcessGroupNCCL watchdog
    # for collectives (e.g., barrier/all_gather_object internal collectives).
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=int(timeout_minutes)))


def _patch_dataparallel_to_single_gpu(device_id: int):
    """
    Context manager to patch DataParallel to use single GPU.
    
    In DDP each process can see all GPUs. Some metric code wraps models with
    `torch.nn.DataParallel()` (default: all visible GPUs), which can break/oom.
    This context patches DataParallel to default to a single GPU.
    """
    from contextlib import contextmanager
    
    @contextmanager
    def _context():
        orig_dp = torch.nn.DataParallel

        class _SingleDeviceDataParallel(orig_dp):
            def __init__(self, module, device_ids=None, output_device=None, *args, **kwargs):
                if device_ids is None:
                    device_ids = [device_id]
                if output_device is None and isinstance(device_ids, (list, tuple)) and len(device_ids) > 0:
                    output_device = device_ids[0]
                super().__init__(
                    module,
                    device_ids=device_ids,
                    output_device=output_device,
                    *args,
                    **kwargs,
                )

        torch.nn.DataParallel = _SingleDeviceDataParallel
        try:
            yield
        finally:
            torch.nn.DataParallel = orig_dp
    
    return _context()


def check_unused_params(model):
    """Check for unused parameters in model."""
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params


class StreamingTrainState:
    """Track training progress and dataset streaming state for resumption."""
    def __init__(self):
        self.step: int = 0
        self.dataset_state_dict = {
            "consumed_samples": None,
            "rng_state": None,
            "used_epochs": None,
            "dataset_state_dict": {},
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.dataset_state_dict = state_dict["dataset_state_dict"]
        
    def state_dict(self):
        return {
            "step": self.step,
            "dataset_state_dict": self.dataset_state_dict,
        }

    def update(self, state_dict = None):
        self.step += 1
        if state_dict is None:
            return
        self.dataset_state_dict["consumed_samples"] = state_dict.get("consumed_samples")
        self.dataset_state_dict["used_epochs"] = state_dict.get("used_epochs")
        self.dataset_state_dict["rng_state"] = state_dict.get("rng_state")
        self.dataset_state_dict["dataset_state_dict"] = state_dict.get("dataset_state_dict", {})


def set_requires_grad_optimizer(optimizer, requires_grad):
    """Set requires_grad for all parameters in optimizer."""
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad


def total_params(model):
    """Get total number of trainable parameters in millions."""
    total_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params_count / 1e6
    return int(total_params_in_millions)


def get_exp_name(args):
    """Generate experiment name from args."""
    return f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}-sr{args.sample_rate}-fr{args.num_frames}"


def set_train(modules):
    """Set modules to training mode."""
    for module in modules:
        module.train()


def set_eval(modules):
    """Set modules to eval mode."""
    for module in modules:
        module.eval()


def set_modules_requires_grad(modules, requires_grad):
    """Set requires_grad for modules."""
    for module in modules:
        module.requires_grad_(requires_grad)


def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    train_state_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
):
    """Save training checkpoint."""
    from pathlib import Path
    
    filepath = Path(checkpoint_dir) / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
            "train_state": train_state_state,
        },
        filepath,
    )
    return filepath
