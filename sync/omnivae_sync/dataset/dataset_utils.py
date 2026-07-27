import csv
import os
import random
import shlex
from pathlib import Path
from glob import glob
import shutil
import logging

import torchaudio
import torchvision
from omnivae_sync.utils.utils import get_fixed_off_fname
import subprocess
import torch

from omnivae_sync.utils.utils import which_ffmpeg  # 根据你的项目结构调整导入路径


import tempfile as _tempfile
TMP_DIR = os.path.join(_tempfile.gettempdir(), 'synchformer_reencode')


def get_fixed_offsets(transforms, split, splits_path, dataset_name):
    '''dataset_name: `vggsound` or `lrs3`'''
    logging.info(f'Using fixed offset for {split}')
    vid2offset_params = {}
    fixed_offset_fname = get_fixed_off_fname(transforms, split)
    if fixed_offset_fname is None:
        raise ValueError('Cant find fixed offsets for given params. Perhaps you need to make it first?')
    fixed_offset_path = os.path.join(splits_path, f'fixed_offsets_{dataset_name}', fixed_offset_fname)
    fixed_offset_paths = sorted(glob(fixed_offset_path.replace(split, '*')))
    assert len(fixed_offset_paths) > 0, f'Perhaps: {fixed_offset_path} does not exist. Make fixed offsets'

    for fix_off_path in fixed_offset_paths:
        reader = csv.reader(open(fix_off_path))
        # k700_2020 has no header, and also `vstart` comes before `offset_sec`
        if dataset_name == 'k700_2020':
            header = ['path', 'vstart_sec', 'offset_sec', 'oos_target']
        else:
            header = next(reader)
        for line in reader:
            data = dict()
            for f, value in zip(header, line):
                if f == 'path':
                    v = value
                elif f == 'offset_sec':
                    data[f] = float(value)
                elif f in ['vstart_sec', 'v_start_sec']:
                    f = 'v_start_i_sec'
                    data[f] = float(value)
                elif f == 'oos_target':
                    data[f] = int(value)
                else:
                    data[f] = value
            # assert v not in vid2offset_params, 'otherwise, offs from other splits will override each other'

            # even if we have multiple splits (val=test), we want to make sure that the offsets are the same
            if v in vid2offset_params:
                assert all([vid2offset_params[v][k] == data[k] for k in data]), f'{v} isnt unique and vary'

            vid2offset_params[v] = data
    return vid2offset_params


def maybe_cache_file(path: os.PathLike):
    '''Motivation: if every job reads from a shared disk it`ll get very slow, consider an image can
    be 2MB, then with batch size 32, 16 workers in dataloader you`re already requesting 1GB!! -
    imagine this for all users and all jobs simultaneously.'''
    # checking if we are on cluster, not on a local machine
    if 'LOCAL_SCRATCH' in os.environ:
        cache_dir = os.environ.get('LOCAL_SCRATCH')
        # a bit ugly but we need not just fname to be appended to `cache_dir` but parent folders,
        # otherwise the same fnames in multiple folders will create a bug (the same input for multiple paths)
        cache_path = os.path.join(cache_dir, Path(path).relative_to('/'))
        if not os.path.exists(cache_path):
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            shutil.copyfile(path, cache_path)
        return cache_path
    else:
        return path

# version2
def get_video_and_audio(path, get_meta=False, start_sec=0, end_sec=None, do_preprocess=False,
                        target_vfps=25, target_afps=16000, target_size=256):
    orig_path = path
    path = maybe_cache_file(path)
    temp_path = None
    
    if do_preprocess:
        # 先检查是否需要预处理
        v, _, info = torchvision.io.read_video(str(path), pts_unit='sec')
        _, H, W, _ = v.shape
        
        need_vfps = info['video_fps'] != target_vfps
        need_afps = info['audio_fps'] != target_afps
        need_resize = min(H, W) != target_size
        
        if need_vfps or need_afps or need_resize:
            temp_path = _reencode_video(
                path, 
                target_vfps=target_vfps if need_vfps else None,
                target_afps=target_afps if need_afps else None,
                target_size=target_size if need_resize else None
            )
            path = temp_path
    
    try:
        # (Tv, 3, H, W) [0, 255, uint8]; (Ca, Ta)
        rgb, audio, meta = torchvision.io.read_video(str(path), start_sec, end_sec, 'sec', output_format='TCHW')
        if 'video_fps' not in meta.keys():
            print(f'No video fps for {orig_path}')
            meta['video_fps'] = target_vfps
        if 'audio_fps' not in meta.keys():
            print(f'No audio fps for {orig_path}')
            meta['audio_fps'] = target_afps
        if rgb.ndim != 4 or rgb.shape[0] == 0 or min(rgb.shape[2], rgb.shape[3]) < 2:
            raise RuntimeError(
                f'Invalid video tensor shape {tuple(rgb.shape)} for {orig_path} '
                f'(read from {"temp" if temp_path else "original"} path={path})'
            )
        # (Ta) <- (Ca, Ta)
        audio = audio.mean(dim=0)
        # FIXME: this is legacy format of meta as it used to be loaded by VideoReader.
        meta = {'video': {'fps': [meta['video_fps']]}, 'audio': {'framerate': [meta['audio_fps']]}}
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)
    
    if get_meta:
        return rgb, audio, meta
    return rgb, audio, meta


def _reencode_video(path, target_vfps=None, target_afps=None, target_size=None):
    """使用 ffmpeg 重新编码视频，只处理需要修改的部分
    
    Args:
        path: 输入视频路径
        target_vfps: 目标视频帧率，None 表示保持原样
        target_afps: 目标音频采样率，None 表示保持原样
        target_size: 目标短边尺寸，None 表示保持原样
    """
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    assert any([target_vfps, target_afps, target_size]), "At least one target must be specified"
    
    # 确保临时目录存在
    os.makedirs(TMP_DIR, exist_ok=True)
    
    # 使用唯一文件名避免冲突
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    
    # 构建文件名后缀，标明处理了哪些内容
    suffix_parts = []
    if target_vfps is not None:
        suffix_parts.append(f'{target_vfps}fps')
    if target_size is not None:
        suffix_parts.append(f'{target_size}side')
    if target_afps is not None:
        suffix_parts.append(f'{target_afps}hz')
    suffix = '_'.join(suffix_parts)
    
    new_path = Path(TMP_DIR) / f'{Path(path).stem}_{unique_id}_{suffix}.mp4'
    new_path = str(new_path)
    
    cmd_parts = [which_ffmpeg(), '-hide_banner', '-y', '-i', str(path)]
    
    # 构建视频滤镜
    vf_filters = []
    if target_vfps is not None:
        vf_filters.append(f'fps={target_vfps}')
    if target_size is not None:
        vf_filters.append(
            f"scale='iw*{target_size}/min(iw,ih)':'ih*{target_size}/min(iw,ih)'"
        )
        vf_filters.append("crop='trunc(iw/2)*2':'trunc(ih/2)*2'")
    
    if vf_filters:
        cmd_parts += ['-vf', ','.join(vf_filters)]
    else:
        cmd_parts += ['-c:v', 'copy']
    
    if target_afps is not None:
        cmd_parts += ['-ar', str(target_afps)]
    else:
        cmd_parts += ['-c:a', 'copy']
    
    cmd_parts.append(new_path)
    
    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(new_path) or os.path.getsize(new_path) == 0:
        if os.path.exists(new_path):
            os.remove(new_path)
        stderr_lines = result.stderr.strip().splitlines() if result.stderr else []
        stderr_head = '\n'.join(stderr_lines[:20])
        stderr_tail = '\n'.join(stderr_lines[-5:]) if len(stderr_lines) > 20 else ''
        stderr_msg = stderr_head + ('\n...\n' + stderr_tail if stderr_tail else '')
        raise RuntimeError(
            f'ffmpeg re-encode failed (exit={result.returncode}) for {path}. '
            f'stderr:\n{stderr_msg or "<no stderr>"}'
        )
    
    return new_path


def get_audio_stream(path, get_meta=False):
    '''Used only in feature extractor training'''
    path = str(Path(path).with_suffix('.wav'))
    path = maybe_cache_file(path)
    waveform, _ = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if get_meta:
        info = torchaudio.info(path)
        duration = info.num_frames / info.sample_rate
        meta = {'audio': {'duration': [duration], 'framerate': [info.sample_rate]}}
        return waveform, meta
    else:
        return waveform

def subsample_dataset(dataset: list, size_ratio: float, shuffle: bool = False):
    if size_ratio is not None and 0.0 < size_ratio < 1.0:
        logging.info(f'Subsampling dataset to {size_ratio}')
        # shuffling is important only during subsampling (sometimes paths are sorted by class)
        if shuffle:
            random.shuffle(dataset)
        cut_off = int(len(dataset) * size_ratio)
        # making sure that we have at least one example
        dataset = dataset[:max(1, cut_off)]
        logging.info(f'Subsampled dataset to {size_ratio} (size: {len(dataset)})')
    return dataset
