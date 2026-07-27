import json
import logging
from pathlib import Path
import os
import math
import random
import fcntl

import torch

from omnivae_sync.dataset.dataset_utils import get_video_and_audio


class JsonlDataset(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 metadata_path,
                 transforms=None,
                 to_filter_bad_examples=False,
                 bad_examples_path=None,
                 do_preprocess=True,
                 vfps=25,
                 afps=16000,
                 size_before_crop=256,
                 input_size=224,
                 min_duration=7.5,
                 padding_to_min_duration=True):
        super().__init__()
        self.vfps = vfps
        self.afps = afps
        self.size_before_crop = size_before_crop
        self.input_size = input_size
        self.max_clip_len_sec = None
        self.split = split
        self.metadata_path = metadata_path
        self.transforms = transforms
        self.to_filter_bad_examples = to_filter_bad_examples
        self.bad_examples_path = bad_examples_path
        self.do_preprocess = do_preprocess
        self.min_duration = min_duration
        self.padding_to_min_duration = padding_to_min_duration
        
        # 输出所有参数
        logging.info(f'JsonlDataset parameters:')
        logging.info(f'split: {self.split}')
        logging.info(f'metadata_path: {self.metadata_path}')
        logging.info(f'transforms: {self.transforms}')
        logging.info(f'to_filter_bad_examples: {self.to_filter_bad_examples}')
        logging.info(f'bad_examples_path: {self.bad_examples_path}')
        logging.info(f'do_preprocess: {self.do_preprocess}')
        logging.info(f'vfps: {self.vfps}')
        logging.info(f'afps: {self.afps}')
        logging.info(f'size_before_crop: {self.size_before_crop}')
        logging.info(f'input_size: {self.input_size}')
        logging.info(f'max_clip_len_sec: {self.max_clip_len_sec}')
        logging.info(f'min_duration: {self.min_duration}')
        logging.info(f'padding_to_min_duration: {self.padding_to_min_duration}')
        # 处理 bad_examples_path 可能为 None 的情况
        if bad_examples_path:
            os.makedirs(bad_examples_path, exist_ok=True)
            self.video_too_short_path = os.path.join(bad_examples_path, 'video_too_short.txt')
            self.audio_too_short_path = os.path.join(bad_examples_path, 'audio_too_short.txt')
        else:
            self.video_too_short_path = None
            self.audio_too_short_path = None
        
        # 加载坏样本列表
        self.bad_examples = self.load_bad_examples() if to_filter_bad_examples else set()

        # 读取 JSONL 文件
        self.lines = self.load_lines(metadata_path)
        logging.info(f'Loaded {len(self.lines)} lines from {metadata_path}')

    def load_lines(self, path):
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def load_bad_examples(self):
        bad = set()
        if not self.bad_examples_path:
            return bad
        bad_examples_dir = Path(self.bad_examples_path)
        if bad_examples_dir.exists():
            for p in sorted(bad_examples_dir.glob('*.txt')):
                bad.update(open(p).read().splitlines())
        return bad

    def parse_line(self, index):
        line = self.lines[index]
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            logging.error(f'Line {index}: JSON decode error: {e}')
            return None

    def is_bad_example(self, item_meta):
        if not self.to_filter_bad_examples or not self.bad_examples:
            return False
        video_path = item_meta.get('video_path', '')
        return video_path in self.bad_examples

    def safe_write(self, filepath, content):
        """线程安全的文件写入"""
        if not filepath:
            return
        with open(filepath, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(content + '\n')
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def __getitem__(self, index):
        max_retries = 10
        
        for attempt in range(max_retries):
            current_index = index if attempt == 0 else random.randint(0, len(self) - 1)
            
            # try:
            item_meta = self.parse_line(current_index)
            if item_meta is None or self.is_bad_example(item_meta):
                continue
            
            path = item_meta['video_path']
            rgb, audio, meta = self.load_media(path)
            if rgb is None or audio is None:
                logging.warning(f'Failed to load media for {path}, choosing random next item...')
                continue
            item = self.make_datapoint(item_meta, rgb, audio, meta)
            if self.transforms:
                try:
                    item = self.transforms(item) # video: [Num_Seg, T, C, H, W], audio: [Num_Seg, T, F]
                except Exception as e:
                    logging.warning(f'Failed to apply transforms to item {path}: {type(e).__name__}: {e}')
                    continue
            return item        
        
        logging.error(f'Failed after {max_retries} retries for index {index}')
        return None

    def make_datapoint(self, item_meta, rgb, audio, meta):
        path = item_meta['video_path']
        return {
            'video': rgb,
            'audio': audio,
            'meta': meta,
            'path': str(path),
            'split': self.split,
            'item_meta': {"video_path": item_meta['video_path']},
            'targets': {'vggsound_target': {}, 'vggsound_label': {}},
        }

    def load_media(self, path):
        rgb, audio, meta = get_video_and_audio(
            path,
            get_meta=True,
            start_sec=0,
            end_sec=self.max_clip_len_sec,
            do_preprocess=self.do_preprocess,
            target_vfps=self.vfps,
            target_afps=self.afps,
            target_size=self.size_before_crop
        )
        if rgb is None or audio is None:
            return None, None, None
        
        if self.padding_to_min_duration:
            min_video_frames = math.ceil(self.min_duration * self.vfps)
            min_audio_samples = math.ceil(self.min_duration * self.afps)
            
            if rgb.shape[0] < min_video_frames:
                padding_frames = min_video_frames - rgb.shape[0]
                rgb = torch.cat([rgb, rgb[-1:].repeat(padding_frames, 1, 1, 1)], dim=0)
                logging.info(f'Video too short, padded: {path}')
                if self.video_too_short_path and path not in self.bad_examples:
                    self.safe_write(self.video_too_short_path, path)

            if audio.shape[0] < min_audio_samples:
                padding_samples = min_audio_samples - audio.shape[0]
                audio = torch.cat([audio, audio[-1:].repeat(padding_samples)], dim=0)
                logging.info(f'Audio too short, padded: {path}')
                if self.audio_too_short_path and path not in self.bad_examples:
                    self.safe_write(self.audio_too_short_path, path)

        return rgb, audio, meta

    def __len__(self):
        return len(self.lines)
