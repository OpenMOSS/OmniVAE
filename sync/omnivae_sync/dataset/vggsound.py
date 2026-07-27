import csv
import logging
import os
import random
from collections import Counter
from glob import glob
from pathlib import Path

import torch

from omnivae_sync.dataset.dataset_utils import get_fixed_offsets, get_video_and_audio, subsample_dataset

_DEBUG_DATA = os.environ.get('SYNCHFORMER_DEBUG_DATA', '0') == '1'


class VGGSound(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 to_filter_bad_examples=True,
                 splits_path='./data',
                 meta_path='./data/vggsound.csv',
                 seed=1337,
                 load_fixed_offsets_on=['valid', 'test'],
                 vis_load_backend='read_video',
                 size_ratio=None,
                 attr_annot_path=None,
                 max_attr_per_vid=None,
                 do_preprocess=True, 
                 vfps=25,
                 afps=16000,
                 size_before_crop=256,
                 input_size=224,
                 target_fps=None):
        super().__init__()
        self.vfps = vfps
        self.afps = afps
        self.target_fps = target_fps
        self.subsample_step = round(vfps / target_fps) if (target_fps and target_fps < vfps) else 1
        self.size_before_crop = size_before_crop
        self.input_size = input_size
        self.max_clip_len_sec = None
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.to_filter_bad_examples = to_filter_bad_examples
        self.splits_path = splits_path
        self.meta_path = meta_path
        self.seed = seed
        self.load_fixed_offsets_on = [] if load_fixed_offsets_on is None else load_fixed_offsets_on
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio
        self.do_preprocess = do_preprocess
        vggsound_meta = list(csv.reader(open(meta_path), quotechar='"'))

        # filter "bad" examples
        if to_filter_bad_examples:
            vggsound_meta = self.filter_bad_examples(vggsound_meta)

        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta}

        split_clip_ids_path = os.path.join(splits_path, f'vggsound_{split}.txt')
        if not os.path.exists(split_clip_ids_path):
            self.make_split_files()
        # the ugly string converts ['AdfsGsfII2yQ', '1'] into `AdfsGsfII2yQ_1000_11000`
        meta_available = set([f'{r[0]}_{int(r[1])*1000}_{(int(r[1])+10)*1000}' for r in vggsound_meta])
        within_split = set(open(split_clip_ids_path).read().splitlines())
        clip_paths = [os.path.join(vids_dir, v + '.mp4') for v in meta_available.intersection(within_split)]
        clip_paths = sorted(clip_paths)

        if split in self.load_fixed_offsets_on:
            logging.info(f'Using fixed offset for {split}')
            self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, 'vggsound')

        # making sure that all classes have at least one example
        counter = Counter([self.video2target[Path(p).stem[:11]] for p in clip_paths])
        assert all(counter[c] > 0 for c in self.target2label.keys()), \
            f'Some classes have 0 count: {dict(counter)}'

        self.dataset = clip_paths
        self.dataset = subsample_dataset(self.dataset, size_ratio, shuffle=split == 'train')

    def filter_bad_examples(self, vggsound_meta):
        bad = set()
        base_path = Path('./data/filtered_examples_vggsound')
        lists = [open(p).read().splitlines() for p in sorted(glob(str(base_path / '*.txt')))]
        for s in lists:
            bad = bad.union(s)
        # the ugly string converts '---g-f_I2yQ', '1' into `---g-f_I2yQ_1000_11000`
        vggsound_meta = [r for r in vggsound_meta if f'{r[0]}_{int(r[1])*1000}_{(int(r[1])+10)*1000}' not in bad]
        return vggsound_meta

    def __getitem__(self, index):
        max_retries = 1 if _DEBUG_DATA else 10
        for attempt in range(max_retries):
            current_index = index if attempt == 0 else random.randint(0, len(self) - 1)
            path = self.dataset[current_index]
            rgb, audio, meta = self.load_media(path)
            if rgb is None or audio is None:
                if _DEBUG_DATA:
                    raise RuntimeError(f'load_media returned None for {path}')
                continue
            item = self.make_datapoint(path, rgb, audio, meta)
            if self.transforms is not None:
                try:
                    item = self.transforms(item)
                except Exception as e:
                    if _DEBUG_DATA:
                        raise
                    logging.warning(f'Failed to apply transforms to item {path}: {type(e).__name__}: {e}')
                    continue
            return item
        raise RuntimeError(f"Failed to load data after {max_retries} retries, original index: {index}")

    def make_datapoint(self, path, rgb, audio, meta):
        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        target = self.video2target[Path(path).stem[:11]]
        item = {
            'video': rgb,
            'audio': audio,
            'meta': meta,
            'path': str(path),
            'targets': {'vggsound_target': target, 'vggsound_label': self.target2label[target]},
            'split': self.split,
        }

        if self.split in self.load_fixed_offsets_on:
            unique_id = Path(path).stem
            offset_params = self.vid2offset_params[unique_id]
            item['targets']['offset_sec'] = self.vid2offset_params[unique_id]['offset_sec']
            item['targets']['v_start_i_sec'] = self.vid2offset_params[unique_id]['v_start_i_sec']
            if 'oos_target' in offset_params:
                item['targets']['offset_target'] = {
                    'oos': offset_params['oos_target'], 'offset': item['targets']['offset_sec'],
                }

        return item

    def load_media(self, path):
        rgb, audio, meta = get_video_and_audio(path, get_meta=True, start_sec=0, end_sec=self.max_clip_len_sec, do_preprocess=self.do_preprocess, target_vfps=self.vfps, target_afps=self.afps, target_size=self.size_before_crop)
        
        if rgb is None or audio is None:
            return None, None, None

        if self.subsample_step > 1:
            rgb = rgb[::self.subsample_step]
            if meta and 'video' in meta and 'fps' in meta['video']:
                meta['video']['fps'] = [float(self.target_fps)]

        effective_fps = self.target_fps or self.vfps
        min_video_frames = round(7.2 * effective_fps)
        min_audio_samples = round(5.5 * self.afps)
        if rgb.shape[0] < min_video_frames:
            rgb = torch.cat([rgb, rgb[-1:].repeat(min_video_frames - rgb.shape[0], 1, 1, 1)], dim=0)
            logging.info(f'video length is too short, padding to {min_video_frames}, path: {path}')

        if audio.shape[0] < min_audio_samples:
            audio = torch.cat([audio, audio[-1:].repeat(min_audio_samples - audio.shape[0])], dim=0)
            logging.info(f'audio length is too short, padding to {min_audio_samples}, path: {path}')
        return rgb, audio, meta

    def make_split_files(self):
        if self.to_filter_bad_examples:
            logging.warning('`to_filter_bad_examples` is True. `make_split_files` expects otherwise')

        logging.info(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')
        # The downloaded videos (some went missing on YouTube and no longer available)
        available_vid_paths = sorted(glob(os.path.join(self.vids_dir, '*.mp4')))
        logging.info(f'The number of clips available after download: {len(available_vid_paths)}')

        # original (full) train and test sets
        vggsound_meta = list(csv.reader(open(self.meta_path), quotechar='"'))
        train_vids = {row[0] for row in vggsound_meta if row[3] == 'train'}
        test_vids = {row[0] for row in vggsound_meta if row[3] == 'test'}

        # # the cleaned test set
        # vggsound_meta_test_v2 = list(csv.reader(open(self.meta_path_clean_test), quotechar='"'))
        logging.info(f'The number of videos in vggsound train set: {len(train_vids)}')
        logging.info(f'The number of videos in vggsound test set: {len(test_vids)}')

        # class counts in test set. We would like to have the same distribution in valid
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        label2target = {label: target for target, label in enumerate(unique_classes)}
        video2target = {row[0]: label2target[row[2]] for row in vggsound_meta}
        test_vid_classes = [video2target[vid] for vid in test_vids]
        test_target2count = Counter(test_vid_classes)

        # now given the counts from test set, sample the same count for validation and the rest leave in train
        train_vids_wo_valid, valid_vids = set(), set()
        for target, label in enumerate(label2target.keys()):
            class_train_vids = [vid for vid in sorted(list(train_vids)) if video2target[vid] == target]
            random.Random(self.seed).shuffle(class_train_vids)
            count = test_target2count[target]
            valid_vids.update(class_train_vids[:count])
            train_vids_wo_valid.update(class_train_vids[count:])

        # make file with a list of available test videos (each video should contain timestamps as well)
        train_i = valid_i = test_i = 0
        with open(os.path.join(self.splits_path, 'vggsound_train.txt'), 'w') as train_file, \
             open(os.path.join(self.splits_path, 'vggsound_valid.txt'), 'w') as valid_file, \
             open(os.path.join(self.splits_path, 'vggsound_test.txt'), 'w') as test_file:
            #  open(os.path.join(self.splits_path, 'vggsound_test_v2.txt'), 'w') as test_file_v2:
            for path in available_vid_paths:
                path = path.replace('.mp4', '')
                vid_name = Path(path).name
                # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
                if vid_name[:11] in train_vids_wo_valid:
                    train_file.write(vid_name + '\n')
                    train_i += 1
                elif vid_name[:11] in valid_vids:
                    valid_file.write(vid_name + '\n')
                    valid_i += 1
                elif vid_name[:11] in test_vids:
                    test_file.write(vid_name + '\n')
                    test_i += 1
                # else:
                #     raise Exception(f'Clip {vid_name} is neither in train, valid nor test. Strange.')

        logging.info(f'Put {train_i} clips to the train set and saved it to ./data/vggsound_train.txt')
        logging.info(f'Put {valid_i} clips to the valid set and saved it to ./data/vggsound_valid.txt')
        logging.info(f'Put {test_i} clips to the test set and saved it to ./data/vggsound_test.txt')

    def __len__(self):
        return len(self.dataset)

class VGGSoundSparse(VGGSound):
    '''
        The same as VGGSound, except the list of videos is filtered for sparse sounds (sparse_meta_path)
    '''

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True,
                 splits_path='./data', meta_path='./data/vggsound.csv',
                 sparse_meta_path='./data/sparse_classes.csv', seed=1337, load_fixed_offsets_on=['valid', 'test'],
                 vis_load_backend='read_video', size_ratio=None, attr_annot_path=None, max_attr_per_vid=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path, meta_path, seed,
                         load_fixed_offsets_on, vis_load_backend, size_ratio)
        self.sparse_meta_path = sparse_meta_path
        sparse_meta = list(csv.reader(open(sparse_meta_path), quotechar='"', delimiter='\t'))
        sparse_classes = set([row[0] for row in sparse_meta if row[1] == 'y'])
        label2new_target = {label: target for target, label in enumerate(sorted(list(sparse_classes)))}
        new_target2label = {target: label for label, target in label2new_target.items()}

        sparse_dataset = []
        video2new_target = {}
        for path in self.dataset:
            vid_id = Path(path).stem[:11]
            vid_target = self.video2target[vid_id]
            vid_label = self.target2label[vid_target]
            if vid_label in sparse_classes:
                sparse_dataset.append(path)
                video2new_target[vid_id] = label2new_target[vid_label]

        self.dataset = sparse_dataset
        logging.debug(f'Filtered VGGSound dataset to sparse classes from {Path(sparse_meta_path).name}.')

        # redefining the label <-> target variable
        self.label2target = label2new_target
        self.target2label = new_target2label
        self.video2target = video2new_target
        logging.debug('Redefined label <-> target mapping to sparse classes.')

        counter = Counter([self.video2target[Path(p).stem[:11]] for p in self.dataset])
        assert len(self.dataset) < 1000 or all(counter[c] > 0 for c in self.target2label.keys()), \
            f'Some classes have 0 count: {dict(counter)}'


class VGGSoundSparsePicked(VGGSoundSparse):

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True,
                 splits_path='./data', meta_path='./data/vggsound.csv',
                 sparse_meta_path='./data/picked_sparse_classes.csv', seed=1337,
                 load_fixed_offsets_on=['valid', 'test'], vis_load_backend='read_video', size_ratio=None,
                 attr_annot_path=None, max_attr_per_vid=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path,
                         meta_path, sparse_meta_path, seed, load_fixed_offsets_on, vis_load_backend,
                         size_ratio)


class VGGSoundSparsePickedCleanTest(VGGSoundSparse):

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True,
                 splits_path='./data', meta_path='./data/vggsound.csv',
                 sparse_meta_path='./data/picked_sparse_classes.csv', seed=1337,
                 load_fixed_offsets_on=['valid', 'test'], vis_load_backend='read_video', size_ratio=None,
                 attr_annot_path=None, max_attr_per_vid=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path,
                         meta_path, sparse_meta_path, seed, load_fixed_offsets_on, vis_load_backend,
                         size_ratio)

    def filter_bad_examples(self, vggsound_meta):
        bad = set()
        base_path1 = Path('./data/filtered_examples_vggsound')
        base_path2 = Path('./data/filtered_examples_vggsound_extra')
        lists1 = [open(p).read().splitlines() for p in sorted(glob(str(base_path1 / '*.txt')))]
        lists2 = [open(p).read().splitlines() for p in sorted(glob(str(base_path2 / '*.txt')))]
        lists = lists1 + lists2
        for s in lists:
            bad = bad.union(s)
        # the ugly string converts '---g-f_I2yQ', '1' into `---g-f_I2yQ_1000_11000`
        vggsound_meta = [r for r in vggsound_meta if f'{r[0]}_{int(r[1])*1000}_{(int(r[1])+10)*1000}' not in bad]
        return vggsound_meta


class VGGSoundSparsePickedCleanTestFixedOffsets(VGGSoundSparse):
    '''This dataset only operates on manually annotated fixed offsets. Meant for evaluation purpose.'''

    def __init__(self, split, vids_dir, transforms=None, to_filter_bad_examples=True,
                 splits_path='./data', meta_path='./data/vggsound.csv',
                 sparse_meta_path='./data/picked_sparse_classes.csv', seed=1337,
                 load_fixed_offsets_on=['valid', 'test'], vis_load_backend='read_video', size_ratio=None,
                 attr_annot_path=None, max_attr_per_vid=None):
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path,
                         meta_path, sparse_meta_path, seed, load_fixed_offsets_on, vis_load_backend,
                         size_ratio)
        # redefine the dataset to only use fixed offsets
        fix_off_path = './data/vggsound_sparse_clean_fixed_offsets.csv'
        self.vid2offset_params = {}
        # lines have the format: dataset_name,video_id,vstart_sec,offset_sec,is_sync
        reader = csv.reader(open(fix_off_path))
        next(reader)  # skipping the header
        for _, vi, st, of, sy in reader:
            # FIXME: if there are more than one fixed offset per video, we will need to change the logic here
            assert vi not in self.vid2offset_params, 'offsets from other splits will override each other'
            if sy == '1':
                self.vid2offset_params[vi] = {'offset_sec': float(of), 'v_start_i_sec': float(st)}
        logging.debug(f'Loaded {len(self.vid2offset_params)} fixed offsets from {Path(fix_off_path).name}.')
        # since we care only about the videos in the fixed offset split, we filter the dataset
        self.dataset = [p for p in self.dataset if Path(p).stem in self.vid2offset_params]
        logging.debug(f'Filtered VGGSoundSparse dataset to fixed offsets from {Path(fix_off_path).name}.')


class LongerVGGSound(VGGSound):
    '''This class is different to the parent in the extra filtering it does. If the parent was
    making the splits with filtering for shorter than 9 second, this class filters for shorter than 9.5 sec.
    by applying extra filtering.
    '''

    def __init__(self,
                 split,
                 vids_dir,
                 transforms=None,
                 to_filter_bad_examples=True,
                 splits_path='./data',
                 meta_path='./data/vggsound.csv',
                 seed=1337,
                 load_fixed_offsets_on=['valid', 'test'],
                 vis_load_backend='read_video',
                 size_ratio=None,
                 attr_annot_path=None,
                 max_attr_per_vid=None):
        # size_ratio=None and fixed_offsets_on=[] to avoid double subsampling and loading fixed offsets
        super().__init__(split, vids_dir, transforms, to_filter_bad_examples, splits_path, meta_path, seed,
                         [], vis_load_backend, None, attr_annot_path, max_attr_per_vid)
        # redefining the load_fixed_offsets_on because the parent class does not load them (see above)
        self.load_fixed_offsets_on = load_fixed_offsets_on
        # doing the extra filtering for longer than 9.5 sec
        if to_filter_bad_examples:
            filt_ex_path = Path('./data/filtered_examples_vggsound_shorter/less_than_9.5s.txt')
            bad = set([p for p in open(filt_ex_path).read().splitlines()])
            self.dataset = [p for p in self.dataset if Path(p).stem not in bad]

        # longer vid clips require new fixed offsets, loading them here again (VGGSound loads them in init)
        if split in self.load_fixed_offsets_on:
            logging.info(f'Using fixed offset for {split}')
            self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, 'vggsound')

        self.dataset = subsample_dataset(self.dataset, size_ratio, shuffle=split == 'train')
        logging.info(f'{split} has {len(self.dataset)} items')


class VGGSoundList(torch.utils.data.Dataset):
    """
    自定义列表数据集：
    - 读取一个 txt（每行一个 clip_id，形如 video_id_startMS_endMS）
    - 在 `vids_dir` 下寻找 `${clip_id}.mp4`
    - 其余逻辑尽量复用 VGGSound 的媒体加载 + transforms + offset 目标生成流程
      （offset 相关目标由 `TemporalCropAndOffset` transform 生成，无需固定 offset 文件）

    目的：让分布式评测可以只跑一小撮 clip（比如 release/eval/data/vggsound_sparse_clean_validfmt.txt）
    """

    def __init__(
        self,
        split,
        vids_dir,
        list_txt,
        transforms=None,
        to_filter_bad_examples=False,
        meta_path='./data/vggsound.csv',
        load_fixed_offsets_on=None,
        vis_load_backend='read_video',
        size_ratio=None,
        attr_annot_path=None,
        max_attr_per_vid=None,
        do_preprocess=True,
        vfps=25,
        afps=16000,
        size_before_crop=256,
        input_size=224,
        seed=1337,
        target_fps=None,
    ):
        super().__init__()
        self.vfps = vfps
        self.afps = afps
        self.target_fps = target_fps
        self.subsample_step = round(vfps / target_fps) if (target_fps and target_fps < vfps) else 1
        self.size_before_crop = size_before_crop
        self.input_size = input_size
        self.max_clip_len_sec = None
        self.split = split
        self.vids_dir = vids_dir
        self.transforms = transforms
        self.to_filter_bad_examples = to_filter_bad_examples
        self.meta_path = meta_path
        self.seed = seed
        self.load_fixed_offsets_on = [] if load_fixed_offsets_on is None else load_fixed_offsets_on
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio
        self.do_preprocess = do_preprocess

        # label map（尽量保持与 VGGSound 一致；虽然后续 offset 任务不依赖分类标签）
        vggsound_meta = list(csv.reader(open(meta_path), quotechar='"'))
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta}

        # list
        list_txt_path = Path(list_txt)
        clip_ids = [s.strip() for s in list_txt_path.read_text(encoding='utf-8').splitlines() if s.strip()]

        clip_paths = [os.path.join(vids_dir, f"{cid}.mp4") for cid in clip_ids]
        # 过滤不存在的文件（避免 dataloader 无限重试）
        clip_paths = [p for p in clip_paths if os.path.exists(p)]
        if not clip_paths:
            raise RuntimeError(f"VGGSoundList: no existing mp4 found under vids_dir={vids_dir} for list_txt={list_txt}")

        # 可选 subsample
        self.dataset = sorted(clip_paths)
        self.dataset = subsample_dataset(self.dataset, size_ratio, shuffle=split == 'train')
        logging.info(f'VGGSoundList({split}): loaded {len(self.dataset)} items from {list_txt_path}')

    def __getitem__(self, index):
        max_retries = 1 if _DEBUG_DATA else 10
        for attempt in range(max_retries):
            current_index = index if attempt == 0 else random.randint(0, len(self) - 1)
            path = self.dataset[current_index]
            rgb, audio, meta = self.load_media(path)
            if rgb is None or audio is None:
                if _DEBUG_DATA:
                    raise RuntimeError(f'load_media returned None for {path}')
                continue
            item = self.make_datapoint(path, rgb, audio, meta)
            if self.transforms is not None:
                try:
                    item = self.transforms(item)
                except Exception as e:
                    if _DEBUG_DATA:
                        raise
                    logging.warning(f'Failed to apply transforms to item {path}: {type(e).__name__}: {e}')
                    continue
            return item
        raise RuntimeError(f"VGGSoundList: Failed to load data after {max_retries} retries, original index: {index}")

    def make_datapoint(self, path, rgb, audio, meta):
        # (Tv, 3, H, W) in [0, 225], (Ta,) in [-1, 1]
        vid_id = Path(path).stem[:11]
        target = self.video2target.get(vid_id, 0)
        label = self.target2label.get(target, "unknown")
        return {
            'video': rgb,
            'audio': audio,
            'meta': meta,
            'path': str(path),
            'targets': {'vggsound_target': target, 'vggsound_label': label},
            'split': self.split,
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
            target_size=self.size_before_crop,
        )

        if rgb is None or audio is None:
            return None, None, None

        if self.subsample_step > 1:
            rgb = rgb[::self.subsample_step]
            if meta and 'video' in meta and 'fps' in meta['video']:
                meta['video']['fps'] = [float(self.target_fps)]

        effective_fps = self.target_fps or self.vfps
        min_video_frames = round(7.2 * effective_fps)
        min_audio_samples = round(5.5 * self.afps)
        if rgb.shape[0] < min_video_frames:
            rgb = torch.cat([rgb, rgb[-1:].repeat(min_video_frames - rgb.shape[0], 1, 1, 1)], dim=0)
            logging.info(f'video length is too short, padding to {min_video_frames}, path: {path}')

        if audio.shape[0] < min_audio_samples:
            audio = torch.cat([audio, audio[-1:].repeat(min_audio_samples - audio.shape[0])], dim=0)
            logging.info(f'audio length is too short, padding to {min_audio_samples}, path: {path}')

        return rgb, audio, meta

    def __len__(self):
        return len(self.dataset)



if __name__ == '__main__':
    from omegaconf import OmegaConf
    from omnivae_sync.training.train_utils import get_transforms
    from omnivae_sync.utils.utils import cfg_sanity_check_and_patch
    cfg = OmegaConf.load('./configs/transformer.yaml')
    vis_load_backend = 'read_video'

    # set logging for info level
    logging.basicConfig(level=logging.INFO)

    transforms = get_transforms(cfg)

    vids_path = 'PLACEHOLDER'

    # cfg.data.dataset.params.size_ratio = 0.1

    cfg_sanity_check_and_patch(cfg)

    datasets = {
        'train': VGGSound('train', vids_path, transforms['train'], vis_load_backend=vis_load_backend,
                          size_ratio=cfg.data.dataset.params.size_ratio),
        'valid': VGGSound('valid', vids_path, transforms['test'], vis_load_backend=vis_load_backend),
        'test': VGGSound('test', vids_path, transforms['test'], vis_load_backend=vis_load_backend),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    print(datasets['train'][0]['meta'])
    print(datasets['valid'][0]['audio'].shape, datasets['valid'][0]['video'].shape)
    print(datasets['valid'][0]['meta'])
    print(datasets['test'][0]['audio'].shape, datasets['test'][0]['video'].shape)
    print(datasets['test'][0]['meta'])

    # for i in range(300, 1000):
    #     datasets['train'][i]['path']
    #     print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    #     print(datasets['train'][0]['meta'])

    datasets = {
        'train': VGGSoundSparse('train', vids_path, transforms['train']),
        'valid': VGGSoundSparse('valid', vids_path, transforms['test']),
        'test': VGGSoundSparse('test', vids_path, transforms['test']),
    }
    for phase in ['train', 'valid', 'test']:
        print(phase, len(datasets[phase]))

    print(datasets['train'][0]['audio'].shape, datasets['train'][0]['video'].shape)
    print(datasets['train'][0]['meta'])
    print(datasets['valid'][0]['audio'].shape, datasets['valid'][0]['video'].shape)
    print(datasets['valid'][0]['meta'])
    print(datasets['test'][0]['audio'].shape, datasets['test'][0]['video'].shape)
    print(datasets['test'][0]['meta'])

    datasets = {
        'test': VGGSoundSparsePickedCleanTestFixedOffsets('test', vids_path, transforms['test']),
    }
    for phase in ['test']:
        print(phase, len(datasets[phase]))

    for i in range(len(datasets['test'])):
        datasets['test'][i]['path']
        print(datasets['test'][i]['audio'].shape, datasets['test'][i]['video'].shape)
