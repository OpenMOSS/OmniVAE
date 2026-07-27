import os
import subprocess
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf

from syncformer.dataset.dataset_utils import get_video_and_audio
from syncformer.dataset.transforms import make_class_grid, quantize_offset
from syncformer.utils.utils import check_if_file_exists_else_download, which_ffmpeg
from syncformer.scripts.train_utils import get_model, get_transforms, prepare_inputs


def reencode_video(path, vfps=25, afps=16000, in_size=256):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    new_path = Path.cwd() / 'vis' / f'{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4'
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    cmd = f'{which_ffmpeg()}'
    # no info/error printing
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {path}'
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f' {new_path}'
    subprocess.call(cmd.split())
    cmd = f'{which_ffmpeg()}'
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {new_path}'
    cmd += f' -acodec pcm_s16le -ac 1'
    cmd += f' {new_path.replace(".mp4", ".wav")}'
    subprocess.call(cmd.split())
    return new_path


def decode_single_video_prediction(off_logits, grid, item):
    off_probs = torch.softmax(off_logits, dim=-1)
    k = min(off_probs.shape[-1], 5)
    topk_logits, topk_preds = torch.topk(off_logits, k)
    assert len(topk_logits) == 1, 'batch is larger than 1'
    topk_preds = topk_preds[0]
    return grid[topk_preds[0]].cpu().item()


def patch_config(cfg):
    # the FE ckpts are already in the model ckpt
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    # old checkpoints have different names
    cfg.model.params.transformer.target = cfg.model.params.transformer.target \
        .replace('.modules.feature_selector.', '.sync_model.')
    return cfg


class SyncformerInferencer:
    def __init__(self, model_path):
        self.vfps = 25
        self.afps = 16000
        self.in_size = 256
        self.offset_sec = 0.0
        self.v_start_i_sec = 0.0
        cfg_path = os.path.join(os.path.dirname(__file__),'cfg.yaml')
        ckpt_path = f'{model_path}/24-01-04T16-39-21.pt'
        cfg = OmegaConf.load(cfg_path)
        self.cfg = patch_config(cfg)
        self.device = torch.device("cuda")
        _, model = get_model(self.cfg, self.device)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        self.model = model

    def infer(self, vid_path):
        print(f'Using video: {vid_path}')
        v, _, info = torchvision.io.read_video(vid_path, pts_unit='sec')
        _, H, W, _ = v.shape
        if info['video_fps'] != self.vfps or info['audio_fps'] != self.afps or min(H, W) != self.in_size:
            print(f'Reencoding. vfps: {info["video_fps"]} -> {self.vfps};', end=' ')
            print(f'afps: {info["audio_fps"]} -> {self.afps};', end=' ')
            print(f'{(H, W)} -> min(H, W)={self.in_size}')
            vid_path = reencode_video(vid_path, self.vfps, self.afps, self.in_size)
        else:
            print(
                f'Skipping reencoding. vfps: {info["video_fps"]}; afps: {info["audio_fps"]}; min(H, W)={self.in_size}')
        rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)
        item = dict(
            video=rgb, audio=audio, meta=meta, path=vid_path, split='test',
            targets={'v_start_i_sec': self.v_start_i_sec, 'offset_sec': self.offset_sec, },
        )
        max_off_sec = self.cfg.data.max_off_sec
        num_cls = self.cfg.model.params.transformer.params.off_head_cfg.params.out_features
        grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
        if not (min(grid) <= item['targets']['offset_sec'] <= max(grid)):
            print(f'WARNING: offset_sec={item["targets"]["offset_sec"]} is outside the trained grid: {grid}')

        # applying the test-time transform
        # self.cfg['transform_sequence_test'][2]['params']['crop_len_sec'] = rgb.shape[0] / self.vfps
        item = get_transforms(self.cfg, ['test'])['test'](item)

        # prepare inputs for inference
        batch = torch.utils.data.default_collate([item])
        aud, vid, targets = prepare_inputs(batch, self.device)
        with torch.set_grad_enabled(False):
            with torch.autocast('cuda', enabled=self.cfg.training.use_half_precision):
                _, logits = self.model(vid, aud)
        off = decode_single_video_prediction(logits, grid, item)
        return off


