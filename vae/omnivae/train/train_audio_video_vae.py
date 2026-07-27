"""
Audio-Video VAE Joint Trainer — Entry Point

保持原始脚本路径不变，供 train_local.sh 调用。
实际逻辑已拆分到 omnivae.train.av_vae 子包。
"""

import os
import sys
import yaml
import logging
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_omnivae_root = os.path.join(_project_root, "omnivae")
if _omnivae_root not in sys.path:
    sys.path.insert(0, _omnivae_root)

from omnivae.utils.helpers import suppress_useless_warnings
suppress_useless_warnings()

from omnivae.utils.helpers import waiting_for_debug, set_logging
from omnivae.train.av_vae.cli import build_arg_parser, merge_cli_to_config, build_experiment_tag
from omnivae.train.av_vae.utils import resolve_config_paths, resolve_path_value
from omnivae.train.av_vae.trainer import AudioVideoVAETrainer


def main():
    set_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.debug == 1:
        waiting_for_debug(args.debug_ip, args.debug_port)

    rank = int(os.environ.get('RANK', 0))
    args.config = resolve_path_value(args.config)
    config_path = Path(args.config)

    with config_path.open('r') as f:
        cfg = yaml.safe_load(f) or {}

    merge_cli_to_config(args, cfg)
    cfg = resolve_config_paths(cfg)

    for path_arg in (
        'checkpoint',
        'pretrained_checkpoint',
        'pretrained_video_checkpoint',
        'pretrained_audio_checkpoint',
        'pretrained_contrastive_checkpoint',
        'pretrained_disc_checkpoint',
    ):
        value = getattr(args, path_arg, None)
        if value:
            setattr(args, path_arg, resolve_path_value(value))

    build_experiment_tag(args, cfg, rank)

    trainer = AudioVideoVAETrainer(
        cfg, tag=args.tag, continue_train=args.continue_train,
        pretrained_checkpoint=args.pretrained_checkpoint,
        keep_audio_vae_pretrained=args.keep_audio_vae_pretrained,
        pretrained_video_checkpoint=args.pretrained_video_checkpoint,
        pretrained_audio_checkpoint=args.pretrained_audio_checkpoint,
        pretrained_contrastive_checkpoint=args.pretrained_contrastive_checkpoint,
        pretrained_disc_checkpoint=args.pretrained_disc_checkpoint,
        pretrained_disc_load_optim=args.pretrained_disc_load_optim,
    )

    if rank == 0:
        logging.info(f"this exp dir path: {trainer.exp_dir}")

    if args.valid_only:
        trainer.validate_only(checkpoint_path=args.checkpoint)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
