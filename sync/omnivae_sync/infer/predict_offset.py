import argparse
import json
from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import OmegaConf

from omnivae_sync.dataset.dataset_utils import get_video_and_audio
from omnivae_sync.dataset.transforms import make_class_grid, quantize_offset
from omnivae_sync.train import _normalize_cli, register_resolvers
from omnivae_sync.training.train_utils import get_model, get_transforms, prepare_inputs


TARGET_REPLACEMENTS = {
    "model.sync_model_vae.": "omnivae_sync.model.sync_model_vae.",
    "model.sync_model.": "omnivae_sync.model.sync_model.",
    "model.modules.transformer.": "omnivae_sync.model.modules.transformer.",
    "dataset.transforms.": "omnivae_sync.dataset.transforms.",
    "dataset.vggsound.": "omnivae_sync.dataset.vggsound.",
    "dataset.jsonl_dataset.": "omnivae_sync.dataset.jsonl_dataset.",
}


def patch_config_targets(cfg):
    def _patch_value(value):
        if isinstance(value, str):
            for old, new in TARGET_REPLACEMENTS.items():
                if value.startswith(old):
                    return f"{new}{value[len(old):]}"
            return value
        if isinstance(value, dict):
            return {k: _patch_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_patch_value(v) for v in value]
        return value

    return OmegaConf.create(_patch_value(OmegaConf.to_container(cfg, resolve=False)))


def load_cfg(cfg_path: Optional[str], ckpt: Dict[str, Any], overrides: Iterable[str]):
    register_resolvers()
    if cfg_path:
        cfg = OmegaConf.load(cfg_path)
    elif "args" in ckpt:
        cfg = ckpt["args"]
    else:
        raise ValueError("Checkpoint has no saved args. Please pass --cfg_path.")
    cfg = patch_config_targets(cfg)
    dotlist = _normalize_cli(overrides)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg.training.resume = False
    cfg.training.finetune = False
    cfg.training.run_test_only = False
    OmegaConf.resolve(cfg)
    return cfg


def subsample_video_if_needed(rgb: torch.Tensor, meta: Dict[str, Any], cfg):
    vfps = float(cfg.data.get("vfps", meta["video"]["fps"][0]))
    target_fps = cfg.data.get("target_fps", None)
    if target_fps is None or float(target_fps) >= vfps:
        return rgb, meta
    step = max(1, round(vfps / float(target_fps)))
    rgb = rgb[::step]
    meta["video"]["fps"] = [float(target_fps)]
    return rgb, meta


def decode_prediction(logits: torch.Tensor, grid: torch.Tensor, item: Dict[str, Any], topk: int):
    probs = torch.softmax(logits.float(), dim=-1)[0].cpu()
    logits = logits.float()[0].cpu()
    k = min(topk, logits.shape[-1])
    topk_logits, topk_idx = torch.topk(logits, k)

    offset_sec = float(item["targets"]["offset_sec"])
    target_grid_value, target_idx = quantize_offset(grid, offset_sec)
    result = {
        "target_offset_sec": offset_sec,
        "target_grid_sec": float(target_grid_value),
        "target_index": int(target_idx),
        "predictions": [],
    }
    for logit, idx in zip(topk_logits, topk_idx):
        result["predictions"].append(
            {
                "index": int(idx),
                "offset_sec": float(grid[idx]),
                "prob": float(probs[idx]),
                "logit": float(logit),
            }
        )
    return result


def print_prediction(result: Dict[str, Any]) -> None:
    print(
        "Target offset: "
        f"{result['target_offset_sec']:.2f}s "
        f"(grid={result['target_grid_sec']:.2f}s, index={result['target_index']})"
    )
    print("Top predictions:")
    for pred in result["predictions"]:
        print(
            f"  index={pred['index']:>3d} "
            f"offset={pred['offset_sec']:>6.2f}s "
            f"prob={pred['prob']:.4f} "
            f"logit={pred['logit']:.4f}"
        )


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Predict audio-video sync offset for one video.")
    parser.add_argument("--ckpt_path", required=True, help="Path to OmniVAE-Sync checkpoint.")
    parser.add_argument("--cfg_path", default=None, help="Path to the training config. Optional if saved in ckpt.")
    parser.add_argument("--vid_path", required=True, help="Input video path.")
    parser.add_argument("--offset_sec", type=float, default=0.0, help="Audio offset injected before prediction.")
    parser.add_argument("--v_start_i_sec", type=float, default=0.0, help="Video crop start time in seconds.")
    parser.add_argument("--device", default="cuda:0", help="Device, for example cuda:0 or cpu.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable table.")
    parser.add_argument("--do_preprocess", action="store_true", default=None)
    parser.add_argument("--no_preprocess", action="store_false", dest="do_preprocess")
    args, overrides = parser.parse_known_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_cfg(args.cfg_path, ckpt, overrides)
    cfg.ckpt_path = args.ckpt_path

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _, model = get_model(cfg, device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    do_preprocess = cfg.data.get("do_preprocess", True) if args.do_preprocess is None else args.do_preprocess
    rgb, audio, meta = get_video_and_audio(
        args.vid_path,
        get_meta=True,
        do_preprocess=do_preprocess,
        target_vfps=int(cfg.data.get("vfps", 24)),
        target_afps=int(cfg.data.get("afps", 48000)),
        target_size=int(cfg.data.get("size_before_crop", 256)),
    )
    rgb, meta = subsample_video_if_needed(rgb, meta, cfg)

    item = {
        "video": rgb,
        "audio": audio,
        "meta": meta,
        "path": args.vid_path,
        "split": "test",
        "targets": {"v_start_i_sec": args.v_start_i_sec, "offset_sec": args.offset_sec},
    }
    item = get_transforms(cfg, ["test"])["test"](item)
    batch = torch.utils.data.default_collate([item])
    aud, vid, _ = prepare_inputs(batch, device)

    with torch.no_grad():
        amp = bool(cfg.training.get("use_half_precision", False)) and device.type == "cuda"
        with torch.autocast(device.type, enabled=amp):
            _, logits = model(vid, aud)

    num_cls = int(cfg.data.get("num_off_cls", logits.shape[-1]))
    grid = make_class_grid(-float(cfg.data.max_off_sec), float(cfg.data.max_off_sec), num_cls)
    result = decode_prediction(logits, grid, item, args.topk)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_prediction(result)


if __name__ == "__main__":
    main_cli()
