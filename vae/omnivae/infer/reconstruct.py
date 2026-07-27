from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import soundfile as sf
import torch
from tqdm import tqdm

from omnivae.dataset.audio_video_streaming_dataset import (
    build_video_transform,
    load_audio_from_path,
    load_audio_from_video_path,
    load_video_from_path,
)
from omnivae.dataset.video_utils import save_video_tensor_to_mp4
from omnivae.eval.reconstruction.common import (
    build_reconstruction_model,
    load_config,
    load_model_weights,
    require_config,
    resolve_path,
    setup_logging,
    write_json,
)


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _media_path(raw: str, *, metadata_path: Optional[Path], data_root: Optional[str]) -> Path:
    p = Path(os.path.expanduser(os.path.expandvars(str(raw))))
    if p.is_absolute():
        return p
    roots = []
    if data_root:
        roots.append(resolve_path(data_root))
    if metadata_path is not None:
        roots.append(metadata_path.parent)
    roots.append(resolve_path("."))
    for root in roots:
        candidate = (root / p).resolve()
        if candidate.exists():
            return candidate
    return (roots[0] / p).resolve()


def _safe_stem(path: Path, index: int) -> str:
    stem = path.stem or f"sample_{index:06d}"
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
    return f"{index:06d}_{stem}"


def _scan_dir(input_dir: str, mode: str) -> List[Dict[str, Any]]:
    root = resolve_path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {root}")
    exts = VIDEO_EXTENSIONS if mode in {"video", "av"} else AUDIO_EXTENSIONS
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            key = "video_path" if mode in {"video", "av"} else "audio_path"
            rows.append({key: str(path)})
    return rows


def _read_jsonl(path: str, max_examples: Optional[int]) -> tuple[List[Dict[str, Any]], Path]:
    jsonl_path = resolve_path(path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"input_jsonl does not exist: {jsonl_path}")
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows, jsonl_path


def _build_inputs(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Optional[Path]]:
    sources = [bool(args.input_file), bool(args.input_dir), bool(args.input_jsonl)]
    if sum(sources) != 1:
        raise ValueError("Pass exactly one of --input_file, --input_dir, or --input_jsonl")

    if args.input_jsonl:
        return _read_jsonl(args.input_jsonl, args.max_examples)
    if args.input_dir:
        rows = _scan_dir(args.input_dir, args.mode)
        if args.max_examples is not None:
            rows = rows[: args.max_examples]
        return rows, None

    input_file = resolve_path(args.input_file)
    if args.mode == "audio":
        row = {"audio_path": str(input_file)}
    else:
        row = {"video_path": str(input_file)}
    return [row], None


def _record_audio_path(record: Dict[str, Any], metadata_path: Optional[Path], data_root: Optional[str]) -> Optional[Path]:
    raw = record.get("audio_path") or record.get("audio")
    if raw:
        return _media_path(str(raw), metadata_path=metadata_path, data_root=data_root)
    return None


def _record_video_path(record: Dict[str, Any], metadata_path: Optional[Path], data_root: Optional[str]) -> Optional[Path]:
    raw = record.get("video_path") or record.get("video") or record.get("path")
    if raw:
        return _media_path(str(raw), metadata_path=metadata_path, data_root=data_root)
    return None


def _load_video_tensor(path: Path, args: argparse.Namespace) -> torch.Tensor:
    raw = load_video_from_path(
        str(path),
        num_frames=args.num_frames,
        sample_rate=1,
        target_fps=args.target_fps,
        use_torchcodec=not args.no_torchcodec,
        random_start=False,
    )
    video = raw.permute(3, 0, 1, 2).contiguous()
    transform = build_video_transform(
        args.resolution,
        spatial_transform_mode=args.spatial_transform_mode,
        spatial_roundtrip_short_edge=args.spatial_roundtrip_short_edge,
    )
    return transform(video)


def _load_audio_tensor(path: Path, args: argparse.Namespace, *, from_video: bool = False) -> torch.Tensor:
    if from_video:
        return load_audio_from_video_path(
            str(path),
            target_sample_rate=args.sample_rate,
            max_duration=args.max_duration,
        )
    return load_audio_from_path(
        str(path),
        target_sample_rate=args.sample_rate,
        max_duration=args.max_duration,
    )


def _write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _infer_modality_for_model(mode: str) -> str:
    if mode == "audio":
        return "audio"
    if mode == "video":
        return "video"
    return "av"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniVAE reconstruction inference")
    parser.add_argument("--mode", choices=["audio", "video", "av"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or Trainer_* directory")
    parser.add_argument("--config", default=None, help="Config YAML. If omitted, inferred from checkpoint layout.")
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--data_root", default=os.environ.get("OMNIVAE_DATA_ROOT"))
    parser.add_argument("--output_dir", default="$OMNIVAE_EXP_ROOT/infer/reconstruct")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--no_sample_posterior", action="store_true")
    parser.add_argument("--save_inputs", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--max_duration", type=float, default=None)

    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--target_fps", type=float, default=24.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--spatial_transform_mode", default="resize_center_crop")
    parser.add_argument("--spatial_roundtrip_short_edge", type=int, default=None)
    parser.add_argument("--no_torchcodec", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = resolve_path(args.output_dir)
    setup_logging(None if args.dry_run else output_dir / "run.log")

    config_path = require_config(args.config, args.checkpoint)
    rows, metadata_path = _build_inputs(args)
    logging.info("mode=%s", args.mode)
    logging.info("checkpoint=%s", args.checkpoint)
    logging.info("config=%s", config_path)
    logging.info("inputs=%d", len(rows))
    logging.info("output_dir=%s", output_dir)

    if args.dry_run:
        return 0

    cfg = load_config(config_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_reconstruction_model(cfg, modality=_infer_modality_for_model(args.mode))
    load_stats = load_model_weights(model, args.checkpoint, use_ema=args.use_ema)
    logging.info("load_stats=%s", load_stats)
    model.to(device).eval()

    audio_out_dir = output_dir / "audio_recon"
    video_out_dir = output_dir / "video_recon"
    input_out_dir = output_dir / "inputs"
    manifest: List[Dict[str, Any]] = []

    for index, record in enumerate(tqdm(rows, desc=f"infer {args.mode}")):
        video_path = _record_video_path(record, metadata_path, args.data_root)
        audio_path = _record_audio_path(record, metadata_path, args.data_root)
        stem_source = video_path or audio_path
        if stem_source is None:
            raise KeyError("input record must contain audio_path or video_path")
        stem = _safe_stem(stem_source, index)

        video_batch = None
        audio_batch = None
        video_input = None
        audio_input = None

        if args.mode in {"video", "av"}:
            if video_path is None:
                raise KeyError("video or av mode requires video_path")
            video_input = _load_video_tensor(video_path, args)
            video_batch = video_input.unsqueeze(0).to(device)

        if args.mode in {"audio", "av"}:
            if audio_path is not None:
                audio_input = _load_audio_tensor(audio_path, args, from_video=False)
                audio_source = audio_path
            elif video_path is not None:
                audio_input = _load_audio_tensor(video_path, args, from_video=True)
                audio_source = video_path
            else:
                raise KeyError("audio or av mode requires audio_path or video_path with audio")
            audio_batch = audio_input.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(
                video_batch,
                audio_batch,
                sample_posterior=not args.no_sample_posterior,
            )

        row_out: Dict[str, Any] = {"index": index, "input": str(stem_source)}

        if video_batch is not None and "video" in outputs:
            video_out_dir.mkdir(parents=True, exist_ok=True)
            video_out = video_out_dir / f"{stem}.mp4"
            save_video_tensor_to_mp4(
                outputs["video"]["recon"].squeeze(0).detach().cpu(),
                video_out,
                fps=args.target_fps,
            )
            row_out["video_recon"] = str(video_out)
            if args.save_inputs and video_input is not None:
                input_out_dir.mkdir(parents=True, exist_ok=True)
                input_video = input_out_dir / f"{stem}_input.mp4"
                save_video_tensor_to_mp4(video_input.detach().cpu(), input_video, fps=args.target_fps)
                row_out["video_input"] = str(input_video)

        if audio_batch is not None and "audio" in outputs:
            audio_out_dir.mkdir(parents=True, exist_ok=True)
            audio_out = audio_out_dir / f"{stem}.wav"
            recon = outputs["audio"]["recon"].squeeze(0).squeeze(0).detach().cpu().numpy()
            sf.write(str(audio_out), recon, args.sample_rate, format="WAV")
            row_out["audio_recon"] = str(audio_out)
            row_out["audio_input_source"] = str(audio_source)
            if args.save_inputs and audio_input is not None:
                input_out_dir.mkdir(parents=True, exist_ok=True)
                input_audio = input_out_dir / f"{stem}_input.wav"
                sf.write(str(input_audio), audio_input.squeeze(0).detach().cpu().numpy(), args.sample_rate, format="WAV")
                row_out["audio_input"] = str(input_audio)

        manifest.append(row_out)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_manifest(output_dir / "manifest.jsonl", manifest)
    write_json(
        output_dir / "summary.json",
        {
            "mode": args.mode,
            "checkpoint": str(args.checkpoint),
            "config": str(config_path),
            "count": len(manifest),
            "load_stats": load_stats,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

