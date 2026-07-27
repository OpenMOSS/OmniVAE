"""Batch T2AV inference for OmniVAE joint-AV checkpoints.

Single-process, single-GPU CLI that walks a list of prompts and writes
one ``.av.mp4`` (or ``.mp4`` / ``.wav`` for single-modality modes) per
prompt. Reuses :func:`t2av_pipeline.generate_one_av` so each sample is
numerically identical to a one-prompt pass of
``omnivae_generation.trainer.joint_av.validation.run_joint_av_validation`` with the same
parameters.

Usage
-----

    # Read a jsonl manifest (one row per prompt):
    python infer/t2av/infer_t2av.py \
        --ckpt /.../checkpoints/snapshots/checkpoint-00060000 \
        --prompt-manifest data/av_valid/vabench_valid.jsonl \
        --output-dir runs/t2av/cli/run_001 \
        --modes joint_av video_only audio_only \
        --cfg-mode dual \
        --num-inference-steps 50 \
        --video-guidance-scale 4 --audio-guidance-scale 4

    # Or pass prompts inline:
    python infer/t2av/infer_t2av.py \
        --ckpt /.../checkpoint-00060000 \
        --prompts "a busy night market with sizzling food and chatter" \
                  "ocean waves at sunset, gulls calling overhead" \
        --output-dir runs/t2av/cli/inline_001 \
        --modes joint_av

Output layout (mirrors omnivae_generation.trainer.joint_av.validation):
::

    <output_dir>/step<NNNNNNNN>/<mode>/cfg_<cfg_mode>/
        sample-NNNN[-<source>][-<type>].{mp4,wav,av.mp4} + .json sidecar
    <output_dir>/samples.jsonl     # one record per (sample, mode) pair
    <output_dir>/run.json          # full request manifest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from tqdm import tqdm


EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from t2av_pipeline import (  # noqa: E402
    check_ffmpeg_available,
    generate_one_av,
    load_joint_av_pipeline,
)


_VALID_MODES = ("joint_av", "video_only", "audio_only")
_VALID_CFG_MODES = ("simple", "dual")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive float, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--ckpt", "--checkpoint-dir", dest="ckpt", type=str, required=True,
        help="Path to a t2av checkpoint-XXXXXXXX directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run directory containing resolved_config.json/yaml.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root directory; per-sample outputs land at "
        "<output-dir>/step<step>/<mode>/cfg_<cfg_mode>/.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: cuda:0 if available else cpu).",
    )

    # Prompts -- pick exactly one source.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prompt-manifest", type=str, default=None,
        help="JSONL file with one prompt per row.",
    )
    group.add_argument(
        "--prompts", type=str, nargs="+", default=None,
        help="Inline prompt strings (each becomes one sample).",
    )

    parser.add_argument(
        "--prompt-key", type=str, default="av_caption",
        help="Field name in the manifest that holds the prompt text "
        "(default: av_caption; falls back to other common fields).",
    )
    parser.add_argument(
        "--prompt-fallback-keys", type=str,
        default="prompt,video_caption,caption,text",
        help="Comma-separated fallback fields used when --prompt-key is "
        "missing or empty. Set to '' to disable.",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default="",
        help="Single negative prompt applied to every sample (overridden "
        "per-row when the manifest has --negative-prompt-key).",
    )
    parser.add_argument(
        "--negative-prompt-key", type=str, default="negative_prompt",
        help="Optional manifest field with a per-row negative prompt.",
    )
    parser.add_argument(
        "--index-key", type=str, default="index",
        help="Optional manifest field used as the per-sample id. Falls "
        "back to the row's 0-based offset when missing.",
    )
    parser.add_argument(
        "--source-name-key", type=str, default="source",
        help="Optional manifest field used to tag the output filename "
        "(handy when the jsonl mixes multiple sources).",
    )
    parser.add_argument(
        "--type-key", type=str, default="type",
        help="Optional manifest field added to the output filename.",
    )

    # Selection / limiting.
    parser.add_argument(
        "--limit", type=positive_int, default=None,
        help="Cap on number of prompts to process.",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first N prompts from the manifest.",
    )

    # Sampling parameters.
    parser.add_argument(
        "--modes", type=str, nargs="+", default=["joint_av"], choices=list(_VALID_MODES),
        help="Generation modes to run for every prompt (default: joint_av).",
    )
    parser.add_argument(
        "--cfg-mode", type=str, default="simple", choices=list(_VALID_CFG_MODES),
        help="CFG variant. 'dual' only applies in joint_av; collapses to "
        "simple in single-modality modes (matches trainer validation).",
    )
    parser.add_argument("--num-inference-steps", type=positive_int, default=50)
    parser.add_argument("--video-guidance-scale", type=float, default=4.0)
    parser.add_argument("--audio-guidance-scale", type=float, default=4.0)
    parser.add_argument("--cfg-normalization", action="store_true")
    parser.add_argument(
        "--video-text-guidance", type=float, default=None,
        help="Dual CFG: text-side video guidance (default: --video-guidance-scale).",
    )
    parser.add_argument(
        "--video-modality-guidance", type=float, default=None,
        help="Dual CFG: modality-side video guidance (default: --video-guidance-scale).",
    )
    parser.add_argument(
        "--audio-text-guidance", type=float, default=None,
        help="Dual CFG: text-side audio guidance (default: --audio-guidance-scale).",
    )
    parser.add_argument(
        "--audio-modality-guidance", type=float, default=None,
        help="Dual CFG: modality-side audio guidance (default: --audio-guidance-scale).",
    )

    # Shape / seed.
    parser.add_argument("--num-frames", type=positive_int, default=None)
    parser.add_argument("--fps", type=positive_float, default=None)
    parser.add_argument("--height", type=positive_int, default=None)
    parser.add_argument("--width", type=positive_int, default=None)
    parser.add_argument(
        "--audio-duration-seconds", type=positive_float, default=None,
        help="Audio target duration before pad/trim to num_frames/fps (default: from validation block).",
    )
    parser.add_argument(
        "--seed", type=int, default=20260508,
        help="Base seed; each sample uses (--seed + manifest row index * 100003) "
        "so two samples never collide and an offset reproduces older subsets.",
    )
    parser.add_argument("--video-quality", type=int, default=8)

    # VAE overrides.
    parser.add_argument("--vae-type", type=str, default=None)
    parser.add_argument("--vae-path", type=str, default=None)
    parser.add_argument("--audio-vae-type", type=str, default=None)
    parser.add_argument("--audio-vae-path", type=str, default=None)

    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (sample_id, mode) pairs already present in samples.jsonl.",
    )

    parser.add_argument(
        "--no-task-prefix", action="store_true",
        help=(
            "Disable the automatic t2av instruction template wrapping. By "
            "default every prompt is wrapped with one of 10 t2av templates "
            "(matches omnivae_generation.trainer.joint_av.validation), which is required for "
            "distilled checkpoints to produce on-distribution outputs. "
            "Only set this when you've already wrapped the prompt yourself."
        ),
    )
    parser.add_argument(
        "--no-duration-suffix", action="store_true",
        help=(
            "Disable the automatic ' duration: <X.Xs>' suffix. Mirrors the "
            "trainer's ``validation.joint_av_prompts.append_duration_suffix`` "
            "knob; default ON because the joint model was trained / "
            "validated against prompts that included it."
        ),
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Manifest loading
# ----------------------------------------------------------------------
def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
            if item and str(item).strip():
                return str(item).strip()
        return ""
    return str(value).strip()


def _resolve_prompt(
    record: dict,
    *,
    primary_key: str,
    fallback_keys: list[str],
) -> str:
    for key in [primary_key, *fallback_keys]:
        if not key:
            continue
        text = _coerce_text(record.get(key))
        if text:
            return text
    return ""


def load_manifest(path: Path, *, prompt_key: str, fallback_keys: list[str]) -> list[dict]:
    """Read a JSONL file; warn (don't crash) on malformed rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Prompt manifest not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                print(f"[manifest] skipping malformed line {line_idx}", file=sys.stderr)
                continue
            if not isinstance(entry, dict):
                continue
            prompt = _resolve_prompt(entry, primary_key=prompt_key, fallback_keys=fallback_keys)
            if not prompt:
                continue
            entry.setdefault("_row_index", line_idx)
            entry.setdefault("_resolved_prompt", prompt)
            rows.append(entry)
    return rows


def build_inline_records(prompts: list[str]) -> list[dict]:
    return [
        {"_row_index": i, "_resolved_prompt": text.strip(), "index": i}
        for i, text in enumerate(prompts)
        if text and text.strip()
    ]


# Fallback offset for rows whose ``index`` field is missing or non-numeric
# (e.g. versebench set2 rows with ``index="clip_05f5760d"``). Without an
# offset the fallback uses ``_row_index`` directly, which can collide with
# legitimate numeric ``index`` values from sibling rows in the same
# ``type`` -- on versebench this clobbered ~64 of the 600 samples on disk
# (two records racing to write the same ``sample-versebench-0309-set2.*``
# file). Adding ~10M to the fallback puts it in a strictly higher
# namespace than any real numeric index in any of our jsonl sources, so
# the two ranges become disjoint by construction.
#
# Eval-side mirror: ``generation/infer/t2av/build_sample_manifest.load_valid_jsonl``
# MUST use the same constant or
# the eval-side ``(type, index)`` lookup won't match what's actually on
# disk. Keep these two in lockstep.
INDEX_FALLBACK_OFFSET = 10_000_000


def _filename_safe(value: Any, *, fallback: str = "x") -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(value)).strip("_")
    return cleaned or fallback


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_sample_index(record_index: Any, row_index: int) -> int:
    """Mirror of the ``make_stem`` / ``load_valid_jsonl`` fallback logic.

    Returns ``int(record_index)`` when it parses as an int, else
    ``INDEX_FALLBACK_OFFSET + row_index`` (see the constant's docstring
    for why the offset is needed).
    """
    try:
        return int(record_index)
    except (TypeError, ValueError):
        return INDEX_FALLBACK_OFFSET + int(row_index)


def make_file_stem(
    record: dict,
    *,
    row_index: int,
    source_key: str,
    type_key: str,
    index_key: str,
) -> str:
    """Construct a sortable, collision-free filename stem per (row, mode)."""
    sample_index = _resolve_sample_index(record.get(index_key), row_index)
    parts = [f"sample-{sample_index:04d}"]
    source = record.get(source_key)
    if source:
        parts.append(_filename_safe(source))
    type_label = record.get(type_key)
    if type_label:
        parts.append(_filename_safe(type_label))
    return "-".join(parts)


def existing_pairs(samples_path: Path) -> set[tuple[str, str]]:
    if not samples_path.is_file():
        return set()
    done: set[tuple[str, str]] = set()
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            stem = rec.get("file_stem")
            mode = rec.get("mode")
            if isinstance(stem, str) and isinstance(mode, str):
                done.add((stem, mode))
    return done


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def resolve_device(arg: Optional[str]) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def iter_records(records: list[dict], *, offset: int, limit: Optional[int]) -> Iterable[dict]:
    selected = records[max(0, int(offset)):]
    if limit is not None:
        selected = selected[: int(limit)]
    yield from selected


def main() -> None:
    args = parse_args()

    if not check_ffmpeg_available():
        print(
            "[t2av] WARNING: ffmpeg is not on PATH; joint_av mode will save "
            ".mp4 + .wav side-by-side without producing the muxed .av.mp4.",
            file=sys.stderr,
        )

    fallback_keys = [
        token.strip() for token in (args.prompt_fallback_keys or "").split(",")
        if token.strip()
    ]
    if args.prompt_manifest is not None:
        records = load_manifest(
            Path(args.prompt_manifest).expanduser().resolve(),
            prompt_key=args.prompt_key,
            fallback_keys=fallback_keys,
        )
    else:
        records = build_inline_records(list(args.prompts or []))
    if not records:
        raise SystemExit("[t2av] no usable prompts found in the provided source.")

    device = resolve_device(args.device)

    pipe = load_joint_av_pipeline(
        args.ckpt,
        device=device,
        run_dir=args.run_dir,
        vae_type_override=args.vae_type,
        vae_path_override=args.vae_path,
        audio_vae_type_override=args.audio_vae_type,
        audio_vae_path_override=args.audio_vae_path,
    )

    val_cfg = pipe.run_config.get("validation", {}) or {}
    dataset_cfg = pipe.run_config.get("dataset", {}) or {}
    frame_size = val_cfg.get("video_frame_size") or dataset_cfg.get("frame_size") or [256, 256]
    num_frames = int(args.num_frames or val_cfg.get("video_num_frames") or dataset_cfg.get("num_frames") or 121)
    fps = float(args.fps or val_cfg.get("video_fps") or dataset_cfg.get("target_fps") or 24.0)
    height = int(args.height or frame_size[0])
    width = int(args.width or frame_size[1])
    audio_duration_seconds = float(
        args.audio_duration_seconds
        or val_cfg.get("audio_duration_seconds", 5.04)
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    step_root = output_root / f"step{pipe.checkpoint_step:08d}"
    step_root.mkdir(parents=True, exist_ok=True)
    samples_path = output_root / "samples.jsonl"
    run_json_path = output_root / "run.json"

    run_manifest = {
        "checkpoint_dir": str(pipe.checkpoint_dir),
        "checkpoint_step": int(pipe.checkpoint_step),
        "run_dir": str(pipe.run_dir),
        "modes": list(args.modes),
        "cfg_mode": args.cfg_mode,
        "num_inference_steps": int(args.num_inference_steps),
        "video_guidance_scale": float(args.video_guidance_scale),
        "audio_guidance_scale": float(args.audio_guidance_scale),
        "cfg_normalization": bool(args.cfg_normalization),
        "video_text_guidance": args.video_text_guidance,
        "video_modality_guidance": args.video_modality_guidance,
        "audio_text_guidance": args.audio_text_guidance,
        "audio_modality_guidance": args.audio_modality_guidance,
        "shift_v": pipe.shift_v,
        "shift_a": pipe.shift_a,
        "num_frames": num_frames,
        "fps": fps,
        "height": height,
        "width": width,
        "audio_duration_seconds": audio_duration_seconds,
        "base_seed": int(args.seed),
        "prompt_manifest": str(args.prompt_manifest) if args.prompt_manifest else None,
        "prompts_inline_count": len(records) if args.prompts is not None else None,
        "device": str(device),
        "vae_type_override": args.vae_type,
        "vae_path_override": args.vae_path,
        "audio_vae_type_override": args.audio_vae_type,
        "audio_vae_path_override": args.audio_vae_path,
        "started_at": time.strftime("%Y%m%d_%H%M%S"),
    }
    run_json_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    done_pairs = existing_pairs(samples_path) if args.resume else set()
    samples_mode = "a" if args.resume and samples_path.exists() else "w"
    samples_handle = samples_path.open(samples_mode, encoding="utf-8")

    selected = list(iter_records(records, offset=args.offset, limit=args.limit))
    pair_total = len(selected) * len(args.modes)
    progress = tqdm(total=pair_total, desc=f"t2av step={pipe.checkpoint_step}")

    try:
        for record in selected:
            row_index = int(record.get("_row_index", 0))
            prompt = str(record.get("_resolved_prompt", "")).strip()
            if not prompt:
                progress.update(len(args.modes))
                continue
            negative_prompt = _coerce_text(record.get(args.negative_prompt_key)) or args.negative_prompt

            sample_seed = int(args.seed) + row_index * 100003
            file_stem = make_file_stem(
                record,
                row_index=row_index,
                source_key=args.source_name_key,
                type_key=args.type_key,
                index_key=args.index_key,
            )

            for mode in args.modes:
                pair_key = (file_stem, mode)
                if pair_key in done_pairs:
                    progress.update(1)
                    progress.set_postfix_str(f"skip {file_stem}/{mode}")
                    continue

                effective_cfg_mode = args.cfg_mode
                if mode != "joint_av" and effective_cfg_mode == "dual":
                    effective_cfg_mode = "simple"
                mode_dir = step_root / mode / f"cfg_{effective_cfg_mode}"
                mode_dir.mkdir(parents=True, exist_ok=True)

                progress.set_postfix_str(f"{file_stem}/{mode}")
                try:
                    result = generate_one_av(
                        pipe,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        mode=mode,
                        cfg_mode=effective_cfg_mode,
                        num_inference_steps=int(args.num_inference_steps),
                        video_guidance_scale=float(args.video_guidance_scale),
                        audio_guidance_scale=float(args.audio_guidance_scale),
                        cfg_normalization=bool(args.cfg_normalization),
                        video_text_guidance=args.video_text_guidance,
                        video_modality_guidance=args.video_modality_guidance,
                        audio_text_guidance=args.audio_text_guidance,
                        audio_modality_guidance=args.audio_modality_guidance,
                        num_frames=num_frames,
                        fps=fps,
                        height=height,
                        width=width,
                        audio_duration_seconds=audio_duration_seconds,
                        seed=sample_seed,
                        output_dir=mode_dir,
                        file_stem=file_stem,
                        video_quality=int(args.video_quality),
                        wrap_task_prefix=not bool(args.no_task_prefix),
                        append_duration_suffix=not bool(args.no_duration_suffix),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[t2av] FAILED file_stem={file_stem!r} mode={mode!r}: {exc!r}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    progress.update(1)
                    continue

                result.update(
                    file_stem=file_stem,
                    row_index=row_index,
                    sample_seed=sample_seed,
                    source_record=record,
                    checkpoint_step=int(pipe.checkpoint_step),
                )
                sidecar = mode_dir / f"{file_stem}.json"
                sidecar.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                samples_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                samples_handle.flush()
                done_pairs.add(pair_key)
                progress.update(1)
    finally:
        samples_handle.close()
        progress.close()

    finished = {
        "samples_path": str(samples_path),
        "generated_pairs": len(done_pairs),
        "finished_at": time.strftime("%Y%m%d_%H%M%S"),
    }
    print(json.dumps(finished, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
