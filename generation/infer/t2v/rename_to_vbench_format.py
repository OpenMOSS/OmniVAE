"""Rename the videos produced by ``export_video_checkpoint_samples.py`` so they
follow VBench's expected filename convention.

VBench's ``vbench_standard`` mode looks for filenames ``<prompt>-<i>.mp4`` where
``<prompt>`` is the original prompt text and ``<i>`` is an integer in [0, N-1].
Our replicated jsonl already encoded the ``video_basename`` for each record;
this script walks the sampler output and renames / hard-links files
accordingly.

Source layout (input):
    <output-dir>/
        sample_000000_seed*.mp4
        sample_000001_seed*.mp4
        ...
        samples.jsonl

Result layout (output):
    <output-dir>/
        videos/<prompt>-0.mp4
        videos/<prompt>-1.mp4
        ...
        samples.jsonl   (untouched)
        rename_manifest.jsonl  (audit trail)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True, help="Directory produced by export_video_checkpoint_samples.py")
    parser.add_argument("--metadata-jsonl", type=Path, required=True, help="The replicated jsonl that was fed to the sampler. Must contain video_basename per record.")
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy", "move"),
        default="symlink",
        help="How to materialise the renamed files. symlink/hardlink avoid duplicating data.",
    )
    parser.add_argument("--videos-subdir", type=str, default="videos", help="Subdirectory under samples-dir for the renamed videos.")
    parser.add_argument("--strict", action="store_true", help="Fail when a samples.jsonl record cannot be matched to the metadata jsonl.")
    return parser.parse_args()


def materialise(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    args = parse_args()
    samples_dir = args.samples_dir.expanduser().resolve()
    metadata_jsonl = args.metadata_jsonl.expanduser().resolve()
    samples_jsonl = samples_dir / "samples.jsonl"

    if not samples_dir.is_dir():
        raise SystemExit(f"samples_dir not found: {samples_dir}")
    if not samples_jsonl.is_file():
        raise SystemExit(f"samples.jsonl not found in {samples_dir}")
    if not metadata_jsonl.is_file():
        raise SystemExit(f"metadata jsonl not found: {metadata_jsonl}")

    metadata_records: list[dict] = []
    with metadata_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                metadata_records.append(json.loads(line))

    samples_records: list[dict] = []
    with samples_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples_records.append(json.loads(line))

    target_dir = samples_dir / args.videos_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    rename_log_path = samples_dir / "rename_manifest.jsonl"
    written = 0
    skipped = 0
    with rename_log_path.open("w", encoding="utf-8") as audit:
        for sample in samples_records:
            sample_index = sample.get("sample_index")
            if sample_index is None or sample_index >= len(metadata_records):
                skipped += 1
                if args.strict:
                    raise SystemExit(f"sample_index {sample_index} out of range relative to metadata jsonl ({len(metadata_records)} records)")
                continue
            metadata_record = metadata_records[int(sample_index)]
            video_basename = metadata_record.get("video_basename")
            if not video_basename:
                skipped += 1
                if args.strict:
                    raise SystemExit(f"metadata record at sample_index {sample_index} missing video_basename")
                continue
            src = Path(sample["video_path"]).expanduser().resolve()
            if not src.is_file():
                skipped += 1
                if args.strict:
                    raise SystemExit(f"source video missing: {src}")
                continue
            dst = target_dir / video_basename
            materialise(src, dst, args.mode)
            audit.write(json.dumps({
                "sample_index": int(sample_index),
                "src": str(src),
                "dst": str(dst),
                "prompt": metadata_record.get("prompt", ""),
                "repeat_idx": metadata_record.get("repeat_idx"),
                "mode": args.mode,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"Renamed {written} videos into {target_dir} (mode={args.mode})", file=sys.stderr)
    if skipped:
        print(f"Skipped {skipped} samples without a matching metadata record.", file=sys.stderr)
    print(str(target_dir))


if __name__ == "__main__":
    main()
