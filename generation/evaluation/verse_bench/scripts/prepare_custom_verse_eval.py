#!/usr/bin/env python
import argparse
import json
import os
import re
import shutil
from pathlib import Path


SAMPLE_RE = re.compile(r"sample-versebench-(\d+)-(set[123])")


def read_jsonl(path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            set_name = item.get("type")
            raw_index = item.get("index")
            if set_name is None or raw_index is None:
                raise ValueError(f"{path}:{line_no} missing type/index")
            rows[(set_name, str(raw_index))] = item
            try:
                rows[(set_name, int(raw_index))] = item
            except (TypeError, ValueError):
                pass
    return rows


def find_manifest(samples_root, experiment, step, mode, cfg):
    root = samples_root / experiment / "samples" / step / mode / cfg
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    return manifest


def sample_base(sample):
    for key in ("av_path", "video_path", "audio_path"):
        path = sample.get(key)
        if not path:
            continue
        match = SAMPLE_RE.search(Path(path).name)
        if match:
            return match.group(1), match.group(2), int(match.group(1))
    raise ValueError(f"cannot parse sample id from sample entry: {sample}")


def link_file(source, target):
    source = Path(source)
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(str(source), str(target))
    return True


def clean_output(output_root):
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "inputs").mkdir(parents=True)
    for set_name in ("set1", "set2", "set3"):
        json_dir = output_root / "verse_bench" / set_name
        if set_name in ("set2", "set3"):
            json_dir = json_dir / "data"
        json_dir.mkdir(parents=True)


def build_item(row, prompt_field):
    prompt = row.get(prompt_field) or row.get("av_caption") or row.get("prompt") or ""
    speech_prompt = row.get("speech_prompt") or {"speaker": "", "text": ""}
    if not isinstance(speech_prompt, dict):
        speech_prompt = {"speaker": "", "text": str(speech_prompt)}
    return {
        "video_prompt": prompt,
        "audio_prompt": [prompt] if prompt else [],
        "speech_prompt": {
            "speaker": speech_prompt.get("speaker", ""),
            "text": speech_prompt.get("text", ""),
        },
        "prompt": prompt,
        "source_index": row.get("index"),
        "source_type": row.get("type"),
        "first_frame_path": row.get("first_frame_path", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment", default="2_t2av_recon")
    parser.add_argument("--step", default="step-00000000")
    parser.add_argument("--mode", default="joint_av")
    parser.add_argument("--cfg", default="cfg_simple")
    parser.add_argument("--prompt-field", default="av_caption")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata = read_jsonl(args.metadata_jsonl)
    manifest_path = find_manifest(args.samples_root, args.experiment, args.step, args.mode, args.cfg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_output(args.output_root)

    prepared = []
    skipped = []
    for sample in manifest.get("samples", []):
        base, set_name, entry_index = sample_base(sample)
        row = metadata.get((set_name, entry_index))
        if row is None:
            skipped.append({"base": base, "reason": "metadata_missing"})
            continue

        video_source = sample.get("av_path") or sample.get("video_path")
        audio_source = sample.get("audio_path")
        if not video_source or not Path(video_source).is_file():
            skipped.append({"base": base, "reason": "video_missing"})
            continue
        if not audio_source or not Path(audio_source).is_file():
            skipped.append({"base": base, "reason": "audio_missing"})
            continue

        input_dir = args.output_root / "inputs"
        link_file(video_source, input_dir / f"{base}.mp4")
        link_file(audio_source, input_dir / f"{base}.wav")

        json_dir = args.output_root / "verse_bench" / set_name
        if set_name in ("set2", "set3"):
            json_dir = json_dir / "data"

        item = build_item(row, args.prompt_field)
        (json_dir / f"{base}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        first_frame = row.get("first_frame_path")
        if first_frame:
            link_file(first_frame, json_dir / f"{base}{Path(first_frame).suffix or '.jpg'}")

        prepared.append(
            {
                "base": base,
                "set": set_name,
                "entry_index": entry_index,
                "video": video_source,
                "audio": audio_source,
                "first_frame": first_frame,
            }
        )
        if args.limit and len(prepared) >= args.limit:
            break

    summary = {
        "manifest": str(manifest_path),
        "prompt_field": args.prompt_field,
        "prepared_count": len(prepared),
        "skipped_count": len(skipped),
        "prepared": prepared,
        "skipped": skipped[:100],
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not prepared:
        raise SystemExit("no samples prepared")


if __name__ == "__main__":
    main()
