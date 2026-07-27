#!/usr/bin/env python3
"""Build a sample manifest for one generated sample directory.

The manifest is the single source of truth that downstream T2AV-Compass and
VABench evaluators read. No video copy, no audio re-encoding: we point each
record at the original ``.mp4`` (video) and ``.wav`` (audio) sitting next to
each other in ``sample_dir``. The legacy ``.av.mp4`` (muxed) file is ignored.

For VABench, which insists on a ``data_dir/{video,audio,json}/<category>/``
layout, we additionally emit a thin symlink view: every entry under
``vabench_view/video`` and ``vabench_view/audio`` is a symlink back to the
original file. ``vabench_view/json/<category>.json`` is a regenerated prompt
file derived from manifest contents.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
RELEASE_ROOT = Path(
    os.environ.get(
        "OMNIVAE_RELEASE_ROOT",
        os.environ.get("OPEN_SOURCE_ROOT", str(REPO_ROOT / "open_source")),
    )
)
DEFAULT_VALID_JSONL = (
    RELEASE_ROOT / "eval" / "data" / "t2av" / "versebench_minimal" / "versebench_t2av_infer_minimal.jsonl"
)
DEFAULT_VABENCH_JSON_DIR = RELEASE_ROOT / "eval" / "data" / "t2av" / "vabench_json"
DEFAULT_MAPPING = REPO_ROOT / "VABench" / "mapping" / "final_idx_to_prompt.csv"
DEFAULT_DIMENSIONS = [
    "first_dnsmos",
    "first_nisqa",
    "first_audiobox",
    "second_viclip",
    "second_clap",
    "second_desync",
    "second_lsa",
    "second_imagebind",
    "third_alignment",
    "third_audio_reality",
    "third_visual_reality",
    "third_expressiveness",
    "third_artistry",
]
FILENAME_RE = re.compile(r"^sample-[^-]+-(?P<index>\d+)-(?P<category>.+?)\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sample manifest + VABench symlink view.")
    parser.add_argument("--sample-dir", required=True, help="Directory containing sample .mp4 and matching .wav files.")
    parser.add_argument("--output-dir", required=True, help="Where to drop manifest + vabench_view.")
    parser.add_argument(
        "--valid-jsonl",
        "--input-jsonl",
        dest="valid_jsonl",
        default=str(DEFAULT_VALID_JSONL),
        help=(
            "JSONL with source prompts/metadata. Rows may be original input "
            "records (type/index/av_caption) or generation logs with the "
            "original row nested under source_record."
        ),
    )
    parser.add_argument("--vabench-json-dir", default=str(DEFAULT_VABENCH_JSON_DIR))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--categories", nargs="*", default=None, help="Optional category allow-list.")
    parser.add_argument("--max-per-category", type=int, default=0)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recreate existing symlinks.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# Must match ``generation/infer/t2av/infer_t2av.INDEX_FALLBACK_OFFSET``.
# The inference driver embeds (numeric_index_or_offset_plus_row_index) into
# every sample filename; we recompute the same mapping here so the eval
# (type, index) lookup matches what is actually on disk. The offset keeps
# numeric indices (in practice <100k for vabench/versebench/etc.) disjoint
# from fallback rows, which would otherwise collide on jsonl files like
# versebench_expanded.jsonl that mix numeric ``index="00000309"`` and
# non-numeric ``index="clip_05f5760d"`` rows of the same ``type``.
INDEX_FALLBACK_OFFSET = 10_000_000


def _source_item(item: dict[str, Any]) -> dict[str, Any]:
    source = item.get("source_record")
    if isinstance(source, dict):
        out = dict(source)
        if "file_stem" in item:
            out.setdefault("_generated_file_stem", item.get("file_stem"))
        if "row_index" in item:
            out.setdefault("_generated_row_index", item.get("row_index"))
        return out
    return item


def _category_from_item(item: dict[str, Any]) -> str:
    category = item.get("type")
    if category:
        return str(category)
    source_type = item.get("source_type")
    variant = item.get("camera_rewrite_variant")
    if source_type and variant:
        return f"{source_type}-{variant}"
    if source_type:
        return str(source_type)
    return ""


def _base_set_category(category: str) -> str:
    for prefix in ("set1", "set2", "set3"):
        if category == prefix or category.startswith(f"{prefix}-"):
            return prefix
    return category


def _bucket_category(meta: dict[str, Any], fallback_category: str) -> str:
    # The generated filename category is the evaluation bucket. Do not collapse
    # variants such as set3-large / set3-medium-large back into set3.
    if fallback_category:
        return fallback_category
    source_type = meta.get("source_type")
    if source_type:
        variant = meta.get("camera_rewrite_variant")
        if variant:
            return f"{source_type}-{variant}"
        return str(source_type)
    return ""


def _coerce_prompt_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_coerce_prompt_text(v) for v in value]
        return " ".join(v for v in parts if v).strip()
    if isinstance(value, dict):
        text = value.get("text")
        return str(text).strip() if text is not None else ""
    return str(value).strip()


def _sidecar_meta(video_path: Path, source_category: str, category_index: int) -> dict[str, Any] | None:
    sidecar = video_path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        data = read_json(sidecar)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    source = _source_item(data)
    prompt = (
        _coerce_prompt_text(data.get("prompt"))
        or _coerce_prompt_text(source.get("av_caption"))
        or _coerce_prompt_text(source.get("prompt"))
        or _coerce_prompt_text(source.get("video_prompt"))
    )
    audio_prompt = (
        _coerce_prompt_text(source.get("audio_prompt"))
        or _coerce_prompt_text(data.get("audio_prompt"))
        or prompt
    )
    return {
        "prompt_en": prompt,
        "prompt_vision": prompt,
        "prompt_audio": audio_prompt,
        "dimension": list(DEFAULT_DIMENSIONS),
        "category": _bucket_category(source, source_category),
        "source_category": source_category,
        "category_index": category_index,
        "raw_source": source,
        "sidecar_json": str(sidecar),
    }


def load_valid_jsonl(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Index the JSONL by ``(type, index)``.

    Index resolution mirrors the inference driver
    (``sweep_t2av_ckpts.make_stem`` / ``infer_t2av._resolve_sample_index``):
    use ``int(row["index"])`` when it parses, otherwise fall back to
    ``INDEX_FALLBACK_OFFSET + line_idx``. On-disk filenames embed exactly
    this value as the 4+ digit ``<index>`` segment, so eval-side dict keys
    must follow the same rule or every fallback row gets "missing
    metadata" downstream. We also attach ``_row_index`` to the item dict
    for the same reason ``infer_t2av.load_manifest`` does.
    """
    out: dict[str, dict[int, dict[str, Any]]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            item = _source_item(json.loads(line))
            category = _category_from_item(item)
            if not category:
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                index = INDEX_FALLBACK_OFFSET + line_idx
            item.setdefault("_row_index", line_idx)
            out.setdefault(category, {})[index] = item
    return out


def decode_mapping_prompt(value: str) -> str:
    try:
        return json.loads('"' + value + '"')
    except Exception:
        return value


def load_prompt_to_slug(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            prompt = decode_mapping_prompt(row.get("prompt", ""))
            slug = sanitize_stem(row.get("idx", ""))
            if prompt and slug and prompt not in mapping:
                mapping[prompt] = slug
    return mapping


def sanitize_stem(value: str) -> str:
    value = str(value).strip().replace("/", "_").replace("\\", "_")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\x00-\x1f]", "_", value)
    return value[:180].strip(" .") or "sample"


def existing_path(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    return str(path) if path.is_file() else ""


def load_category_records(
    raw_dir: Path,
    valid: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[int, dict[str, Any]]]:
    categories: dict[str, dict[int, dict[str, Any]]] = {}
    if raw_dir.is_dir():
        for json_file in sorted(raw_dir.glob("*.json")):
            category = json_file.stem
            data = read_json(json_file)
            if not isinstance(data, list):
                continue
            for index, item in enumerate(data, 1):
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                item.setdefault("prompt_en", item.get("prompt") or item.get("av_caption") or "")
                item.setdefault("prompt_vision", item.get("prompt_en", ""))
                item.setdefault("prompt_audio", item.get("prompt_en", ""))
                item.setdefault("dimension", list(DEFAULT_DIMENSIONS))
                item["category"] = category
                item["category_index"] = index
                categories.setdefault(category, {})[index] = item

    for category, by_index in valid.items():
        for index, item in by_index.items():
            if category in categories and index in categories[category]:
                continue
            caption = (
                _coerce_prompt_text(item.get("av_caption"))
                or _coerce_prompt_text(item.get("prompt"))
                or _coerce_prompt_text(item.get("video_prompt"))
            )
            audio_caption = _coerce_prompt_text(item.get("audio_prompt")) or caption
            categories.setdefault(category, {})[index] = {
                "prompt_en": caption,
                "prompt_vision": caption,
                "prompt_audio": audio_caption,
                "dimension": list(DEFAULT_DIMENSIONS),
                "category": _bucket_category(item, category),
                "source_category": category,
                "category_index": index,
                "raw_source": item,
            }
    return categories


def parse_sample_file(path: Path) -> tuple[str, int] | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return match.group("category"), int(match.group("index"))


def make_symlink(target: Path, link: Path, force: bool) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if not force:
            return
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    os.symlink(target, link)


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    valid_jsonl = Path(args.valid_jsonl).expanduser().resolve()
    raw_json_dir = Path(args.vabench_json_dir).expanduser().resolve()
    mapping_path = Path(args.mapping).expanduser().resolve()

    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample directory not found: {sample_dir}")

    valid = load_valid_jsonl(valid_jsonl)
    category_records = load_category_records(raw_json_dir, valid)
    prompt_to_slug = load_prompt_to_slug(mapping_path)
    allow_categories = set(args.categories or [])

    by_category_count: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    used_stems: set[str] = set()

    candidate_videos = sorted(
        path for path in sample_dir.glob("*.mp4")
        if not path.name.endswith(".av.mp4")
    )
    if not candidate_videos:
        raise RuntimeError(
            f"No non-.av video files found in {sample_dir}. Expected files named like "
            "'sample-<dataset>-<index>-<category>.mp4'."
        )

    for video_path in candidate_videos:
        parsed = parse_sample_file(video_path)
        if not parsed:
            skipped.append({"file": str(video_path), "reason": "unrecognized filename"})
            continue
        source_category, category_index = parsed
        parsed_bucket = _bucket_category({}, source_category)
        parsed_base = _base_set_category(source_category)
        if (
            allow_categories
            and source_category not in allow_categories
            and parsed_bucket not in allow_categories
            and parsed_base not in allow_categories
        ):
            continue
        if args.max_total and len(selected) >= args.max_total:
            break
        meta = category_records.get(source_category, {}).get(category_index)
        if not meta:
            meta = _sidecar_meta(video_path, source_category, category_index)
        if not meta:
            skipped.append({"file": str(video_path), "reason": "missing metadata"})
            continue

        audio_path = video_path.with_suffix(".wav")
        if not audio_path.is_file():
            skipped.append({"file": str(video_path), "reason": "missing sidecar .wav"})
            continue

        prompt = (
            _coerce_prompt_text(meta.get("prompt_en"))
            or _coerce_prompt_text(meta.get("prompt"))
            or _coerce_prompt_text(valid.get(source_category, {}).get(category_index, {}).get("av_caption", ""))
            or ""
        )
        category = _bucket_category(meta, source_category)
        if args.max_per_category and by_category_count.get(category, 0) >= args.max_per_category:
            continue
        stem_base = prompt_to_slug.get(prompt) or video_path.stem
        stem = sanitize_stem(stem_base)
        if stem in used_stems:
            stem = sanitize_stem(f"{stem_base}-{source_category}-{category_index:04d}")
        suffix = 2
        while stem in used_stems:
            stem = sanitize_stem(f"{stem_base}-{source_category}-{category_index:04d}-{suffix}")
            suffix += 1
        used_stems.add(stem)

        by_category_count[category] = by_category_count.get(category, 0) + 1
        seq = len(selected) + 1
        raw_source = meta.get("raw_source") if isinstance(meta.get("raw_source"), dict) else {}
        speech_prompt = raw_source.get("speech_prompt") or meta.get("speech_prompt") or {}
        reference_image_path = (
            existing_path(raw_source.get("first_frame_path"))
            or existing_path(meta.get("first_frame_path"))
            or existing_path(raw_source.get("reference_image_path"))
            or existing_path(meta.get("reference_image_path"))
        )
        reference_audio_path = (
            existing_path(raw_source.get("reference_audio_path"))
            or existing_path(meta.get("reference_audio_path"))
            or existing_path(raw_source.get("audio_path"))
            or existing_path(meta.get("audio_path"))
        )
        record = {
            "seq": seq,
            "category": category,
            "category_index": category_index,
            "source_category": source_category,
            "file_stem": stem,
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "prompt": prompt,
            "av_caption": prompt,
            "video_prompt": _coerce_prompt_text(meta.get("prompt_vision")) or prompt,
            "audio_prompt": _coerce_prompt_text(meta.get("prompt_audio")) or prompt,
            "speech_prompt": speech_prompt if isinstance(speech_prompt, dict) else {},
            "speech_text": _coerce_prompt_text(speech_prompt.get("text")) if isinstance(speech_prompt, dict) else "",
            "reference_image_path": reference_image_path,
            "reference_audio_path": reference_audio_path,
            "raw_meta": meta,
        }
        selected.append(record)

    if not selected:
        raise RuntimeError(f"No usable videos selected from {sample_dir}")

    metadata_dir = output_dir / "metadata"
    manifest_json = metadata_dir / "manifest.json"
    manifest_csv = metadata_dir / "manifest.csv"
    vabench_view = output_dir / "vabench_view"
    vabench_video = vabench_view / "video"
    vabench_audio = vabench_view / "audio"
    vabench_json = vabench_view / "json"

    per_category_json: dict[str, list[dict[str, Any]]] = {}
    for rec in selected:
        src_video = Path(rec["video_path"])
        src_audio = Path(rec["audio_path"])
        link_video = vabench_video / rec["category"] / f"{rec['file_stem']}.mp4"
        link_audio = vabench_audio / rec["category"] / f"{rec['file_stem']}.wav"
        make_symlink(src_video, link_video, args.force)
        make_symlink(src_audio, link_audio, args.force)

        vabench_item = dict(rec["raw_meta"])
        vabench_item["file_stem"] = rec["file_stem"]
        vabench_item["category"] = rec["category"]
        vabench_item["category_index"] = rec["category_index"]
        vabench_item.setdefault("prompt_en", rec["prompt"])
        vabench_item.setdefault("prompt_vision", rec["video_prompt"])
        vabench_item.setdefault("prompt_audio", rec["audio_prompt"])
        per_category_json.setdefault(rec["category"], []).append(vabench_item)

    for category, items in per_category_json.items():
        write_json(vabench_json / f"{category}.json", items)

    target_info = infer_target(sample_dir)
    payload_records = [
        {key: value for key, value in rec.items() if key != "raw_meta"}
        for rec in selected
    ]
    manifest_payload = {
        "target": target_info,
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "vabench_view_dir": str(vabench_view),
        "num_selected": len(selected),
        "categories": dict(sorted(by_category_count.items())),
        "skipped": skipped,
        "records": payload_records,
    }
    write_json(manifest_json, manifest_payload)
    write_manifest_csv(manifest_csv, payload_records)

    summary = {key: value for key, value in manifest_payload.items() if key != "records"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def infer_target(sample_dir: Path) -> dict[str, str]:
    parts = sample_dir.resolve().parts
    target = {"experiment": "", "step": "", "cfg": sample_dir.name}
    if len(parts) >= 4 and parts[-2] == "joint_av":
        target["step"] = parts[-3]
        target["experiment"] = parts[-5] if len(parts) >= 5 else ""
    elif len(parts) >= 2:
        target["cfg"] = parts[-1]
    return target


def write_manifest_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seq",
        "category",
        "category_index",
        "file_stem",
        "video_path",
        "audio_path",
        "prompt",
        "video_prompt",
        "audio_prompt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec.get(key, "") for key in fields})


if __name__ == "__main__":
    raise SystemExit(main())
