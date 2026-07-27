"""Resolve Verse-Bench reference assets/text for my_eval metric tasks."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_VERSE_BENCH_DIR = REPO_ROOT / "generation" / "evaluation" / "verse_bench"
DEFAULT_VERSE_DATASET_DIR = DEFAULT_VERSE_BENCH_DIR / "verse_bench"


def verse_dataset_root() -> Path:
    explicit = os.environ.get("MY_EVAL_VERSE_DATASET_ROOT") or os.environ.get("VERSE_BENCH_DATASET_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    return DEFAULT_VERSE_DATASET_DIR


def _base_set_category(category: str) -> str:
    for prefix in ("set1", "set2", "set3"):
        if category == prefix or category.startswith(f"{prefix}-"):
            return prefix
    return category


def _path_if_file(value: Any) -> str | None:
    if not value:
        return None
    p = Path(str(value)).expanduser()
    return str(p) if p.is_file() else None


def _candidate_base_names(rec: dict[str, Any]) -> list[str]:
    names: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        text = str(value)
        if text and text not in names:
            names.append(text)

    idx = rec.get("category_index")
    try:
        idx_int = int(idx)
    except (TypeError, ValueError):
        idx_int = None
    if idx_int is not None:
        add(idx_int)
        add(f"{idx_int:05d}")
        add(f"{idx_int:04d}")
    else:
        add(idx)

    stem = str(rec.get("file_stem") or "")
    match = re.search(r"-(\d+)-[^-/]+$", stem)
    if match:
        raw = match.group(1)
        add(raw)
        try:
            raw_int = int(raw)
        except ValueError:
            raw_int = None
        if raw_int is not None:
            add(raw_int)
            add(f"{raw_int:05d}")

    return names


def _candidate_dirs(rec: dict[str, Any]) -> list[Path]:
    root = verse_dataset_root()
    source_category = str(rec.get("source_category") or rec.get("category") or "")
    set_name = _base_set_category(source_category)
    set_root = root / set_name
    dirs = [set_root, set_root / "data", set_root / "clips"]
    unique: list[Path] = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


def _candidate_files(rec: dict[str, Any], suffixes: Iterable[str]) -> Iterable[Path]:
    for directory in _candidate_dirs(rec):
        for base in _candidate_base_names(rec):
            for suffix in suffixes:
                yield directory / f"{base}{suffix}"


@lru_cache(maxsize=8192)
def _read_json_cached(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def resolve_reference_json(rec: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    explicit = _path_if_file(rec.get("reference_json_path"))
    if explicit:
        return explicit, _read_json_cached(explicit)
    for candidate in _candidate_files(rec, [".json"]):
        if candidate.is_file():
            path_str = str(candidate)
            return path_str, _read_json_cached(path_str)
    return None, None


def resolve_reference_image(rec: dict[str, Any]) -> str | None:
    for key in ("reference_image_path", "first_frame_path", "image_path"):
        found = _path_if_file(rec.get(key))
        if found:
            return found
    _, item = resolve_reference_json(rec)
    if item:
        for key in ("first_frame_path", "image_path", "reference_image_path"):
            found = _path_if_file(item.get(key))
            if found:
                return found
    for candidate in _candidate_files(rec, [".jpg", ".jpeg", ".png"]):
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_reference_audio(rec: dict[str, Any]) -> str | None:
    for key in ("reference_audio_path", "ref_audio_path", "source_audio_path"):
        found = _path_if_file(rec.get(key))
        if found:
            return found
    _, item = resolve_reference_json(rec)
    if item:
        for key in ("reference_audio_path", "ref_audio_path", "source_audio_path", "audio_path"):
            found = _path_if_file(item.get(key))
            if found:
                return found
    for candidate in _candidate_files(rec, [".wav", ".mp3", ".m4a", ".flac"]):
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_speech_text(rec: dict[str, Any]) -> str:
    for key in ("speech_text", "target_text"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    speech_prompt = rec.get("speech_prompt")
    if isinstance(speech_prompt, dict):
        value = speech_prompt.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    _, item = resolve_reference_json(rec)
    if item:
        speech_prompt = item.get("speech_prompt")
        if isinstance(speech_prompt, dict):
            value = speech_prompt.get("text")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""
