"""Per-sample / summary JSON writers and consolidation helpers."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np


def per_sample_dir(target_dir: Path, kind: str) -> Path:
    return target_dir / "per_sample" / kind


def summary_path(target_dir: Path, kind: str) -> Path:
    return target_dir / "summary" / f"{kind}.json"


def per_sample_path(target_dir: Path, kind: str, file_stem: str) -> Path:
    return per_sample_dir(target_dir, kind) / f"{file_stem}.json"


def already_done(target_dir: Path, kind: str, file_stem: str, required_keys: Optional[Iterable[str]] = None) -> bool:
    p = per_sample_path(target_dir, kind, file_stem)
    if not p.is_file():
        return False
    if p.stat().st_size == 0:
        return False
    if required_keys:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        skipped = set(data.get("_skipped_metrics") or [])
        for key in required_keys:
            if key not in data and key not in skipped:
                return False
    return True


def write_per_sample(target_dir: Path, kind: str, file_stem: str, payload: Dict[str, Any]) -> None:
    out = per_sample_path(target_dir, kind, file_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(out)


def _is_real(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _bucket_for_record(rec: dict) -> str:
    cat = rec.get("category", "")
    return cat if cat else "all"


def _ordered_count_buckets(*maps: Dict[str, Any]) -> List[str]:
    preferred = ["set1", "set2", "set3", "set3-large", "set3-medium-large", "all"]
    seen = set()
    for m in maps:
        seen.update(m.keys())
    return [b for b in preferred if b in seen] + sorted(b for b in seen if b not in preferred)


def consolidate_summary(
    target_dir: Path,
    kind: str,
    manifest: dict,
    metric_keys: List[str],
    extra_payload: Optional[Dict[str, Any]] = None,
    eligible_categories: Optional[Set[str]] = None,
) -> Path:
    """Walk per_sample/<kind>/*.json, group scores by manifest record's category,
    and write summary/<kind>.json with per-set means + overall mean."""
    stem_to_cat = {rec["file_stem"]: rec.get("category", "") for rec in manifest.get("records", [])}
    per_kind_dir = per_sample_dir(target_dir, kind)
    scores: Dict[str, Dict[str, List[float]]] = {k: defaultdict(list) for k in metric_keys}
    successes: Dict[str, Dict[str, int]] = {k: defaultdict(int) for k in metric_keys}
    failures: Dict[str, Dict[str, int]] = {k: defaultdict(int) for k in metric_keys}
    skipped: Dict[str, Dict[str, int]] = {k: defaultdict(int) for k in metric_keys}
    counts: Dict[str, int] = defaultdict(int)
    num_files = 0
    if per_kind_dir.is_dir():
        for json_file in sorted(per_kind_dir.glob("*.json")):
            num_files += 1
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            stem = json_file.stem
            bucket = stem_to_cat.get(stem, "")
            counts[bucket] += 1
            counts["all"] += 1
            is_skipped = eligible_categories is not None and bucket not in eligible_categories
            skipped_metrics = set(data.get("_skipped_metrics") or [])
            for key in metric_keys:
                if is_skipped or key in skipped_metrics:
                    skipped[key][bucket] += 1
                    skipped[key]["all"] += 1
                    continue
                val = data.get(key)
                if _is_real(val):
                    scores[key][bucket].append(float(val))
                    scores[key]["all"].append(float(val))
                    successes[key][bucket] += 1
                    successes[key]["all"] += 1
                else:
                    failures[key][bucket] += 1
                    failures[key]["all"] += 1

    summary: Dict[str, Any] = {
        "metric_kind": kind,
        "scores": {},
        "num_samples": {b: int(counts.get(b, 0)) for b in _ordered_count_buckets(counts)},
        "num_success": {},
        "num_failed": {},
        "num_skipped": {},
    }
    for key in metric_keys:
        bucket_means: Dict[str, Optional[float]] = {}
        for bucket, vals in scores[key].items():
            bucket_means[bucket] = float(np.mean(vals)) if vals else None
        summary["scores"][key] = bucket_means

        buckets = _ordered_count_buckets(
            counts,
            successes[key],
            failures[key],
            skipped[key],
        )
        summary["num_success"][key] = {b: int(successes[key].get(b, 0)) for b in buckets}
        summary["num_failed"][key] = {b: int(failures[key].get(b, 0)) for b in buckets}
        summary["num_skipped"][key] = {b: int(skipped[key].get(b, 0)) for b in buckets}
    if extra_payload:
        summary.update(extra_payload)
    out = summary_path(target_dir, kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out


def merge_all_metrics_summary(target_dir: Path) -> Optional[Path]:
    summary_dir = target_dir / "summary"
    if not summary_dir.is_dir():
        return None
    merged: Dict[str, Any] = {}
    for json_file in sorted(summary_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            data = {"error": str(exc)}
        merged[json_file.stem] = data
    out = target_dir / "all_metrics_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return out


def iter_per_sample_dicts(target_dir: Path, kind: str) -> Iterable[tuple[str, Dict[str, Any]]]:
    d = per_sample_dir(target_dir, kind)
    if not d.is_dir():
        return
    for jf in sorted(d.glob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        yield jf.stem, payload
