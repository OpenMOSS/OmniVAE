"""Aggregate VBench per-dimension results into CSV / JSON / Markdown.

Walks an output directory produced by ``evaluate.sh`` (or ``evaluate_vbench.py``),
collects every ``summary.json`` and per-dimension ``*_eval_results.json`` it finds,
and produces a flat per-dimension table plus VBench's two top-level aggregates
(``Quality Score`` and ``Semantic Score``) using the official weighting [1].

[1] https://github.com/Vchitect/VBench/blob/master/vbench/utils.py#L13 (weights
    are mirrored here so we don't depend on a specific vbench version layout).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# Mirror VBench's official weights so we can compute the headline numbers
# without round-tripping through the vbench package.
QUALITY_WEIGHTS: dict[str, float] = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "temporal_flickering": 1.0,
    "motion_smoothness": 1.0,
    "dynamic_degree": 0.5,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
}
SEMANTIC_WEIGHTS: dict[str, float] = {
    "object_class": 1.0,
    "multiple_objects": 1.0,
    "human_action": 1.0,
    "color": 1.0,
    "spatial_relationship": 1.0,
    "scene": 1.0,
    "appearance_style": 1.0,
    "temporal_style": 1.0,
    "overall_consistency": 1.0,
}
TOTAL_QUALITY_WEIGHT = 0.5  # for the final "Total Score"
TOTAL_SEMANTIC_WEIGHT = 0.5


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if value:
            return _coerce_score(value[0])
    if isinstance(value, dict):
        for key in ("score", "value"):
            if key in value:
                return _coerce_score(value[key])
    return None


def _collect_summary_jsons(results_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("**/summary*.json")):
        if path.name == "summary.csv":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            summaries.append({"path": path, "payload": payload})
    return summaries


def _collect_eval_jsons(results_dir: Path) -> dict[str, float]:
    """Fallback: read VBench's raw <name>_<dim>_eval_results.json files."""

    scores: dict[str, float] = {}
    for path in sorted(results_dir.glob("**/*eval_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for dim, value in payload.items():
            score = _coerce_score(value)
            if score is None:
                continue
            # Prefer earlier files so users can override by renaming.
            scores.setdefault(dim, score)
    return scores


def aggregate(results_dir: Path) -> dict[str, Any]:
    summaries = _collect_summary_jsons(results_dir)
    merged_scores: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for entry in summaries:
        sources.append(str(entry["path"]))
        results = entry["payload"].get("results", {})
        for dim, info in results.items():
            score = _coerce_score(info if not isinstance(info, dict) else info.get("score"))
            if score is None:
                continue
            existing = merged_scores.get(dim)
            if existing is None:
                merged_scores[dim] = {"score": score, "source": str(entry["path"])}

    raw_scores = _collect_eval_jsons(results_dir)
    for dim, score in raw_scores.items():
        merged_scores.setdefault(dim, {"score": score, "source": "raw_eval_results.json"})

    quality_components = [(d, w) for d, w in QUALITY_WEIGHTS.items() if d in merged_scores]
    semantic_components = [(d, w) for d, w in SEMANTIC_WEIGHTS.items() if d in merged_scores]

    def _weighted_mean(components: Iterable[tuple[str, float]]) -> float | None:
        items = list(components)
        if not items:
            return None
        total_w = sum(w for _, w in items)
        if total_w == 0:
            return None
        return sum(merged_scores[d]["score"] * w for d, w in items) / total_w

    quality_score = _weighted_mean(quality_components)
    semantic_score = _weighted_mean(semantic_components)
    if quality_score is not None and semantic_score is not None:
        total_score = (
            TOTAL_QUALITY_WEIGHT * quality_score + TOTAL_SEMANTIC_WEIGHT * semantic_score
        )
    elif quality_score is not None:
        total_score = quality_score
    elif semantic_score is not None:
        total_score = semantic_score
    else:
        total_score = None

    return {
        "results_dir": str(results_dir),
        "sources": sources,
        "scores": merged_scores,
        "headline": {
            "quality_score": quality_score,
            "semantic_score": semantic_score,
            "total_score": total_score,
        },
    }


def write_csv(aggregate_payload: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for dim, info in sorted(aggregate_payload["scores"].items()):
        rows.append({"dimension": dim, "score": info["score"], "source": info["source"]})
    headline = aggregate_payload["headline"]
    rows.append({"dimension": "__quality_score__", "score": headline.get("quality_score"), "source": ""})
    rows.append({"dimension": "__semantic_score__", "score": headline.get("semantic_score"), "source": ""})
    rows.append({"dimension": "__total_score__", "score": headline.get("total_score"), "source": ""})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dimension", "score", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(aggregate_payload: dict[str, Any], path: Path) -> None:
    headline = aggregate_payload["headline"]
    lines: list[str] = []
    lines.append("# VBench Evaluation Summary")
    lines.append("")
    lines.append(f"- Source: `{aggregate_payload['results_dir']}`")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("| :-- | --: |")
    lines.append(f"| Quality Score | {headline.get('quality_score')!s} |")
    lines.append(f"| Semantic Score | {headline.get('semantic_score')!s} |")
    lines.append(f"| **Total Score** | {headline.get('total_score')!s} |")
    lines.append("")
    lines.append("## Per-dimension")
    lines.append("")
    lines.append("| Dimension | Score | Source |")
    lines.append("| :-- | --: | :-- |")
    for dim, info in sorted(aggregate_payload["scores"].items()):
        lines.append(f"| {dim} | {info['score']:.6f} | `{info['source']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="VBench eval output directory.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir not found: {results_dir}")
    payload = aggregate(results_dir)
    if not payload["scores"]:
        print("WARNING: no scores found under results_dir.", file=sys.stderr)

    if args.output_csv is not None:
        write_csv(payload, args.output_csv.expanduser().resolve())
        print(f"Wrote {args.output_csv}")
    if args.output_json is not None:
        args.output_json.expanduser().resolve().write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Wrote {args.output_json}")
    if args.output_md is not None:
        write_markdown(payload, args.output_md.expanduser().resolve())
        print(f"Wrote {args.output_md}")

    print(json.dumps(payload["headline"], indent=2))


if __name__ == "__main__":
    main()
