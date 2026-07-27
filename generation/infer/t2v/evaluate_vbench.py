"""Run VBench evaluation against a directory of generated videos.

The script wraps the ``vbench`` Python package. It supports two modes:

* ``vbench_standard`` (default): uses VBench's official prompt list and expects
  videos named ``<prompt>-<i>.mp4``. ``generate_vbench_videos.sh`` produces the
  expected layout.

* ``custom_input``: evaluate any directory + a JSONL/JSON list of
  ``{"video_path": ..., "prompt": ...}`` records. Useful for quick sanity
  checks on a small subset.

Each requested dimension is evaluated independently. If multiple dimensions
are requested they are scored serially with the same VBench instance to
amortise the model loading.

Outputs:
    <output-dir>/<dimension>_eval_results.json  (raw VBench output)
    <output-dir>/summary.json                   (collected scores)
    <output-dir>/run.json                       (call args)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

VBENCH_DIMENSIONS_QUALITY = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]
VBENCH_DIMENSIONS_SEMANTIC = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]
VBENCH_ALL_DIMENSIONS = VBENCH_DIMENSIONS_QUALITY + VBENCH_DIMENSIONS_SEMANTIC


def parse_dimensions(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return list(VBENCH_ALL_DIMENSIONS)
    if value.lower() == "quality":
        return list(VBENCH_DIMENSIONS_QUALITY)
    if value.lower() == "semantic":
        return list(VBENCH_DIMENSIONS_SEMANTIC)
    dims = [d.strip() for d in value.split(",") if d.strip()]
    invalid = [d for d in dims if d not in VBENCH_ALL_DIMENSIONS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown VBench dimension(s): {invalid}. Valid: {VBENCH_ALL_DIMENSIONS}"
        )
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory containing the generated videos to score.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write VBench raw + summary JSONs.")
    parser.add_argument(
        "--dimensions",
        type=str,
        default="all",
        help="Comma-separated subset of VBench dimensions, or 'all' / 'quality' / 'semantic'.",
    )
    parser.add_argument(
        "--mode",
        choices=("vbench_standard", "custom_input"),
        default="vbench_standard",
        help="Whether to evaluate against VBench's standard prompt suite or a custom prompt list.",
    )
    parser.add_argument(
        "--full-info-json",
        type=Path,
        default=None,
        help="Path to VBench_full_info.json. If unset we point vbench at its bundled copy.",
    )
    parser.add_argument(
        "--custom-input-json",
        type=Path,
        default=None,
        help="Required when --mode=custom_input. JSON or JSONL of {video_path, prompt} records.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="anytok_t2v",
        help="Run name (becomes a subdir inside --output-dir for VBench's raw outputs).",
    )
    parser.add_argument(
        "--read-frame",
        action="store_true",
        help="Pre-decode frames into PIL images instead of re-decoding at every step. Recommended for short videos.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local cached weights only; do not attempt to redownload.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory for VBench checkpoints (defaults to ~/.cache/vbench).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If a single dimension fails, log the failure and keep going.",
    )
    return parser.parse_args()


def configure_cache(cache_dir: Path | None) -> Path:
    """Direct VBench / HuggingFace caches to the requested location.

    VBench downloads its own checkpoints into ``~/.cache/vbench``. We honour
    the same convention but allow overriding via ``--cache-dir`` to keep the
    project's data on shared / large disks.
    """

    target = cache_dir or Path(os.environ.get("VBENCH_CACHE_DIR", str(Path.home() / ".cache" / "vbench")))
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VBENCH_CACHE_DIR", str(target))
    # Newer VBench checkpoints come from HuggingFace; keep their cache nearby.
    hf_cache = target / "hf"
    hf_cache.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    return target


def load_custom_input(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"--custom-input-json not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict) and "videos" in payload:
            records = payload["videos"]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported custom input shape in {path}")
    cleaned = []
    for record in records:
        if not isinstance(record, dict):
            continue
        video = record.get("video_path") or record.get("video") or record.get("video_basename")
        prompt = record.get("prompt") or record.get("prompt_en") or record.get("text")
        if not video or not prompt:
            continue
        cleaned.append({"video_path": str(video), "prompt_en": str(prompt)})
    return cleaned


def evaluate_dimensions(args: argparse.Namespace, dimensions: Iterable[str], output_dir: Path) -> dict[str, Any]:
    try:
        from vbench import VBench
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("vbench is not installed. Run `pip install vbench` (or `pip install -r requirements.txt`).") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    full_info = args.full_info_json
    if full_info is None:
        try:
            from vbench import VBenchFullInfo  # type: ignore[attr-defined]
            full_info = Path(VBenchFullInfo)
        except Exception:
            full_info = None
    if full_info is None:
        # Fall back to the json we downloaded under prompts/.
        candidate = Path("data/t2v/vbench/prompts/VBench_full_info.json")
        if candidate.is_file():
            full_info = candidate
    if full_info is not None and not Path(full_info).is_file():
        raise SystemExit(f"--full-info-json points at non-existent path: {full_info}")
    print(f"VBench full_info: {full_info}", file=sys.stderr)

    vb_runner = VBench(args.device, str(full_info) if full_info else "", str(output_dir))

    summary: dict[str, Any] = {
        "videos_dir": str(args.videos_dir),
        "name": args.name,
        "mode": args.mode,
        "results": {},
    }

    for dimension in dimensions:
        print(f"\n=== Evaluating dimension: {dimension} ===", file=sys.stderr)
        start = time.time()
        try:
            kwargs: dict[str, Any] = dict(
                videos_path=str(args.videos_dir),
                name=f"{args.name}_{dimension}",
                dimension_list=[dimension],
                local=bool(args.local),
                read_frame=bool(args.read_frame),
                mode=args.mode,
            )
            if args.mode == "custom_input" and args.custom_input_json is not None:
                kwargs["custom_input"] = load_custom_input(args.custom_input_json.expanduser().resolve())
            vb_runner.evaluate(**kwargs)
        except Exception as exc:
            print(f"Dimension {dimension} failed: {exc}", file=sys.stderr)
            summary["results"][dimension] = {"error": str(exc)}
            if not args.continue_on_error:
                raise
            continue

        # VBench writes <name>_eval_results.json into output_dir.
        result_files = sorted(output_dir.glob(f"{args.name}_{dimension}_eval_results.json"))
        if not result_files:
            # New VBench versions name files differently; fall back to all matching files
            result_files = sorted(output_dir.glob(f"*{dimension}*eval_results.json"))
        if not result_files:
            summary["results"][dimension] = {"error": "no result file produced"}
            print(f"WARNING: no result file produced for {dimension}.", file=sys.stderr)
            continue

        target = result_files[-1]
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary["results"][dimension] = {"error": f"invalid json in {target}: {exc}"}
            continue
        score = None
        if isinstance(payload, dict) and dimension in payload:
            entry = payload[dimension]
            if isinstance(entry, list) and entry:
                score = entry[0]
            elif isinstance(entry, (int, float)):
                score = float(entry)
            elif isinstance(entry, dict):
                # Some versions store {"score": .., "video_results": [...]}
                score = entry.get("score")
        summary["results"][dimension] = {
            "score": score,
            "raw_path": str(target),
            "elapsed_seconds": round(time.time() - start, 2),
        }
        print(f"  -> score = {score} (raw: {target.name})", file=sys.stderr)

    # Compute aggregate statistics.
    valid = [v["score"] for v in summary["results"].values() if isinstance(v, dict) and isinstance(v.get("score"), (int, float))]
    summary["aggregates"] = {
        "n_dimensions": len(summary["results"]),
        "n_dimensions_with_score": len(valid),
        "mean_score": round(sum(valid) / len(valid), 6) if valid else None,
    }

    return summary


def main() -> None:
    args = parse_args()
    cache_root = configure_cache(args.cache_dir)
    print(f"VBench cache root: {cache_root}", file=sys.stderr)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.videos_dir.expanduser().resolve()
    if not videos_dir.is_dir():
        raise SystemExit(f"--videos-dir not found: {videos_dir}")

    dimensions = parse_dimensions(args.dimensions)
    print(f"Evaluating {len(dimensions)} dimension(s): {dimensions}", file=sys.stderr)

    run_record = {
        "videos_dir": str(videos_dir),
        "output_dir": str(output_dir),
        "dimensions": dimensions,
        "mode": args.mode,
        "name": args.name,
        "read_frame": bool(args.read_frame),
        "local": bool(args.local),
        "device": str(args.device),
        "cache_dir": str(cache_root),
    }
    (output_dir / "run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8")

    summary = evaluate_dimensions(args, dimensions, output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote summary to {summary_path}", file=sys.stderr)

    # Pretty-print to stdout for shell capture.
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
