"""Convert VBench's prompt files into OmniVAE-compatible JSONL metadata.

The downstream `scripts/eval/export_video_checkpoint_samples.py`
expects a JSON or JSONL file with one record per prompt. Each record must contain
a `prompt` field (configurable) and we add VBench-specific fields so the evaluation
side can correlate the generated video back to the dimensions it covers.

Output schema (one JSON object per line):
    {
        "prompt_id":   <int, 0-indexed within the source list>,
        "prompt":      <str>,
        "dimension":   <list[str], the dimension(s) this prompt is part of>,
        "negative_prompt": "",
        "source_file": <str, basename of the file in the VBench prompts folder>
    }

When VBench_full_info.json is available we additionally populate
`auxiliary_info` (e.g. object class, color, etc.) so it can be carried into the
generation manifest and surfaced again at evaluation time.

Default I/O:
    in:   data/t2v/vbench/prompts/
    out:  examples/metadata/vbench.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data/t2v/vbench/prompts"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "examples/metadata/vbench.jsonl"

# Map filename -> dimension(s) it counts toward.
PER_DIMENSION_FILE_TO_DIMENSIONS: dict[str, list[str]] = {
    "subject_consistency.txt": [
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
    ],
    "scene.txt": ["scene", "background_consistency"],
    "temporal_flickering.txt": ["temporal_flickering"],
    "overall_consistency.txt": [
        "overall_consistency",
        "aesthetic_quality",
        "imaging_quality",
    ],
    "object_class.txt": ["object_class"],
    "multiple_objects.txt": ["multiple_objects"],
    "human_action.txt": ["human_action"],
    "color.txt": ["color"],
    "spatial_relationship.txt": ["spatial_relationship"],
    "appearance_style.txt": ["appearance_style"],
    "temporal_style.txt": ["temporal_style"],
}


def read_prompt_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_full_info(path: Path) -> dict[str, dict[str, Any]]:
    """Index VBench_full_info.json by prompt_en for quick lookup."""

    if not path.is_file():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"WARNING: failed to parse {path}: {exc}", file=sys.stderr)
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(records, list):
        return indexed
    for entry in records:
        if not isinstance(entry, dict):
            continue
        prompt = entry.get("prompt_en") or entry.get("prompt") or ""
        if not prompt:
            continue
        indexed[prompt] = entry
    return indexed


def gather_prompt_to_dimensions(prompts_dir: Path) -> dict[str, set[str]]:
    """Walk per-dimension files to learn which dimensions each prompt belongs to."""

    mapping: dict[str, set[str]] = defaultdict(set)
    per_dim_dir = prompts_dir / "prompts_per_dimension"
    if not per_dim_dir.is_dir():
        print(f"WARNING: {per_dim_dir} does not exist; falling back to all_dimension.txt only.", file=sys.stderr)
        return mapping
    for filename, dimensions in PER_DIMENSION_FILE_TO_DIMENSIONS.items():
        file_path = per_dim_dir / filename
        prompts = read_prompt_file(file_path)
        for prompt in prompts:
            for dimension in dimensions:
                mapping[prompt].add(dimension)
    return mapping


def select_source_prompts(prompts_dir: Path, source: str) -> tuple[list[str], str]:
    """Choose the master prompt list."""

    if source == "all_dimension":
        path = prompts_dir / "all_dimension.txt"
    elif source == "all_dimension_longer":
        path = prompts_dir / "augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt"
    else:
        path = prompts_dir / source
    prompts = read_prompt_file(path)
    if not prompts:
        raise FileNotFoundError(f"Source prompt file is empty or missing: {path}")
    return prompts, str(path.relative_to(prompts_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR, help="Directory containing the downloaded VBench prompts.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Where to write the OmniVAE-compatible JSONL.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all_dimension",
        help="Which top-level prompt list to use: 'all_dimension', 'all_dimension_longer', or a relative file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prompts (debug).",
    )
    parser.add_argument(
        "--full-info",
        type=Path,
        default=None,
        help="Path to VBench_full_info.json (defaults to <prompts-dir>/VBench_full_info.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts_dir = args.prompts_dir.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()

    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"prompts_dir does not exist: {prompts_dir}")

    full_info_path = args.full_info or (prompts_dir / "VBench_full_info.json")
    full_info = load_full_info(full_info_path.expanduser().resolve())
    if full_info:
        print(f"Loaded {len(full_info)} entries from {full_info_path}")
    else:
        print(f"No VBench_full_info.json available at {full_info_path}; auxiliary_info will be empty.")

    prompt_to_dims = gather_prompt_to_dimensions(prompts_dir)

    prompts, source_rel = select_source_prompts(prompts_dir, args.source)
    print(f"Selected {len(prompts)} prompts from {source_rel}")

    if args.limit is not None:
        prompts = prompts[: int(args.limit)]
        print(f"Truncated to {len(prompts)} prompts due to --limit")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen: set[str] = set()
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts):
            if prompt in seen:
                continue
            seen.add(prompt)
            dims_from_files = sorted(prompt_to_dims.get(prompt, set()))
            full_entry = full_info.get(prompt, {})
            full_dims = full_entry.get("dimension") or []
            if isinstance(full_dims, list):
                merged = sorted(set(dims_from_files) | set(full_dims))
            else:
                merged = dims_from_files
            payload = {
                "prompt_id": index,
                "prompt": prompt,
                "negative_prompt": "",
                "dimension": merged,
                "source_file": source_rel,
                "auxiliary_info": full_entry.get("auxiliary_info") or {},
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompt records to {output_jsonl}")
    by_dimension: dict[str, int] = defaultdict(int)
    for prompt in prompts:
        for dim in prompt_to_dims.get(prompt, set()):
            by_dimension[dim] += 1
    if by_dimension:
        print("Per-dimension prompt counts (subset that has dimension info):")
        for dim, count in sorted(by_dimension.items()):
            print(f"  {dim:<25s} {count}")


if __name__ == "__main__":
    main()
