"""Diagnostics helper - run this before generate / evaluate to verify that
prompts, metadata, dependencies and (optionally) checkpoint paths are all in
order.

Usage:
    python verify_setup.py                                  # check everything
    python verify_setup.py --checkpoint-dir /path/to/ckpt   # also check ckpt
    python verify_setup.py --strict                         # exit non-zero on warnings
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "data/t2v/vbench/prompts"
METADATA_JSONL = REPO_ROOT / "examples/metadata/vbench.jsonl"
SAMPLER_SCRIPT = REPO_ROOT / "scripts/eval/export_video_checkpoint_samples.py"

REQUIRED_TOP_LEVEL = ["all_dimension.txt"]
REQUIRED_PER_DIMENSION = [
    "subject_consistency.txt",
    "scene.txt",
    "temporal_flickering.txt",
    "overall_consistency.txt",
    "object_class.txt",
    "multiple_objects.txt",
    "human_action.txt",
    "color.txt",
    "spatial_relationship.txt",
    "appearance_style.txt",
    "temporal_style.txt",
]
REQUIRED_PYPI_PACKAGES = [
    "vbench",
    "torch",
    "imageio",
    "decord",
    "requests",
]


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK " if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = 0
    warnings = 0

    print(f"Repo root:   {REPO_ROOT}")
    print(f"Prompts:     {PROMPTS_DIR}")
    print(f"Metadata:    {METADATA_JSONL}\n")

    print("[1/4] Filesystem layout")
    for filename in REQUIRED_TOP_LEVEL:
        ok = (PROMPTS_DIR / filename).is_file()
        if not ok:
            failures += 1
        check(f"prompts/{filename}", ok, str(PROMPTS_DIR / filename))

    pd = PROMPTS_DIR / "prompts_per_dimension"
    for filename in REQUIRED_PER_DIMENSION:
        ok = (pd / filename).is_file()
        if not ok:
            failures += 1
        check(f"prompts/prompts_per_dimension/{filename}", ok)

    info_path = PROMPTS_DIR / "VBench_full_info.json"
    if not check("VBench_full_info.json", info_path.is_file(), str(info_path)):
        warnings += 1

    if not check("metadata jsonl", METADATA_JSONL.is_file(), str(METADATA_JSONL)):
        failures += 1
    else:
        with METADATA_JSONL.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        check("metadata jsonl prompt count > 0", count > 0, f"{count} prompts")

    print("\n[2/4] Sampler script")
    if not check("export_video_checkpoint_samples.py", SAMPLER_SCRIPT.is_file(), str(SAMPLER_SCRIPT)):
        failures += 1

    print("\n[3/4] Python dependencies")
    for pkg in REQUIRED_PYPI_PACKAGES:
        try:
            importlib.import_module(pkg)
            check(f"import {pkg}", True)
        except Exception as exc:
            level = "FAIL" if pkg in {"vbench", "torch"} else "WARN"
            if level == "FAIL":
                failures += 1
            else:
                warnings += 1
            check(f"import {pkg}", False, f"{type(exc).__name__}: {exc}")

    print("\n[4/4] Optional checks")
    if args.checkpoint_dir is not None:
        ckpt = args.checkpoint_dir.expanduser().resolve()
        ok = ckpt.is_dir()
        if not ok:
            failures += 1
        check(f"checkpoint dir {ckpt}", ok)

    print()
    print(f"Failures: {failures}, Warnings: {warnings}")
    if failures > 0 or (args.strict and warnings > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
