#!/usr/bin/env python3
"""Compare T2AV my_eval outputs against a previous reference run.

The script accepts either the direct local layout:

    <eval-root>/<experiment>/step-00200000/cfg_dual_g4/

or the submitted/sharded layout:

    <eval-root>/cfg_4/shard_00/<experiment>/step-00200000/cfg_dual_g4/

It writes three comparison tables:

* summary_compare.csv/json: all_metrics_summary.json values.
* per_sample_compare.csv/json: metrics matched by per-sample JSON filename.
* subset_mean_compare.csv/json: per-sample means over the matched subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXPERIMENT_MAP = {
    "t2av_recon": "2_t2av_recon_lr2",
    "t2av_recon_distill_avclip": "2_t2av_recon_distill_avclip_lr2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--new-eval-root", required=True)
    parser.add_argument("--reference-eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["t2av_recon", "t2av_recon_distill_avclip"],
    )
    parser.add_argument(
        "--experiment-map",
        action="append",
        default=[],
        help="Map release name to reference name, e.g. t2av_recon=2_t2av_recon_lr2.",
    )
    parser.add_argument("--step", type=int, default=200000)
    parser.add_argument("--cfg-dir", default="cfg_dual_g4")
    parser.add_argument(
        "--require-per-sample-match",
        action="store_true",
        help="Fail when no per-sample files match for any requested experiment.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def diff_payload(new: float, ref: float) -> dict[str, float | None]:
    diff = float(new) - float(ref)
    return {
        "diff": diff,
        "abs_diff": abs(diff),
        "rel_diff": None if float(ref) == 0.0 else diff / float(ref),
    }


def parse_experiment_map(raw_items: Iterable[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_EXPERIMENT_MAP)
    for raw in raw_items:
        if "=" not in raw:
            raise SystemExit(f"ERROR: --experiment-map expects NEW=REFERENCE, got {raw!r}")
        new, ref = raw.split("=", 1)
        new = new.strip()
        ref = ref.strip()
        if not new or not ref:
            raise SystemExit(f"ERROR: --experiment-map expects non-empty NEW and REFERENCE, got {raw!r}")
        mapping[new] = ref
    return mapping


def find_metric_dirs(root: Path, experiment: str, step: int, cfg_dir: str) -> list[Path]:
    step_dir = f"step-{step:08d}"
    direct = root / experiment / step_dir / cfg_dir
    out: list[Path] = []
    if (direct / "all_metrics_summary.json").is_file() or (direct / "per_sample").is_dir():
        out.append(direct)
    pattern = f"**/{experiment}/{step_dir}/{cfg_dir}"
    for path in root.glob(pattern):
        if path == direct:
            continue
        if (path / "all_metrics_summary.json").is_file() or (path / "per_sample").is_dir():
            out.append(path)
    return sorted(set(out))


def flatten_summary(summary_path: Path, *, experiment: str, old_experiment: str, root_kind: str) -> list[dict[str, Any]]:
    if not summary_path.is_file():
        return []
    data = load_json(summary_path)
    rows: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return rows
    for metric_kind, payload in sorted(data.items()):
        if not isinstance(payload, dict):
            continue
        scores = payload.get("scores")
        if not isinstance(scores, dict):
            continue
        for metric, by_split in sorted(scores.items()):
            if not isinstance(by_split, dict):
                continue
            for split, value in sorted(by_split.items()):
                if not is_finite_number(value):
                    continue
                rows.append(
                    {
                        "root_kind": root_kind,
                        "experiment": experiment,
                        "old_experiment": old_experiment,
                        "metric_kind": metric_kind,
                        "metric": metric,
                        "split": split,
                        "value": float(value),
                        "summary_path": str(summary_path),
                    }
                )
    return rows


def compare_summary(
    new_dir: Path,
    ref_dir: Path,
    *,
    experiment: str,
    old_experiment: str,
) -> list[dict[str, Any]]:
    new_rows = flatten_summary(
        new_dir / "all_metrics_summary.json",
        experiment=experiment,
        old_experiment=old_experiment,
        root_kind="new",
    )
    ref_rows = flatten_summary(
        ref_dir / "all_metrics_summary.json",
        experiment=experiment,
        old_experiment=old_experiment,
        root_kind="reference",
    )
    ref_by_key = {
        (row["metric_kind"], row["metric"], row["split"]): row
        for row in ref_rows
    }
    rows: list[dict[str, Any]] = []
    for new_row in new_rows:
        key = (new_row["metric_kind"], new_row["metric"], new_row["split"])
        ref_row = ref_by_key.get(key)
        row = {
            "experiment": experiment,
            "old_experiment": old_experiment,
            "metric_kind": new_row["metric_kind"],
            "metric": new_row["metric"],
            "split": new_row["split"],
            "new": new_row["value"],
            "reference": None,
            "diff": None,
            "abs_diff": None,
            "rel_diff": None,
            "new_summary_path": new_row["summary_path"],
            "reference_summary_path": str(ref_dir / "all_metrics_summary.json"),
        }
        if ref_row is not None:
            row["reference"] = ref_row["value"]
            row.update(diff_payload(float(new_row["value"]), float(ref_row["value"])))
        rows.append(row)
    return rows


def flatten_numeric(payload: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if is_finite_number(payload):
        if prefix:
            out[prefix] = float(payload)
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric(value, next_prefix))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            out.update(flatten_numeric(value, next_prefix))
    return out


def read_per_sample(metric_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    root = metric_dir / "per_sample"
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not root.is_dir():
        return rows
    for sample_path in sorted(root.glob("*/*.json")):
        metric_kind = sample_path.parent.name
        sample = sample_path.stem
        try:
            metrics = flatten_numeric(load_json(sample_path))
        except Exception as exc:  # noqa: BLE001 - keep compare robust after partial eval jobs.
            rows[(metric_kind, sample, "__read_error__")] = {
                "value": None,
                "path": str(sample_path),
                "error": str(exc),
            }
            continue
        for metric, value in metrics.items():
            rows[(metric_kind, sample, metric)] = {
                "value": value,
                "path": str(sample_path),
                "error": None,
            }
    return rows


def compare_per_sample(
    new_dirs: list[Path],
    ref_dirs: list[Path],
    *,
    experiment: str,
    old_experiment: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    new_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    ref_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for metric_dir in new_dirs:
        new_rows.update(read_per_sample(metric_dir))
    for metric_dir in ref_dirs:
        ref_rows.update(read_per_sample(metric_dir))

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for key, new_payload in sorted(new_rows.items()):
        metric_kind, sample, metric = key
        ref_payload = ref_rows.get(key)
        if ref_payload is None:
            missing.append(
                {
                    "experiment": experiment,
                    "old_experiment": old_experiment,
                    "metric_kind": metric_kind,
                    "sample": sample,
                    "metric": metric,
                    "new_path": new_payload.get("path"),
                    "reason": "missing_reference_sample_metric",
                }
            )
            continue
        if new_payload.get("value") is None or ref_payload.get("value") is None:
            continue
        row: dict[str, Any] = {
            "experiment": experiment,
            "old_experiment": old_experiment,
            "sample": sample,
            "metric_kind": metric_kind,
            "metric": metric,
            "new": float(new_payload["value"]),
            "reference": float(ref_payload["value"]),
            "new_path": new_payload.get("path"),
            "reference_path": ref_payload.get("path"),
        }
        row.update(diff_payload(float(new_payload["value"]), float(ref_payload["value"])))
        rows.append(row)
    return rows, missing


def subset_mean_rows(per_sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        grouped[(row["experiment"], row["old_experiment"], row["metric_kind"], row["metric"])].append(row)
    out: list[dict[str, Any]] = []
    for (experiment, old_experiment, metric_kind, metric), rows in sorted(grouped.items()):
        new_values = [float(row["new"]) for row in rows]
        ref_values = [float(row["reference"]) for row in rows]
        abs_values = [float(row["abs_diff"]) for row in rows]
        new_mean = sum(new_values) / len(new_values)
        ref_mean = sum(ref_values) / len(ref_values)
        row: dict[str, Any] = {
            "experiment": experiment,
            "old_experiment": old_experiment,
            "metric_kind": metric_kind,
            "metric": metric,
            "num_samples": len(rows),
            "new_mean": new_mean,
            "reference_subset_mean": ref_mean,
            "diff": new_mean - ref_mean,
            "abs_diff": abs(new_mean - ref_mean),
            "mean_abs_per_sample_diff": sum(abs_values) / len(abs_values),
            "max_abs_per_sample_diff": max(abs_values),
        }
        out.append(row)
    return out


def pick_summary_pairs(new_dirs: list[Path], ref_dirs: list[Path]) -> list[tuple[Path, Path]]:
    if not new_dirs or not ref_dirs:
        return []
    pairs: list[tuple[Path, Path]] = []
    for new_dir in new_dirs:
        pairs.append((new_dir, ref_dirs[0]))
    return pairs


def main() -> int:
    args = parse_args()
    new_root = Path(args.new_eval_root).expanduser().resolve()
    ref_root = Path(args.reference_eval_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_map = parse_experiment_map(args.experiment_map)

    all_summary: list[dict[str, Any]] = []
    all_per_sample: list[dict[str, Any]] = []
    all_subset: list[dict[str, Any]] = []
    all_missing: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "new_eval_root": str(new_root),
        "reference_eval_root": str(ref_root),
        "output_dir": str(output_dir),
        "step": int(args.step),
        "cfg_dir": args.cfg_dir,
        "experiments": {},
    }

    for experiment in args.experiments:
        old_experiment = experiment_map.get(experiment, experiment)
        new_dirs = find_metric_dirs(new_root, experiment, int(args.step), args.cfg_dir)
        ref_dirs = find_metric_dirs(ref_root, old_experiment, int(args.step), args.cfg_dir)
        manifest["experiments"][experiment] = {
            "old_experiment": old_experiment,
            "new_metric_dirs": [str(path) for path in new_dirs],
            "reference_metric_dirs": [str(path) for path in ref_dirs],
        }
        if not new_dirs:
            all_missing.append(
                {
                    "experiment": experiment,
                    "old_experiment": old_experiment,
                    "reason": "missing_new_metric_dir",
                    "expected_pattern": f"**/{experiment}/step-{int(args.step):08d}/{args.cfg_dir}",
                }
            )
            continue
        if not ref_dirs:
            all_missing.append(
                {
                    "experiment": experiment,
                    "old_experiment": old_experiment,
                    "reason": "missing_reference_metric_dir",
                    "expected_pattern": f"**/{old_experiment}/step-{int(args.step):08d}/{args.cfg_dir}",
                }
            )
            continue

        for new_dir, ref_dir in pick_summary_pairs(new_dirs, ref_dirs):
            all_summary.extend(
                compare_summary(
                    new_dir,
                    ref_dir,
                    experiment=experiment,
                    old_experiment=old_experiment,
                )
            )
        per_sample, missing = compare_per_sample(
            new_dirs,
            ref_dirs,
            experiment=experiment,
            old_experiment=old_experiment,
        )
        all_per_sample.extend(per_sample)
        all_missing.extend(missing)
    all_subset = subset_mean_rows(all_per_sample)

    write_csv(output_dir / "summary_compare.csv", all_summary)
    write_csv(output_dir / "per_sample_compare.csv", all_per_sample)
    write_csv(output_dir / "subset_mean_compare.csv", all_subset)
    write_csv(output_dir / "missing_compare_items.csv", all_missing)
    write_json(output_dir / "summary_compare.json", all_summary)
    write_json(output_dir / "per_sample_compare.json", {"per_sample": all_per_sample})
    write_json(output_dir / "subset_mean_compare.json", all_subset)
    write_json(output_dir / "missing_compare_items.json", all_missing)
    write_json(output_dir / "compare_manifest.json", manifest)

    print("T2AV eval comparison written:")
    print(f"  summary     : {output_dir / 'summary_compare.csv'}")
    print(f"  per-sample  : {output_dir / 'per_sample_compare.csv'}")
    print(f"  subset mean : {output_dir / 'subset_mean_compare.csv'}")
    print(f"  missing     : {output_dir / 'missing_compare_items.csv'}")
    print(f"  matched per-sample rows: {len(all_per_sample)}")
    print(f"  subset mean rows       : {len(all_subset)}")
    print(f"  missing rows           : {len(all_missing)}")

    if args.require_per_sample_match and not all_per_sample:
        raise SystemExit("ERROR: no per-sample metrics matched the reference run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
