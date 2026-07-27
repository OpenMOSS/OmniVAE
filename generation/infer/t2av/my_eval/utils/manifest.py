"""Discovery of <sample-root>/<exp>/samples/step-*/joint_av/<cfg>/ targets and
manifest I/O.

We deliberately reuse generation/infer/t2av/build_sample_manifest.py
as the single source of truth for sample selection and prompt resolution -- the
exact same manifest format the existing eval pipeline already consumes.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set


STEP_RE = re.compile(r"^step-(\d+)$")
CFG_PREFIXES = {
    "dual": ("cfg_dual",),
    "simple": ("cfg_simple",),
    "both": ("cfg_dual", "cfg_simple"),
}


@dataclass
class Target:
    experiment: str
    step: str
    cfg: str
    sample_dir: Path

    @property
    def step_num(self) -> int:
        match = STEP_RE.match(self.step)
        return int(match.group(1)) if match else -1

    @property
    def label(self) -> str:
        return f"{self.experiment}/{self.step}/{self.cfg}"


def discover_targets(
    sample_root: Path,
    cfg_filter: str = "dual",
    experiments: Optional[Iterable[str]] = None,
    steps_whitelist: Optional[Set[int]] = None,
    verbose: bool = True,
) -> List[Target]:
    cfg_allowed = CFG_PREFIXES[cfg_filter]
    exp_filter = set(experiments) if experiments else None
    targets: List[Target] = []

    def _log(msg: str) -> None:
        if verbose:
            print(f"[discover] {msg}", flush=True)

    if not sample_root.is_dir():
        _log(f"sample-root is not a directory: {sample_root}")
        return targets

    n_exps = 0
    n_samples_present = 0
    n_step_dirs = 0
    n_joint_av_present = 0
    n_cfg_candidates = 0
    n_cfg_matched = 0
    n_cfg_with_pairs = 0

    for exp_dir in sorted(p for p in sample_root.iterdir() if p.is_dir()):
        n_exps += 1
        if exp_filter is not None and exp_dir.name not in exp_filter:
            _log(f"skip exp '{exp_dir.name}' (not in --experiments filter)")
            continue
        samples_dir = exp_dir / "samples"
        if not samples_dir.is_dir():
            _log(f"skip exp '{exp_dir.name}': no samples/ subdir")
            continue
        n_samples_present += 1
        step_glob = sorted(samples_dir.glob("step-*"))
        if not step_glob:
            _log(f"exp '{exp_dir.name}': no step-* subdirs (looked under {samples_dir})")
            continue
        for step_dir in step_glob:
            match = STEP_RE.match(step_dir.name)
            if not match:
                _log(f"skip {exp_dir.name}/{step_dir.name}: doesn't match 'step-<digits>'")
                continue
            step_num = int(match.group(1))
            if steps_whitelist is not None and step_num not in steps_whitelist:
                continue
            n_step_dirs += 1
            joint_av = step_dir / "joint_av"
            if not joint_av.is_dir():
                _log(
                    f"skip {exp_dir.name}/{step_dir.name}: no joint_av/ "
                    f"(found {[p.name for p in step_dir.iterdir() if p.is_dir()]})"
                )
                continue
            n_joint_av_present += 1
            cfg_dirs_here = [p for p in joint_av.iterdir() if p.is_dir()]
            n_cfg_candidates += len(cfg_dirs_here)
            for cfg_dir in sorted(cfg_dirs_here):
                if not any(
                    cfg_dir.name == prefix or cfg_dir.name.startswith(f"{prefix}_")
                    for prefix in cfg_allowed
                ):
                    _log(
                        f"skip {exp_dir.name}/{step_dir.name}/{cfg_dir.name}: "
                        f"cfg '{cfg_dir.name}' not in --cfg filter ({list(cfg_allowed)})"
                    )
                    continue
                n_cfg_matched += 1
                mp4s = [p for p in cfg_dir.glob("*.mp4") if not p.name.endswith(".av.mp4")]
                paired = [p for p in mp4s if p.with_suffix(".wav").is_file()]
                if not paired:
                    _log(
                        f"skip {exp_dir.name}/{step_dir.name}/{cfg_dir.name}: "
                        f"{len(mp4s)} non-.av.mp4 found, but none have a sibling .wav"
                    )
                    continue
                n_cfg_with_pairs += 1
                targets.append(
                    Target(
                        experiment=exp_dir.name,
                        step=step_dir.name,
                        cfg=cfg_dir.name,
                        sample_dir=cfg_dir.resolve(),
                    )
                )
                _log(
                    f"add target {exp_dir.name}/{step_dir.name}/{cfg_dir.name} "
                    f"({len(paired)} mp4+wav pairs)"
                )

    _log(
        f"summary: exps_seen={n_exps}, exps_with_samples/={n_samples_present}, "
        f"steps_kept={n_step_dirs}, with_joint_av={n_joint_av_present}, "
        f"cfg_dirs_total={n_cfg_candidates}, cfg_matched_filter={n_cfg_matched}, "
        f"cfg_with_pairs={n_cfg_with_pairs}, returning={len(targets)}"
    )
    return targets


def parse_steps_arg(raw: str) -> Optional[Set[int]]:
    if not raw or not raw.strip():
        return None
    tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
    bad = [t for t in tokens if not t.isdigit()]
    if bad:
        raise ValueError(f"--steps got non-integer tokens: {bad}")
    return {int(t) for t in tokens}


def target_output_dir(eval_output_root: Path, target: Target) -> Path:
    return eval_output_root / target.experiment / target.step / target.cfg


def manifest_path(eval_output_root: Path, target: Target) -> Path:
    return target_output_dir(eval_output_root, target) / "metadata" / "manifest.json"


def _category_sort_key(category: str) -> tuple[int, int, str]:
    match = re.match(r"^set(\d+)(.*)$", category)
    if match:
        suffix = match.group(2).lstrip("-")
        return (0, int(match.group(1)), suffix)
    return (1, 0, category)


def _normalise_manifest_categories(manifest: dict) -> bool:
    """Use source_category as the authoritative evaluation bucket when present.

    Older manifests collapsed filename categories like set3-large into set3.
    The filename/source category is the bucket users compare, so normalise on
    read and when deciding whether an existing manifest is stale.
    """
    records = manifest.get("records")
    if not isinstance(records, list):
        return False

    changed = False
    counts: Counter[str] = Counter()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        source_category = rec.get("source_category")
        if isinstance(source_category, str) and source_category:
            category = source_category
        else:
            category = str(rec.get("category") or "")
        counts[category] += 1
        if rec.get("category") != category:
            rec["category"] = category
            changed = True

    if counts:
        categories = {
            cat: counts[cat]
            for cat in sorted(counts.keys(), key=_category_sort_key)
        }
        if manifest.get("categories") != categories:
            manifest["categories"] = categories
            changed = True
    return changed


def _normalise_manifest_file(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if _normalise_manifest_categories(data):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def ensure_manifest(
    target: Target,
    eval_output_root: Path,
    build_script: Path,
    valid_jsonl: Path,
    force: bool = False,
) -> Path:
    """Run build_sample_manifest.py for one target if its manifest does not exist."""
    out_path = manifest_path(eval_output_root, target)
    if out_path.is_file() and out_path.stat().st_size > 0 and not force:
        if _normalise_manifest_file(out_path):
            return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(build_script),
        "--sample-dir", str(target.sample_dir),
        "--output-dir", str(target_output_dir(eval_output_root, target)),
        "--valid-jsonl", str(valid_jsonl),
    ]
    if force:
        cmd.append("--force")
    print(f"[manifest] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return out_path


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    _normalise_manifest_categories(manifest)
    return manifest


def limit_records(records: List[dict], limit: int) -> List[dict]:
    if limit and limit > 0:
        return records[:limit]
    return records
