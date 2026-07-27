"""Plot ``my_eval`` checkpoint-sweep evaluation results into comparison curves.

Inputs
------
``--eval-root`` points at the EVAL_OUTPUT_ROOT used by ``run_my_eval.py``.
Expected flat layout:

    <eval_root>/<experiment>/step-<NNNNNNNN>/<cfg>/per_sample/<kind>/*.json
    <eval_root>/<experiment>/step-<NNNNNNNN>/<cfg>/all_metrics_summary.json
    <eval_root>/<experiment>/step-<NNNNNNNN>/<cfg>/summary/<kind>.json

The target discovery is recursive, so sharded QZ submission outputs also work:

    <eval_root>/cfg_<value>/shard_<NN>/<experiment>/step-<NNNNNNNN>/<cfg>/...

By default the script reads summaries for backwards-compatible behavior. Pass
``--from-per-sample`` to recompute plot aggregates from per-sample JSONs; that
mode uses filename suffixes as category names, so samples such as ``set3``,
``set3-large`` and ``set3-medium-large`` stay separate even if old summary files
collapsed them into ``set3``. Summary files are still used as a fallback for
metric kinds that do not have per-sample data.

Each summary JSON has the shape::

    {
      "metric_kind": "av_sync_imagebind",
      "scores": {
        "DeSync":   {"set1": ..., "set2": ..., "set3": ..., "all": ...},
        "AV-Align": {...},
        ...
      },
      "num_samples": {"set1": 205, "set2": 295, "set3": 100, "all": 600}
    }

so each (metric_kind, sub_metric, set) cell becomes one long-format row.

Outputs
-------
``--output-dir`` (default ``<eval_root>/_plots``) populated with PNG files
organised by view:

    <output>/<view>/<metric>.png                  one curve per experiment
    <output>/all_sets/<metric>.png                grid: every discovered set + all
    <output>/_all_metrics/<view>.png              every metric in one figure
    <output>/_groups/<group>.png                  thematic group dashboards

A consolidated ``<output>/metrics_long.csv`` is also written for downstream
analysis (long-format: experiment, step, cfg, metric_id, category, value,
n_samples).

Each plot title carries ``[↑ higher is better]`` / ``[↓ lower is better]`` /
``[— descriptive]`` so the direction is unambiguous when comparing curves.

Parallelism
-----------
File parsing and plotting are both fanned out across ``--workers`` processes
(default = cpu_count). Parsing JSON is IO-bound on shared FS and plotting is
matplotlib-bound; both benefit from process-level parallelism.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None


_STEP_RE = re.compile(r"^step-(\d+)$")
_CATEGORY_SUFFIX_RE = re.compile(r"-(set\d+(?:-.+)?)$")
_DISCOVERY_PRUNE_DIRS = {
    "_plots",
    "_timing",
    "metadata",
    "monitor",
    "per_sample",
    "summary",
}


# --------------------------------------------------------------------------
# Metric registry
# --------------------------------------------------------------------------
DIR_UP = "up"           # higher is better
DIR_DOWN = "down"       # lower is better
DIR_NEUTRAL = "neutral" # descriptive only, no good/bad


@dataclass(frozen=True)
class MetricSpec:
    """One scalar metric extracted from ``summary/<kind>.json``.

    ``metric_id`` is the canonical key used in filenames / CSV columns and is
    unique across all kinds. ``kind`` selects which JSON file to read,
    ``sub_metric`` is the key under ``scores`` in that JSON.
    """

    metric_id: str        # canonical key, e.g. "av_sync_imagebind__DeSync"
    display: str          # human-friendly title chunk
    kind: str             # one of KIND_ORDER
    sub_metric: str       # key inside JSON ``scores``
    direction: str        # DIR_UP / DIR_DOWN / DIR_NEUTRAL


def _mid(kind: str, sub: str) -> str:
    """Canonical metric id used in filenames and CSV."""
    return f"{kind}__{sub}"


PE_AV_METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(_mid("pe_av", "PE-TV"),                 "PE-AV: text↔video dot",           "pe_av", "PE-TV",    DIR_UP),
    MetricSpec(_mid("pe_av", "PE-TA"),                 "PE-AV: text↔audio dot",           "pe_av", "PE-TA",    DIR_UP),
    MetricSpec(_mid("pe_av", "PE-TAV"),                "PE-AV: text↔audio-video dot",     "pe_av", "PE-TAV",   DIR_UP),
)


BASE_METRIC_SPECS: tuple[MetricSpec, ...] = (
    # -- av_sync_synchformer / av_sync_imagebind --------------------------
    MetricSpec(_mid("av_sync_synchformer", "DeSync"),    "AV-Sync |offset| (sec)",          "av_sync_synchformer", "DeSync",   DIR_DOWN),
    MetricSpec(_mid("av_sync_synchformer", "AV-Align"),  "AV-Align (optical-flow / onset)", "av_sync_synchformer", "AV-Align", DIR_UP),
    MetricSpec(_mid("av_sync_imagebind", "IB-AV"),     "ImageBind: video↔audio cos",      "av_sync_imagebind", "IB-AV",    DIR_UP),
    MetricSpec(_mid("av_sync_imagebind", "IB-TV"),     "ImageBind: text↔video cos",       "av_sync_imagebind", "IB-TV",    DIR_UP),
    MetricSpec(_mid("av_sync_imagebind", "IB-TA"),     "ImageBind: text↔audio cos",       "av_sync_imagebind", "IB-TA",    DIR_UP),
    # -- lip_sync (set3 only) ---------------------------------------------
    MetricSpec(_mid("lip_sync", "LSE-D"),              "LSE-D (SyncNet distance)",        "lip_sync", "LSE-D", DIR_DOWN),
    MetricSpec(_mid("lip_sync", "LSE-C"),              "LSE-C (SyncNet confidence)",      "lip_sync", "LSE-C", DIR_UP),
    # -- audio_clap -------------------------------------------------------
    MetricSpec(_mid("audio_clap", "CLAP"),             "CLAP: text↔audio cos",            "audio_clap", "CLAP", DIR_UP),
    # -- video_motion -----------------------------------------------------
    MetricSpec(_mid("video_motion", "MS"),              "Motion Score (RAFT flow)",        "video_motion", "MS", DIR_UP),
    # -- video_aesthetic --------------------------------------------------
    MetricSpec(_mid("video_aesthetic", "Aesthetic"),   "Aesthetic v2.5",                  "video_aesthetic", "Aesthetic", DIR_UP),
    MetricSpec(_mid("video_aesthetic", "MusiQ"),       "MusiQ",                            "video_aesthetic", "MusiQ",     DIR_UP),
    MetricSpec(_mid("video_aesthetic", "ManiQA"),      "ManiQA",                           "video_aesthetic", "ManiQA",    DIR_UP),
    MetricSpec(_mid("video_aesthetic", "AS"),          "Aesthetic Score (mean of 3)",     "video_aesthetic", "AS",        DIR_UP),
    # -- identity_dino ----------------------------------------------------
    MetricSpec(_mid("identity_dino", "ID"),             "Identity/reference consistency",  "identity_dino", "ID", DIR_UP),
    # -- audio_fd_kl ------------------------------------------------------
    MetricSpec(_mid("audio_fd_kl", "FD"),               "Fréchet Distance (CLAP refs)",    "audio_fd_kl", "FD", DIR_DOWN),
    MetricSpec(_mid("audio_fd_kl", "KL"),               "KL Divergence (PaSST refs)",      "audio_fd_kl", "KL", DIR_DOWN),
    # -- audio_box (CE / CU / PC / PQ) ------------------------------------
    MetricSpec(_mid("audio_box", "CE"),                "AudioBox: Content Enjoyment (CE)",     "audio_box", "CE", DIR_UP),
    MetricSpec(_mid("audio_box", "CU"),                "AudioBox: Content Usefulness (CU)",    "audio_box", "CU", DIR_UP),
    MetricSpec(_mid("audio_box", "PC"),                "AudioBox: Production Complexity (PC)", "audio_box", "PC", DIR_DOWN),
    MetricSpec(_mid("audio_box", "PQ"),                "AudioBox: Production Quality (PQ)",    "audio_box", "PQ", DIR_UP),
    # -- speech_wer -------------------------------------------------------
    MetricSpec(_mid("speech_wer", "WER"),              "Speech WER",                      "speech_wer", "WER", DIR_DOWN),
    # -- audio_dnsmos -----------------------------------------------------
    MetricSpec(_mid("audio_dnsmos", "P808_MOS"),       "DNSMOS P.808 (MOS)",              "audio_dnsmos", "P808_MOS", DIR_UP),
    # -- audio_is ---------------------------------------------------------
    MetricSpec(_mid("audio_is", "IS"),                 "Inception Score (PANNs CNN14)",   "audio_is", "IS", DIR_UP),
    # -- audio_amplitude (descriptive) ------------------------------------
    MetricSpec(_mid("audio_amplitude", "amplitude_rms"), "Audio RMS amplitude",         "audio_amplitude", "amplitude_rms", DIR_NEUTRAL),
    MetricSpec(_mid("audio_amplitude", "loudness_lufs"), "Audio loudness (LUFS)",       "audio_amplitude", "loudness_lufs", DIR_NEUTRAL),
)


METRIC_SPECS: tuple[MetricSpec, ...] = BASE_METRIC_SPECS
_SPEC_BY_ID: dict[str, MetricSpec] = {s.metric_id: s for s in METRIC_SPECS}


# --------------------------------------------------------------------------
# Thematic groups. Each group renders ONE consolidated figure laying out
# several metric panels side-by-side. ``view`` per-item chooses which
# category slice to use (default "all" = dataset-wide sample-weighted mean).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupItem:
    metric_id: str
    view: str          # "all" / "set1" / "set2" / "set3"


@dataclass(frozen=True)
class MetricGroup:
    name: str          # filename-safe id
    display: str       # human-friendly title
    items: tuple[GroupItem, ...]


# lip_sync runs only on set3, so panel uses view="set3" so we don't waste a
# subplot on the (always-NaN) "all" view averaged over non-eligible samples.
PE_AV_GROUP_ITEMS: tuple[GroupItem, ...] = (
    GroupItem(_mid("pe_av", "PE-TV"),                 "all"),
    GroupItem(_mid("pe_av", "PE-TA"),                 "all"),
    GroupItem(_mid("pe_av", "PE-TAV"),                "all"),
)


BASE_METRIC_GROUPS: tuple[MetricGroup, ...] = (
    MetricGroup(
        name="av_alignment",
        display="AV Synchronization & Alignment",
        items=(
            GroupItem(_mid("av_sync_synchformer", "DeSync"), "all"),
            GroupItem(_mid("lip_sync", "LSE-C"),             "set3"),
        ),
    ),
    MetricGroup(
        name="cross_modal_alignment",
        display="Cross-modal Alignment (Text / Video / Audio)",
        items=(
            GroupItem(_mid("audio_clap", "CLAP"),             "all"),
            GroupItem(_mid("av_sync_imagebind", "IB-AV"),     "all"),
            GroupItem(_mid("av_sync_imagebind", "IB-TV"),     "all"),
            GroupItem(_mid("av_sync_imagebind", "IB-TA"),     "all"),
        ),
    ),
    MetricGroup(
        name="video_quality",
        display="Video Quality",
        items=(
            GroupItem(_mid("video_aesthetic", "Aesthetic"), "all"),
            GroupItem(_mid("video_aesthetic", "MusiQ"),     "all"),
            GroupItem(_mid("video_aesthetic", "ManiQA"),    "all"),
            GroupItem(_mid("video_aesthetic", "AS"),        "all"),
            GroupItem(_mid("video_motion", "MS"),           "all"),
            GroupItem(_mid("identity_dino", "ID"),          "all"),
        ),
    ),
    MetricGroup(
        name="audio_aesthetics",
        display="Audio Aesthetics (AudioBox CE / CU / PC / PQ)",
        items=(
            GroupItem(_mid("audio_box", "CE"), "all"),
            GroupItem(_mid("audio_box", "CU"), "all"),
            GroupItem(_mid("audio_box", "PC"), "all"),
            GroupItem(_mid("audio_box", "PQ"), "all"),
        ),
    ),
    MetricGroup(
        name="audio_quality",
        display="Audio Quality / Distribution / Speech",
        items=(
            GroupItem(_mid("audio_dnsmos", "P808_MOS"),       "all"),
            GroupItem(_mid("audio_is", "IS"),                 "all"),
            GroupItem(_mid("audio_fd_kl", "FD"),              "all"),
            GroupItem(_mid("audio_fd_kl", "KL"),              "all"),
            GroupItem(_mid("speech_wer", "WER"),              "all"),
        ),
    ),
)


METRIC_GROUPS: tuple[MetricGroup, ...] = BASE_METRIC_GROUPS


def configure_metric_registry(include_pe_av: bool) -> None:
    global METRIC_SPECS, _SPEC_BY_ID, METRIC_GROUPS

    if include_pe_av:
        specs: list[MetricSpec] = []
        inserted = False
        for spec in BASE_METRIC_SPECS:
            specs.append(spec)
            if spec.metric_id == _mid("av_sync_imagebind", "IB-TA"):
                specs.extend(PE_AV_METRIC_SPECS)
                inserted = True
        if not inserted:
            specs.extend(PE_AV_METRIC_SPECS)
        METRIC_SPECS = tuple(specs)

        groups: list[MetricGroup] = []
        for group in BASE_METRIC_GROUPS:
            if group.name == "cross_modal_alignment":
                groups.append(MetricGroup(
                    name=group.name,
                    display=group.display,
                    items=group.items + PE_AV_GROUP_ITEMS,
                ))
            else:
                groups.append(group)
        METRIC_GROUPS = tuple(groups)
    else:
        METRIC_SPECS = BASE_METRIC_SPECS
        METRIC_GROUPS = BASE_METRIC_GROUPS

    _SPEC_BY_ID = {s.metric_id: s for s in METRIC_SPECS}


# --------------------------------------------------------------------------
# Discovery + parsing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Target:
    experiment: str
    step: int
    cfg: str
    target_dir: Path


def discover_targets(
    eval_root: Path,
    cfg_filter: Optional[set[str]],
    experiments_filter: Optional[set[str]],
    from_per_sample: bool,
) -> list[_Target]:
    """Collect every ``.../<exp>/step-N/<cfg>/`` dir with usable metric data.

    ``eval_root`` may be either the historical flat my_eval root or the QZ
    submitter's sharded root (``cfg_*/shard_*/<exp>/step-*/<cfg>``).
    """
    if not eval_root.is_dir():
        raise SystemExit(f"--eval-root not found: {eval_root}")

    def _has_metric_data(cfg_dir: Path) -> bool:
        has_all = (cfg_dir / "all_metrics_summary.json").is_file()
        has_summary_dir = (cfg_dir / "summary").is_dir() and any(
            (cfg_dir / "summary").glob("*.json")
        )
        has_per_sample = (cfg_dir / "per_sample").is_dir() and any(
            (cfg_dir / "per_sample").glob("*/*.json")
        )
        return (from_per_sample and has_per_sample) or has_all or has_summary_dir

    def _iter_step_dirs() -> list[Path]:
        step_dirs: list[Path] = []
        for root, dirs, _files in os.walk(eval_root):
            dirs[:] = sorted(d for d in dirs if d not in _DISCOVERY_PRUNE_DIRS)
            path = Path(root)
            if _STEP_RE.match(path.name):
                step_dirs.append(path)
                # A step directory's children are cfg target dirs; do not walk
                # into their per-sample JSON trees during discovery.
                dirs[:] = []
        return sorted(step_dirs)

    out: list[_Target] = []
    seen: set[Path] = set()
    for step_dir in _iter_step_dirs():
        m = _STEP_RE.match(step_dir.name)
        if not m:
            continue
        exp_dir = step_dir.parent
        if experiments_filter is not None and exp_dir.name not in experiments_filter:
            continue
        step = int(m.group(1))
        for cfg_dir in sorted(p for p in step_dir.iterdir() if p.is_dir()):
            resolved = cfg_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if cfg_filter is not None and cfg_dir.name not in cfg_filter:
                continue
            if not _has_metric_data(cfg_dir):
                continue
            out.append(_Target(
                experiment=exp_dir.name,
                step=step,
                cfg=cfg_dir.name,
                target_dir=cfg_dir,
            ))
    return out


def _is_real(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _category_from_stem(stem: str, fallback: Any = None) -> str:
    """Recover the category suffix from filenames like
    ``sample-versebench-1060-set3-medium-large``.

    The payload's ``category`` field is only a fallback because older manifests
    may have collapsed all set3 variants to ``set3``.
    """
    m = _CATEGORY_SUFFIX_RE.search(stem)
    if m:
        return m.group(1)
    if isinstance(fallback, str) and fallback:
        return fallback
    return "unknown"


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals))


def _progress(
    iterable: Any,
    *,
    total: Optional[int] = None,
    desc: str,
    unit: str,
) -> Any:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _normalised_prob(prob: Any) -> Optional[list[float]]:
    if not isinstance(prob, list) or not prob:
        return None
    vals: list[float] = []
    total = 0.0
    for v in prob:
        if not _is_real(v):
            return None
        fv = float(v)
        if fv < 0:
            return None
        vals.append(fv)
        total += fv
    if total <= 0:
        return None
    return [v / total for v in vals]


def _calculate_inception_score(probs: list[list[float]], eps: float = 1e-10) -> Optional[float]:
    if not probs:
        return None
    dim = len(probs[0])
    if dim == 0 or any(len(p) != dim for p in probs):
        return None
    n = float(len(probs))
    p_y = [0.0] * dim
    for p in probs:
        for i, v in enumerate(p):
            p_y[i] += v / n
    kl = 0.0
    for p in probs:
        row_kl = 0.0
        for i, v in enumerate(p):
            row_kl += v * (math.log(v + eps) - math.log(p_y[i] + eps))
        kl += row_kl / n
    return float(math.exp(kl))


def _read_per_sample_rows_for_target(
    target_dir: Path,
    experiment: str,
    step: int,
    cfg: str,
) -> list[dict[str, Any]]:
    """Aggregate ``per_sample/<kind>/*.json`` into the same long row format as
    summaries. Filename suffixes are authoritative for category names."""
    rows: list[dict[str, Any]] = []
    specs_by_kind: dict[str, list[MetricSpec]] = defaultdict(list)
    for spec in METRIC_SPECS:
        specs_by_kind[spec.kind].append(spec)

    per_sample_root = target_dir / "per_sample"
    if not per_sample_root.is_dir():
        return rows

    for kind, specs in specs_by_kind.items():
        kind_dir = per_sample_root / kind
        if not kind_dir.is_dir():
            continue

        counts: dict[str, int] = defaultdict(int)
        scalar_values: dict[str, dict[str, list[float]]] = {
            spec.metric_id: defaultdict(list) for spec in specs
        }
        is_probs: dict[str, list[list[float]]] = defaultdict(list)
        saw_file = False

        for json_file in sorted(kind_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logging.warning("failed to parse %s: %r", json_file, exc)
                continue
            if not isinstance(data, dict):
                continue
            saw_file = True
            cat = _category_from_stem(json_file.stem, data.get("category"))
            counts[cat] += 1
            counts["all"] += 1

            for spec in specs:
                if spec.kind == "audio_is":
                    prob = _normalised_prob(data.get("prob"))
                    if prob is not None:
                        is_probs[cat].append(prob)
                        is_probs["all"].append(prob)
                    continue
                val = data.get(spec.sub_metric)
                if _is_real(val):
                    scalar_values[spec.metric_id][cat].append(float(val))
                    scalar_values[spec.metric_id]["all"].append(float(val))

        if not saw_file:
            continue

        for spec in specs:
            if spec.kind == "audio_is":
                for cat, probs in sorted(is_probs.items(), key=lambda kv: _category_sort_key(kv[0])):
                    score = _calculate_inception_score(probs)
                    if score is None:
                        continue
                    rows.append({
                        "experiment": experiment,
                        "step": int(step),
                        "cfg": cfg,
                        "metric_id": spec.metric_id,
                        "category": cat,
                        "value": score,
                        "n_samples": int(counts.get(cat, len(probs))),
                    })
                continue

            for cat, vals in sorted(
                scalar_values[spec.metric_id].items(),
                key=lambda kv: _category_sort_key(kv[0]),
            ):
                if not vals:
                    continue
                rows.append({
                    "experiment": experiment,
                    "step": int(step),
                    "cfg": cfg,
                    "metric_id": spec.metric_id,
                    "category": cat,
                    "value": _mean(vals),
                    "n_samples": int(counts.get(cat, len(vals))),
                })
    return rows


def _read_summary_for_target(target_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``{kind: {"scores": {...}, "num_samples": {...}}}``.

    Prefer ``all_metrics_summary.json`` (one file IO instead of N); fall back
    to walking ``summary/<kind>.json`` files when the consolidated file is
    missing or unreadable (partial / in-progress runs).
    """
    all_path = target_dir / "all_metrics_summary.json"
    if all_path.is_file():
        try:
            data = json.loads(all_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001
            logging.warning("failed to parse %s: %r -- falling back to summary/", all_path, exc)
    out: dict[str, dict[str, Any]] = {}
    summary_dir = target_dir / "summary"
    if summary_dir.is_dir():
        for jf in sorted(summary_dir.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logging.warning("failed to parse %s: %r", jf, exc)
                continue
            kind = data.get("metric_kind") or jf.stem
            out[kind] = data
    return out


def _parse_worker(payload: tuple) -> list[dict[str, Any]]:
    target_dir_str, experiment, step, cfg, from_per_sample = payload
    target_dir = Path(target_dir_str)
    rows: list[dict[str, Any]] = []
    if from_per_sample:
        rows = _read_per_sample_rows_for_target(target_dir, experiment, int(step), cfg)
    per_sample_metric_ids = {r["metric_id"] for r in rows}

    summaries = _read_summary_for_target(target_dir)
    for spec in METRIC_SPECS:
        if spec.metric_id in per_sample_metric_ids:
            continue
        bundle = summaries.get(spec.kind)
        if not isinstance(bundle, dict):
            continue
        scores = bundle.get("scores") or {}
        num_samples = bundle.get("num_samples") or {}
        per_cat = scores.get(spec.sub_metric)
        if not isinstance(per_cat, dict):
            continue
        for cat, val in per_cat.items():
            if not isinstance(val, (int, float)):
                continue
            if val != val:  # NaN guard
                continue
            n = num_samples.get(cat)
            rows.append({
                "experiment": experiment,
                "step": int(step),
                "cfg": cfg,
                "metric_id": spec.metric_id,
                "category": cat,
                "value": float(val),
                "n_samples": int(n) if isinstance(n, (int, float)) else None,
            })
    return rows


def collect_long_records(
    eval_root: Path,
    cfg_filter: Optional[set[str]],
    experiments_filter: Optional[set[str]],
    workers: int,
    from_per_sample: bool,
) -> list[dict[str, Any]]:
    targets = discover_targets(eval_root, cfg_filter, experiments_filter, from_per_sample)
    if not targets:
        logging.warning("no (experiment, step, cfg) targets discovered under %s", eval_root)
        return []
    logging.info(
        "discovered %d (exp, step, cfg) targets; parsing with %d workers",
        len(targets), workers,
    )
    payloads = [
        (str(t.target_dir), t.experiment, t.step, t.cfg, from_per_sample)
        for t in targets
    ]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for p in _progress(payloads, total=len(payloads), desc="parse targets", unit="target"):
            rows.extend(_parse_worker(p))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            chunks = ex.map(_parse_worker, payloads, chunksize=1)
            for chunk in _progress(chunks, total=len(payloads), desc="parse targets", unit="target"):
                rows.extend(chunk)
    return rows


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------
def _category_sort_key(cat: str) -> tuple[int, int, str]:
    if cat == "all":
        return (1, 999999, "")
    m = re.match(r"^set(\d+)(.*)$", cat)
    if m:
        suffix = m.group(2).lstrip("-")
        return (0, int(m.group(1)), suffix)
    return (0, 999998, cat)


def _ordered_categories(categories: set[str], *, all_first: bool = False) -> list[str]:
    has_all = "all" in categories
    concrete = sorted((c for c in categories if c != "all"), key=_category_sort_key)
    if all_first:
        return (["all"] if has_all else []) + concrete
    return concrete + (["all"] if has_all else [])


def _views_for(records: list[dict[str, Any]]) -> list[str]:
    return _ordered_categories({r["category"] for r in records}, all_first=True)


def _grid_subviews_for(records: list[dict[str, Any]]) -> list[str]:
    return _ordered_categories({r["category"] for r in records}, all_first=False)


def _direction_arrow(d: str) -> str:
    return {DIR_UP: "↑", DIR_DOWN: "↓", DIR_NEUTRAL: "—"}.get(d, "?")


def _direction_label(d: str) -> str:
    return {
        DIR_UP: "higher is better",
        DIR_DOWN: "lower is better",
        DIR_NEUTRAL: "descriptive",
    }.get(d, "?")


def _series_for(
    records: list[dict[str, Any]],
    metric_id: str,
    category: str,
    cfg: Optional[str] = None,
) -> dict[str, list[tuple[int, float]]]:
    """Group filtered records into ``{experiment: sorted [(step, value), ...]}``."""
    out: dict[str, list[tuple[int, float]]] = {}
    for r in records:
        if r["metric_id"] != metric_id or r["category"] != category:
            continue
        if cfg is not None and r["cfg"] != cfg:
            continue
        out.setdefault(r["experiment"], []).append((int(r["step"]), float(r["value"])))
    for exp in out:
        out[exp].sort()
    return out


def _all_experiments(records: list[dict[str, Any]]) -> list[str]:
    return sorted({r["experiment"] for r in records})


def _all_cfgs(records: list[dict[str, Any]]) -> list[str]:
    return sorted({r["cfg"] for r in records})


def _build_color_map(experiments: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {exp: cmap(i % 10) for i, exp in enumerate(experiments)}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_into(
    ax: plt.Axes,
    series: dict[str, list[tuple[int, float]]],
    spec: MetricSpec,
    category_label: str,
    color_map: dict[str, str],
    *,
    title_fontsize: int = 11,
) -> None:
    for exp in sorted(series.keys()):
        pts = series[exp]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys,
            marker="o", linewidth=1.5, markersize=4,
            color=color_map.get(exp),
            label=exp,
        )
    arrow = _direction_arrow(spec.direction)
    ax.set_title(
        f"{spec.display} ({category_label}) [{arrow} {_direction_label(spec.direction)}]",
        fontsize=title_fontsize,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    if series:
        ax.legend(fontsize=8, loc="best")


# --------------------------------------------------------------------------
# Per-metric × per-view single plots
# --------------------------------------------------------------------------
def _plot_single_view(
    records: list[dict[str, Any]],
    spec: MetricSpec,
    cfg: str,
    view: str,
    grid_subviews: list[str],
    output_dir: Path,
    color_map: dict[str, str],
) -> Optional[Path]:
    cfg_records = [r for r in records if r["cfg"] == cfg]
    if not cfg_records:
        return None
    if view == "all_sets":
        panels: list[tuple[str, dict[str, list[tuple[int, float]]]]] = []
        for sub in grid_subviews:
            series = _series_for(cfg_records, spec.metric_id, sub)
            if any(series.values()):
                panels.append((sub, series))
        if not panels:
            return None
        ncols = min(len(panels), 4)
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 3.6), squeeze=False)
        for i, (sub, series) in enumerate(panels):
            ax = axes[i // ncols][i % ncols]
            _plot_into(ax, series, spec, sub, color_map, title_fontsize=10)
        for j in range(len(panels), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.suptitle(
            f"{spec.display}   [cfg={cfg}]   "
            f"({_direction_arrow(spec.direction)} {_direction_label(spec.direction)})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = output_dir / cfg / "all_sets" / f"{spec.metric_id}.png"
        _save(fig, out)
        return out

    category = view
    series = _series_for(cfg_records, spec.metric_id, category)
    if not any(series.values()):
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_into(ax, series, spec, category, color_map)
    fig.tight_layout()
    out = output_dir / cfg / view / f"{spec.metric_id}.png"
    _save(fig, out)
    return out


def _plot_single_worker(payload: tuple) -> Optional[str]:
    records, spec_dict, cfg, view, grid_subviews, output_dir_str, color_map = payload
    spec = MetricSpec(**spec_dict)
    p = _plot_single_view(records, spec, cfg, view, grid_subviews, Path(output_dir_str), color_map)
    return str(p) if p is not None else None


# --------------------------------------------------------------------------
# Group dashboards
# --------------------------------------------------------------------------
def _short_view_tag(view: str) -> str:
    return {
        "all": "all (sample-weighted)",
        "set1": "set1",
        "set2": "set2",
        "set3": "set3",
    }.get(view, view)


def _plot_group(
    records: list[dict[str, Any]],
    group: MetricGroup,
    cfg: str,
    output_dir: Path,
    color_map: dict[str, str],
) -> Optional[Path]:
    cfg_records = [r for r in records if r["cfg"] == cfg]
    if not cfg_records:
        return None
    panels: list[tuple[MetricSpec, GroupItem, dict[str, list[tuple[int, float]]]]] = []
    for item in group.items:
        spec = _SPEC_BY_ID.get(item.metric_id)
        if spec is None:
            logging.warning(
                "group %s references unknown metric_id=%s (skipping panel)",
                group.name, item.metric_id,
            )
            continue
        series = _series_for(cfg_records, item.metric_id, item.view)
        if any(series.values()):
            panels.append((spec, item, series))
    if not panels:
        return None

    out_path = output_dir / cfg / "_groups" / f"{group.name}.png"
    ncols = min(len(panels), 4)
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.5, nrows * 3.4), squeeze=False,
    )
    handles_seen: dict[str, Any] = {}
    for i, (spec, item, series) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        for exp in sorted(series.keys()):
            pts = series[exp]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            (line,) = ax.plot(
                xs, ys, marker="o", linewidth=1.6, markersize=4,
                color=color_map.get(exp), label=exp,
            )
            handles_seen.setdefault(exp, line)
        arrow = _direction_arrow(spec.direction)
        ax.set_title(
            f"{spec.display}  [{arrow}]\n({_short_view_tag(item.view)})",
            fontsize=10,
        )
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("value", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"{group.display}   [cfg={cfg}]\n"
        f"(↑ = higher is better,   ↓ = lower is better,   — = descriptive)",
        fontsize=14, fontweight="bold",
    )
    if handles_seen:
        sorted_exps = sorted(handles_seen.keys())
        fig.legend(
            [handles_seen[e] for e in sorted_exps],
            sorted_exps,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(sorted_exps), 6),
            fontsize=10,
            frameon=True,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out_path)
    return out_path


def _plot_group_worker(payload: tuple) -> Optional[str]:
    records, group_dict, cfg, output_dir_str, color_map = payload
    group = MetricGroup(
        name=group_dict["name"],
        display=group_dict["display"],
        items=tuple(GroupItem(**it) for it in group_dict["items"]),
    )
    p = _plot_group(records, group, cfg, Path(output_dir_str), color_map)
    return str(p) if p is not None else None


# --------------------------------------------------------------------------
# Consolidated all-metrics grid (one giant figure per (cfg, view))
# --------------------------------------------------------------------------
def _plot_all_metrics_grid(
    records: list[dict[str, Any]],
    cfg: str,
    view: str,
    output_dir: Path,
    color_map: dict[str, str],
) -> Optional[Path]:
    """One giant figure per (cfg, view), laying out subplots **one group per
    row** (in METRIC_GROUPS order). Each row's metrics follow the group's
    internal order. Group name appears as a rotated label at the left margin.

    Note
    ----
    For the ``view`` argument we IGNORE the per-item view overrides inside
    METRIC_GROUPS and use the same ``view`` (one of all / set1 / set2 / set3)
    for every subplot, so all panels are slice-comparable.
    """
    cfg_records = [r for r in records if r["cfg"] == cfg]
    if not cfg_records:
        return None
    rendered_groups: list[tuple[MetricGroup, list[tuple[MetricSpec, dict[str, list[tuple[int, float]]]]]]] = []
    for grp in METRIC_GROUPS:
        panels: list[tuple[MetricSpec, dict[str, list[tuple[int, float]]]]] = []
        for item in grp.items:
            spec = _SPEC_BY_ID.get(item.metric_id)
            if spec is None:
                continue
            series = _series_for(cfg_records, spec.metric_id, view)
            if any(series.values()):
                panels.append((spec, series))
        if panels:
            rendered_groups.append((grp, panels))
    if not rendered_groups:
        return None

    out_path = output_dir / cfg / "_all_metrics" / f"{view}.png"
    ncols = max(len(panels) for _, panels in rendered_groups)
    nrows = len(rendered_groups)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.2 + 1.0, nrows * 3.2),
        squeeze=False,
    )
    handles_seen: dict[str, Any] = {}
    for ri, (grp, panels) in enumerate(rendered_groups):
        for ci in range(ncols):
            ax = axes[ri][ci]
            if ci >= len(panels):
                ax.set_visible(False)
                continue
            spec, series = panels[ci]
            for exp in sorted(series.keys()):
                pts = series[exp]
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                (line,) = ax.plot(
                    xs, ys, marker="o", linewidth=1.5, markersize=4,
                    color=color_map.get(exp), label=exp,
                )
                handles_seen.setdefault(exp, line)
            arrow = _direction_arrow(spec.direction)
            ax.set_title(f"{spec.display}  [{arrow}]", fontsize=10)
            ax.set_xlabel("step", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"All metrics × all experiments   [cfg={cfg}]   view: {_short_view_tag(view)}\n"
        f"(↑ = higher is better,   ↓ = lower is better,   — = descriptive)",
        fontsize=14, fontweight="bold",
    )
    if handles_seen:
        sorted_exps = sorted(handles_seen.keys())
        fig.legend(
            [handles_seen[e] for e in sorted_exps],
            sorted_exps,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(len(sorted_exps), 6),
            fontsize=10,
            frameon=True,
        )
    fig.tight_layout(rect=(0.045, 0, 1, 0.92))
    for ri, (grp, _) in enumerate(rendered_groups):
        pos = axes[ri][0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(
            0.012, y_center, grp.display,
            rotation=90, fontsize=12, fontweight="bold",
            va="center", ha="left",
        )
    _save(fig, out_path)
    return out_path


def _plot_all_metrics_worker(payload: tuple) -> Optional[str]:
    records, cfg, view, output_dir_str, color_map = payload
    p = _plot_all_metrics_grid(records, cfg, view, Path(output_dir_str), color_map)
    return str(p) if p is not None else None


def render_all_plots(
    records: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
) -> list[str]:
    experiments = _all_experiments(records)
    cfgs = _all_cfgs(records)
    if not experiments or not cfgs:
        return []
    color_map = _build_color_map(experiments)

    # 1. Per-metric × per-view (all / set1 / set2 / set3 / all_sets grid)
    single_plan: list[tuple] = []
    for cfg in cfgs:
        for spec in METRIC_SPECS:
            cfg_metric_records = [
                r for r in records
                if r["cfg"] == cfg and r["metric_id"] == spec.metric_id
            ]
            if not cfg_metric_records:
                continue
            views = _views_for(cfg_metric_records)
            grid_subviews = _grid_subviews_for(cfg_metric_records)
            for view in views + ["all_sets"]:
                single_plan.append((
                    cfg_metric_records,
                    spec.__dict__,
                    cfg,
                    view,
                    grid_subviews,
                    str(output_dir),
                    color_map,
                ))

    # 2. Group dashboards (one figure per (cfg, group))
    group_plan: list[tuple] = []
    for cfg in cfgs:
        for grp in METRIC_GROUPS:
            group_plan.append((
                records,
                {
                    "name": grp.name,
                    "display": grp.display,
                    "items": [{"metric_id": it.metric_id, "view": it.view} for it in grp.items],
                },
                cfg,
                str(output_dir),
                color_map,
            ))

    # 3. Consolidated all-metrics grid (one figure per (cfg, view))
    grid_plan: list[tuple] = []
    for cfg in cfgs:
        cfg_records = [r for r in records if r["cfg"] == cfg]
        for view in _views_for(cfg_records):
            grid_plan.append((
                records,
                cfg,
                view,
                str(output_dir),
                color_map,
            ))

    logging.info(
        "rendering %d single-metric + %d group(s) + %d all-metrics grid(s) "
        "with %d workers",
        len(single_plan), len(group_plan), len(grid_plan), workers,
    )

    written: list[str] = []
    if workers <= 1:
        for task in _progress(
            single_plan, total=len(single_plan), desc="single-metric plots", unit="plot",
        ):
            r = _plot_single_worker(task)
            if r:
                written.append(r)
        for task in _progress(
            group_plan, total=len(group_plan), desc="group plots", unit="plot",
        ):
            r = _plot_group_worker(task)
            if r:
                written.append(r)
        for task in _progress(
            grid_plan, total=len(grid_plan), desc="all-metrics grids", unit="plot",
        ):
            r = _plot_all_metrics_worker(task)
            if r:
                written.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            single_results = ex.map(_plot_single_worker, single_plan, chunksize=2)
            for r in _progress(
                single_results, total=len(single_plan), desc="single-metric plots", unit="plot",
            ):
                if r:
                    written.append(r)
            group_results = ex.map(_plot_group_worker, group_plan, chunksize=1)
            for r in _progress(
                group_results, total=len(group_plan), desc="group plots", unit="plot",
            ):
                if r:
                    written.append(r)
            grid_results = ex.map(_plot_all_metrics_worker, grid_plan, chunksize=1)
            for r in _progress(
                grid_results, total=len(grid_plan), desc="all-metrics grids", unit="plot",
            ):
                if r:
                    written.append(r)
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval-root", required=True,
                   help="EVAL_OUTPUT_ROOT from run_my_eval.py, or a sharded eval root "
                        "produced by validate_checkpoints.py.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write plots + CSV. Default: <eval-root>/_plots/")
    p.add_argument("--cfg", nargs="*", default=None,
                   help="Restrict to these cfg dir names (e.g. cfg_dual cfg_simple). "
                        "Default: every cfg dir found.")
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Restrict to these experiment names. Default: all under eval-root.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)),
                   help="Process-pool size for parsing + plotting (default: cpu_count).")
    p.add_argument("--csv-only", action="store_true",
                   help="Skip rendering plots; only write metrics_long.csv.")
    p.add_argument("--from-per-sample", action="store_true",
                   help="Recompute metric/category aggregates from per_sample/<kind>/*.json "
                        "instead of using summary JSONs. This preserves filename-derived "
                        "categories such as set3-large and set3-medium-large.")
    p.add_argument("--include-pe-av", action="store_true",
                   help="Include PE-AV plots/CSV rows (PE-TV, PE-TA, PE-TAV). Disabled by default.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else eval_root / "_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_filter = set(args.cfg) if args.cfg else None
    experiments_filter = set(args.experiments) if args.experiments else None
    configure_metric_registry(include_pe_av=bool(args.include_pe_av))
    logging.info("PE-AV plotting: %s", "enabled" if args.include_pe_av else "disabled")

    records = collect_long_records(
        eval_root,
        cfg_filter,
        experiments_filter,
        args.workers,
        args.from_per_sample,
    )
    logging.info("collected %d long-format records", len(records))
    if not records:
        logging.warning("no records found; check --eval-root / --cfg / --experiments")
        return 1

    csv_path = output_dir / "metrics_long.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("experiment,step,cfg,metric_id,category,value,n_samples\n")
        for r in records:
            ns = r.get("n_samples")
            ns_str = "" if ns is None else f"{int(ns)}"
            f.write(
                f'{r["experiment"]},{r["step"]},{r["cfg"]},{r["metric_id"]},'
                f'{r["category"]},{r["value"]:.10g},{ns_str}\n'
            )
    logging.info("wrote %s (%d rows)", csv_path, len(records))

    if args.csv_only:
        return 0

    written = render_all_plots(records, output_dir, args.workers)
    logging.info("wrote %d PNG plot(s) under %s", len(written), output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
