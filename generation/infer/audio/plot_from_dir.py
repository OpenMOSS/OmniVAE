"""Re-plot WER summary figures from an existing sweep output directory.

Use this when you already have a sweep output (the directory written by
``run_eval.py``) and want to (re)generate aggregate plots without rerunning
inference. Reads each ``cfg-<slug>/<experiment>__checkpoint-<step>/<set>/wer_summary.json``
and renders:

1. The **global** plot (WER vs CFG, line per ckpt, col per set) -- same file
   ``run_eval.py`` writes (``wer_summary.png``).
2. **Per-experiment** plots (WER vs step, line per cfg, col per set) -- same
   files ``run_eval.py`` writes (``wer_summary__<experiment>.png``).
3. **Shared-step cross-experiment** plots (NEW): for every training step that
   is present in ``--min-experiments-per-step`` or more experiments, one PNG
   with ``WER vs CFG`` and one line per experiment
   (``wer_summary__step-<NNNNNNNN>.png``).

Usage::

    cd <repo-root>
    python -m infer.audio.plot_from_dir \
        --output-dir /path/to/sweep-output

Or directly (the script auto-injects the repo root into sys.path so that
``from infer.audio.plot import ...`` resolves either way)::

    python infer/audio/plot_from_dir.py --output-dir /path/to/sweep-output

The script is read-only with respect to the per-set ``wer_summary.json``
files; it only writes PNGs at the top level of ``--output-dir``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

# Allow running as a plain script: `python infer/audio/plot_from_dir.py ...`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from infer.audio.plot import (  # noqa: E402
    _import_matplotlib,
    _read_summary,
    _set_dir,
    build_summary_plot,
)


# Directory naming used by run_eval.py (see plot.py:_set_dir):
#   <output_dir>/cfg-<cfg_slug>/<experiment>__checkpoint-<step>/<set_slug>/wer_summary.json
# The step is zero-padded but we only require ``\d+`` here so we can also pick
# up checkpoints written by older runs that didn't pad to 8 digits.
CKPT_NAME_RE = re.compile(r"^(?P<exp>.+)__checkpoint-(?P<step>\d+)$")


# ---------------------------------------------------------------------------
# scan helpers
# ---------------------------------------------------------------------------


def _parse_cfg_dir(name: str) -> float | None:
    """Reverse ``cfg-<slug>`` produced by ``plot._format_cfg_for_dir`` + ``_slug``.

    ``_format_cfg_for_dir`` writes integers as ``"3"`` and non-integers via
    ``"%g"`` (e.g. ``"1.5"``, ``"-2.5"`` -> ``"m2.5"``); ``_slug`` then
    collapses ``"."`` -> ``"_"``. This reverses both transforms; returns
    ``None`` when ``name`` doesn't look like a CFG bucket.
    """
    if not name.startswith("cfg-"):
        return None
    body = name[len("cfg-"):]
    if not body:
        return None
    if body.startswith("m") and len(body) > 1 and body[1].isdigit():
        signed = "-" + body[1:]
    else:
        signed = body
    signed = signed.replace("_", ".")
    try:
        return float(signed)
    except ValueError:
        return None


def _scan_sweep(output_dir: Path) -> dict:
    """Walk ``output_dir`` and surface everything we need to plot.

    The walk is purely directory-driven: any ``cfg-*`` child of
    ``output_dir`` becomes a CFG bucket, any ``<exp>__checkpoint-<step>`` child
    of that bucket becomes a checkpoint, and any subdir holding a
    ``wer_summary.json`` becomes a prompt-set entry. Missing files are
    silently skipped so this script also works on partial / interrupted sweeps.
    """
    cfg_values: set[float] = set()
    ckpt_meta: dict[str, tuple[str, int]] = {}  # ckpt_name -> (experiment, step)
    set_names: set[str] = set()
    summaries: dict[tuple[float, str, str], dict] = {}

    if not output_dir.is_dir():
        return {
            "cfg_values": [],
            "ckpts": [],
            "ckpt_to_exp": {},
            "set_names": [],
            "summaries": {},
        }

    for cfg_child in sorted(output_dir.iterdir()):
        if not cfg_child.is_dir():
            continue
        cfg_v = _parse_cfg_dir(cfg_child.name)
        if cfg_v is None:
            continue
        # Don't record the CFG until we actually find at least one summary
        # under it (otherwise an empty cfg-X dir would show up as an x-tick
        # with no data).
        cfg_has_data = False

        for ckpt_child in sorted(cfg_child.iterdir()):
            if not ckpt_child.is_dir():
                continue
            m = CKPT_NAME_RE.match(ckpt_child.name)
            if m is None:
                continue
            exp = m.group("exp")
            step = int(m.group("step"))

            for set_child in sorted(ckpt_child.iterdir()):
                if not set_child.is_dir():
                    continue
                summary_path = set_child / "wer_summary.json"
                if not summary_path.is_file():
                    continue
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                # The set_dir name is already the slug used by _set_dir, so we
                # can hand it back to plot.py untouched.
                set_key = set_child.name
                set_names.add(set_key)
                summaries[(cfg_v, ckpt_child.name, set_key)] = summary
                ckpt_meta[ckpt_child.name] = (exp, step)
                cfg_has_data = True

        if cfg_has_data:
            cfg_values.add(cfg_v)

    ckpts_sorted = sorted(
        ((name, step) for name, (_exp, step) in ckpt_meta.items()),
        key=lambda kv: (int(kv[1]), str(kv[0])),
    )
    return {
        "cfg_values": sorted(cfg_values),
        "ckpts": ckpts_sorted,
        "ckpt_to_exp": {name: exp for name, (exp, _step) in ckpt_meta.items()},
        "set_names": sorted(set_names),
        "summaries": summaries,
    }


def _group_by_experiment(
    ckpts: Sequence[tuple[str, int]],
    ckpt_to_exp: dict[str, str],
) -> list[tuple[str, list[tuple[str, int]]]]:
    """``[(experiment_name, [(ckpt_name, step), ...]), ...]`` for per-exp plots.

    Each per-experiment list is sorted by step, ascending, so the plot reads
    left-to-right as training progresses.
    """
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, step in ckpts:
        grouped[ckpt_to_exp.get(name, "default")].append((name, step))
    out = []
    for exp_name in sorted(grouped):
        rows = sorted(grouped[exp_name], key=lambda kv: int(kv[1]))
        out.append((exp_name, rows))
    return out


# ---------------------------------------------------------------------------
# the new "shared-step cross-experiment" plot
# ---------------------------------------------------------------------------


def _build_step_compare_plot(
    *,
    output_dir: Path,
    cfg_list: Sequence[float],
    exp_to_ckpt: dict[str, str],
    prompt_sets: Sequence[str],
    metrics: Sequence[str],
    filename: str,
    title: str | None,
) -> Path | None:
    """One PNG comparing experiments **at one fixed training step**.

    Layout: rows = metrics, cols = prompt sets, x = CFG, line = experiment.
    ``exp_to_ckpt[exp]`` is the directory name of that experiment's checkpoint
    at this step, i.e. ``<exp>__checkpoint-<step>`` -- exactly what
    ``_set_dir`` expects for ``ckpt_name``.
    """
    if not cfg_list or not exp_to_ckpt or not prompt_sets:
        return None
    plt = _import_matplotlib()
    if plt is None:
        return None

    n_sets = len(prompt_sets)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        n_metrics,
        n_sets,
        figsize=(max(4.0, 4.0 * n_sets), max(3.0, 3.2 * n_metrics)),
        squeeze=False,
        sharex=True,
    )

    cmap = plt.get_cmap("tab10")
    found_any = False
    handles_for_legend: list = []
    labels_for_legend: list[str] = []

    exp_names = sorted(exp_to_ckpt.keys())
    for col, set_name in enumerate(prompt_sets):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            for e_idx, exp_name in enumerate(exp_names):
                ckpt_name = exp_to_ckpt[exp_name]
                xs: list[float] = []
                ys: list[float] = []
                for cfg_v in cfg_list:
                    summary = _read_summary(_set_dir(output_dir, cfg_v, ckpt_name, set_name))
                    if summary is None:
                        continue
                    val = summary.get(metric)
                    if val is None:
                        continue
                    xs.append(float(cfg_v))
                    ys.append(float(val))
                if not xs:
                    continue
                found_any = True
                line, = ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=cmap(e_idx % 10),
                    label=exp_name,
                )
                if row == 0 and col == 0:
                    handles_for_legend.append(line)
                    labels_for_legend.append(exp_name)

            if row == 0:
                ax.set_title(set_name, fontsize=11)
            if col == 0:
                ax.set_ylabel(metric)
            if row == n_metrics - 1:
                ax.set_xlabel("CFG  (pred = pos + s * (pos - neg))")
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            ax.set_xticks(list(cfg_list))

    if not found_any:
        plt.close(fig)
        return None

    if title:
        fig.suptitle(title, fontsize=12)
    if handles_for_legend:
        fig.legend(
            handles_for_legend,
            labels_for_legend,
            loc="lower center",
            ncol=min(4, len(handles_for_legend)),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.96 if title else 1.0))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1.0))

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_step_compare_plots(
    *,
    output_dir: Path,
    cfg_list: Sequence[float],
    ckpts: Sequence[tuple[str, int]],
    ckpt_to_exp: dict[str, str],
    prompt_sets: Sequence[str],
    metrics: Sequence[str] = ("mean_wer", "mean_wer_below_50"),
    require_min_experiments: int = 2,
) -> dict[int, Path]:
    """For every step shared by ``>= require_min_experiments`` experiments,
    render one comparison PNG. Returns ``{step: png_path}``.

    Steps that don't meet the threshold or where every requested metric is
    missing on disk are silently skipped.
    """
    by_step: dict[int, dict[str, str]] = defaultdict(dict)
    for ckpt_name, step in ckpts:
        exp = ckpt_to_exp.get(ckpt_name, "default")
        # If the same experiment somehow has two ckpt dirs at the same step
        # (unlikely but possible after manual renames), keep whichever sorts
        # last so we deterministically pick one.
        by_step[int(step)][exp] = ckpt_name

    out: dict[int, Path] = {}
    for step in sorted(by_step):
        exp_to_ckpt = by_step[step]
        if len(exp_to_ckpt) < require_min_experiments:
            continue
        path = _build_step_compare_plot(
            output_dir=Path(output_dir),
            cfg_list=cfg_list,
            exp_to_ckpt=exp_to_ckpt,
            prompt_sets=prompt_sets,
            metrics=metrics,
            filename=f"wer_summary__step-{step:08d}.png",
            title=(
                f"step={step}: WER vs CFG  (lines = experiment, "
                f"{len(exp_to_ckpt)} experiments)"
            ),
        )
        if path is not None:
            out[step] = path
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Re-render WER summary plots from an existing sweep output dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Sweep output dir written by run_eval.py (the dir holding cfg-*/).",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=(
            "WER metric key to plot. Repeat to add more (e.g. "
            "'--metric mean_wer --metric mean_wer_below_50'). Default: mean_wer."
        ),
    )
    parser.add_argument(
        "--min-experiments-per-step",
        type=int,
        default=2,
        help=(
            "Only emit a step-compare plot for steps with at least N distinct "
            "experiments present. Default: 2 (i.e. only render the figure when "
            "there is something to compare)."
        ),
    )
    parser.add_argument(
        "--no-global",
        action="store_true",
        help="Skip the global (cross-ckpt) WER vs CFG plot.",
    )
    parser.add_argument(
        "--no-per-experiment",
        action="store_true",
        help="Skip the per-experiment WER vs step plots.",
    )
    parser.add_argument(
        "--no-step-compare",
        action="store_true",
        help="Skip the shared-step cross-experiment plots.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise SystemExit(f"[plot_from_dir] output-dir not found: {output_dir}")

    metrics: tuple[str, ...]
    if args.metric:
        metrics = tuple(args.metric)
    else:
        metrics = ("mean_wer",)

    print(f"[plot_from_dir] scanning: {output_dir}")
    scan = _scan_sweep(output_dir)
    cfg_list = scan["cfg_values"]
    ckpts = scan["ckpts"]
    set_names = scan["set_names"]
    ckpt_to_exp = scan["ckpt_to_exp"]

    if not cfg_list or not ckpts or not set_names:
        raise SystemExit(
            f"[plot_from_dir] nothing to plot. "
            f"cfg={cfg_list}, ckpts={len(ckpts)}, sets={set_names}"
        )

    experiments = _group_by_experiment(ckpts, ckpt_to_exp)
    print(f"[plot_from_dir]   cfg values : {cfg_list}")
    print(f"[plot_from_dir]   prompt sets: {set_names}")
    print(f"[plot_from_dir]   metrics    : {list(metrics)}")
    print(f"[plot_from_dir]   ckpts      : {len(ckpts)} across {len(experiments)} experiments")
    for exp_name, group in experiments:
        steps = ", ".join(str(step) for _name, step in group)
        print(f"[plot_from_dir]     - {exp_name}: {len(group)} ckpt(s) [steps={steps}]")

    # Plots 1 + 2: existing global + per-experiment.
    summary_plots = build_summary_plot(
        output_dir=output_dir,
        cfg_list=cfg_list,
        ckpts=ckpts,
        prompt_sets=set_names,
        experiments=None if args.no_per_experiment else experiments,
        metrics=metrics,
    )

    if not args.no_global:
        gp = summary_plots.get("global_plot")
        if gp is not None:
            print(f"[plot_from_dir] wrote global plot   : {gp}")
        else:
            print("[plot_from_dir] global plot skipped (no data or matplotlib missing)")
    if not args.no_per_experiment:
        for exp_name, png in summary_plots.get("experiments", {}).items():
            print(f"[plot_from_dir] wrote per-exp plot  : {png}  ({exp_name})")

    # Plot 3 (NEW): shared-step cross-experiment.
    if not args.no_step_compare:
        step_plots = build_step_compare_plots(
            output_dir=output_dir,
            cfg_list=cfg_list,
            ckpts=ckpts,
            ckpt_to_exp=ckpt_to_exp,
            prompt_sets=set_names,
            metrics=metrics,
            require_min_experiments=int(args.min_experiments_per_step),
        )
        if step_plots:
            for step, png in sorted(step_plots.items()):
                print(f"[plot_from_dir] wrote step-compare  : {png}  (step={step})")
        else:
            print(
                "[plot_from_dir] no shared step met "
                f"--min-experiments-per-step={args.min_experiments_per_step}; "
                "no step-compare plot written."
            )

    print("[plot_from_dir] done.")


if __name__ == "__main__":
    main()
