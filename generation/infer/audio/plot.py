"""Aggregate WER plots across the (cfg, ckpt, set) sweep grid.

Reads the per-set ``wer_summary.json`` files written by ``run_eval.py`` and
renders:

1. A **global** figure (one row per metric, one column per prompt set) where
   each line is one checkpoint, x = cfg. This is the cross-experiment compare
   view (``wer_summary.png``).
2. (Optional) **per-experiment** figures: one PNG per training run grouped by
   ``CkptEntry.experiment``. For each experiment the layout is the same
   (rows = metrics, cols = prompt sets) but the **x-axis is the training
   step** and **each line is a CFG value**, so you can see how WER evolves as
   the run trains, separately for every cfg setting
   (``wer_summary__<experiment-slug>.png``).

The plot files are written next to the sweep output and embedded into
``report.html`` (see ``infer.audio.report``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence


_TYPE_SLUG_RE = re.compile(r"[^0-9a-z]+")


def _slug(label) -> str:
    text = str(label).strip().lower()
    text = _TYPE_SLUG_RE.sub("_", text).strip("_")
    return text or "unknown"


def _format_cfg_for_dir(cfg_value: float) -> str:
    if float(cfg_value).is_integer():
        return str(int(cfg_value))
    return ("%g" % float(cfg_value)).replace("-", "m")


def _set_dir(output_dir: Path, cfg_value: float, ckpt_name: str, set_name: str) -> Path:
    return output_dir / f"cfg-{_slug(_format_cfg_for_dir(cfg_value))}" / ckpt_name / _slug(set_name)


def _read_summary(set_dir: Path) -> dict | None:
    path = set_dir / "wer_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _build_global_plot(
    *,
    output_dir: Path,
    cfg_list: Sequence[float],
    ckpts: Sequence[tuple[str, int]],
    prompt_sets: Sequence[str],
    metrics: Sequence[str],
    filename: str,
    title: str | None,
) -> Path | None:
    """One PNG with rows=metrics, cols=sets; each line = one ckpt; x = cfg.

    This is the cross-experiment compare view that existed before
    `--checkpoint <root>/*` / per-experiment plotting was added.
    """
    if not cfg_list or not ckpts or not prompt_sets:
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

    for col, set_name in enumerate(prompt_sets):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            for k_idx, (ckpt_name, ckpt_step) in enumerate(ckpts):
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
                    color=cmap(k_idx % 10),
                    label=f"{ckpt_name} (step {ckpt_step})",
                )
                if row == 0 and col == 0:
                    handles_for_legend.append(line)
                    labels_for_legend.append(f"{ckpt_name} (step {ckpt_step})")

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

    out_path = Path(output_dir) / filename
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_experiment_plot(
    *,
    output_dir: Path,
    cfg_list: Sequence[float],
    ckpts: Sequence[tuple[str, int]],
    prompt_sets: Sequence[str],
    metrics: Sequence[str],
    filename: str,
    title: str | None,
) -> Path | None:
    """One PNG per training run: x = step, one line per cfg.

    Used when a single sweep mixes ckpts from multiple training runs (the
    ``--checkpoint <root>/*`` use-case): each experiment gets its own plot
    that shows how WER evolves over training steps for each cfg.
    """
    if not cfg_list or not ckpts or not prompt_sets:
        return None
    plt = _import_matplotlib()
    if plt is None:
        return None

    # Sort ckpts by step so the line plot reads left-to-right as training progresses.
    ckpts_sorted = sorted(ckpts, key=lambda kv: (int(kv[1]), str(kv[0])))
    if not ckpts_sorted:
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

    for col, set_name in enumerate(prompt_sets):
        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            for c_idx, cfg_v in enumerate(cfg_list):
                xs: list[int] = []
                ys: list[float] = []
                for ckpt_name, ckpt_step in ckpts_sorted:
                    summary = _read_summary(_set_dir(output_dir, cfg_v, ckpt_name, set_name))
                    if summary is None:
                        continue
                    val = summary.get(metric)
                    if val is None:
                        continue
                    xs.append(int(ckpt_step))
                    ys.append(float(val))
                if not xs:
                    continue
                found_any = True
                line, = ax.plot(
                    xs,
                    ys,
                    marker="o",
                    color=cmap(c_idx % 10),
                    label=f"cfg={cfg_v}",
                )
                if row == 0 and col == 0:
                    handles_for_legend.append(line)
                    labels_for_legend.append(f"cfg={cfg_v}")

            if row == 0:
                ax.set_title(set_name, fontsize=11)
            if col == 0:
                ax.set_ylabel(metric)
            if row == n_metrics - 1:
                ax.set_xlabel("training step")
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            steps_unique = sorted({int(s) for _, s in ckpts_sorted})
            ax.set_xticks(steps_unique)
            # Step values can be five+ digits → tilt to keep them readable.
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_horizontalalignment("right")

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
            ncol=min(6, len(handles_for_legend)),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.96 if title else 1.0))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1.0))

    out_path = Path(output_dir) / filename
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_summary_plot(
    *,
    output_dir: Path,
    cfg_list: Iterable[float],
    ckpts: Sequence[tuple[str, int]],
    prompt_sets: Sequence[str],
    experiments: Sequence[tuple[str, Sequence[tuple[str, int]]]] | None = None,
    metrics: Sequence[str] = ("mean_wer", "mean_wer_below_50"),
    filename: str = "wer_summary.png",
    title: str | None = "WER vs CFG (per set / per ckpt)",
) -> dict:
    """Render the global mixed plot and (if given) one plot per experiment.

    Returns a dict with::

        {
            "global_plot": Path | None,
            "experiments": {exp_name: Path, ...},
        }

    A missing entry means matplotlib was unavailable or no
    ``wer_summary.json`` had been written under that scope yet.
    """
    output_dir = Path(output_dir).resolve()
    cfg_list = list(cfg_list)

    global_plot = _build_global_plot(
        output_dir=output_dir,
        cfg_list=cfg_list,
        ckpts=ckpts,
        prompt_sets=prompt_sets,
        metrics=metrics,
        filename=filename,
        title=title,
    )

    exp_plots: dict[str, Path] = {}
    if experiments:
        for exp_name, exp_ckpts in experiments:
            exp_filename = f"wer_summary__{_slug(exp_name)}.png"
            exp_path = _build_experiment_plot(
                output_dir=output_dir,
                cfg_list=cfg_list,
                ckpts=exp_ckpts,
                prompt_sets=prompt_sets,
                metrics=metrics,
                filename=exp_filename,
                title=f"{exp_name}: WER vs step (per cfg / per set)",
            )
            if exp_path is not None:
                exp_plots[exp_name] = exp_path

    return {"global_plot": global_plot, "experiments": exp_plots}


__all__ = ["build_summary_plot"]
