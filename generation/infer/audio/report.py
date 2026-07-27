"""HTML report writer for the (cfg x ckpt) inference sweep.

Layout of the generated ``report.html``:

1. Sweep summary table (mean WER per set, per (cfg, ckpt) pair).
2. Per-set comparison cards. Each card picks up to ``samples_per_set`` prompts
   and shows a grid of audio cells: rows = cfg, columns = ckpt. This makes it
   trivial to A/B same prompt across different cfgs and across different
   training steps.

Audio cells use ``<audio preload="none">`` with relative paths so the report
loads instantly regardless of how many prompts are surfaced; the browser only
fetches a wav when the user actually presses play.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Iterable


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


def _read_meta_jsonl(meta_path: Path) -> list[dict]:
    if not meta_path.is_file():
        return []
    out: list[dict] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _read_wer_summary(set_dir: Path) -> dict | None:
    path = set_dir / "wer_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_wer_jsonl(set_dir: Path) -> dict[int, dict]:
    """Map global_idx -> per-record dict for quick lookup of hyp / wer."""
    path = set_dir / "wer.jsonl"
    if not path.is_file():
        return {}
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        idx = record.get("global_idx")
        if idx is None:
            continue
        out[int(idx)] = record
    return out


def _pick_indices(total: int, k: int, *, strategy: str) -> list[int]:
    if total <= 0 or k <= 0:
        return []
    k = min(k, total)
    if strategy == "first":
        return list(range(k))
    if strategy == "evenly_spaced":
        if k == 1:
            return [total // 2]
        # Approx evenly spaced including endpoints.
        return [round(i * (total - 1) / (k - 1)) for i in range(k)]
    return list(range(k))


def _pick_per_type(records: list[dict], k: int, *, strategy: str) -> list[dict]:
    """Try to keep type variety: pick `ceil(k / num_types)` from each, then
    fall back to filling up to `k` from the remainder.
    """
    if not records or k <= 0:
        return []

    by_type: dict[str, list[dict]] = {}
    type_order: list[str] = []
    for record in records:
        type_label = str(record.get("type", "all"))
        if type_label not in by_type:
            type_order.append(type_label)
            by_type[type_label] = []
        by_type[type_label].append(record)

    n_types = max(1, len(type_order))
    base_quota = max(1, k // n_types)
    chosen: list[dict] = []
    for type_label in type_order:
        bucket = by_type[type_label]
        idxs = _pick_indices(len(bucket), base_quota, strategy=strategy)
        chosen.extend(bucket[i] for i in idxs)
    if len(chosen) > k:
        chosen = chosen[:k]
    elif len(chosen) < k:
        seen = {(r.get("type"), r.get("global_idx")) for r in chosen}
        for record in records:
            key = (record.get("type"), record.get("global_idx"))
            if key in seen:
                continue
            chosen.append(record)
            seen.add(key)
            if len(chosen) >= k:
                break
    return chosen


def _rel_audio_src(report_root: Path, wav_path: str | None) -> str | None:
    if not wav_path:
        return None
    candidate = Path(wav_path)
    try:
        rel = candidate.resolve().relative_to(report_root.resolve())
    except ValueError:
        return None
    return str(rel)


def _format_float(value, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return format(float(value), f".{digits}f")
    except (TypeError, ValueError):
        return str(value)


def build_report_html(
    *,
    output_dir: Path,
    cfg_list: Iterable[float],
    ckpts: list[tuple[str, int]],
    prompt_sets: list[str],
    samples_per_set: int,
    sample_strategy: str,
    title: str = "T2A inference sweep",
    summary_plot_filename: str | None = None,
    per_experiment_plot_filenames: dict[str, str] | None = None,
) -> Path:
    output_dir = Path(output_dir).resolve()
    cfg_list = list(cfg_list)
    if not ckpts:
        raise ValueError("No checkpoints provided to build_report_html.")
    if not prompt_sets:
        raise ValueError("No prompt sets provided to build_report_html.")

    # ------------------------- WER summary table ---------------------------
    summary_rows: list[dict] = []
    for cfg_value in cfg_list:
        for ckpt_name, ckpt_step in ckpts:
            row: dict = {
                "cfg": cfg_value,
                "ckpt": ckpt_name,
                "step": ckpt_step,
                "by_set": {},
            }
            for set_name in prompt_sets:
                set_dir = _set_dir(output_dir, cfg_value, ckpt_name, set_name)
                summary = _read_wer_summary(set_dir)
                row["by_set"][set_name] = summary
            summary_rows.append(row)

    # ------------------------- Per-set audio grid --------------------------
    set_blocks: list[str] = []
    for set_name in prompt_sets:
        # Pick the master prompt list from the first (cfg, ckpt) that has data.
        master_records: list[dict] = []
        for cfg_value in cfg_list:
            for ckpt_name, _ in ckpts:
                set_dir = _set_dir(output_dir, cfg_value, ckpt_name, set_name)
                records = _read_meta_jsonl(set_dir / "meta.jsonl")
                if records:
                    master_records = records
                    break
            if master_records:
                break

        chosen = _pick_per_type(master_records, samples_per_set, strategy=sample_strategy)
        if not chosen:
            set_blocks.append(
                f'<section class="set-block"><h2>{html.escape(set_name)}</h2>'
                f'<p class="empty">No generated samples found for this set.</p></section>'
            )
            continue

        # Pre-load WER per (cfg, ckpt, set) once (avoid quadratic parsing).
        wer_map: dict[tuple[float, str], dict[int, dict]] = {}
        for cfg_value in cfg_list:
            for ckpt_name, _ in ckpts:
                set_dir = _set_dir(output_dir, cfg_value, ckpt_name, set_name)
                wer_map[(cfg_value, ckpt_name)] = _read_wer_jsonl(set_dir)

        # Index meta.jsonl per (cfg, ckpt, set) once for quick wav_path lookup.
        meta_map: dict[tuple[float, str], dict[int, dict]] = {}
        for cfg_value in cfg_list:
            for ckpt_name, _ in ckpts:
                set_dir = _set_dir(output_dir, cfg_value, ckpt_name, set_name)
                records = _read_meta_jsonl(set_dir / "meta.jsonl")
                meta_map[(cfg_value, ckpt_name)] = {
                    int(r["global_idx"]): r for r in records if "global_idx" in r
                }

        rows_html: list[str] = []
        for record in chosen:
            global_idx = int(record.get("global_idx", -1))
            type_label = str(record.get("type", "all"))
            index_label = record.get("index", global_idx)
            text = str(record.get("text", "")).strip()

            header_html = (
                f'<div class="prompt-header">'
                f'<span class="badge type">{html.escape(type_label)}</span>'
                f'<span class="badge idx">#{html.escape(str(index_label))}</span>'
                f'<span class="prompt-text">{html.escape(text)}</span>'
                f"</div>"
            )

            grid_rows: list[str] = []
            # Header row: ckpts as columns
            ckpt_header_cells = "".join(
                f'<th>{html.escape(ckpt_name)}<br><span class="step">step {ckpt_step}</span></th>'
                for ckpt_name, ckpt_step in ckpts
            )
            grid_rows.append(f"<tr><th>cfg \\ ckpt</th>{ckpt_header_cells}</tr>")

            for cfg_value in cfg_list:
                cells: list[str] = [f'<th>{_format_float(cfg_value, digits=2)}</th>']
                for ckpt_name, _ in ckpts:
                    record_for_cell = meta_map.get((cfg_value, ckpt_name), {}).get(global_idx)
                    wav_rel = (
                        _rel_audio_src(output_dir, record_for_cell.get("wav_path"))
                        if record_for_cell
                        else None
                    )
                    wer_record = wer_map.get((cfg_value, ckpt_name), {}).get(global_idx)
                    parts: list[str] = []
                    if wav_rel:
                        parts.append(
                            f'<audio controls preload="none" src="{html.escape(wav_rel)}"></audio>'
                        )
                    else:
                        parts.append('<span class="missing">no wav</span>')
                    if wer_record is not None:
                        wer_value = wer_record.get("wer")
                        hyp = str(wer_record.get("hyp_norm", "")).strip()
                        parts.append(
                            f'<div class="wer-line">WER {_format_float(wer_value, digits=3)}</div>'
                        )
                        if hyp:
                            parts.append(
                                f'<div class="hyp" title="{html.escape(hyp)}">{html.escape(hyp[:120])}{"…" if len(hyp) > 120 else ""}</div>'
                            )
                    cells.append(f"<td>{''.join(parts)}</td>")
                grid_rows.append("<tr>" + "".join(cells) + "</tr>")

            rows_html.append(
                '<div class="prompt-card">'
                + header_html
                + '<div class="grid-scroll"><table class="grid">'
                + "".join(grid_rows)
                + "</table></div>"
                + "</div>"
            )

        set_blocks.append(
            f'<section class="set-block">'
            f'<h2>{html.escape(set_name)}</h2>'
            + "".join(rows_html)
            + "</section>"
        )

    # ------------------------- Top summary table HTML ----------------------
    summary_header_cells = (
        "<tr>"
        '<th rowspan="2">CFG</th>'
        '<th rowspan="2">Checkpoint</th>'
        '<th rowspan="2">Step</th>'
        + "".join(f'<th colspan="2">{html.escape(name)}</th>' for name in prompt_sets)
        + "</tr>"
        + "<tr>"
        + "".join("<th>mean WER</th><th>mean WER (≤0.5)</th>" for _ in prompt_sets)
        + "</tr>"
    )

    summary_body: list[str] = []
    for row in summary_rows:
        cells = [
            f'<td>{_format_float(row["cfg"], digits=2)}</td>',
            f'<td>{html.escape(row["ckpt"])}</td>',
            f'<td>{row["step"]}</td>',
        ]
        for set_name in prompt_sets:
            summary = row["by_set"].get(set_name)
            if summary is None:
                cells.append('<td class="missing">-</td><td class="missing">-</td>')
            else:
                cells.append(f'<td>{_format_float(summary.get("mean_wer"))}</td>')
                cells.append(f'<td>{_format_float(summary.get("mean_wer_below_50"))}</td>')
        summary_body.append("<tr>" + "".join(cells) + "</tr>")

    summary_table = (
        '<table class="summary"><thead>'
        + summary_header_cells
        + "</thead><tbody>"
        + "".join(summary_body)
        + "</tbody></table>"
    )

    css = """
:root {
  --bg: #0f1216;
  --fg: #e7eaee;
  --muted: #9aa3ad;
  --accent: #56b4ff;
  --card: #161a20;
  --border: #232a33;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 32px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: var(--bg); color: var(--fg); }
h1 { margin: 0 0 8px 0; font-size: 22px; }
h2 { margin: 24px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); font-size: 18px; }
.subtle { color: var(--muted); font-size: 12px; margin-bottom: 16px; }
section.set-block { margin-bottom: 36px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
table.summary { margin: 0 0 24px; background: var(--card); border: 1px solid var(--border); }
table.summary th, table.summary td { border: 1px solid var(--border); padding: 6px 10px; text-align: center; }
table.summary thead th { background: #1c222a; }
.grid-scroll { overflow-x: auto; margin-top: 8px; }
table.grid { margin: 0; background: var(--card); border: 1px solid var(--border); min-width: 100%; }
table.grid th, table.grid td { border: 1px solid var(--border); padding: 6px; text-align: center; vertical-align: middle; min-width: 220px; }
table.grid th:first-child, table.grid td:first-child { min-width: 80px; position: sticky; left: 0; background: #1c222a; }
table.grid thead th, table.grid th { background: #1c222a; font-weight: 600; }
.prompt-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; }
.prompt-header { margin-bottom: 8px; line-height: 1.5; }
.prompt-text { margin-left: 8px; }
.badge { display: inline-block; font-size: 11px; padding: 2px 6px; border-radius: 4px; background: #1f2730; color: var(--accent); }
.badge.type { margin-right: 4px; }
.badge.idx { margin-right: 4px; background: #1c222a; color: var(--muted); }
.step { color: var(--muted); font-size: 11px; font-weight: 400; }
audio { width: 100%; min-width: 220px; }
.missing { color: var(--muted); font-style: italic; }
.wer-line { font-size: 11px; color: var(--muted); margin-top: 4px; }
.hyp { font-size: 11px; color: var(--muted); margin-top: 2px; word-break: break-word; max-width: 320px; }
.summary-plot { background: #ffffff; border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin: 0 0 24px; text-align: center; }
.summary-plot img { max-width: 100%; height: auto; }
.exp-title { margin: 0 0 8px; font-size: 14px; color: #0f1216; text-align: left; }
"""

    summary_plot_html = ""
    if summary_plot_filename:
        plot_path = output_dir / summary_plot_filename
        if plot_path.is_file():
            summary_plot_html = (
                '<h2>WER vs CFG (per set / per ckpt)</h2>'
                f'<div class="summary-plot"><img src="{html.escape(summary_plot_filename)}"'
                f' alt="WER vs CFG"/></div>'
            )

    per_experiment_html = ""
    if per_experiment_plot_filenames:
        cards: list[str] = []
        for exp_name, fname in per_experiment_plot_filenames.items():
            if not fname:
                continue
            if not (output_dir / fname).is_file():
                continue
            cards.append(
                '<div class="summary-plot">'
                f'<h3 class="exp-title">{html.escape(exp_name)}</h3>'
                f'<img src="{html.escape(fname)}" alt="{html.escape(exp_name)}"/>'
                "</div>"
            )
        if cards:
            per_experiment_html = (
                '<h2>WER vs step (per experiment)</h2>'
                + "".join(cards)
            )

    page = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>" + css + "</style>"
        "</head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f'<div class="subtle">cfg list: {html.escape(", ".join(_format_float(v, digits=2) for v in cfg_list))}'
        f' &nbsp;|&nbsp; checkpoints: {html.escape(", ".join(name for name, _ in ckpts))}'
        f' &nbsp;|&nbsp; sets: {html.escape(", ".join(prompt_sets))}'
        f'</div>'
        + summary_plot_html
        + per_experiment_html
        + f'<h2>WER summary</h2>{summary_table}'
        + "".join(set_blocks)
        + "</body></html>"
    )

    out_path = output_dir / "report.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


__all__ = ["build_report_html"]
