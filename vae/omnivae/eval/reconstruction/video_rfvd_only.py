from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from omnivae.eval.reconstruction.common import resolve_path, setup_logging, write_json


def _run_name(vr_dir: Path) -> str:
    return vr_dir.parents[1].name


def _root(vr_dir: Path) -> Path:
    return vr_dir.parents[2]


def discover_tasks(output_roots: List[Path], max_tasks: int = 0) -> List[Path]:
    tasks: List[Path] = []
    for root in output_roots:
        if not root.is_dir():
            logging.warning("output root not found, skip: %s", root)
            continue
        for candidate in sorted(root.glob("*/no_ema/video_recon")):
            if (candidate / "gt").is_dir() and (candidate / "recon").is_dir():
                tasks.append(candidate)
    tasks.sort(key=lambda p: (str(_root(p)), _run_name(p)))
    return tasks[:max_tasks] if max_tasks > 0 else tasks


def _progress_names(vr_dir: Path) -> List[str]:
    names: List[str] = []
    seen = set()
    for progress in sorted((vr_dir / ".progress").glob("rank*.jsonl")):
        with progress.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = rec.get("name")
                if not name and rec.get("path"):
                    name = Path(str(rec["path"])).stem + ".mp4"
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def build_pairs(vr_dir: Path, max_videos: int = 0) -> List[Tuple[str, Path, Path]]:
    gt_dir = vr_dir / "gt"
    recon_dir = vr_dir / "recon"
    ordered = _progress_names(vr_dir)
    if not ordered:
        ordered = [p.name for p in sorted(recon_dir.glob("*.mp4"))]

    pairs: List[Tuple[str, Path, Path]] = []
    seen = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        gt = gt_dir / name
        recon = recon_dir / name
        if gt.is_file() and recon.is_file():
            pairs.append((name, gt, recon))

    for recon in sorted(recon_dir.glob("*.mp4")):
        if recon.name in seen:
            continue
        gt = gt_dir / recon.name
        if gt.is_file():
            pairs.append((recon.name, gt, recon))

    return pairs[:max_videos] if max_videos > 0 else pairs


def _load_group_map(jsonl_paths: List[Path]) -> Dict[str, str]:
    """Map saved video names/paths to dataset groups from metadata JSONL files."""
    groups: Dict[str, str] = {}
    for jsonl_path in jsonl_paths:
        group = jsonl_path.stem
        try:
            with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw = rec.get("video_path") or rec.get("video") or rec.get("path")
                    if not raw:
                        continue
                    source = Path(str(raw)).stem
                    video_id = str(rec.get("video_id", "")).strip()
                    for key in (source, video_id):
                        if key:
                            groups[key] = group
        except OSError as exc:
            logging.warning("could not read group JSONL %s: %s", jsonl_path, exc)
    return groups


def _pair_group(name: str, group_map: Dict[str, str]) -> str:
    stem = Path(name).stem
    # Saved names commonly have an index prefix (000123_original.mp4).
    candidates = [name, stem]
    if "_" in stem:
        candidates.append(stem.split("_", 1)[1])
    for key in candidates:
        if key in group_map:
            return group_map[key]
    return "default"


def _prepare_i3d_weight(i3d_torchscript_pt: Optional[str]) -> None:
    override = i3d_torchscript_pt or os.environ.get("I3D_TORCHSCRIPT_PT")
    if not override:
        return
    from omnivae.models.causalvideovae.eval.fvd.styleganv import fvd as fvd_module

    src = resolve_path(override)
    if not src.is_file():
        raise FileNotFoundError(f"I3D_TORCHSCRIPT_PT does not exist: {src}")
    if src.stat().st_size == 0:
        raise RuntimeError(f"I3D_TORCHSCRIPT_PT is empty: {src}")
    dst = Path(fvd_module.__file__).resolve().parent / "i3d_torchscript.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    logging.info("I3D weight ready: %s", dst)


def _read_video_01(path: Path) -> torch.Tensor:
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    if len(vr) == 0:
        raise ValueError(f"empty video: {path}")
    arr = vr.get_batch(list(range(len(vr)))).asnumpy()
    video = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    if video.shape[0] > 3:
        video = video[:3]
    if video.shape[0] == 1:
        video = video.repeat(3, 1, 1, 1)
    return video.float().div(255.0)


def _extract_features(
    pairs: List[Tuple[str, Path, Path]],
    *,
    i3d,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    from omnivae.models.causalvideovae.eval.fvd.styleganv.fvd import get_fvd_feats

    names_out: List[str] = []
    gt_out: List[np.ndarray] = []
    recon_out: List[np.ndarray] = []
    buckets: Dict[int, List[Tuple[str, torch.Tensor, torch.Tensor]]] = {}

    def flush(bucket: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> None:
        if not bucket:
            return
        names, gt_videos, recon_videos = zip(*bucket)
        gt_batch = torch.stack(gt_videos, 0)
        recon_batch = torch.stack(recon_videos, 0)
        with torch.no_grad():
            gt_feat = get_fvd_feats(gt_batch, i3d=i3d, device=device, bs=batch_size)
            recon_feat = get_fvd_feats(recon_batch, i3d=i3d, device=device, bs=batch_size)
        for name, fg, fr in zip(names, gt_feat, recon_feat):
            names_out.append(name)
            gt_out.append(fg.astype(np.float32))
            recon_out.append(fr.astype(np.float32))
        bucket.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    skipped = 0
    for name, gt_path, recon_path in tqdm(pairs, desc="rFVD pairs"):
        try:
            gt = _read_video_01(gt_path)
            recon = _read_video_01(recon_path)
            t = min(gt.shape[1], recon.shape[1])
            if t < 10:
                skipped += 1
                continue
            bucket = buckets.setdefault(t, [])
            bucket.append((name, gt[:, :t], recon[:, :t]))
            if len(bucket) >= batch_size:
                flush(bucket)
        except Exception as exc:
            skipped += 1
            logging.warning("skip %s: %s", name, exc)

    for bucket in buckets.values():
        flush(bucket)

    if skipped:
        logging.warning("skipped %d video pairs", skipped)
    if len(gt_out) < 2:
        raise RuntimeError(f"rFVD requires at least 2 valid pairs, got {len(gt_out)}")
    return names_out, np.stack(gt_out, 0), np.stack(recon_out, 0)


def _update_results(vr_dir: Path, rfvd: float, count: int, by_group: Dict[str, Dict[str, Any]]) -> Path:
    result_path = vr_dir.parents[1] / "results.json"
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    vr = data.setdefault("no_ema", {}).setdefault("video_recon", {})
    total = vr.setdefault("total", {})
    vr["rfvd"] = rfvd
    vr["rfvd_count"] = count
    total["rfvd"] = rfvd
    total["rfvd_count"] = count
    vr["by_group"] = by_group
    total["by_group"] = by_group
    write_json(result_path, data)
    return result_path


def _row_for(
    vr_dir: Path,
    rfvd: float,
    count: int,
    pairs: int,
    result_path: Path,
    by_group: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    root = _root(vr_dir)
    run_name = _run_name(vr_dir)
    return {
        "root": root.name,
        "run_name": run_name,
        "rfvd": rfvd,
        "rfvd_count": count,
        "pairs": pairs,
        "video_recon_dir": str(vr_dir),
        "results_json": str(result_path),
        "by_group": by_group,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["root", "run_name", "group", "rfvd", "rfvd_count", "pairs", "results_json", "video_recon_dir"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            base = {k: row.get(k, "") for k in fields}
            base["group"] = "overall"
            base["rfvd"] = f"{float(row['rfvd']):.6f}"
            writer.writerow(base)
            for group, group_result in sorted(row.get("by_group", {}).items()):
                group_row = {k: row.get(k, "") for k in fields}
                group_row.update(
                    group=group,
                    rfvd=f"{float(group_result['rfvd']):.6f}",
                    rfvd_count=group_result.get("rfvd_count", ""),
                    pairs=group_result.get("pairs", ""),
                )
                writer.writerow(group_row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute rFVD from existing OmniVAE video_recon outputs")
    parser.add_argument("--output_root", action="append", default=[], help="Root containing */no_ema/video_recon")
    parser.add_argument(
        "--group_jsonl", action="append", default=[],
        help="Metadata JSONL(s) used to split rFVD by dataset; group is the JSONL filename stem.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--i3d_torchscript_pt", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging()
    roots = [resolve_path(p) for p in args.output_root]
    if not roots:
        roots = [resolve_path("$OMNIVAE_EXP_ROOT/eval/video_recon")]

    tasks = discover_tasks(roots, max_tasks=args.max_tasks)
    logging.info("discovered rFVD tasks: %d", len(tasks))
    for idx, task in enumerate(tasks):
        logging.info("task[%d]=%s", idx, task)
    if args.dry_run:
        return 0
    if not tasks:
        raise SystemExit("No video_recon tasks found")

    _prepare_i3d_weight(args.i3d_torchscript_pt)
    from omnivae.models.causalvideovae.eval.fvd.styleganv.fvd import (
        frechet_distance,
        load_i3d_pretrained,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    i3d = load_i3d_pretrained(device=device).eval()
    group_map = _load_group_map([resolve_path(p) for p in args.group_jsonl])

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        pairs = build_pairs(task, max_videos=args.max_videos)
        logging.info("run=%s pairs=%d", _run_name(task), len(pairs))
        names, gt_feats, recon_feats = _extract_features(
            pairs,
            i3d=i3d,
            device=device,
            batch_size=max(1, args.batch_size),
        )
        by_group: Dict[str, Dict[str, Any]] = {}
        for group in sorted({_pair_group(name, group_map) for name in names}):
            indices = [i for i, name in enumerate(names) if _pair_group(name, group_map) == group]
            if len(indices) < 2:
                logging.warning("group=%s has fewer than 2 valid pairs; skipping rFVD", group)
                continue
            group_gt = gt_feats[indices]
            group_recon = recon_feats[indices]
            by_group[group] = {
                "rfvd": float(frechet_distance(group_recon.astype(np.float64), group_gt.astype(np.float64))),
                "rfvd_count": int(len(indices)),
                "pairs": int(len(indices)),
            }
        rfvd = float(frechet_distance(recon_feats.astype(np.float64), gt_feats.astype(np.float64)))
        result_path = _update_results(task, rfvd, int(gt_feats.shape[0]), by_group)
        row = _row_for(task, rfvd, int(gt_feats.shape[0]), len(names), result_path, by_group)
        write_json(task / ".rfvd_only" / "rfvd.json", row)
        rows.append(row)

    for root in roots:
        root_rows = [r for r in rows if r["root"] == root.name]
        if root_rows:
            _write_csv(root / "rfvd_summary.csv", root_rows)
    if roots:
        _write_csv(roots[0].parent / "rfvd_summary_all.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
