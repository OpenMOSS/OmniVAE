"""Multi-GPU T2I inference + FID evaluation on COCO val2014.

Pipeline (all run under ``accelerate launch`` so prompts shard across ranks):

  1. Build a deterministic ``num_samples``-long schedule from
     ``captions_val2014.json`` -- one prompt per image, image_id sorted,
     first caption per image. ``prompts.jsonl`` lives at the run root.
  2. Build (and cache) a single reference npz of real val images, center-
     cropped + resized to ``image_size``. Multi-rank shard preprocess +
     main-rank merge; the cache key is
     ``(annotations_json, num_samples, image_size)``.
  3. For each checkpoint: load a Z-Image pipeline (``ZImagePipeline``) via
     the existing ``omnivae_generation.trainer.eval.guided_diffusion.load_pipeline_for_checkpoint``,
     shard prompts across ranks, run ``pipeline(...)`` to produce uint8
     samples, merge per-rank shards into a single npz on the main rank,
     then call torch-fidelity for ``FID(real, samples)``.
  4. (Optional) ``--enable-recon-fid``: encode the reference real images
     through the ckpt's VAE, decode, and compute ``FID(real, recon)``. This
     gives an upper bound for what the DiT sweep above could ever reach,
     since recon FID is purely the VAE's responsibility.
  5. Aggregate per-ckpt ``metrics.json`` into ``summary.json`` /
     ``summary.csv`` at the run root, sorted by training step.

Outputs
-------
``<output_dir>/`` layout::

    run_manifest.json                        # CLI args + ckpt list + signatures
    prompts.jsonl                            # CocoEntry rows
    reference/
      reference_manifest.json
      coco_val2014_<N>x<S>x<S>x3.npz
    checkpoint-XXXXXXXX/
      sample_manifest.json
      samples_<N>x<S>x<S>x3.npz
      recon_<N>x<S>x<S>x3.npz                # only with --enable-recon-fid
      metrics.json
    summary.json
    summary.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Resolve repo root before importing trainer.* so that running this script
# from anywhere still imports the in-repo trainer package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omnivae_generation.trainer.runtime_env import ensure_hf_home  # noqa: E402

ensure_hf_home()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from omnivae_generation.trainer.config import load_config  # noqa: E402
from omnivae_generation.trainer.eval.guided_diffusion import (  # noqa: E402
    convert_images_to_uint8,
    extract_checkpoint_step,
    load_pipeline_for_checkpoint,
    make_torch_generator,
    merge_sample_shards_to_npz,
    save_local_sample_shard,
    shard_sample_indices,
)
from omnivae_generation.trainer.modeling import (  # noqa: E402
    decode_latents_to_images,
    encode_images_to_latents,
    resolve_dtype,
)
from omnivae_generation.trainer.runtime_patches import (  # noqa: E402
    patch_diffusers_zimage_forward_block_stacks,
    patch_diffusers_zimage_real_rope,
    patch_transformers_qwen3_5_disable_fast_path,
)
from omnivae_generation.trainer.utils import ensure_dir, is_checkpoint_complete, save_json  # noqa: E402

from infer.image.coco import (  # noqa: E402
    CocoEntry,
    load_coco_schedule,
    preprocess_reference_shard,
    write_coco_schedule_jsonl,
)
from infer.image.fid import (  # noqa: E402
    DEFAULT_FID_BATCH_SIZE,
    compute_fid,
    ensure_torch_fidelity_available,
)


DEFAULT_ANNOTATIONS = (
    "data/image/annotations/captions_val2014.json"
)
DEFAULT_IMAGES_DIR = (
    "data/image/val2014"
)


# ---------------------------------------------------------------------------
# Checkpoint discovery (lightweight version of the audio sweep's resolver)
# ---------------------------------------------------------------------------


@dataclass
class CkptEntry:
    path: Path
    name: str
    step: int


def _parse_ckpt_arg(values: list[str]) -> list[Path]:
    """Expand each ``--checkpoint`` value with optional glob support.

    Glob metacharacters (``* ? [``) are passed through ``glob.glob``; matches
    are filtered down to **directories whose basename starts with
    'checkpoint'**, mirroring ``infer.audio.run_eval._parse_ckpt_arg``.
    """
    out: list[Path] = []
    for value in values:
        for chunk in re.split(r"[,\s]+", value.strip()):
            if not chunk:
                continue
            chunk_expanded = str(Path(chunk).expanduser())
            if any(meta in chunk_expanded for meta in ("*", "?", "[")):
                hits = sorted(glob.glob(chunk_expanded))
                if not hits:
                    raise SystemExit(
                        f"--checkpoint glob {chunk!r} did not match anything"
                    )
                kept: list[Path] = []
                for hit in hits:
                    p = Path(hit)
                    if p.is_dir() and p.name.startswith("checkpoint"):
                        kept.append(p)
                if not kept:
                    raise SystemExit(
                        f"--checkpoint glob {chunk!r} expanded to {len(hits)} "
                        f"path(s) but none of them is a directory whose name "
                        f"starts with 'checkpoint'"
                    )
                out.extend(kept)
            else:
                out.append(Path(chunk_expanded))
    return out


def _resolve_ckpts(args: argparse.Namespace) -> list[CkptEntry]:
    raw_paths: list[Path] = []
    if args.checkpoint:
        raw_paths.extend(_parse_ckpt_arg(args.checkpoint))
    if args.checkpoint_root:
        for root_str in args.checkpoint_root:
            root = Path(root_str).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"--checkpoint-root does not exist: {root}")
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                if not (entry / "transformer" / "config.json").is_file():
                    continue
                if entry.name.startswith("checkpoint-") and not is_checkpoint_complete(entry):
                    continue
                raw_paths.append(entry)

    if not raw_paths:
        raise SystemExit(
            "No checkpoints provided. Pass --checkpoint <path> (repeatable) "
            "and/or --checkpoint-root <dir> pointing at a checkpoints/ folder."
        )

    seen: set[str] = set()
    resolved: list[CkptEntry] = []
    for path in raw_paths:
        path = path.resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not (path / "transformer" / "config.json").is_file():
            print(
                f"warning: skipping {path} (missing transformer/config.json)",
                file=sys.stderr,
            )
            continue
        try:
            step = int(extract_checkpoint_step(path))
        except (ValueError, TypeError, FileNotFoundError):
            match = re.search(r"(\d+)$", path.name)
            step = int(match.group(1)) if match else 0
        resolved.append(CkptEntry(path=path, name=path.name, step=step))

    if not resolved:
        raise SystemExit(
            "No valid checkpoints found after resolution; check that each path "
            "contains a transformer/config.json."
        )

    if args.sort_checkpoints == "step":
        resolved.sort(key=lambda entry: (entry.step, entry.name))
    elif args.sort_checkpoints == "step-desc":
        resolved.sort(key=lambda entry: (entry.step, entry.name), reverse=True)
    elif args.sort_checkpoints == "name":
        resolved.sort(key=lambda entry: entry.name)
    elif args.sort_checkpoints == "name-desc":
        resolved.sort(key=lambda entry: entry.name, reverse=True)
    return resolved


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate COCO val2014 T2I samples + compute FID (and optional "
            "reconstruction FID) for one or more Z-Image checkpoints."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training YAML used to build text encoder / VAE / scheduler / image_size.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Checkpoint dir (containing transformer/, scheduler/, tokenizer/, "
            "metadata.json). Repeatable; comma-separated also OK; glob "
            "metacharacters (* ? [) are expanded by Python (quote them so "
            "the shell does not pre-expand)."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=[],
        help="Root dir under which to auto-discover checkpoint-XXXXXXXX/ folders.",
    )
    parser.add_argument(
        "--sort-checkpoints",
        choices=["step", "step-desc", "name", "name-desc", "given"],
        default="step-desc",
        help="Order to iterate checkpoints (default: step-desc).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="How many (image, caption) pairs to use for FID (default: 5000).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override train.validation_num_inference_steps.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="Guidance scale (default: train.validation_guidance_scale).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for sample generation (default: train.seed).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-rank batch size for sampling and VAE recon (default: 8).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Resolution H=W for both real and generated images (default: dataset.image_size).",
    )
    parser.add_argument(
        "--annotations-json",
        type=str,
        default=DEFAULT_ANNOTATIONS,
        help="Path to COCO captions_val2014.json.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help="Directory holding the COCO val2014 jpg files referenced by the JSON.",
    )
    parser.add_argument(
        "--vae-type",
        "--vae_type",
        dest="vae_type",
        type=str,
        default=None,
        help=(
            "Override yaml ``vae.type`` (e.g. ``omnivae`` when the ckpt was "
            "trained against a OmniVAE training bundle but the yaml still "
            "lists ``wan2_2_native_vae``). Default: keep yaml value. "
            "Both ``--vae-type`` and ``--vae_type`` are accepted to match "
            "the trainer's CLI hint in the wan2_2_native VAE error message."
        ),
    )
    parser.add_argument(
        "--vae-path",
        "--vae_path",
        dest="vae_path",
        type=str,
        default=None,
        help=(
            "Override the VAE checkpoint path used to load the decoder. "
            "Has highest priority: beats ``<ckpt>/vae/`` and "
            "``metadata.json.vae_model_name_or_path``. Useful when sweeping "
            "ckpts trained against a VAE whose original location moved or "
            "when ablating a different VAE. Default: keep the metadata-driven "
            "fallback. Both ``--vae-path`` and ``--vae_path`` are accepted."
        ),
    )
    parser.add_argument(
        "--enable-recon-fid",
        dest="enable_recon_fid",
        action="store_true",
        default=False,
        help="Also compute reconstruction FID (encode->decode of real images via ckpt's VAE).",
    )
    parser.add_argument(
        "--reference-workers",
        type=int,
        default=8,
        help=(
            "Threads per rank for COCO reference preprocessing "
            "(JPEG decode + center-crop + bicubic resize). Both libjpeg "
            "and PIL release the GIL, so 4-8 threads give a 2-4x speedup "
            "on HDD/NFS-backed images dirs. Set 0 or 1 to fall back to the "
            "original single-thread loop. Default: 8."
        ),
    )
    parser.add_argument(
        "--no-recon-fid",
        dest="enable_recon_fid",
        action="store_false",
        help="Disable reconstruction FID (the default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Where to write outputs. Default: <experiment.output_dir>/infer/image."
        ),
    )
    parser.add_argument(
        "--no-reuse-existing",
        dest="reuse_existing",
        action="store_false",
        default=True,
        help=(
            "Disable cache-hit reuse of reference / sample npz files. By default "
            "we skip regeneration when an existing manifest's signature matches."
        ),
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=DEFAULT_FID_BATCH_SIZE,
        help="Batch size for the Inception feature extractor (default: 64).",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep per-rank shard intermediates after merging (default: delete).",
    )
    parser.add_argument(
        "--save-image-previews",
        type=int,
        default=0,
        metavar="K",
        help=(
            "If >0, dump K preview PNGs per ckpt for visual inspection (in "
            "addition to the npz used for FID). Indices are evenly spaced "
            "across [0, num_samples) so the previews cover the full prompt "
            "range. Reference previews go to <output>/reference/previews/, "
            "sample previews to <output>/<ckpt>/previews/, and (with "
            "--enable-recon-fid) recon previews to <output>/<ckpt>/previews_recon/. "
            "Default: 0 (no PNGs are written)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Runtime patches (mirror infer/audio/run_eval.py)
# ---------------------------------------------------------------------------


def _apply_runtime_patches(config: dict) -> None:
    patch_diffusers_zimage_real_rope()
    patch_diffusers_zimage_forward_block_stacks()
    if config.get("text_encoder", {}).get("disable_qwen3_5_fast_path", False):
        patch_transformers_qwen3_5_disable_fast_path()


# ---------------------------------------------------------------------------
# Reference npz: build + cache
# ---------------------------------------------------------------------------


def _reference_signature(*, annotations_json: Path, num_samples: int, image_size: int, seed: int) -> dict[str, Any]:
    """Cache key for a reference npz; mirrors the layout used by
    :func:`omnivae_generation.trainer.eval.guided_diffusion.build_sampling_signature`.
    """
    return {
        "annotations_json": str(annotations_json),
        "num_samples": int(num_samples),
        "image_size": int(image_size),
        "seed": int(seed),
        "preprocessing": "center_crop+resize_bicubic",
    }


def _reference_npz_path(reference_dir: Path, *, num_samples: int, image_size: int) -> Path:
    return reference_dir / f"coco_val2014_{int(num_samples)}x{int(image_size)}x{int(image_size)}x3.npz"


def _existing_reference_is_valid(
    *,
    reference_dir: Path,
    expected_signature: dict[str, Any],
    expected_npz: Path,
) -> bool:
    manifest_path = reference_dir / "reference_manifest.json"
    if not manifest_path.is_file() or not expected_npz.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(manifest.get(key) == value for key, value in expected_signature.items())


def ensure_reference_npz(
    *,
    schedule: list[CocoEntry],
    annotations_json: Path,
    images_dir: Path,
    image_size: int,
    output_dir: Path,
    seed: int,
    accelerator: Accelerator,
    reuse_existing: bool,
    keep_shards: bool,
    num_workers: int = 0,
) -> Path:
    """Multi-rank shard preprocess + main-rank merge for the COCO real-image
    reference batch. Returns the resolved path to the merged npz.
    """
    reference_dir = ensure_dir(output_dir / "reference")
    expected_signature = _reference_signature(
        annotations_json=annotations_json,
        num_samples=len(schedule),
        image_size=image_size,
        seed=seed,
    )
    expected_npz = _reference_npz_path(
        reference_dir, num_samples=len(schedule), image_size=image_size,
    )

    if reuse_existing and _existing_reference_is_valid(
        reference_dir=reference_dir,
        expected_signature=expected_signature,
        expected_npz=expected_npz,
    ):
        if accelerator.is_main_process:
            print(
                f"[reference] reusing cached real-image npz: {expected_npz}",
                flush=True,
            )
        accelerator.wait_for_everyone()
        return expected_npz

    shards_dir = ensure_dir(reference_dir / "shards")
    local_indices = shard_sample_indices(
        len(schedule), accelerator.process_index, accelerator.num_processes,
    )
    if accelerator.is_main_process:
        # PIL open + center-crop + bicubic resize is GIL-released so we
        # default to threaded preprocessing (--reference-workers=8). Print
        # an explicit start banner regardless so the run never looks hung.
        per_rank = (len(schedule) + accelerator.num_processes - 1) // accelerator.num_processes
        worker_msg = (
            f"with {int(num_workers)} threads per rank"
            if int(num_workers) > 1
            else "single-threaded"
        )
        print(
            f"[reference] preprocessing {len(schedule)} real images on "
            f"{accelerator.num_processes} ranks "
            f"(~{per_rank} per rank, {worker_msg}); "
            f"PIL center-crop + bicubic resize.",
            flush=True,
        )
    progress_desc = (
        f"reference[rank={accelerator.process_index}]"
        if accelerator.is_local_main_process
        else None
    )
    images, indices = preprocess_reference_shard(
        schedule=schedule,
        images_dir=images_dir,
        sample_indices=local_indices,
        image_size=image_size,
        progress_desc=progress_desc,
        num_workers=int(num_workers),
    )
    save_local_sample_shard(
        shards_dir / f"rank-{accelerator.process_index:05d}",
        sample_indices=indices,
        class_indices=np.zeros((indices.size,), dtype=np.int32),
        images=images,
    )
    if accelerator.is_main_process:
        print(
            "[reference] rank=0 done preprocessing + shard write; "
            "waiting for sibling ranks...",
            flush=True,
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        approx_gb = (len(schedule) * int(image_size) * int(image_size) * 3) / (1024 ** 3)
        print(
            f"[reference] all ranks done; main rank merging "
            f"{accelerator.num_processes} shards into a single npz "
            f"(~{approx_gb:.1f} GB write to disk; this is a single "
            f"``np.savez`` call so HDD ~1-3 min, SSD ~10-30 s; no progress "
            f"bar during the write)...",
            flush=True,
        )
        merge_sample_shards_to_npz(
            shards_dir,
            expected_npz,
            num_samples=len(schedule),
            image_height=image_size,
            image_width=image_size,
            keep_intermediates=bool(keep_shards),
        )
        manifest = dict(expected_signature)
        manifest["reference_npz_path"] = str(expected_npz)
        manifest["images_dir"] = str(images_dir)
        save_json(reference_dir / "reference_manifest.json", manifest)
        print(f"[reference] wrote {expected_npz} ({len(schedule)} images)", flush=True)

    accelerator.wait_for_everyone()
    return expected_npz


# ---------------------------------------------------------------------------
# Per-ckpt sampling
# ---------------------------------------------------------------------------


def _sample_signature(
    *,
    ckpt: CkptEntry,
    num_samples: int,
    image_size: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    annotations_json: Path,
    vae_type_override: str | None = None,
    vae_path_override: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_dir": str(ckpt.path),
        "checkpoint_step": int(ckpt.step),
        "num_samples": int(num_samples),
        "image_size": int(image_size),
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "annotations_json": str(annotations_json),
        "prompt_signature": "coco_val2014_first_caption_per_image",
        "vae_type_override": (str(vae_type_override) if vae_type_override else None),
        "vae_path_override": (str(vae_path_override) if vae_path_override else None),
    }


def _sample_npz_path(ckpt_out: Path, *, num_samples: int, image_size: int) -> Path:
    return ckpt_out / f"samples_{int(num_samples)}x{int(image_size)}x{int(image_size)}x3.npz"


def _resize_uint8_batch_if_needed(images: np.ndarray, image_size: int) -> np.ndarray:
    if images.shape[1:] == (int(image_size), int(image_size), 3):
        return images
    resized = np.empty((images.shape[0], int(image_size), int(image_size), 3), dtype=np.uint8)
    for idx in range(images.shape[0]):
        image = Image.fromarray(np.ascontiguousarray(images[idx]))
        resized[idx] = np.asarray(
            image.resize((int(image_size), int(image_size)), Image.BICUBIC),
            dtype=np.uint8,
        )
    return resized


def _existing_sample_is_valid(
    *,
    ckpt_out: Path,
    expected_signature: dict[str, Any],
    expected_npz: Path,
) -> bool:
    manifest_path = ckpt_out / "sample_manifest.json"
    if not manifest_path.is_file() or not expected_npz.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(manifest.get(key) == value for key, value in expected_signature.items())


def generate_samples_for_ckpt(
    *,
    ckpt: CkptEntry,
    schedule: list[CocoEntry],
    image_size: int,
    cfg: float,
    num_inference_steps: int,
    seed: int,
    batch_size: int,
    config: dict,
    accelerator: Accelerator,
    output_dir: Path,
    annotations_json: Path,
    reuse_existing: bool,
    keep_shards: bool,
    vae_type_override: str | None = None,
    vae_path_override: str | Path | None = None,
):
    """Generate ``len(schedule)`` images for a single checkpoint, returning
    ``(pipeline, sample_npz_path, was_reused)``.

    The pipeline is left loaded and returned so the optional recon-FID branch
    can reuse the same VAE without reloading. Caller is responsible for
    calling ``del pipeline`` and freeing GPU memory afterwards.
    """
    ckpt_out = ensure_dir(output_dir / ckpt.name)
    expected_signature = _sample_signature(
        ckpt=ckpt,
        num_samples=len(schedule),
        image_size=image_size,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg,
        annotations_json=annotations_json,
        vae_type_override=vae_type_override,
        vae_path_override=vae_path_override,
    )
    sample_npz = _sample_npz_path(ckpt_out, num_samples=len(schedule), image_size=image_size)

    pipeline = None
    was_reused = False
    if reuse_existing and _existing_sample_is_valid(
        ckpt_out=ckpt_out,
        expected_signature=expected_signature,
        expected_npz=sample_npz,
    ):
        was_reused = True
        if accelerator.is_main_process:
            print(
                f"[ckpt={ckpt.name}] reusing cached samples: {sample_npz}",
                flush=True,
            )

    if not was_reused:
        pipeline = load_pipeline_for_checkpoint(
            ckpt.path,
            config,
            accelerator.device,
            vae_type_override=vae_type_override,
            vae_path_override=vae_path_override,
        )
        local_indices = shard_sample_indices(
            len(schedule), accelerator.process_index, accelerator.num_processes,
        )
        local_entries = [schedule[int(i)] for i in local_indices.tolist()]
        shards_dir = ensure_dir(ckpt_out / "shards")

        max_seq_len = int(config["text_encoder"]["max_sequence_length"])
        local_images: list[np.ndarray] = []
        batch_starts = range(0, len(local_entries), int(batch_size))
        if accelerator.is_local_main_process:
            batch_starts = tqdm(
                batch_starts,
                total=(len(local_entries) + int(batch_size) - 1) // int(batch_size),
                desc=f"sample[{ckpt.name}]",
                leave=False,
                dynamic_ncols=True,
            )

        with torch.no_grad():
            for batch_start in batch_starts:
                batch_entries = local_entries[batch_start : batch_start + int(batch_size)]
                prompts = [entry.caption for entry in batch_entries]
                generators = [
                    make_torch_generator(accelerator.device, entry.sample_seed)
                    for entry in batch_entries
                ]
                result = pipeline(
                    prompt=prompts,
                    height=int(image_size),
                    width=int(image_size),
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(cfg),
                    generator=generators,
                    max_sequence_length=max_seq_len,
                    output_type="np",
                )
                local_images.append(
                    _resize_uint8_batch_if_needed(
                        convert_images_to_uint8(result.images),
                        int(image_size),
                    )
                )

        local_images_arr = (
            np.concatenate(local_images, axis=0)
            if local_images
            else np.zeros((0, image_size, image_size, 3), dtype=np.uint8)
        )
        save_local_sample_shard(
            shards_dir / f"rank-{accelerator.process_index:05d}",
            sample_indices=local_indices,
            class_indices=np.zeros((local_indices.size,), dtype=np.int32),
            images=local_images_arr,
        )

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            merge_sample_shards_to_npz(
                shards_dir,
                sample_npz,
                num_samples=len(schedule),
                image_height=image_size,
                image_width=image_size,
                keep_intermediates=bool(keep_shards),
            )
            manifest = dict(expected_signature)
            manifest["sample_npz_path"] = str(sample_npz)
            save_json(ckpt_out / "sample_manifest.json", manifest)
            print(f"[ckpt={ckpt.name}] wrote {sample_npz}", flush=True)

        accelerator.wait_for_everyone()

    return pipeline, sample_npz, was_reused


# ---------------------------------------------------------------------------
# Reconstruction npz (main rank only)
# ---------------------------------------------------------------------------


def compute_recon_npz(
    *,
    pipeline,
    real_npz: Path,
    image_size: int,
    batch_size: int,
    output_path: Path,
    config: dict,
    device: torch.device,
) -> Path:
    """Run the ckpt's VAE encode->decode on the real npz batch on a single
    rank (main rank). Saves a sibling ``recon_*.npz`` whose images are
    aligned 1:1 with the real npz so ``FID(real, recon)`` is meaningful.
    """
    if pipeline is None:
        raise RuntimeError(
            "compute_recon_npz needs a live pipeline (cached-sample reuse "
            "should have triggered re-loading the pipeline)."
        )
    vae = pipeline.vae
    vae_dtype = resolve_dtype(config.get("vae", {}).get("torch_dtype"), fallback=torch.float32)

    real_handle = np.load(real_npz, mmap_mode="r")
    real_images = real_handle["arr_0"]
    if real_images.dtype != np.uint8 or real_images.ndim != 4:
        raise TypeError(
            f"Expected real npz uint8 [N,H,W,3]; got dtype={real_images.dtype} shape={real_images.shape}"
        )
    n = int(real_images.shape[0])

    recon_buffer = np.empty((n, int(image_size), int(image_size), 3), dtype=np.uint8)

    iterator = range(0, n, int(batch_size))
    iterator = tqdm(
        iterator,
        total=(n + int(batch_size) - 1) // int(batch_size),
        desc="recon",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for start in iterator:
            end = min(start + int(batch_size), n)
            batch_uint8 = np.ascontiguousarray(real_images[start:end])  # [B,H,W,3] uint8
            batch_tensor = torch.from_numpy(batch_uint8).to(device)
            # HWC uint8 -> CHW float in [-1, 1]
            batch_tensor = batch_tensor.permute(0, 3, 1, 2).contiguous()
            pixel_values = batch_tensor.to(dtype=vae_dtype).mul_(1.0 / 127.5).sub_(1.0)

            latents = encode_images_to_latents(pixel_values, vae)
            latents = latents.to(dtype=getattr(vae, "dtype", latents.dtype))
            recon = decode_latents_to_images(latents, vae)
            recon_uint8 = (
                ((recon.detach().float().clamp(-1.0, 1.0) + 1.0) * (255.0 / 2.0))
                .round()
                .clamp(0.0, 255.0)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            # Recon may differ from input H/W if the VAE up/downsamples
            # asymmetrically (rare for square inputs); enforce shape.
            if recon_uint8.shape[1:] != (int(image_size), int(image_size), 3):
                resized = np.empty(
                    (recon_uint8.shape[0], int(image_size), int(image_size), 3),
                    dtype=np.uint8,
                )
                for i in range(recon_uint8.shape[0]):
                    img = Image.fromarray(recon_uint8[i]).resize(
                        (int(image_size), int(image_size)), Image.BICUBIC
                    )
                    resized[i] = np.asarray(img, dtype=np.uint8)
                recon_uint8 = resized
            recon_buffer[start:end] = recon_uint8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        recon_buffer,
        np.zeros((n,), dtype=np.int32),
    )
    return output_path


# ---------------------------------------------------------------------------
# Optional preview PNG export (not used for FID; only for human inspection)
# ---------------------------------------------------------------------------


def select_preview_indices(total: int, k: int) -> list[int]:
    """Pick ``k`` evenly-spaced indices from ``[0, total)``.

    Even spacing makes the previews cover the full schedule (image_id range)
    rather than collapsing to the alphabetic head -- handy when ``--num-samples``
    is much larger than ``k``.
    """
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(int(total)))
    return [int(round(i * (total - 1) / (k - 1))) for i in range(int(k))]


def save_preview_pngs_from_npz(
    *,
    npz_path: Path,
    indices: list[int],
    out_dir: Path,
    prefix: str,
    progress_desc: str | None = None,
) -> list[Path]:
    """Load ``npz_path["arr_0"]`` (uint8 [N,H,W,3]) and write the rows at
    ``indices`` as ``<out_dir>/<prefix>-XXXXXX.png`` files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    handle = np.load(npz_path)
    images = handle["arr_0"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise TypeError(
            f"Expected uint8 [N,H,W,3] images in {npz_path}; "
            f"got dtype={images.dtype} shape={images.shape}"
        )
    iterator = indices
    if progress_desc is not None:
        iterator = tqdm(
            iterator,
            total=len(indices),
            desc=progress_desc,
            leave=False,
            dynamic_ncols=True,
        )
    written: list[Path] = []
    for sample_index in iterator:
        out_path = out_dir / f"{prefix}-{int(sample_index):06d}.png"
        Image.fromarray(np.ascontiguousarray(images[int(sample_index)])).save(out_path)
        written.append(out_path)
    return written


def write_preview_captions_jsonl(
    *,
    schedule: list[CocoEntry],
    indices: list[int],
    out_path: Path,
) -> Path:
    """Write a sidecar ``captions.jsonl`` listing ``(sample_index, image_id,
    file_name, caption)`` for the previewed indices, so human reviewers can
    pair each PNG with its prompt without crawling the global ``prompts.jsonl``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample_index in indices:
            entry = schedule[int(sample_index)]
            handle.write(
                json.dumps(
                    {
                        "sample_index": int(entry.sample_index),
                        "image_id": int(entry.image_id),
                        "file_name": entry.file_name,
                        "caption": entry.caption,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def _summary_csv_fieldnames(rows: list[dict]) -> list[str]:
    base = [
        "checkpoint_step",
        "checkpoint_name",
        "fid",
        "recon_fid",
        "num_samples",
        "image_size",
        "num_inference_steps",
        "guidance_scale",
    ]
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key in base or key in extras:
                continue
            extras.append(key)
    return base + extras


def write_summary(rows: list[dict], output_dir: Path) -> dict[str, Path]:
    rows_sorted = sorted(rows, key=lambda r: int(r.get("checkpoint_step") or -1))
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    save_json(json_path, {"checkpoints": rows_sorted})
    fieldnames = _summary_csv_fieldnames(rows_sorted)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)
    return {"summary_json": json_path, "summary_csv": csv_path}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    _apply_runtime_patches(config)
    ensure_torch_fidelity_available()

    accelerator = Accelerator()
    device = accelerator.device

    image_size = int(args.image_size or config["dataset"]["image_size"])
    cfg = float(
        args.cfg if args.cfg is not None else config["train"]["validation_guidance_scale"]
    )
    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else config["train"]["validation_num_inference_steps"]
    )
    base_seed = int(
        args.seed if args.seed is not None else config["train"].get("seed", 42)
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(config["experiment"]["output_dir"]) / "infer" / "image"
    )
    output_dir = ensure_dir(output_dir)

    annotations_json = Path(args.annotations_json).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()

    requested_num_samples = int(args.num_samples)
    if not accelerator.is_main_process:
        # Same UserWarning fires on every rank; only let main rank print it
        # so the launcher logs aren't spammed with N copies.
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"infer\.image\.coco"
        )
    schedule = load_coco_schedule(
        annotations_json,
        num_samples=requested_num_samples,
        seed=base_seed,
    )
    capped_num_samples = (
        len(schedule) < requested_num_samples
    )
    if capped_num_samples and accelerator.is_main_process:
        print(
            "[warn] requested --num-samples="
            f"{requested_num_samples} exceeds COCO val2014 unique "
            f"image+caption pairs ({len(schedule)}); "
            f"falling back to the maximum {len(schedule)}.",
            flush=True,
        )

    ckpts = _resolve_ckpts(args)

    if accelerator.is_main_process:
        write_coco_schedule_jsonl(schedule, output_dir / "prompts.jsonl")
        manifest = {
            "config": str(Path(args.config).resolve()),
            "annotations_json": str(annotations_json),
            "images_dir": str(images_dir),
            "output_dir": str(output_dir),
            "image_size": int(image_size),
            "num_samples": int(len(schedule)),
            "num_samples_requested": requested_num_samples,
            "num_samples_capped": bool(capped_num_samples),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(cfg),
            "seed": int(base_seed),
            "batch_size": int(args.batch_size),
            "fid_batch_size": int(args.fid_batch_size),
            "enable_recon_fid": bool(args.enable_recon_fid),
            "save_image_previews": int(args.save_image_previews),
            "reference_workers": int(args.reference_workers),
            "vae_type_override": args.vae_type,
            "vae_path_override": args.vae_path,
            "world_size": int(accelerator.num_processes),
            "checkpoints": [
                {"name": c.name, "step": c.step, "path": str(c.path)} for c in ckpts
            ],
        }
        save_json(output_dir / "run_manifest.json", manifest)
        print("Sweep:")
        if capped_num_samples:
            print(
                f"  num_samples: {len(schedule)} (capped from "
                f"requested {requested_num_samples})"
            )
        else:
            print(f"  num_samples: {len(schedule)}")
        print(f"  image_size: {image_size}")
        print(f"  num_inference_steps: {num_inference_steps}")
        print(f"  guidance_scale: {cfg}")
        print(f"  base_seed: {base_seed}")
        print(f"  enable_recon_fid: {bool(args.enable_recon_fid)}")
        print(f"  save_image_previews: {int(args.save_image_previews)}")
        if args.vae_type:
            print(f"  vae_type_override: {args.vae_type}")
        if args.vae_path:
            print(f"  vae_path_override: {args.vae_path}")
        print(f"  output_dir: {output_dir}")
        print(f"  checkpoints ({len(ckpts)}):")
        for c in ckpts:
            print(f"    - {c.name} (step={c.step})  <- {c.path}")

    accelerator.wait_for_everyone()

    real_npz = ensure_reference_npz(
        schedule=schedule,
        annotations_json=annotations_json,
        images_dir=images_dir,
        image_size=int(image_size),
        output_dir=output_dir,
        seed=base_seed,
        accelerator=accelerator,
        reuse_existing=bool(args.reuse_existing),
        keep_shards=bool(args.keep_shards),
        num_workers=int(args.reference_workers),
    )

    preview_indices = select_preview_indices(
        len(schedule), int(args.save_image_previews),
    )
    if preview_indices and accelerator.is_main_process:
        # Reference previews are written once globally; sample / recon
        # previews below all reuse `preview_indices` so a real PNG and the
        # corresponding sample / recon PNG share the same image_id and can
        # be paired up trivially by sample_index.
        reference_preview_dir = output_dir / "reference" / "previews"
        save_preview_pngs_from_npz(
            npz_path=real_npz,
            indices=preview_indices,
            out_dir=reference_preview_dir,
            prefix="real",
            progress_desc="reference-previews",
        )
        write_preview_captions_jsonl(
            schedule=schedule,
            indices=preview_indices,
            out_path=reference_preview_dir / "captions.jsonl",
        )
        print(
            f"[reference] wrote {len(preview_indices)} preview PNGs to "
            f"{reference_preview_dir}",
            flush=True,
        )

    summary_rows: list[dict] = []
    for ckpt in ckpts:
        if accelerator.is_main_process:
            print(f"== ckpt={ckpt.name} (step={ckpt.step}) ==", flush=True)

        pipeline = None
        try:
            pipeline, sample_npz, samples_were_reused = generate_samples_for_ckpt(
                ckpt=ckpt,
                schedule=schedule,
                image_size=int(image_size),
                cfg=cfg,
                num_inference_steps=num_inference_steps,
                seed=base_seed,
                batch_size=int(args.batch_size),
                config=config,
                accelerator=accelerator,
                output_dir=output_dir,
                annotations_json=annotations_json,
                reuse_existing=bool(args.reuse_existing),
                keep_shards=bool(args.keep_shards),
                vae_type_override=args.vae_type,
                vae_path_override=args.vae_path,
            )

            metrics_path = output_dir / ckpt.name / "metrics.json"
            metrics_payload: dict[str, Any] = {
                "checkpoint_name": ckpt.name,
                "checkpoint_step": int(ckpt.step),
                "num_samples": int(len(schedule)),
                "image_size": int(image_size),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(cfg),
                "seed": int(base_seed),
                "real_npz": str(real_npz),
                "sample_npz": str(sample_npz),
            }

            recon_npz_path: Path | None = None
            if accelerator.is_main_process:
                fid_metrics = compute_fid(
                    real_npz=real_npz,
                    fake_npz=sample_npz,
                    device=device,
                    batch_size=int(args.fid_batch_size),
                )
                metrics_payload["fid"] = float(fid_metrics.get("fid", float("nan")))
                metrics_payload["fid_raw"] = fid_metrics
                print(
                    f"[ckpt={ckpt.name}] generation FID = {metrics_payload['fid']:.4f}",
                    flush=True,
                )

                if preview_indices:
                    sample_preview_dir = output_dir / ckpt.name / "previews"
                    save_preview_pngs_from_npz(
                        npz_path=sample_npz,
                        indices=preview_indices,
                        out_dir=sample_preview_dir,
                        prefix="sample",
                        progress_desc=f"previews[{ckpt.name}]",
                    )
                    write_preview_captions_jsonl(
                        schedule=schedule,
                        indices=preview_indices,
                        out_path=sample_preview_dir / "captions.jsonl",
                    )
                    metrics_payload["sample_preview_dir"] = str(sample_preview_dir)

                if bool(args.enable_recon_fid):
                    if pipeline is None:
                        # Cached samples but we still need the VAE; load it.
                        pipeline = load_pipeline_for_checkpoint(
                            ckpt.path,
                            config,
                            device,
                            vae_type_override=args.vae_type,
                            vae_path_override=args.vae_path,
                        )
                    recon_npz_path = compute_recon_npz(
                        pipeline=pipeline,
                        real_npz=real_npz,
                        image_size=int(image_size),
                        batch_size=int(args.batch_size),
                        output_path=output_dir
                        / ckpt.name
                        / f"recon_{len(schedule)}x{image_size}x{image_size}x3.npz",
                        config=config,
                        device=device,
                    )
                    recon_metrics = compute_fid(
                        real_npz=real_npz,
                        fake_npz=recon_npz_path,
                        device=device,
                        batch_size=int(args.fid_batch_size),
                    )
                    metrics_payload["recon_fid"] = float(
                        recon_metrics.get("fid", float("nan"))
                    )
                    metrics_payload["recon_fid_raw"] = recon_metrics
                    metrics_payload["recon_npz"] = str(recon_npz_path)
                    print(
                        f"[ckpt={ckpt.name}] reconstruction FID = "
                        f"{metrics_payload['recon_fid']:.4f}",
                        flush=True,
                    )

                    if preview_indices:
                        recon_preview_dir = output_dir / ckpt.name / "previews_recon"
                        save_preview_pngs_from_npz(
                            npz_path=recon_npz_path,
                            indices=preview_indices,
                            out_dir=recon_preview_dir,
                            prefix="recon",
                            progress_desc=f"recon-previews[{ckpt.name}]",
                        )
                        write_preview_captions_jsonl(
                            schedule=schedule,
                            indices=preview_indices,
                            out_path=recon_preview_dir / "captions.jsonl",
                        )
                        metrics_payload["recon_preview_dir"] = str(recon_preview_dir)

                save_json(metrics_path, metrics_payload)
                summary_rows.append(metrics_payload)
            else:
                # Non-main ranks still need to participate in the next
                # ckpt's barrier; pipeline is already on GPU though, so
                # release it before that.
                pass
        finally:
            if pipeline is not None:
                del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

    if accelerator.is_main_process and summary_rows:
        paths = write_summary(summary_rows, output_dir)
        print(f"Wrote summary to {paths['summary_json']}", flush=True)
        print(f"Wrote summary CSV to {paths['summary_csv']}", flush=True)

    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
