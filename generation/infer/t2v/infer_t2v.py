"""Parallel T2V inference for OmniVAE Z-Image checkpoints.

This script wraps the single-process sampling loop in
``scripts/eval/export_video_checkpoint_samples.py`` (which
implements the *exact* validation-time z-image diffusion loop) with
torchrun-driven multi-rank sharding.

* **Single-node multi-GPU**::

      torchrun --nproc_per_node=8 eval/video/t2v/infer_t2v.py \
          --checkpoint-dir <ckpt> \
          --prompt-manifest data/t2v/valid/inference_manifest.jsonl \
          --output-dir <out>/step50000/steps50_cfg4.0 \
          --num-inference-steps 50 --guidance-scale 4.0

  ``--prompt-manifest`` accepts multiple JSONL paths (space-separated); rows
  are concatenated in the order given, every row is stamped with
  ``_source_manifest`` (the input file stem) and ``manifest_row_index`` (the
  global concatenated index), and unified ids are auto-namespaced by
  ``dataset`` / ``_source_manifest`` to avoid cross-file collisions.

* **Multi-node**: launched via ``eval/video/t2v/scripts/infer.sh`` which
  resolves the cluster's ``PET_*`` envs into ``torchrun --nnodes / --node_rank
  / --master_addr / --master_port`` flags. The Python entrypoint stays the
  same: it auto-detects ``RANK / WORLD_SIZE / LOCAL_RANK`` from the env.

* **Single GPU / no distributed**: works unchanged. ``WORLD_SIZE`` defaults to
  1 and no ``init_process_group`` is called.

Workflow:
    1. Read the unified manifest (rows already carry dataset / unified_id /
       same_source_id / source_real_video / category fields from
       ``build_unified_manifest.py``).
    2. Modulo-shard rows across ranks (rank ``r`` of ``W`` keeps every row
       whose 0-based index satisfies ``i % W == r``). This matches
       ``omnivae_generation.trainer.video_validation.run_video_validation``'s sharding scheme.
    3. Each rank loads the pipeline + transformer once, then iterates its
       shard, running ``generate_batch`` (imported verbatim from the existing
       export script) and writing per-row records to ``samples.rank{R}.jsonl``.
    4. After ``barrier()``, rank 0 merges all ``samples.rank*.jsonl`` files
       into ``samples.jsonl`` (sorted by ``unified_id``) and removes the
       per-rank shards. ``--resume`` reads the union of ``samples.jsonl`` and
       any leftover ``samples.rank*.jsonl`` to skip ``unified_id``\s already
       done -- so a previous run that crashed *before* the merge step still
       gets credit for whatever each rank had flushed to its shard.
       In resume mode the per-rank shard is opened in append (``"a"``) mode
       to preserve those partial rows; the historical mp4 files for skipped
       ids are not re-encoded.

Output schema for each line in ``samples.jsonl`` (also matches each
``samples.rank{R}.jsonl`` line):

    * Every field from the unified manifest row (dataset, unified_id,
      row_index, prompt, negative_prompt, video_caption, category, source,
      same_source_id?, source_real_video?, ...).
    * ``video_path``       absolute path of the generated mp4.
    * ``sample_index``     stable per-row int (the manifest row index after
                           sharding-aware enumeration); only used as a
                           secondary sort key after ``unified_id``.
    * ``seed``             ``base_seed + manifest_row_index``.
    * ``num_inference_steps`` / ``guidance_scale`` / ``cfg_normalization``
      / ``cfg_truncation``  resolved values (CLI > yaml).
    * ``height`` / ``width`` / ``num_frames`` / ``fps`` /
      ``requested_num_frames`` / ``decoded_num_frames`` / ``latent_frames``.
    * ``checkpoint_dir`` / ``checkpoint_step`` / ``run_dir``.

Because the unified manifest preserves all original fields, downstream
metric scripts can ``filter dataset=='vbench_sampled'`` and pull
``same_source_id`` directly from each row -- no extra bookkeeping needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omnivae_generation.trainer.eval.guided_diffusion import (  # noqa: E402
    extract_checkpoint_step,
    load_pipeline_for_checkpoint,
    load_run_config_for_eval,
    resolve_run_dir,
)
from omnivae_generation.trainer.utils import ensure_dir, save_json  # noqa: E402
from omnivae_generation.trainer.video_validation import (  # noqa: E402
    _video_tensor_to_uint8_frames,
    _video_validation_latent_shape,
)


def _load_core_module():
    """Load scripts/eval/export_video_checkpoint_samples.py as a
    module so we can reuse its ``generate_batch`` / ``build_request_config``
    verbatim (single source of truth for the validation diffusion loop)."""

    import importlib.util

    core_path = PROJECT_ROOT / "scripts" / "eval" / "export_video_checkpoint_samples.py"
    if not core_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {core_path}; expected the export script inside this repository."
        )
    spec = importlib.util.spec_from_file_location("_anytok_t2v_export_core", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _load_core_module()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive float, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--prompt-manifest",
        type=str,
        required=True,
        nargs="+",
        help="One or more JSONL manifests to infer over. When multiple are given "
        "they are concatenated in the listed order; the global concatenated row "
        "index is used for shard assignment / seed and stamped as "
        "'manifest_row_index'. Each row carries '_source_manifest' = manifest "
        "stem so downstream tooling can split per-source. Records without "
        "'dataset' fall back to the manifest stem when synthesizing unified_id "
        "so cross-file collisions are avoided by default.",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Primary manifest field for the positive prompt (default: prompt; "
        "build_unified_manifest already standardizes this).",
    )
    parser.add_argument(
        "--prompt-fallback-keys",
        type=str,
        default="video_caption,caption,text",
        help="Comma-separated fallback fields used when --prompt-key is missing "
        "or empty for a row. Default 'video_caption,caption,text' covers the "
        "raw {from_train,train2valid_gen,vbench_sampled,valid_when_train}.jsonl "
        "manifests (which store the prompt under 'video_caption'). Set to '' to "
        "disable the fallback chain.",
    )
    parser.add_argument("--negative-prompt-key", type=str, default="negative_prompt")
    parser.add_argument(
        "--negative-prompt-fallback-keys",
        type=str,
        default="neg_prompt,negative",
        help="Comma-separated fallback fields for negative prompt; default empty "
        "string when nothing matches.",
    )
    parser.add_argument(
        "--unified-id-key",
        type=str,
        default="unified_id",
        help="Manifest field used as the merge / dedup key.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--limit",
        "--max-examples",
        "--max_examples",
        dest="limit",
        type=positive_int,
        default=None,
        help="Cap the total number of manifest rows that are inferred. Aliases: --max-examples / --max_examples.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--num-frames", type=positive_int, default=None)
    parser.add_argument("--height", type=positive_int, default=None)
    parser.add_argument("--width", type=positive_int, default=None)
    parser.add_argument("--fps", type=positive_float, default=None)
    parser.add_argument("--num-inference-steps", type=positive_int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--cfg-normalization", action="store_true")
    parser.add_argument("--cfg-truncation", type=float, default=None)
    parser.add_argument("--quality", type=int, default=8, help="imageio/libx264 output quality.")
    parser.add_argument(
        "--vae-type",
        "--vae_type",
        dest="vae_type",
        type=str,
        default=None,
        help="Override run_config['vae']['type'] at load time. Use this when the "
        "metadata.json's vae_model_name_or_path points at a OmniVAE Trainer state "
        "dict (Trainer_NNNNN/state_dict.pt) -- you'll typically want '--vae-type "
        "omnivae' in that case (see omnivae_generation.trainer.vae.omnivae). Other valid values: "
        "autoencoder_kl, wan2_2_vae, wan2_2_native_vae, omnivae.",
    )
    parser.add_argument(
        "--vae-path",
        "--vae_path",
        dest="vae_path",
        type=str,
        default=None,
        help="Override the VAE path. Takes precedence over checkpoint_dir/vae/ "
        "and metadata.json. Combine with --vae-type when the auto-detected type "
        "doesn't match (e.g. OmniVAE Trainer state_dict.pt + --vae-type omnivae).",
    )
    parser.add_argument(
        "--vae-path-2",
        "--vae_path_2",
        dest="vae_path_2",
        type=str,
        default=None,
        help="Optional second VAE path. When provided, each batch is decoded with "
        "BOTH VAEs and the two resulting videos are concatenated (left=main VAE, "
        "right=second VAE by default; see --dual-vae-stack-axis). The diffusion "
        "loop only runs once; only the decoder is duplicated. REQUIRES the two "
        "VAEs to share the same latent layout (channels / spatial scale / "
        "temporal scale) -- otherwise the same latent tensor cannot drive both. "
        "Omit this flag to keep the original single-VAE behavior unchanged.",
    )
    parser.add_argument(
        "--vae-type-2",
        "--vae_type_2",
        dest="vae_type_2",
        type=str,
        default=None,
        help="VAE type for --vae-path-2; same semantics / valid values as "
        "--vae-type. Falls back to --vae-type (then run_config['vae']['type']) "
        "when omitted.",
    )
    parser.add_argument(
        "--dual-vae-stack-axis",
        dest="dual_vae_stack_axis",
        type=str,
        default="width",
        choices=("width", "height"),
        help="Axis along which to concatenate the two decoded videos in dual-VAE "
        "mode (only used when --vae-path-2 is set). 'width' = left/right (main "
        "left, second right), 'height' = top/bottom (main top, second bottom).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip unified_ids that already appear in either the merged "
        "samples.jsonl OR any per-rank samples.rank*.jsonl. Appends to the "
        "existing rank shard rather than overwriting it, so progress from a "
        "crashed previous run is preserved (mp4 files for already-done ids "
        "are NOT re-generated). Without --resume, every prompt is "
        "regenerated and the per-rank shard is wiped at start.",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="torch.distributed backend (default: nccl). Forced to gloo if cuda is unavailable.",
    )
    parser.add_argument(
        "--dist-timeout-minutes",
        type=int,
        default=120,
        help="Process group timeout (minutes); generous default to cover first-step compile.",
    )
    return parser.parse_args()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def setup_distributed(args: argparse.Namespace) -> tuple[int, int, int, bool]:
    """Honor torchrun env vars; init NCCL/Gloo only if WORLD_SIZE > 1.

    Returns (rank, world_size, local_rank, is_distributed).
    """

    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)

    if world_size <= 1:
        return rank, world_size, local_rank, False

    backend = args.dist_backend if torch.cuda.is_available() else "gloo"
    if not torch.distributed.is_initialized():
        from datetime import timedelta

        torch.distributed.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=int(args.dist_timeout_minutes)),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def resolve_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    records = core.load_json_records(path)
    return [record for record in records if isinstance(record, dict)]


def load_all_manifest_records(
    paths: list[Path],
    *,
    is_main: bool,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Read N manifests, concatenate, and stamp '_source_manifest' on every row.

    Returns the concatenated records and the resolved manifest paths. Order
    follows ``paths`` exactly; row indices in the concatenated list are what
    drives modulo sharding and seed assignment, so the order is the contract
    callers must preserve across reruns / resume.
    """

    resolved = [Path(p).expanduser().resolve() for p in paths]
    seen = set()
    deduped: list[Path] = []
    for path in resolved:
        if path in seen:
            if is_main:
                print(f"[manifest] WARN: ignoring duplicate manifest path {path}", file=sys.stderr)
            continue
        seen.add(path)
        deduped.append(path)

    merged: list[dict[str, Any]] = []
    per_file_counts: dict[str, int] = {}
    for path in deduped:
        records = load_manifest_records(path)
        stem = path.stem
        for record in records:
            record_copy = dict(record)
            record_copy.setdefault("_source_manifest", stem)
            merged.append(record_copy)
        per_file_counts[stem] = len(records)
        if is_main:
            print(f"[manifest] {path} -> {len(records)} rows (stem={stem!r})", file=sys.stderr)

    if is_main and len(deduped) > 1:
        print(
            f"[manifest] merged {sum(per_file_counts.values())} rows across "
            f"{len(deduped)} manifests",
            file=sys.stderr,
        )

    return merged, deduped


def select_records(records: list[dict[str, Any]], offset: int, limit: int | None) -> list[dict[str, Any]]:
    selected = records[max(0, int(offset)):]
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def existing_unified_ids(samples_path: Path, key: str) -> set[str]:
    if not samples_path.is_file():
        return set()
    out: set[str] = set()
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            value = payload.get(key)
            if isinstance(value, str) and value:
                out.add(value)
    return out


def synthesize_unified_id(record: dict[str, Any], fallback_index: int) -> str:
    dataset = record.get("dataset") or record.get("_source_manifest") or "unknown"
    return f"{dataset}_{int(fallback_index):06d}"


def _parse_key_list(raw: str) -> list[str]:
    return [token.strip() for token in (raw or "").split(",") if token.strip()]


def _resolve_text_field(
    record: dict[str, Any],
    *,
    primary_key: str,
    fallback_keys: list[str],
) -> tuple[str, str | None]:
    """Return (resolved_text, key_used_or_None).

    ``key_used`` is the dict key that produced a non-empty string after
    ``coerce_text``; ``None`` means every key was empty/missing.
    """

    for key in [primary_key, *fallback_keys]:
        if not key:
            continue
        value = record.get(key)
        if value is None:
            continue
        text = core.coerce_text(value)
        if text and text.strip():
            return text, key
    return "", None


def collect_done_unified_ids(
    *,
    output_dir: Path,
    samples_path: Path,
    rank_pattern: str,
    key: str,
) -> set[str]:
    """Union of unified_ids found in ``samples.jsonl`` AND all ``samples.rank*.jsonl``.

    Reading both means a previous run that crashed *before* the rank0 merge
    step (so ``samples.jsonl`` was never written) still gets its per-rank
    progress credited as 'done' on resume. Without this, the next ``--resume``
    would silently regenerate every video again because only the merged file
    is consulted.
    """

    out: set[str] = existing_unified_ids(samples_path, key)
    for shard_path in sorted(output_dir.glob(rank_pattern)):
        out |= existing_unified_ids(shard_path, key)
    return out


def gather_existing_ids_across_ranks(
    samples_path: Path,
    *,
    output_dir: Path,
    rank_pattern: str,
    key: str,
    rank: int,
    world_size: int,
    is_distributed: bool,
    device: torch.device,
) -> set[str]:
    """Rank 0 enumerates done unified_ids (samples.jsonl ∪ rank shards), then
    broadcasts the id set to everyone so all ranks agree on what to skip."""

    if not is_distributed:
        return collect_done_unified_ids(
            output_dir=output_dir,
            samples_path=samples_path,
            rank_pattern=rank_pattern,
            key=key,
        )

    payload: list[str] = (
        sorted(
            collect_done_unified_ids(
                output_dir=output_dir,
                samples_path=samples_path,
                rank_pattern=rank_pattern,
                key=key,
            )
        )
        if rank == 0
        else []
    )
    object_list: list[Any] = [payload]
    torch.distributed.broadcast_object_list(object_list, src=0)
    received = object_list[0] if isinstance(object_list[0], list) else []
    return set(str(item) for item in received)


def merge_rank_shards(
    *,
    output_dir: Path,
    samples_path: Path,
    rank_pattern: str,
    unified_id_key: str,
    keep_existing: bool,
) -> dict[str, int]:
    """Concatenate ``samples.rank*.jsonl`` into ``samples.jsonl``.

    Args:
        keep_existing: when True (resume mode) preserves rows already present
            in ``samples.jsonl``; otherwise overwrites it.

    Returns a small {n_kept_existing, n_added, n_total} dict for logging.
    """

    by_id: dict[str, dict[str, Any]] = {}
    n_kept_existing = 0
    if keep_existing and samples_path.is_file():
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                key = payload.get(unified_id_key)
                if isinstance(key, str) and key:
                    by_id[key] = payload
                    n_kept_existing += 1

    n_added = 0
    for shard_path in sorted(output_dir.glob(rank_pattern)):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                key = payload.get(unified_id_key)
                if not isinstance(key, str) or not key:
                    continue
                if key in by_id and keep_existing:
                    # Existing wins on resume to preserve historical timing
                    continue
                by_id[key] = payload
                n_added += 1

    sorted_records = [by_id[key] for key in sorted(by_id.keys())]
    with samples_path.open("w", encoding="utf-8") as handle:
        for record in sorted_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for shard_path in output_dir.glob(rank_pattern):
        try:
            shard_path.unlink()
        except OSError:
            pass

    return {"n_kept_existing": n_kept_existing, "n_added": n_added, "n_total": len(sorted_records)}


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_distributed = setup_distributed(args)
    is_main = rank == 0
    device = resolve_device(local_rank)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    run_dir = resolve_run_dir(args.run_dir, checkpoint_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_main:
        ensure_dir(output_dir)
        ensure_dir(output_dir / "videos")
    if is_distributed:
        torch.distributed.barrier()

    samples_path = output_dir / "samples.jsonl"
    rank_shard_path = output_dir / f"samples.rank{rank:03d}.jsonl"
    rank_pattern = "samples.rank*.jsonl"

    manifest_arg = args.prompt_manifest
    if isinstance(manifest_arg, str):
        manifest_arg = [manifest_arg]
    manifest_paths_input = [Path(p) for p in manifest_arg]
    all_records, manifest_paths = load_all_manifest_records(
        manifest_paths_input, is_main=is_main
    )
    selected = select_records(all_records, args.offset, args.limit)
    if not selected:
        if is_main:
            print("No prompt records selected; exiting.", file=sys.stderr)
        return

    # Stamp every row with a stable manifest_row_index *before* sharding so
    # that resume / merge / seed assignment all agree across ranks. The index
    # is global across all input manifests (concatenation order) so modulo
    # sharding remains stable when callers pass multiple manifests.
    prompt_fallbacks = _parse_key_list(args.prompt_fallback_keys)
    neg_prompt_fallbacks = _parse_key_list(args.negative_prompt_fallback_keys)
    base_offset = max(0, int(args.offset))
    indexed_records: list[tuple[int, dict[str, Any]]] = []
    seen_unified_ids: dict[str, int] = {}
    prompt_key_hits: dict[str, int] = {}
    n_empty_prompts = 0
    for local_index, record in enumerate(selected):
        manifest_row_index = base_offset + local_index
        record_copy = dict(record)
        if not record_copy.get(args.unified_id_key):
            record_copy[args.unified_id_key] = synthesize_unified_id(record_copy, manifest_row_index)
        record_copy["manifest_row_index"] = manifest_row_index

        prompt_text, prompt_key_used = _resolve_text_field(
            record_copy,
            primary_key=args.prompt_key,
            fallback_keys=prompt_fallbacks,
        )
        neg_prompt_text, _neg_key_used = _resolve_text_field(
            record_copy,
            primary_key=args.negative_prompt_key,
            fallback_keys=neg_prompt_fallbacks,
        )
        record_copy["_resolved_prompt"] = prompt_text
        record_copy["_resolved_negative_prompt"] = neg_prompt_text
        record_copy["_resolved_prompt_key"] = prompt_key_used or "<empty>"
        prompt_key_hits[prompt_key_used or "<empty>"] = (
            prompt_key_hits.get(prompt_key_used or "<empty>", 0) + 1
        )
        if not prompt_text:
            n_empty_prompts += 1

        unified_id_str = str(record_copy.get(args.unified_id_key, ""))
        if unified_id_str:
            if unified_id_str in seen_unified_ids and is_main:
                print(
                    f"[manifest] WARN: duplicate unified_id {unified_id_str!r} at "
                    f"row {manifest_row_index} (also at row {seen_unified_ids[unified_id_str]}); "
                    f"the later row will overwrite the earlier on resume/merge.",
                    file=sys.stderr,
                )
            seen_unified_ids.setdefault(unified_id_str, manifest_row_index)
        indexed_records.append((manifest_row_index, record_copy))

    # ----- prompt-resolution audit -----
    if is_main:
        n_total = len(indexed_records)
        primary_hits = prompt_key_hits.get(args.prompt_key, 0)
        fallback_hits = sum(
            count for key, count in prompt_key_hits.items()
            if key not in {args.prompt_key, "<empty>"}
        )
        empty_hits = prompt_key_hits.get("<empty>", 0)
        breakdown = ", ".join(
            f"{key}={count}" for key, count in sorted(
                prompt_key_hits.items(), key=lambda kv: -kv[1]
            )
        )
        print(
            f"[prompt] resolved {n_total} rows -> primary({args.prompt_key})={primary_hits}, "
            f"fallback={fallback_hits}, empty={empty_hits}  [{breakdown}]",
            file=sys.stderr,
        )
        # Show a few resolved prompts so the user can eyeball them.
        for sample_idx, sample_record in indexed_records[:2]:
            prev = sample_record["_resolved_prompt"]
            prev = prev if len(prev) <= 140 else (prev[:137] + "...")
            print(
                f"[prompt] row{sample_idx:03d}  via='{sample_record['_resolved_prompt_key']}'  "
                f"text={prev!r}",
                file=sys.stderr,
            )

    if n_empty_prompts == len(indexed_records) and len(indexed_records) > 0:
        raise SystemExit(
            f"[prompt] FATAL: every one of {len(indexed_records)} manifest rows resolved "
            f"to an empty prompt under primary key {args.prompt_key!r} and fallbacks "
            f"{prompt_fallbacks!r}. The raw {{from_train,train2valid_gen,vbench_sampled,"
            f"valid_when_train}}.jsonl manifests store the prompt under 'video_caption' "
            f"-- either run eval/video/t2v/build_unified_manifest.py first to standardize "
            f"the field, or pass '--prompt-key video_caption' / '--prompt-fallback-keys "
            f"video_caption' to this script. Refusing to generate videos with empty "
            f"prompts because the model would silently produce unconditional garbage."
        )
    elif n_empty_prompts > 0 and is_main:
        print(
            f"[prompt] WARN: {n_empty_prompts}/{len(indexed_records)} rows still have "
            f"empty prompts after fallback chain; those will run unconditional.",
            file=sys.stderr,
        )

    skip_ids = (
        gather_existing_ids_across_ranks(
            samples_path,
            output_dir=output_dir,
            rank_pattern=rank_pattern,
            key=args.unified_id_key,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
            device=device,
        )
        if args.resume
        else set()
    )
    if args.resume and is_main:
        print(
            f"[resume] {len(skip_ids)} unified_ids already done across "
            f"samples.jsonl + {len(sorted(output_dir.glob(rank_pattern)))} rank shards; "
            f"will skip them.",
            file=sys.stderr,
        )

    # Rank-shard with stable modulo, then drop already-finished rows.
    shard: list[tuple[int, dict[str, Any]]] = []
    for manifest_row_index, record in indexed_records:
        if manifest_row_index % max(1, world_size) != rank:
            continue
        unified_id = str(record.get(args.unified_id_key, ""))
        if unified_id in skip_ids:
            continue
        shard.append((manifest_row_index, record))

    run_config = load_run_config_for_eval(run_dir)
    config = core.build_request_config(run_config, args)
    checkpoint_step = extract_checkpoint_step(checkpoint_dir)

    dual_vae_enabled = args.vae_path_2 is not None and str(args.vae_path_2).strip() != ""

    if is_main:
        save_json(
            output_dir / "run.json",
            {
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_step": int(checkpoint_step),
                "run_dir": str(run_dir),
                "prompt_manifest": [str(p) for p in manifest_paths],
                "prompt_key": str(args.prompt_key),
                "prompt_fallback_keys": prompt_fallbacks,
                "negative_prompt_key": str(args.negative_prompt_key),
                "negative_prompt_fallback_keys": neg_prompt_fallbacks,
                "prompt_key_hits": dict(prompt_key_hits),
                "unified_id_key": str(args.unified_id_key),
                "selected_count": len(selected),
                "shards": int(world_size),
                "offset": int(args.offset),
                "limit": args.limit,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_frames": int(
                    config["train"].get("validation_num_frames")
                    or config["dataset"].get("num_frames")
                ),
                "frame_size": list(
                    config["train"].get("validation_frame_size")
                    or config["dataset"].get("frame_size")
                ),
                "fps": float(
                    config["train"].get("validation_fps")
                    or config["dataset"].get("target_fps")
                ),
                "num_inference_steps": int(config["train"]["validation_num_inference_steps"]),
                "guidance_scale": float(config["train"].get("validation_guidance_scale", 4.0)),
                "cfg_normalization": bool(config["train"].get("validation_cfg_normalization", False)),
                "cfg_truncation": config["train"].get("validation_cfg_truncation", 1.0),
                "world_size": int(world_size),
                "vae_type_override": args.vae_type,
                "vae_path_override": args.vae_path,
                "dual_vae_enabled": bool(dual_vae_enabled),
                "secondary_vae_path": args.vae_path_2,
                "secondary_vae_type": args.vae_type_2,
                "dual_vae_stack_axis": args.dual_vae_stack_axis if dual_vae_enabled else None,
            },
        )

    # Even if this rank has nothing to do (e.g. tiny manifest with W>N), it
    # still has to participate in the final barrier + merge; so do not return
    # early -- just drop into the loop with an empty shard.
    if shard:
        pipeline = load_pipeline_for_checkpoint(
            checkpoint_dir,
            config,
            device,
            vae_type_override=args.vae_type,
            vae_path_override=args.vae_path,
        )
        if pipeline.text_encoder is None:
            raise NotImplementedError(
                "Video checkpoint sampling requires a separate text encoder; "
                "qwen3_vl_dit is not supported."
            )
        pipeline.transformer.eval()
        pipeline.text_encoder.eval()
        pipeline.vae.eval()

        from omnivae_generation.trainer.forward_transformer import build_forward_transformer  # noqa: WPS433

        train_patch_size = int(config["transformer"]["all_patch_size"][0])
        train_f_patch_size = int(config["transformer"]["all_f_patch_size"][0])
        forward_transformer = build_forward_transformer(
            pipeline.transformer,
            pipeline.transformer,
            train_patch_size=train_patch_size,
            train_f_patch_size=train_f_patch_size,
        )
        (
            requested_num_frames,
            latent_frames,
            height,
            width,
            latent_height,
            latent_width,
            fps,
        ) = _video_validation_latent_shape(config, pipeline.vae)

        secondary_vae = None
        if dual_vae_enabled:
            secondary_vae = _load_secondary_vae(
                run_config=run_config,
                vae_path=args.vae_path_2,
                vae_type=args.vae_type_2,
                primary_vae_type=args.vae_type,
                device=device,
            )
            # Sanity-check: same latent layout. If the secondary VAE has a
            # different spatial / temporal scale we would have to either resize
            # the latents or run two diffusion passes -- neither is in scope.
            sec_shape = _video_validation_latent_shape(config, secondary_vae)
            if sec_shape[1:6] != (latent_frames, height, width, latent_height, latent_width):
                raise ValueError(
                    "Secondary VAE has a different latent layout than the main "
                    f"VAE (main latent_frames={latent_frames}, H={height}, W={width}, "
                    f"latent_H={latent_height}, latent_W={latent_width}; secondary "
                    f"latent_frames={sec_shape[1]}, H={sec_shape[2]}, W={sec_shape[3]}, "
                    f"latent_H={sec_shape[4]}, latent_W={sec_shape[5]}). Dual-VAE "
                    "decoding only supports VAEs that share the same latent shape; "
                    "use a single VAE or align the two VAE architectures."
                )
    else:
        pipeline = None
        forward_transformer = None
        secondary_vae = None
        requested_num_frames = int(
            config["train"].get("validation_num_frames")
            or config["dataset"].get("num_frames")
        )
        frame_size = config["train"].get("validation_frame_size") or config["dataset"].get("frame_size")
        height, width = int(frame_size[0]), int(frame_size[1])
        latent_frames = latent_height = latent_width = 0
        fps = float(config["train"].get("validation_fps") or config["dataset"].get("target_fps"))

    n_generated = 0
    rank_started_at = time.time()
    if shard:
        # On resume: append so the partial progress already written to
        # samples.rank{R}.jsonl in the previous (crashed) run is preserved.
        # Without --resume: overwrite, matching the historical "clean rerun"
        # contract -- the merged samples.jsonl is the durable artifact.
        shard_open_mode = "a" if args.resume else "w"
        shard_handle = rank_shard_path.open(shard_open_mode, encoding="utf-8")
        progress = tqdm(
            list(_iter_batches(shard, int(args.batch_size))),
            desc=f"rank{rank:02d} ckpt-{checkpoint_step:08d}",
            position=local_rank,
            leave=False,
            disable=not is_main,
        )
        try:
            for batch_pairs in progress:
                batch_records = [
                    {
                        "sample_index": manifest_row_index,
                        "prompt": str(record.get("_resolved_prompt") or ""),
                        "negative_prompt": str(record.get("_resolved_negative_prompt") or ""),
                        "source_record": record,
                    }
                    for manifest_row_index, record in batch_pairs
                ]
                seeds = [int(args.seed) + int(item["sample_index"]) for item in batch_records]
                if secondary_vae is not None:
                    videos_primary, videos_secondary = generate_batch_dual_decode(
                        batch_records=batch_records,
                        config=config,
                        pipeline=pipeline,
                        secondary_vae=secondary_vae,
                        forward_transformer=forward_transformer,
                        device=device,
                        seeds=seeds,
                        latent_frames=latent_frames,
                        latent_height=latent_height,
                        latent_width=latent_width,
                    )
                else:
                    videos_primary = core.generate_batch(
                        batch_records=batch_records,
                        config=config,
                        pipeline=pipeline,
                        forward_transformer=forward_transformer,
                        device=device,
                        seeds=seeds,
                        latent_frames=latent_frames,
                        latent_height=latent_height,
                        latent_width=latent_width,
                    )
                    videos_secondary = None

                for batch_pos, item in enumerate(batch_records):
                    sample_index = int(item["sample_index"])
                    seed = int(args.seed) + sample_index
                    source_record = item["source_record"]
                    unified_id = str(source_record.get(args.unified_id_key, "") or synthesize_unified_id(source_record, sample_index))
                    video_path = output_dir / "videos" / f"{unified_id}_seed{seed}.mp4"

                    if videos_secondary is not None:
                        video_tensor = _concat_videos_side_by_side(
                            videos_primary[batch_pos],
                            videos_secondary[batch_pos],
                            axis=args.dual_vae_stack_axis,
                        )
                    else:
                        video_tensor = videos_primary[batch_pos]
                    frames = _video_tensor_to_uint8_frames(video_tensor)
                    imageio.mimsave(
                        video_path,
                        list(frames),
                        fps=fps,
                        codec="libx264",
                        quality=int(args.quality),
                        macro_block_size=None,
                    )

                    payload: dict[str, Any] = copy.deepcopy(source_record)
                    payload.pop("_resolved_prompt", None)
                    payload.pop("_resolved_negative_prompt", None)
                    prompt_key_used = payload.pop("_resolved_prompt_key", None)
                    payload[args.unified_id_key] = unified_id
                    payload["sample_index"] = sample_index
                    payload["video_path"] = str(video_path)
                    payload["prompt"] = item["prompt"]
                    payload["negative_prompt"] = item["negative_prompt"]
                    if prompt_key_used is not None:
                        payload["prompt_key_used"] = prompt_key_used
                    payload["seed"] = seed
                    payload["requested_num_frames"] = int(requested_num_frames)
                    payload["decoded_num_frames"] = int(videos_primary[batch_pos].shape[1])
                    payload["latent_frames"] = int(latent_frames)
                    payload["height"] = int(height)
                    payload["width"] = int(width)
                    payload["fps"] = float(fps)
                    payload["num_inference_steps"] = int(config["train"]["validation_num_inference_steps"])
                    payload["guidance_scale"] = float(config["train"].get("validation_guidance_scale", 4.0))
                    payload["cfg_normalization"] = bool(config["train"].get("validation_cfg_normalization", False))
                    payload["cfg_truncation"] = config["train"].get("validation_cfg_truncation", 1.0)
                    payload["checkpoint_dir"] = str(checkpoint_dir)
                    payload["checkpoint_step"] = int(checkpoint_step)
                    payload["run_dir"] = str(run_dir)
                    payload["rank"] = int(rank)
                    if videos_secondary is not None:
                        payload["dual_vae_enabled"] = True
                        payload["secondary_vae_path"] = str(args.vae_path_2)
                        payload["secondary_vae_type"] = args.vae_type_2
                        payload["dual_vae_stack_axis"] = args.dual_vae_stack_axis
                        if args.dual_vae_stack_axis == "width":
                            payload["width"] = int(width) * 2
                        else:
                            payload["height"] = int(height) * 2

                    shard_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    shard_handle.flush()
                    n_generated += 1
        finally:
            shard_handle.close()

    if pipeline is not None and hasattr(pipeline.vae, "clear_cache"):
        pipeline.vae.clear_cache()
    if secondary_vae is not None and hasattr(secondary_vae, "clear_cache"):
        secondary_vae.clear_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rank_elapsed = time.time() - rank_started_at
    print(
        f"[infer_t2v] rank={rank} world={world_size} generated={n_generated} "
        f"elapsed={rank_elapsed:.1f}s shard_path={rank_shard_path}",
        flush=True,
    )

    if is_distributed:
        torch.distributed.barrier()

    if is_main:
        merge_stats = merge_rank_shards(
            output_dir=output_dir,
            samples_path=samples_path,
            rank_pattern=rank_pattern,
            unified_id_key=args.unified_id_key,
            keep_existing=bool(args.resume),
        )
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "samples_path": str(samples_path),
                    "world_size": int(world_size),
                    "selected_total": len(selected),
                    "merged": merge_stats,
                },
                indent=2,
                sort_keys=True,
            )
        )

    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _iter_batches(items, batch_size):
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


def _load_secondary_vae(
    *,
    run_config: dict,
    vae_path: str,
    vae_type: str | None,
    primary_vae_type: str | None,
    device: torch.device,
) -> torch.nn.Module:
    """Load the second VAE used by dual-decode mode.

    Mirrors the VAE-loading branch of ``load_pipeline_for_checkpoint`` (no text
    encoder / transformer / scheduler), so any vae.type
    supported there works here too. Type resolution order:
        --vae-type-2 > --vae-type > run_config['vae']['type']
    """

    from omnivae_generation.trainer.modeling import load_vae  # noqa: WPS433

    vae_config = dict(run_config["vae"])
    resolved_type = vae_type or primary_vae_type or vae_config.get("type")
    if resolved_type:
        vae_config["type"] = str(resolved_type)
    vae_config["model_name_or_path"] = str(vae_path)
    vae_config["subfolder"] = None
    vae_config["local_files_only"] = True
    vae = load_vae(vae_config)
    vae.eval()
    return vae.to(device)


@torch.inference_mode()
def generate_batch_dual_decode(
    *,
    batch_records: list[dict[str, Any]],
    config: dict,
    pipeline,
    secondary_vae,
    forward_transformer,
    device: torch.device,
    seeds: list[int],
    latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the validation diffusion loop ONCE, then decode the resulting latents
    with both ``pipeline.vae`` (primary) and ``secondary_vae``.

    The diffusion body is a faithful copy of ``core.generate_batch`` (which is
    the single source of truth for the validation sampler); only the final
    ``decode_latents_to_images`` step is duplicated. Returns
    ``(videos_primary, videos_secondary)`` -- each a video tensor in the same
    layout as ``core.generate_batch`` returns (``[B, C, T, H, W]`` in ``[-1, 1]``).
    """

    transformer_model = pipeline.transformer
    text_encoder_model = pipeline.text_encoder
    primary_vae = pipeline.vae
    tokenizer = pipeline.tokenizer

    prompts = [core.coerce_text(record.get("prompt", "")) for record in batch_records]
    negative_prompts = [core.coerce_text(record.get("negative_prompt", "")) for record in batch_records]
    formatted_prompts = [core.maybe_format_chat_prompt(p, tokenizer) for p in prompts]
    formatted_negative_prompts = [core.maybe_format_chat_prompt(p, tokenizer) for p in negative_prompts]

    inference_dtype = core.resolve_inference_dtype(config, device)
    with torch.autocast(
        device_type=device.type,
        dtype=inference_dtype,
        enabled=core.autocast_enabled_for(device, inference_dtype),
    ):
        prompt_embeds = core.encode_prompts(
            formatted_prompts,
            tokenizer,
            text_encoder_model,
            device,
            int(config["text_encoder"]["max_sequence_length"]),
            cache_enabled=bool(config["text_encoder"].get("cache_enabled", False)),
        )
        negative_prompt_embeds = core.encode_prompts(
            formatted_negative_prompts,
            tokenizer,
            text_encoder_model,
            device,
            int(config["text_encoder"]["max_sequence_length"]),
            cache_enabled=bool(config["text_encoder"].get("cache_enabled", False)),
        )

    transformer_dtype = getattr(transformer_model, "dtype", inference_dtype)
    prompt_embeds = [e.to(dtype=transformer_dtype) for e in prompt_embeds]
    negative_prompt_embeds = [e.to(dtype=transformer_dtype) for e in negative_prompt_embeds]

    latents = core.make_latents(
        seeds=seeds,
        channels=int(config["transformer"]["in_channels"]),
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
    )
    inference_scheduler = core._build_inference_scheduler(
        config, pipeline.scheduler, transformer_model, device
    )
    guidance_scale = float(config["train"].get("validation_guidance_scale", 4.0))
    cfg_normalization = config["train"].get("validation_cfg_normalization", False)
    cfg_truncation = config["train"].get("validation_cfg_truncation", 1.0)
    cfg_truncation = None if cfg_truncation is None else float(cfg_truncation)

    for timestep_value in inference_scheduler.timesteps:
        timestep = timestep_value.expand(latents.shape[0])
        model_timesteps = (
            float(inference_scheduler.config.num_train_timesteps) - timestep
        ) / float(inference_scheduler.config.num_train_timesteps)
        model_timesteps = model_timesteps.to(device=device, dtype=torch.float32)
        t_norm = float(model_timesteps[0].item())
        current_guidance_scale = guidance_scale
        if cfg_truncation is not None and cfg_truncation <= 1.0 and t_norm > cfg_truncation:
            current_guidance_scale = 0.0

        apply_cfg = current_guidance_scale > 0.0
        if apply_cfg:
            latent_model_input = latents.repeat(2, 1, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
            timestep_model_input = model_timesteps.repeat(2)
        else:
            latent_model_input = latents
            prompt_embeds_model_input = prompt_embeds
            timestep_model_input = model_timesteps

        with torch.autocast(
            device_type=device.type,
            dtype=transformer_dtype,
            enabled=core.autocast_enabled_for(device, transformer_dtype),
        ):
            model_pred, _ = forward_transformer(
                latent_model_input.to(dtype=transformer_dtype),
                timestep_model_input,
                prompt_embeds_model_input,
            )
        if apply_cfg:
            batch_size = len(batch_records)
            pos_pred = model_pred[:batch_size].float()
            neg_pred = model_pred[batch_size:].float()
            model_pred = core.apply_zimage_cfg(
                pos_pred, neg_pred, current_guidance_scale, cfg_normalization
            )
        else:
            model_pred = model_pred.float()
        model_pred = -model_pred
        latents = inference_scheduler.step(
            model_pred.to(torch.float32),
            timestep_value,
            latents,
            return_dict=False,
        )[0].to(torch.float32)

    videos_primary = core.decode_latents_to_images(
        latents.to(dtype=getattr(primary_vae, "dtype", latents.dtype)),
        primary_vae,
    )
    videos_secondary = core.decode_latents_to_images(
        latents.to(dtype=getattr(secondary_vae, "dtype", latents.dtype)),
        secondary_vae,
    )
    return videos_primary, videos_secondary


def _concat_videos_side_by_side(
    video_a: torch.Tensor,
    video_b: torch.Tensor,
    *,
    axis: str,
) -> torch.Tensor:
    """Concatenate two decoded video tensors (layout: ``[C, T, H, W]``) along the
    spatial axis. Pads the shorter one along ``T`` / the off-axis spatial dim with
    its last frame / zeros if the two decoders produced different shapes (e.g.
    slight temporal rounding); raises on channel mismatch.

    ``axis='width'``  -> torch.cat(..., dim=-1)    (left=A, right=B)
    ``axis='height'`` -> torch.cat(..., dim=-2)    (top=A, bottom=B)
    """

    if video_a.ndim != 4 or video_b.ndim != 4:
        raise ValueError(
            f"Expected per-sample video tensors with 4 dims (C,T,H,W); got "
            f"{tuple(video_a.shape)} and {tuple(video_b.shape)}."
        )
    if video_a.shape[0] != video_b.shape[0]:
        raise ValueError(
            f"Cannot stack videos with different channel counts: "
            f"{video_a.shape[0]} vs {video_b.shape[0]}."
        )

    target_t = min(video_a.shape[1], video_b.shape[1])
    video_a = video_a[:, :target_t]
    video_b = video_b[:, :target_t]

    if axis == "width":
        if video_a.shape[2] != video_b.shape[2]:
            raise ValueError(
                f"width-axis stacking requires equal heights, got H={video_a.shape[2]} "
                f"vs H={video_b.shape[2]}."
            )
        return torch.cat([video_a, video_b], dim=-1)
    if axis == "height":
        if video_a.shape[3] != video_b.shape[3]:
            raise ValueError(
                f"height-axis stacking requires equal widths, got W={video_a.shape[3]} "
                f"vs W={video_b.shape[3]}."
            )
        return torch.cat([video_a, video_b], dim=-2)
    raise ValueError(f"Unknown dual_vae_stack_axis={axis!r}; expected 'width' or 'height'.")


if __name__ == "__main__":
    main()
