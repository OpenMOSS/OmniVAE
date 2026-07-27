from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from omnivae.dataset.audio_video_streaming_dataset import (
    load_audio_from_path,
    load_audio_from_video_path,
)
from omnivae.eval.audio.dac_metrics import DEFAULT_SAMPLE_RATES, evaluate_dac_metrics
from omnivae.eval.audio.speaker_similarity import evaluate_sim_pairs
from omnivae.eval.reconstruction.common import (
    build_reconstruction_model,
    load_config,
    load_model_weights,
    read_checkpoint_list,
    read_jsonl,
    require_config,
    resolve_path,
    run_name_for_checkpoint,
    setup_logging,
    write_json,
)


def _media_path(raw: str, *, metadata_path: Path, data_root: Optional[str]) -> Path:
    p = Path(os.path.expanduser(os.path.expandvars(str(raw))))
    if p.is_absolute():
        return p
    roots = []
    if data_root:
        roots.append(resolve_path(data_root))
    roots.append(metadata_path.parent)
    for root in roots:
        candidate = (root / p).resolve()
        if candidate.exists():
            return candidate
    return (roots[0] / p).resolve()


def _audio_source(record: Dict[str, Any], *, metadata_path: Path, data_root: Optional[str]) -> Tuple[Path, bool]:
    raw = record.get("audio_path") or record.get("audio") or record.get("path")
    if raw:
        return _media_path(str(raw), metadata_path=metadata_path, data_root=data_root), False
    raw = record.get("video_path") or record.get("video")
    if raw:
        return _media_path(str(raw), metadata_path=metadata_path, data_root=data_root), True
    raise KeyError("record must contain audio_path or video_path")


def _pad_batch(items: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    lengths = [int(x.shape[-1]) for x in items]
    max_len = max(lengths)
    padded = []
    for x in items:
        if x.shape[-1] < max_len:
            x = F.pad(x, (0, max_len - x.shape[-1]))
        padded.append(x)
    return torch.stack(padded, 0), lengths


def _snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    noise = reference - estimate
    signal_power = torch.mean(reference.float() ** 2).item()
    noise_power = torch.mean(noise.float() ** 2).item()
    return 10.0 * math.log10((signal_power + 1e-12) / (noise_power + 1e-12))


def _mel_loss(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    f_min: float,
    f_max: Optional[float],
    device: torch.device,
) -> float:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=True,
        norm="slaney",
        mel_scale="slaney",
    ).to(device)
    ref = torch.log(mel(reference.to(device).float()) + 1e-7)
    est = torch.log(mel(estimate.to(device).float()) + 1e-7)
    return F.l1_loss(est, ref).item()


def _stft_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> Tuple[float, float, float]:
    ref = reference.reshape(-1).float()
    est = estimate.reshape(-1).float()
    sc_vals = []
    mag_vals = []
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=ref.device)
        ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                              window=window, return_complex=True).abs()
        est_spec = torch.stft(est, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                              window=window, return_complex=True).abs()
        sc = torch.linalg.norm(ref_spec - est_spec) / (torch.linalg.norm(ref_spec) + 1e-8)
        mag = F.l1_loss(torch.log(est_spec + 1e-7), torch.log(ref_spec + 1e-7))
        sc_vals.append(sc.item())
        mag_vals.append(mag.item())
    sc_avg = float(np.mean(sc_vals))
    mag_avg = float(np.mean(mag_vals))
    return sc_avg, mag_avg, 0.5 * (sc_avg + mag_avg)


def _optional_stoi(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> Optional[float]:
    try:
        from pystoi import stoi
    except ImportError:
        return None
    return float(stoi(reference, estimate, sample_rate, extended=False))


def _optional_pesq(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    mode: str,
    target_sample_rate: int,
) -> Optional[float]:
    try:
        from pesq import pesq
    except ImportError:
        return None
    if sample_rate != target_sample_rate:
        reference_t = torch.from_numpy(reference).float().unsqueeze(0)
        estimate_t = torch.from_numpy(estimate).float().unsqueeze(0)
        reference = torchaudio.functional.resample(
            reference_t, sample_rate, target_sample_rate
        ).squeeze(0).numpy()
        estimate = torchaudio.functional.resample(
            estimate_t, sample_rate, target_sample_rate
        ).squeeze(0).numpy()

    length = min(reference.shape[-1], estimate.shape[-1])
    if length <= 0:
        return None
    reference = reference[:length]
    estimate = estimate[:length]
    # The PESQ extension can abort when several workers enter its C code at
    # once. Serializing calls does not change the official whole-file metric.
    lock_path = os.environ.get("OMNIVAE_PESQ_LOCK", "/tmp/omnivae_pesq_full_file.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return float(pesq(target_sample_rate, reference, estimate, mode))
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _safe_name(path: Path, index: int) -> str:
    stem = path.stem or f"sample_{index:06d}"
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
    return f"{index:06d}_{stem}.wav"


def _save_float_pair(
    gt_path: Path,
    recon_path: Path,
    ref: np.ndarray,
    est: np.ndarray,
    sample_rate: int,
    subtype: str,
) -> None:
    sf.write(str(gt_path), ref, sample_rate, format="WAV", subtype=subtype)
    sf.write(str(recon_path), est, sample_rate, format="WAV", subtype=subtype)


def evaluate_one_checkpoint(args: argparse.Namespace, checkpoint: str, index: int) -> Dict[str, Any]:
    output_dir = resolve_path(args.output_dir) / run_name_for_checkpoint(index, checkpoint)
    log_file = None if args.dry_run else output_dir / "run.log"
    setup_logging(log_file)

    config_path = require_config(args.config, checkpoint)
    metadata_path = resolve_path(args.input_jsonl)
    rows = read_jsonl(metadata_path, max_examples=args.max_examples)

    logging.info("checkpoint=%s", checkpoint)
    logging.info("config=%s", config_path)
    logging.info("input_jsonl=%s rows=%d", metadata_path, len(rows))
    logging.info("output_dir=%s", output_dir)

    compute_dac_metrics = (
        args.evaluation_domain in {"audio", "music"}
        if args.compute_dac_metrics is None
        else args.compute_dac_metrics
    )
    compute_speaker_similarity = (
        args.evaluation_domain == "speech"
        if args.compute_speaker_similarity is None
        else args.compute_speaker_similarity
    )
    logging.info(
        "evaluation_domain=%s inference_dtype=%s dac_metrics=%s "
        "speaker_similarity=%s",
        args.evaluation_domain,
        args.inference_dtype,
        compute_dac_metrics,
        compute_speaker_similarity,
    )

    speaker_similarity_model = args.speaker_similarity_model
    if compute_speaker_similarity:
        if not speaker_similarity_model:
            raise ValueError(
                "Speech speaker similarity requires --speaker_similarity_model "
                "or OMNIVAE_SPEAKER_SIM_MODEL"
            )
        speaker_similarity_model = str(
            resolve_path(speaker_similarity_model)
        )
        if not Path(speaker_similarity_model).is_file():
            raise FileNotFoundError(
                f"Speaker-sim checkpoint not found: {speaker_similarity_model}"
            )

    if args.dry_run:
        return {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "input_jsonl": str(metadata_path),
            "output_dir": str(output_dir),
            "count": len(rows),
            "evaluation_domain": args.evaluation_domain,
            "inference_dtype": args.inference_dtype,
            "compute_dac_metrics": compute_dac_metrics,
            "compute_speaker_similarity": compute_speaker_similarity,
            "dry_run": True,
        }

    cfg = load_config(config_path)
    audio_cfg = cfg.get("model", {}).get("audio", {})
    sample_rate = int(args.sample_rate or audio_cfg.get("sample_rate") or audio_cfg.get("audio_sample_rate") or 24000)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.inference_dtype]
    if inference_dtype == torch.bfloat16 and device.type != "cuda":
        raise ValueError("bfloat16 reconstruction currently requires a CUDA device")
    model = build_reconstruction_model(cfg, modality="audio")
    load_stats = load_model_weights(model, checkpoint, use_ema=args.use_ema)
    logging.info("load_stats=%s", load_stats)
    if hasattr(model, "module_dtypes"):
        model.module_dtypes["audio_vae"] = inference_dtype
    model.to(device).eval()

    gt_dir = output_dir / "no_ema" / "gt"
    recon_dir = output_dir / "no_ema" / "recon"
    gt_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    sums = {
        "l1": 0.0,
        "snr": 0.0,
        "mel_loss": 0.0,
        "stft_sc": 0.0,
        "stft_mag": 0.0,
        "stft_dist": 0.0,
        "stoi": 0.0,
        "pesq_wb": 0.0,
        "pesq_nb": 0.0,
    }
    counts = {k: 0 for k in sums}
    errors: Dict[str, List[str]] = {"stoi": [], "pesq_wb": [], "pesq_nb": []}
    samples: List[Dict[str, Any]] = []
    saved_names: List[str] = []
    saved_gt_paths: List[str] = []
    saved_recon_paths: List[str] = []

    pending_audio: List[torch.Tensor] = []
    pending_paths: List[Path] = []
    pending_indices: List[int] = []
    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.save_workers)
    save_futures: set[concurrent.futures.Future[None]] = set()

    def submit_save(name: str, ref_np: np.ndarray, est_np: np.ndarray) -> None:
        future = save_executor.submit(
            _save_float_pair,
            gt_dir / name,
            recon_dir / name,
            ref_np,
            est_np,
            sample_rate,
            args.save_subtype,
        )
        save_futures.add(future)
        if len(save_futures) >= args.max_pending_saves:
            done, pending = concurrent.futures.wait(
                save_futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for completed in done:
                completed.result()
            save_futures.clear()
            save_futures.update(pending)

    def flush() -> None:
        if not pending_audio:
            return
        batch, lengths = _pad_batch(pending_audio)
        batch = batch.to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=inference_dtype,
            enabled=inference_dtype != torch.float32,
        ):
            out = model(None, batch, sample_posterior=not args.no_sample_posterior)
            # WAV serialization and metric consumers use float32 arrays even
            # when the model forward pass uses BF16 autocast.
            recon = out["audio"]["recon"].detach().float().cpu()
        batch_cpu = batch.detach().cpu()
        del out, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for local_i, (src_path, sample_idx, length) in enumerate(zip(pending_paths, pending_indices, lengths)):
            ref = batch_cpu[local_i, :, :length]
            est = recon[local_i, :, :length]
            name = _safe_name(src_path, sample_idx)

            sample = {
                "path": str(src_path),
                "name": name,
            }

            ref_np = ref.squeeze(0).numpy()
            est_np = est.squeeze(0).numpy()
            if not args.reconstruction_only:
                l1 = F.l1_loss(est, ref).item()
                snr = _snr_db(ref, est)
                mel_loss = _mel_loss(
                    ref, est, sample_rate, args.mel_n_mels,
                    args.mel_f_min, args.mel_f_max, device,
                )
                stft_sc, stft_mag, stft_dist = _stft_metrics(ref, est)
                sample.update({
                    "l1": l1,
                    "snr": snr,
                    "mel_loss": mel_loss,
                    "stft_sc": stft_sc,
                    "stft_mag": stft_mag,
                    "stft_dist": stft_dist,
                })
                for key, value in (
                    ("l1", l1), ("snr", snr), ("mel_loss", mel_loss),
                    ("stft_sc", stft_sc), ("stft_mag", stft_mag), ("stft_dist", stft_dist),
                ):
                    sums[key] += float(value)
                    counts[key] += 1
            else:
                counts["l1"] += 1

            if args.compute_stoi and not args.reconstruction_only:
                try:
                    value = _optional_stoi(ref_np, est_np, sample_rate)
                    if value is not None:
                        sums["stoi"] += value
                        counts["stoi"] += 1
                        sample["stoi"] = value
                except Exception as exc:
                    errors["stoi"].append(f"{name}: {exc}")
            if args.compute_pesq and not args.reconstruction_only:
                for mode, key in (("wb", "pesq_wb"), ("nb", "pesq_nb")):
                    try:
                        value = _optional_pesq(
                            ref_np,
                            est_np,
                            sample_rate,
                            mode,
                            args.pesq_sample_rate,
                        )
                        if value is not None:
                            sums[key] += value
                            counts[key] += 1
                            sample[key] = value
                    except Exception as exc:
                        errors[key].append(f"{name}: {exc}")

            if compute_dac_metrics and not args.reconstruction_only:
                sample_dac_errors: Dict[str, str] = {}
                dac_values = evaluate_dac_metrics(
                    ref_np,
                    est_np,
                    sample_rate,
                    sample_rates=args.dac_sample_rates,
                    compute_visqol=args.compute_visqol,
                    visqol_mode=args.visqol_mode,
                    visqol_argument_order=args.visqol_argument_order,
                    metric_errors=sample_dac_errors,
                )
                for key, value in dac_values.items():
                    sample[key] = value
                    sums[key] = sums.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
                for key, message in sample_dac_errors.items():
                    errors.setdefault(key, []).append(f"{name}: {message}")

            # Metrics use the in-memory 48 kHz float reconstruction. WAV files
            # are retained only after all requested metrics have been computed.
            submit_save(name, ref_np, est_np)
            saved_names.append(name)
            saved_gt_paths.append(str(gt_dir / name))
            saved_recon_paths.append(str(recon_dir / name))
            samples.append(sample)
        pending_audio.clear()
        pending_paths.clear()
        pending_indices.clear()

    for sample_idx, record in enumerate(tqdm(rows, desc="audio recon")):
        src_path, from_video = _audio_source(record, metadata_path=metadata_path, data_root=args.data_root)
        if from_video:
            audio = load_audio_from_video_path(
                str(src_path),
                target_sample_rate=sample_rate,
                max_duration=args.max_duration,
            )
        else:
            audio = load_audio_from_path(
                str(src_path),
                target_sample_rate=sample_rate,
                max_duration=args.max_duration,
            )
        pending_audio.append(audio)
        pending_paths.append(src_path)
        pending_indices.append(sample_idx)
        if len(pending_audio) >= args.batch_size:
            flush()
    flush()
    for future in concurrent.futures.as_completed(save_futures):
        future.result()
    save_executor.shutdown(wait=True)

    speaker_similarity: Optional[Dict[str, Any]] = None
    if compute_speaker_similarity and not args.reconstruction_only:
        # Free reconstruction-model memory before loading the WavLM speaker
        # encoder on the same GPU.
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        use_cuda_for_speaker = (
            device.type == "cuda"
            if args.speaker_similarity_device == "auto"
            else args.speaker_similarity_device == "cuda"
        )
        logging.info(
            "Computing speaker similarity for %d pairs on %s",
            len(saved_names),
            "cuda" if use_cuda_for_speaker else "cpu",
        )
        similarities, avg_similarity = evaluate_sim_pairs(
            saved_gt_paths,
            saved_recon_paths,
            model_path=speaker_similarity_model,
            use_cuda=use_cuda_for_speaker,
        )
        per_file_similarity = {
            name: float(value) for name, value in zip(saved_names, similarities)
        }
        for sample in samples:
            sample["speaker_similarity"] = per_file_similarity[sample["name"]]
        sums["speaker_similarity"] = float(np.sum(similarities))
        counts["speaker_similarity"] = len(similarities)
        speaker_similarity = {
            "count": len(similarities),
            "avg_speaker_similarity": avg_similarity,
            "model_path": speaker_similarity_model,
            "device": "cuda" if use_cuda_for_speaker else "cpu",
            "per_file": per_file_similarity,
        }
        write_json(output_dir / "no_ema" / "speaker_similarity.json", speaker_similarity)

    metrics: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "input_jsonl": str(metadata_path),
        "count": counts["l1"],
        "load_stats": load_stats,
        "evaluation_config": {
            "evaluation_domain": args.evaluation_domain,
            "batch_size": args.batch_size,
            "chunking": False,
            "max_duration": args.max_duration,
            "inference_dtype": args.inference_dtype,
            "sample_posterior": not args.no_sample_posterior,
            "metrics_before_save": True,
            "save_subtype": args.save_subtype,
            "reconstruction_only": args.reconstruction_only,
            "stoi_sample_rate": sample_rate,
            "pesq_nb_sample_rate": args.pesq_sample_rate,
            "pesq_wb_sample_rate": args.pesq_sample_rate,
            "pesq_length_policy": "whole_file",
            "dac_metrics": compute_dac_metrics,
            "dac_sample_rates": list(args.dac_sample_rates),
            "visqol": compute_dac_metrics and args.compute_visqol,
            "visqol_mode": args.visqol_mode,
            "visqol_argument_order": args.visqol_argument_order,
            "speaker_similarity": compute_speaker_similarity,
            "speaker_similarity_model": speaker_similarity_model,
        },
    }
    for key in sorted(counts):
        if counts[key] > 0:
            metrics[f"avg_{key}"] = sums[key] / counts[key]
            metrics[f"{key}_count"] = counts[key]
    metrics["samples"] = samples[: args.max_samples_in_results]
    metrics["metric_errors"] = {key: value for key, value in errors.items() if value}
    if speaker_similarity is not None:
        metrics["speaker_similarity_path"] = str(
            output_dir / "no_ema" / "speaker_similarity.json"
        )

    write_json(output_dir / "results.json", {"no_ema": metrics})
    write_json(output_dir / "no_ema" / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniVAE audio reconstruction evaluation")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint file or Trainer_* directory")
    parser.add_argument("--checkpoint_list", default=None, help="Text file with one checkpoint per line")
    parser.add_argument("--config", default=None, help="Config YAML. If omitted, inferred from checkpoint layout.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--data_root", default=os.environ.get("OMNIVAE_DATA_ROOT"))
    parser.add_argument("--output_dir", default="$OMNIVAE_EXP_ROOT/eval/audio_recon")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_samples_in_results", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--evaluation_domain",
        choices=["speech", "audio", "music"],
        default="audio",
        help=(
            "Controls default metrics: speech enables speaker similarity; "
            "audio/music enable DAC Mel, STFT, and ViSQOL."
        ),
    )
    parser.add_argument(
        "--inference_dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Autocast dtype used by the Audio VAE forward pass.",
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--no_sample_posterior", action="store_true")
    parser.add_argument("--compute_stoi", action="store_true")
    parser.add_argument("--compute_pesq", action="store_true")
    parser.add_argument(
        "--compute_dac_metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override domain-based DAC Mel/STFT/ViSQOL metric selection.",
    )
    parser.add_argument(
        "--compute_visqol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute ViSQOL when DAC metrics are enabled.",
    )
    parser.add_argument(
        "--dac_sample_rates",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_RATES),
        help="DAC evaluation sample rates (default: 44100).",
    )
    parser.add_argument("--visqol_mode", choices=["audio", "speech"], default="audio")
    parser.add_argument(
        "--visqol_argument_order",
        choices=["dac", "standard"],
        default="dac",
        help="Use DAC's published ViSQOL call order or standard reference/degraded order.",
    )
    parser.add_argument(
        "--compute_speaker_similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override domain-based WavLM speaker-sim metric selection.",
    )
    parser.add_argument(
        "--speaker_similarity_model",
        default=os.environ.get("OMNIVAE_SPEAKER_SIM_MODEL"),
        help="WavLM speaker-verification checkpoint.",
    )
    parser.add_argument(
        "--speaker_similarity_device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--reconstruction_only", action="store_true")
    parser.add_argument("--pesq_sample_rate", type=int, default=16000)
    parser.add_argument("--save_subtype", default="PCM_16", choices=["PCM_16", "FLOAT"])
    parser.add_argument("--save_workers", type=int, default=2)
    parser.add_argument("--max_pending_saves", type=int, default=8)
    parser.add_argument("--mel_n_mels", type=int, default=80)
    parser.add_argument("--mel_f_min", type=float, default=0.0)
    parser.add_argument("--mel_f_max", type=float, default=8000.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoints = read_checkpoint_list(args.checkpoint, args.checkpoint_list)
    if not checkpoints:
        raise SystemExit("Pass --checkpoint or --checkpoint_list")
    output_dir = resolve_path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for index, checkpoint in enumerate(checkpoints):
        all_results.append(evaluate_one_checkpoint(args, checkpoint, index))
    if not args.dry_run:
        write_json(output_dir / "summary.json", {"runs": all_results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
