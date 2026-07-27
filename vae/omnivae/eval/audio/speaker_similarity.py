import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample

from omnivae.utils.helpers import find_audio_files


def evaluate_sim_pairs(
    ref_paths: Iterable[str],
    syn_paths: Iterable[str],
    *,
    model_path: str | None = None,
    use_cuda: bool = True,
):
    from stopes.eval.vocal_style_similarity.vocal_style_sim_tool import (
        compute_cosine_similarity,
        get_embedder,
    )

    ref_audio_list = [str(path) for path in ref_paths]
    syn_audio_list = [str(path) for path in syn_paths]
    if len(ref_audio_list) != len(syn_audio_list):
        raise ValueError(
            f"Speaker-sim pair count mismatch: ref={len(ref_audio_list)} "
            f"recon={len(syn_audio_list)}"
        )
    if not ref_audio_list:
        raise ValueError("Speaker-sim input is empty")

    model_path = model_path or os.environ.get("OMNIVAE_SPEAKER_SIM_MODEL")
    if not model_path:
        raise RuntimeError(
            "Speaker similarity evaluation requires OMNIVAE_SPEAKER_SIM_MODEL "
            "to point to the WavLM fine-tuned checkpoint."
        )
    if not Path(model_path).expanduser().is_file():
        raise FileNotFoundError(f"Speaker-sim checkpoint not found: {model_path}")

    embedder = get_embedder(
        model_name="valle",
        model_path=str(Path(model_path).expanduser().resolve()),
        use_cuda=use_cuda,
    )

    # Stopes' default loader assumes a one-dimensional waveform. Explicitly
    # convert stereo/multichannel reconstruction outputs to 16 kHz mono.
    def load_mono_audio(audio_path: str) -> torch.Tensor:
        wave, sample_rate = sf.read(
            audio_path, dtype="float32", always_2d=True
        )
        tensor = torch.from_numpy(np.asarray(wave, dtype=np.float32).mean(axis=1)).unsqueeze(0)
        if sample_rate != 16000:
            tensor = Resample(orig_freq=sample_rate, new_freq=16000)(tensor)
        return tensor

    embedder.load_audio = load_mono_audio
    src_embs = embedder(ref_audio_list)
    tgt_embs = embedder(syn_audio_list)
    similarities = np.asarray(
        compute_cosine_similarity(src_embs, tgt_embs), dtype=np.float64
    ).reshape(-1)
    if len(similarities) != len(ref_audio_list):
        raise RuntimeError(
            f"Speaker-sim output count mismatch: {len(similarities)} != "
            f"{len(ref_audio_list)}"
        )
    return similarities, float(np.mean(similarities))


def evaluate_sim(ref_path, syn_path, model_path: str | None = None, use_cuda: bool = True):
    logging.info(
        "Evaluating Speaker Similarity: ref_path=%s syn_path=%s", ref_path, syn_path
    )
    ref_by_name = {Path(path).name: path for path in find_audio_files(ref_path)}
    syn_by_name = {Path(path).name: path for path in find_audio_files(syn_path)}
    if set(ref_by_name) != set(syn_by_name):
        raise ValueError(
            "Speaker-sim reference/reconstruction filenames do not match: "
            f"ref={len(ref_by_name)} recon={len(syn_by_name)} "
            f"common={len(set(ref_by_name) & set(syn_by_name))}"
        )
    names = sorted(ref_by_name)
    return evaluate_sim_pairs(
        [ref_by_name[name] for name in names],
        [syn_by_name[name] for name in names],
        model_path=model_path,
        use_cuda=use_cuda,
    )
