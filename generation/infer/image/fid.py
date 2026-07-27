"""Thin wrapper around ``torch-fidelity`` for FID over npz batches.

We persist real / generated / reconstructed images as the same npz layout that
``omnivae_generation.trainer.eval.guided_diffusion.merge_sample_shards_to_npz`` produces:

    np.savez(file, images_arr, labels_arr)
    # arr_0: uint8 [N, H, W, 3]
    # arr_1: int32 [N]      (unused for FID, kept for layout parity)

torch-fidelity expects ``input1`` / ``input2`` to be either an on-disk image
folder or a ``torch.utils.data.Dataset`` returning ``[3, H, W]`` uint8
tensors. The lightweight :class:`NpzImageDataset` below adapts our npz files
into that dataset interface without writing PNGs to disk.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# Default Inception batch size; tuned to keep memory bounded on a single GPU
# even at 5k samples 256x256. torch-fidelity defaults to 64 too.
DEFAULT_FID_BATCH_SIZE = 64


class NpzImageDataset(Dataset):
    """Read an ``np.savez(file, images_uint8[, labels])`` npz lazily and
    expose it as a torch-fidelity-compatible image dataset.

    Each ``__getitem__`` returns a ``[3, H, W]`` uint8 ``torch.Tensor`` -- the
    exact shape/dtype torch-fidelity's Inception extractor expects. Images
    are mmaped from disk so a 5k-30k sample npz never materializes fully in
    RAM during feature extraction.
    """

    def __init__(self, npz_path: str | Path, *, images_key: str = "arr_0") -> None:
        npz_file = Path(npz_path).expanduser().resolve()
        if not npz_file.is_file():
            raise FileNotFoundError(f"Image npz not found: {npz_file}")
        # mmap_mode='r' so we don't dup the array in RAM; np.load returns
        # a NpzFile and we keep a handle to it for lifetime.
        self._handle = np.load(npz_file, mmap_mode="r")
        if images_key not in self._handle.files:
            raise KeyError(
                f"Expected key {images_key!r} in {npz_file}; got {self._handle.files}"
            )
        self._images = self._handle[images_key]
        if self._images.ndim != 4 or self._images.shape[-1] != 3:
            raise ValueError(
                f"Expected images with shape [N, H, W, 3]; got {self._images.shape}"
            )
        if self._images.dtype != np.uint8:
            raise TypeError(
                f"Expected uint8 images for FID; got dtype={self._images.dtype}"
            )
        self._npz_path = npz_file

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.ascontiguousarray(self._images[int(index)])
        # HWC -> CHW; copy() so the returned tensor doesn't keep an mmap view.
        chw = np.transpose(image, (2, 0, 1)).copy()
        return torch.from_numpy(chw)

    @property
    def path(self) -> Path:
        return self._npz_path


def ensure_torch_fidelity_available() -> None:
    """Mirror the dependency-check style used by
    ``omnivae_generation.trainer.eval.guided_diffusion.ensure_adm_evaluator_dependencies`` so the
    error message points at the right install command.
    """
    if importlib.util.find_spec("torch_fidelity") is None:
        raise RuntimeError(
            "infer/image FID evaluation requires `torch-fidelity`, but it is not "
            "installed in the current Python environment. Install it via "
            "`pip install torch-fidelity` and rerun."
        )


def compute_fid(
    real_npz: str | Path,
    fake_npz: str | Path,
    *,
    device: torch.device,
    batch_size: int = DEFAULT_FID_BATCH_SIZE,
    extra_metrics: bool = False,
) -> dict[str, float]:
    """Run ``torch_fidelity.calculate_metrics`` on two npz batches.

    Returns ``{"fid": float, ...}``. When ``extra_metrics=True`` also asks
    torch-fidelity for Inception Score (computed on ``input2``) and Kernel
    Inception Distance, which are cheap to add once features are extracted.
    """
    ensure_torch_fidelity_available()
    import torch_fidelity  # type: ignore[import-not-found]

    real_dataset = NpzImageDataset(real_npz)
    fake_dataset = NpzImageDataset(fake_npz)

    metrics: dict[str, Any] = torch_fidelity.calculate_metrics(
        input1=fake_dataset,
        input2=real_dataset,
        cuda=bool(device.type == "cuda"),
        fid=True,
        isc=bool(extra_metrics),
        kid=bool(extra_metrics),
        prc=False,
        verbose=False,
        batch_size=int(batch_size),
        samples_shuffle=False,
        save_cpu_ram=True,
    )

    out: dict[str, float] = {}
    for raw_key, value in metrics.items():
        try:
            out[str(raw_key)] = float(value)
        except (TypeError, ValueError):
            continue
    if "frechet_inception_distance" in out and "fid" not in out:
        out["fid"] = out["frechet_inception_distance"]
    return out
