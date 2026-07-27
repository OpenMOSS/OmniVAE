from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, Sized

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader.stateful import Stateful

from omnivae_generation.trainer.audio_data import AudioJsonlT2ADataset, collate_audio_samples
from omnivae_generation.trainer.data import collate_samples
from omnivae_generation.trainer.joint_av.dataset import AVPairedJsonlDataset, collate_av_paired_samples
from omnivae_generation.trainer.relaion_data import RelaionDataset, collate_relaion_samples
from omnivae_generation.trainer.video_data import VideoJsonlDataset, collate_video_samples
from omnivae_generation.trainer.utils import find_latest_complete_checkpoint


DATALOADER_RESUME_STRATEGY = "custom_stateful_dataloader"


@dataclass(frozen=True)
class ResumeState:
    checkpoint_path: Path | None
    global_step: int
    first_epoch: int


class _TensorStatefulRandomSamplerIterator(Iterator[int], Stateful):
    _GENERATOR = "generator"
    _YIELDED = "yielded"

    def __init__(self, sampler: "TensorStatefulRandomSampler") -> None:
        self.sampler = sampler
        self.generator_state = self.sampler.generator.get_state()
        self.yielded = 0
        self.next_yielded = None
        self.n = len(sampler.data_source)
        self.num_samples = sampler.num_samples
        self.perm = torch.randperm(self.n, generator=self.sampler.generator)

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.yielded == self.num_samples:
            raise StopIteration()
        value = int(self.perm[self.yielded])
        self.yielded += 1
        return value

    def state_dict(self) -> dict:
        return {
            self._YIELDED: self.yielded,
            self._GENERATOR: self.generator_state,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.next_yielded = state_dict[self._YIELDED]
        self.generator_state = state_dict[self._GENERATOR]
        self.sampler.generator.set_state(self.generator_state)
        self.perm = torch.randperm(self.n, generator=self.sampler.generator)
        self.yielded = int(self.next_yielded)
        self.next_yielded = None


class TensorStatefulRandomSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        generator=None,
    ) -> None:
        if replacement:
            raise ValueError("TensorStatefulRandomSampler only supports replacement=False.")
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        self.generator = generator
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        return _TensorStatefulRandomSamplerIterator(self)

    def __len__(self) -> int:
        return self.num_samples


class TensorStatefulDistributedSampler(DistributedSampler, Stateful):
    _YIELDED = "yielded"

    def __init__(
        self,
        dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.yielded = 0
        self.next_yielded = None

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator)
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.int64)

        if not self.drop_last:
            padding_size = self.total_size - indices.numel()
            if padding_size <= indices.numel():
                indices = torch.cat((indices, indices[:padding_size]))
            else:
                repeats = math.ceil(padding_size / indices.numel())
                indices = torch.cat((indices, indices.repeat(repeats)[:padding_size]))
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert indices.numel() == self.num_samples

        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = int(self.next_yielded)
            self.next_yielded = None

        for position in range(self.yielded, self.num_samples):
            self.yielded += 1
            yield int(indices[position])

    def state_dict(self) -> dict:
        return {self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        yielded = int(state_dict[self._YIELDED])
        if yielded < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_yielded = yielded


def _validate_cycle_sampler_inputs(
    source_sizes: Sequence[int],
    source_weights: Sequence[float],
) -> None:
    if len(source_sizes) != len(source_weights):
        raise ValueError(
            "source_sizes and source_weights must have the same length: "
            f"{len(source_sizes)} vs {len(source_weights)}"
        )
    if not source_sizes:
        raise ValueError("Cycle sampler requires at least one source.")
    for src_idx, size in enumerate(source_sizes):
        if int(size) <= 0:
            raise ValueError(f"Source {src_idx} has zero lines; cannot sample.")
    weights_tensor = torch.tensor([float(w) for w in source_weights], dtype=torch.float64)
    if torch.any(weights_tensor < 0):
        raise ValueError(f"Source weights must be non-negative, got {source_weights}")
    if float(weights_tensor.sum().item()) <= 0:
        raise ValueError(f"Sum of source weights must be positive, got {source_weights}")


class WeightedShuffledCycleStatefulSampler(Sampler[int], Stateful):
    """Multi-source sampler with weighted cross-source mixing and per-source
    shuffled cycle: each source maintains a shuffled permutation traversed
    without replacement; once exhausted the source is reshuffled (a new pass)
    and the cycle continues.

    Each draw:
      1. ``source_idx`` ~ Categorical(weights)  (cross-source: with replacement)
      2. If ``cursor[source_idx] >= source_size[source_idx]``: increment
         ``pass_count[source_idx]`` and regenerate the deterministic permutation
         for that source.
      3. ``offset_idx = perm[source_idx][cursor[source_idx]]; cursor[source_idx] += 1``
      4. Yield ``source_idx * stride + offset_idx``.

    Each rank gets its own independent draw stream (seeded by ``(seed, rank)``)
    so distributed runs see different samples per rank without cross-rank
    coordination.

    Resume: ``state_dict()`` carries cursors, pass counts, and the categorical
    RNG state. The trainer's existing ``next_yielded`` hot-patch is also
    supported: when it's set, ``__iter__`` replays N no-op draws first to
    advance internal state to that step.
    """

    _YIELDED = "yielded"
    _CURSOR = "cursor"
    _PASS = "pass_count"
    _PICK_STATE = "pick_generator_state"
    _EPOCH = "epoch"

    def __init__(
        self,
        source_sizes: Sequence[int],
        source_weights: Sequence[float],
        *,
        stride: int,
        num_samples_per_rank: int,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        _validate_cycle_sampler_inputs(source_sizes, source_weights)
        self.source_sizes = [int(s) for s in source_sizes]
        self.weights = torch.tensor([float(w) for w in source_weights], dtype=torch.float64)
        self.stride = int(stride)
        if num_samples_per_rank <= 0:
            raise ValueError(f"num_samples_per_rank must be positive, got {num_samples_per_rank}")
        self._num_samples = int(num_samples_per_rank)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        self.cursor = [0 for _ in self.source_sizes]
        self.pass_count = [0 for _ in self.source_sizes]
        self.yielded = 0
        self.next_yielded: int | None = None

        self.pick_generator = torch.Generator()
        self._reseed_pick_generator()

        # Permutations are deterministic functions of (seed, rank, source_idx,
        # pass_count[source_idx]); we materialize them lazily and cache one per
        # source to avoid re-allocation on every draw.
        self._perm_cache: dict[int, np.ndarray] = {}
        for src_idx in range(len(self.source_sizes)):
            self._regen_perm(src_idx)

    def _reseed_pick_generator(self) -> None:
        # Per-rank seeding: distinct draw sequences across ranks while keeping
        # each rank reproducible.
        self.pick_generator.manual_seed(
            self.seed * 1_000_003 + self.epoch * 9973 + self.rank
        )

    def _regen_perm(self, source_idx: int) -> None:
        size = self.source_sizes[source_idx]
        rng = np.random.default_rng(
            (int(self.seed), int(self.rank), int(source_idx), int(self.pass_count[source_idx]))
        )
        self._perm_cache[source_idx] = rng.permutation(size).astype(np.int64, copy=False)

    def set_epoch(self, epoch: int) -> None:
        # Re-seed the cross-source picker per epoch so the source-mix sequence
        # is not identical across passes. We deliberately do NOT reset cursors
        # or pass counters: the user-requested semantic is that each source
        # cycles through its full content (shuffle on exhaust) regardless of
        # nominal "epoch" boundaries set by the trainer.
        self.epoch = int(epoch)
        self._reseed_pick_generator()

    def __len__(self) -> int:
        return self._num_samples

    def _draw_one(self) -> int:
        source_idx = int(
            torch.multinomial(self.weights, 1, replacement=True, generator=self.pick_generator).item()
        )
        if self.cursor[source_idx] >= self.source_sizes[source_idx]:
            self.pass_count[source_idx] += 1
            self._regen_perm(source_idx)
            self.cursor[source_idx] = 0
        offset_idx = int(self._perm_cache[source_idx][self.cursor[source_idx]])
        self.cursor[source_idx] += 1
        return source_idx * self.stride + offset_idx

    def __iter__(self) -> Iterator[int]:
        if self.next_yielded is not None:
            target = int(self.next_yielded)
            self.next_yielded = None
            # Replay as no-op draws so cursor / pass_count / pick RNG match the
            # state an uninterrupted run would have at step ``target``.
            for _ in range(target):
                self._draw_one()
            self.yielded = target
        else:
            self.yielded = 0

        while self.yielded < self._num_samples:
            idx = self._draw_one()
            self.yielded += 1
            yield idx

    def state_dict(self) -> dict:
        return {
            self._YIELDED: self.yielded,
            self._EPOCH: self.epoch,
            self._CURSOR: list(self.cursor),
            self._PASS: list(self.pass_count),
            self._PICK_STATE: self.pick_generator.get_state().tolist(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        has_full_state = self._CURSOR in state_dict
        if self._EPOCH in state_dict:
            self.epoch = int(state_dict[self._EPOCH])
        if self._CURSOR in state_dict:
            cursor = [int(c) for c in state_dict[self._CURSOR]]
            if len(cursor) != len(self.cursor):
                raise ValueError(
                    f"Cursor length mismatch: state has {len(cursor)} sources, "
                    f"sampler has {len(self.cursor)}."
                )
            self.cursor = cursor
        if self._PASS in state_dict:
            pass_count = [int(p) for p in state_dict[self._PASS]]
            if len(pass_count) != len(self.pass_count):
                raise ValueError(
                    f"Pass count length mismatch: state has {len(pass_count)} sources, "
                    f"sampler has {len(self.pass_count)}."
                )
            self.pass_count = pass_count
            for src_idx in range(len(self.source_sizes)):
                self._regen_perm(src_idx)
        if self._PICK_STATE in state_dict:
            pick_state = torch.tensor(state_dict[self._PICK_STATE], dtype=torch.uint8)
            self.pick_generator.set_state(pick_state)
        if self._YIELDED in state_dict:
            yielded = int(state_dict[self._YIELDED])
            if yielded < 0:
                raise ValueError("Cannot load state_dict with negative yielded value")
            if has_full_state:
                # Full state_dict path: cursor / pass_count / pick RNG already
                # capture the post-yielded position, so we must NOT replay.
                self.yielded = yielded
                self.next_yielded = None
            else:
                # Trainer hot-patch path (only yielded is supplied): cursors
                # are still at defaults, so __iter__ replays N draws.
                self.next_yielded = yielded


def _is_weighted_audio_dataset(dataset) -> bool:
    if not isinstance(dataset, AudioJsonlT2ADataset):
        return False
    if len(getattr(dataset, "sources", [])) > 1:
        return True
    weights = getattr(dataset, "source_weights", None)
    if not weights:
        return False
    if len(weights) == 1 and abs(float(weights[0]) - 1.0) > 1e-9:
        return True
    return False


def _is_weighted_av_paired_dataset(dataset) -> bool:
    """Mirror of :func:`_is_weighted_audio_dataset` for the joint AV
    paired dataset. We engage the weighted sampler when there is more
    than one source, or when the single source carries a non-1.0
    weight (in which case the weight only affects the batch size /
    data interleaving across runs but still goes through the
    deterministic shuffled-cycle path so resume is exact).
    """
    if not isinstance(dataset, AVPairedJsonlDataset):
        return False
    if len(getattr(dataset, "sources", [])) > 1:
        return True
    weights = getattr(dataset, "source_weights", None)
    if not weights:
        return False
    if len(weights) == 1 and abs(float(weights[0]) - 1.0) > 1e-9:
        return True
    return False


def build_train_dataloader(accelerator: Accelerator, dataset, config: dict):
    if not config["accelerate"].get("use_stateful_dataloader", True):
        raise ValueError("This trainer requires `accelerate.use_stateful_dataloader=true`.")

    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ImportError as exc:
        raise ImportError("Custom stateful dataloader resume requires `torchdata>=0.8.0`.") from exc

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": config["train"]["per_device_batch_size"],
        "num_workers": config["dataset"]["num_workers"],
        "pin_memory": config["dataset"]["pin_memory"] and accelerator.device.type == "cuda",
        "drop_last": config["dataset"]["drop_last"],
        "persistent_workers": config["dataset"]["num_workers"] > 0,
        "collate_fn": (
            collate_relaion_samples
            if isinstance(dataset, RelaionDataset)
            else collate_video_samples
            if isinstance(dataset, VideoJsonlDataset)
            else collate_audio_samples
            if isinstance(dataset, AudioJsonlT2ADataset)
            else collate_av_paired_samples
            if isinstance(dataset, AVPairedJsonlDataset)
            else collate_samples
        ),
        "snapshot_every_n_steps": int(config["accelerate"].get("stateful_snapshot_every_n_steps", 1)),
    }
    if int(config["dataset"]["num_workers"]) > 0:
        dataloader_kwargs["prefetch_factor"] = int(config["dataset"].get("prefetch_factor", 2))

    seed = config["train"].get("seed")
    use_weighted = _is_weighted_audio_dataset(dataset) or _is_weighted_av_paired_dataset(dataset)
    if use_weighted:
        source_sizes = list(dataset.source_sizes)
        source_weights = list(dataset.source_weights)
        stride = int(getattr(dataset, "stride"))
        if accelerator.num_processes > 1:
            num_samples_per_rank = math.ceil(len(dataset) / accelerator.num_processes)
        else:
            num_samples_per_rank = len(dataset)
        dataloader_kwargs["sampler"] = WeightedShuffledCycleStatefulSampler(
            source_sizes,
            source_weights,
            stride=stride,
            num_samples_per_rank=num_samples_per_rank,
            rank=accelerator.process_index,
            seed=0 if seed is None else int(seed),
        )
        dataloader_kwargs["shuffle"] = False
    elif accelerator.num_processes > 1:
        dataloader_kwargs["sampler"] = TensorStatefulDistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=0 if seed is None else int(seed),
            drop_last=config["dataset"]["drop_last"],
        )
        dataloader_kwargs["shuffle"] = False
    else:
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        dataloader_kwargs["sampler"] = TensorStatefulRandomSampler(dataset, generator=generator)
        dataloader_kwargs["shuffle"] = False

    return StatefulDataLoader(**dataloader_kwargs)


def get_dataloader_state_path(checkpoint_dir: Path, process_index: int) -> Path:
    return checkpoint_dir / f"dataloader_state_rank{process_index}.bin"


def save_dataloader_state(train_dataloader, checkpoint_dir: Path, process_index: int) -> None:
    state_path = get_dataloader_state_path(checkpoint_dir, process_index)
    torch.save(train_dataloader.state_dict(), state_path)


def load_dataloader_state(train_dataloader, checkpoint_dir: Path, process_index: int) -> None:
    state_path = get_dataloader_state_path(checkpoint_dir, process_index)
    if not state_path.exists():
        raise ValueError(
            f"Checkpoint {checkpoint_dir} is missing {state_path.name}. "
            "Every rank must have its own saved dataloader state."
        )
    train_dataloader.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))


def load_resume_metadata(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(
            f"Checkpoint {checkpoint_dir} is missing metadata.json. "
            f"Only checkpoints created with `{DATALOADER_RESUME_STRATEGY}` are supported."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataloader_resume_strategy") != DATALOADER_RESUME_STRATEGY:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} was not created with `{DATALOADER_RESUME_STRATEGY}`. "
            "Old checkpoints are not supported by this stateful dataloader resume flow."
    )
    return metadata


def resolve_latest_resume_checkpoint(checkpoint_root: Path) -> Path | None:
    latest_complete = find_latest_complete_checkpoint(checkpoint_root)
    if latest_complete is not None:
        return latest_complete

    if not checkpoint_root.exists():
        return None

    # FIXME: Remove this legacy metadata-only fallback after older checkpoints without
    # `.checkpoint_complete` markers have been migrated or aged out.
    checkpoints = sorted(
        [path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda path: path.name,
        reverse=True,
    )
    for checkpoint in checkpoints:
        try:
            load_resume_metadata(checkpoint)
        except ValueError:
            continue
        return checkpoint
    return None


def restore_training_state(
    accelerator: Accelerator,
    train_dataloader,
    checkpoint_root: Path,
    resume_path: str | None,
    *,
    num_update_steps_per_epoch: int,
    persistent_checkpoint_root: Path | None = None,
) -> ResumeState:
    if not resume_path:
        return ResumeState(checkpoint_path=None, global_step=0, first_epoch=0)

    resolved_root = checkpoint_root
    if resume_path == "latest_persistent":
        if persistent_checkpoint_root is None:
            raise ValueError("`latest_persistent` requires `persistent_checkpoint_root` to be provided.")
        resolved_root = persistent_checkpoint_root
        resume_path = "latest"

    if resume_path == "latest":
        resolved = resolve_latest_resume_checkpoint(resolved_root)
        if resolved is None:
            return ResumeState(checkpoint_path=None, global_step=0, first_epoch=0)
        resume_checkpoint = resolved
    else:
        resume_checkpoint = Path(resume_path)

    load_resume_metadata(resume_checkpoint)
    accelerator.load_state(str(resume_checkpoint))
    # load_dataloader_state(train_dataloader, resume_checkpoint, accelerator.process_index)

    global_step = int(resume_checkpoint.name.split("-")[-1])

    current_step = global_step % num_update_steps_per_epoch
    train_dataloader.sampler.next_yielded = (
        current_step
        * int(train_dataloader.batch_size or 1)
        * int(accelerator.gradient_accumulation_steps)
    )
    
    first_epoch = global_step // num_update_steps_per_epoch
    if global_step > 0 and global_step % num_update_steps_per_epoch == 0:
        # A dataloader saved exactly at epoch end would resume as an exhausted iterator.
        train_dataloader.load_state_dict({})

    return ResumeState(
        checkpoint_path=resume_checkpoint,
        global_step=global_step,
        first_epoch=first_epoch,
    )
