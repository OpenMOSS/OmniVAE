"""Distributed bookkeeping.

Mirrors the pattern used by the T2AV release evaluation launchers:
* Read RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT (set by torchrun).
* Initialise a gloo process group when WORLD_SIZE > 1. Gloo is enough because the
  collectives we use are barriers and small all_gather_object payloads, and it
  doesn't fight with each metric's GPU usage.
* Provide a few convenience wrappers (barrier, broadcast scalar, all_gather_object).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, List, Optional

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int]:
    """Initialise the process group (gloo) iff WORLD_SIZE > 1.

    Returns (rank, local_rank, world_size).
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )
    return rank, local_rank, world_size


def barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def broadcast_int(value: int, world_size: int, src: int = 0) -> int:
    if world_size <= 1 or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.long)
    dist.broadcast(tensor, src=src)
    return int(tensor.item())


def all_gather_object(obj: Any, world_size: int) -> List[Any]:
    if world_size <= 1 or not dist.is_initialized():
        return [obj]
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def broadcast_object(obj: Any, world_size: int, src: int = 0) -> Any:
    if world_size <= 1 or not dist.is_initialized():
        return obj
    payload = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def slice_for_rank(items: list, rank: int, world_size: int) -> list:
    """Standard rank-strided slice (tasks[rank::world_size])."""
    if world_size <= 1:
        return list(items)
    return items[rank::world_size]


def log(rank: int, message: str) -> None:
    ts = datetime.now().strftime("%F %T")
    print(f"[{ts}][rank{rank}] {message}", flush=True)
