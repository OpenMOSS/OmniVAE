from __future__ import annotations

import os
import socket
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor
try:
    from torch.optim._muon import (
        _adjust_lr as _muon_adjust_lr,
        _zeropower_via_newtonschulz as _muon_zeropower_via_newtonschulz,
        muon as torch_muon,
    )
except ModuleNotFoundError:
    _MUON_IMPORT_ERROR = RuntimeError(
        "torch.optim._muon is unavailable in this PyTorch build. "
        "HybridMuonAdamw requires a PyTorch build that includes Muon support."
    )

    def _raise_missing_muon(*args, **kwargs):
        raise _MUON_IMPORT_ERROR

    _muon_adjust_lr = _raise_missing_muon
    _muon_zeropower_via_newtonschulz = _raise_missing_muon
    torch_muon = _raise_missing_muon
from torch.optim.adamw import adamw as torch_adamw
from torch.optim.optimizer import _get_capturable_supported_devices


_CAPTURABLE_SUPPORTED_DEVICES = frozenset(_get_capturable_supported_devices())
_MUON_SHARD_PROCESS_GROUP_CACHE: dict[tuple[int, ...], Any] = {}


def _as_0d_tensor(value: float | Tensor) -> Tensor:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("Learning rate tensor must contain exactly one element.")
        lr_tensor = value.detach().clone()
        if lr_tensor.ndim != 0:
            lr_tensor = lr_tensor.reshape(())
        if not lr_tensor.is_floating_point():
            lr_tensor = lr_tensor.to(dtype=torch.float32)
        return lr_tensor
    return torch.tensor(float(value), dtype=torch.float32)


def _can_use_capturable(params: Sequence[Tensor]) -> bool:
    return len(params) > 0 and all(p.device.type in _CAPTURABLE_SUPPORTED_DEVICES for p in params)


def _can_shard_muon_across_ranks() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _normalize_muon_shard_across_ranks(raw_value: Any) -> int:
    value = 0 if raw_value is None else int(raw_value)
    if value < 0:
        raise ValueError(
            "train.muon_shard_across_ranks must be 0 (all ranks), 1 (disable sharding), "
            f"or a positive divisor of world_size; got {value}."
        )
    return value


def _resolve_muon_shard_world_size(muon_shard_across_ranks: int, world_size: int) -> int:
    if world_size <= 1:
        return 1
    if muon_shard_across_ranks == 0:
        return world_size
    if muon_shard_across_ranks == 1:
        return 1
    shard_world_size = min(muon_shard_across_ranks, world_size)
    if world_size % shard_world_size != 0:
        raise ValueError(
            "train.muon_shard_across_ranks must be 0 (all ranks), 1 (disable sharding), "
            "or clamp to a positive divisor of "
            f"world_size={world_size}; got {muon_shard_across_ranks}, effective={shard_world_size}."
        )
    return shard_world_size


def _maybe_get_env_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return None
    return int(raw_value)


def _gather_muon_rank_metadata() -> list[dict[str, Any]]:
    rank = dist.get_rank()
    explicit_node_rank = None
    for env_name in ("GROUP_RANK", "NODE_RANK", "MACHINE_RANK"):
        explicit_node_rank = _maybe_get_env_int(env_name)
        if explicit_node_rank is not None:
            break

    local_rank = _maybe_get_env_int("LOCAL_RANK")
    local_metadata = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "node_rank": explicit_node_rank,
        "local_rank": local_rank,
    }
    gathered_metadata: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_metadata, local_metadata)
    if any(item is None for item in gathered_metadata):
        raise RuntimeError("Failed to gather Muon shard rank metadata from all ranks.")
    return [item for item in gathered_metadata if item is not None]


def _order_node_members(rank_metadata: list[dict[str, Any]]) -> list[int]:
    local_ranks = [item["local_rank"] for item in rank_metadata]
    if all(local_rank is not None for local_rank in local_ranks) and len(set(local_ranks)) == len(local_ranks):
        return [item["rank"] for item in sorted(rank_metadata, key=lambda item: (item["local_rank"], item["rank"]))]
    return [item["rank"] for item in sorted(rank_metadata, key=lambda item: item["rank"])]


def _build_node_priority_muon_rank_groups(shard_world_size: int) -> list[tuple[int, ...]]:
    world_size = dist.get_world_size()
    if shard_world_size == world_size:
        return [tuple(range(world_size))]

    gathered_metadata = _gather_muon_rank_metadata()
    per_node_ranks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in gathered_metadata:
        node_rank = item["node_rank"]
        node_key = f"node_rank:{node_rank}" if node_rank is not None else f"hostname:{item['hostname']}"
        per_node_ranks[node_key].append(item)

    explicit_node_order_available = True
    node_order_by_key: dict[str, int] = {}
    for node_key, node_items in per_node_ranks.items():
        node_ranks = {item["node_rank"] for item in node_items}
        if len(node_ranks) != 1 or None in node_ranks:
            explicit_node_order_available = False
            break
        node_order_by_key[node_key] = next(iter(node_ranks))
    if explicit_node_order_available and len(set(node_order_by_key.values())) != len(node_order_by_key):
        explicit_node_order_available = False

    if explicit_node_order_available:
        ordered_node_keys = sorted(
            per_node_ranks.keys(),
            key=lambda node_key: (node_order_by_key[node_key], min(item["rank"] for item in per_node_ranks[node_key])),
        )
    else:
        ordered_node_keys = sorted(
            per_node_ranks.keys(),
            key=lambda node_key: min(item["rank"] for item in per_node_ranks[node_key]),
        )

    ordered_ranks: list[int] = []
    for node_key in ordered_node_keys:
        ordered_ranks.extend(_order_node_members(per_node_ranks[node_key]))

    if len(ordered_ranks) != world_size or set(ordered_ranks) != set(range(world_size)):
        raise RuntimeError("Failed to build a valid node-priority Muon shard rank layout.")
    return [tuple(ordered_ranks[start : start + shard_world_size]) for start in range(0, world_size, shard_world_size)]


def _get_muon_shard_process_group(shard_world_size: int) -> tuple[Any, int]:
    rank = dist.get_rank()
    if shard_world_size == dist.get_world_size():
        return dist.group.WORLD, rank
    rank_groups = _build_node_priority_muon_rank_groups(shard_world_size)
    current_group_ranks: tuple[int, ...] | None = None
    current_shard_rank = 0

    for rank_group in rank_groups:
        if rank_group not in _MUON_SHARD_PROCESS_GROUP_CACHE:
            _MUON_SHARD_PROCESS_GROUP_CACHE[rank_group] = dist.new_group(ranks=list(rank_group))
        if rank in rank_group:
            current_group_ranks = rank_group
            current_shard_rank = rank_group.index(rank)

    if current_group_ranks is None:
        raise RuntimeError(f"Rank {rank} was not assigned to any Muon shard process group.")
    process_group = _MUON_SHARD_PROCESS_GROUP_CACHE[current_group_ranks]
    return process_group, current_shard_rank

# @torch.compile(dynamic=True)
def _muon_update(
    grad: Tensor,
    momentum_buffer: Tensor,
    *,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    momentum_buffer.lerp_(grad, 1 - momentum)
    update = grad.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    return _muon_zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)


def _muon_bucket_sort_key(item: tuple[tuple[tuple[int, ...], str, str], list[Tensor]]) -> tuple[Any, ...]:
    shape, dtype_str, device_str = item[0]
    return (len(shape), shape, dtype_str, device_str)


def _is_embedding_like_param_name(name: str) -> bool:
    n = str(name).lower()
    return bool(
        ("pos_embed" in n)
        or ("position_embedding" in n)
        or ("position_embeddings" in n)
        or ("embed_tokens" in n)
        or ("tok_embeddings" in n)
        or ("token_embedding" in n)
        or ("token_embeddings" in n)
        or ("word_embedding" in n)
        or ("word_embeddings" in n)
        or (n.startswith("embedding."))
        or (n.startswith("embeddings."))
        or (".embedding." in n)
        or (".embeddings." in n)
        or (n.endswith(".embedding.weight"))
        or (n.endswith(".embeddings.weight"))
        or ("query_embed" in n)
        or ("queries" in n)
    )


def _is_output_head_like_param_name(name: str) -> bool:
    n = str(name).lower()
    return bool(
        n.startswith("lm_head.")
        or n.startswith("classifier.")
        or n.startswith("class_head.")
        or n.startswith("cls_head.")
        or n.startswith("to_logits.")
        or n.startswith("logits_head.")
        or n.startswith("output_head.")
    )


def _is_no_decay_param(name: str, p: torch.nn.Parameter, *, exclude_embedding_from_wd: bool) -> bool:
    n = name.lower()
    is_bias = n.endswith(".bias") or n == "bias"
    looks_like_norm = ("norm" in n) or (".bn" in n) or ("_bn" in n) or ("layernorm" in n) or (".ln" in n)
    is_1d_or_scalar = p.ndim <= 1
    looks_like_embedding = bool(exclude_embedding_from_wd and _is_embedding_like_param_name(n))
    return bool(is_bias or looks_like_norm or is_1d_or_scalar or looks_like_embedding)


def _should_route_to_adamw(name: str, p: torch.nn.Parameter, *, muon_adamw_prefixes: list[str]) -> bool:
    nl = str(name).lower()
    if p.ndim != 2:
        return True
    if _is_embedding_like_param_name(nl):
        return True
    if _is_output_head_like_param_name(nl):
        return True
    if any(nl.startswith(prefix) for prefix in muon_adamw_prefixes):
        return True
    return False


def _build_hybrid_param_groups(
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    lr: float | Tensor,
    weight_decay: float,
    exclude_norm_and_bias_from_wd: bool,
    exclude_embedding_from_wd: bool,
    muon_adamw_prefixes: list[str],
    adam_betas: tuple[float, float],
    adam_eps: float,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_ns_coefficients: tuple[float, float, float],
    muon_ns_steps: int,
    muon_adjust_lr_fn: str | None,
    muon_eps: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    buckets: dict[tuple[str, str], list[torch.nn.Parameter]] = {
        ("muon", "decay"): [],
        ("muon", "no_decay"): [],
        ("adamw", "decay"): [],
        ("adamw", "no_decay"): [],
    }
    bucket_numel: dict[tuple[str, str], int] = {key: 0 for key in buckets}
    lr_tensor = _as_0d_tensor(lr)

    for name, p in named_params:
        if not p.requires_grad:
            continue
        backend = "adamw" if _should_route_to_adamw(name, p, muon_adamw_prefixes=muon_adamw_prefixes) else "muon"
        no_decay = False
        if exclude_norm_and_bias_from_wd and weight_decay > 0:
            no_decay = _is_no_decay_param(name, p, exclude_embedding_from_wd=exclude_embedding_from_wd)
        kind = "no_decay" if no_decay else "decay"
        buckets[(backend, kind)].append(p)
        bucket_numel[(backend, kind)] += int(p.numel())

    param_groups: list[dict[str, Any]] = []
    for backend in ("muon", "adamw"):
        for kind in ("decay", "no_decay"):
            params = buckets[(backend, kind)]
            if len(params) == 0:
                continue
            group: dict[str, Any] = {
                "params": params,
                "lr": lr_tensor.clone(),
                "weight_decay": float(weight_decay if kind == "decay" else 0.0),
                "backend": backend,
                "use_muon": backend == "muon",
                "maximize": False,
                "foreach": None,
                "capturable": backend == "adamw",
                "differentiable": False,
                "fused": None,
            }
            if backend == "muon":
                group.update(
                    {
                        "momentum": float(muon_momentum),
                        "nesterov": bool(muon_nesterov),
                        "ns_coefficients": tuple(float(x) for x in muon_ns_coefficients),
                        "ns_steps": int(muon_ns_steps),
                        "adjust_lr_fn": muon_adjust_lr_fn,
                        "eps": float(muon_eps),
                    }
                )
            else:
                group.update(
                    {
                        "betas": tuple(float(x) for x in adam_betas),
                        "eps": float(adam_eps),
                        "amsgrad": False,
                    }
                )
            param_groups.append(group)

    stats = {
        "muon_tensors": float(len(buckets[("muon", "decay")]) + len(buckets[("muon", "no_decay")])),
        "adamw_tensors": float(len(buckets[("adamw", "decay")]) + len(buckets[("adamw", "no_decay")])),
        "muon_numel": float(bucket_numel[("muon", "decay")] + bucket_numel[("muon", "no_decay")]),
        "adamw_numel": float(bucket_numel[("adamw", "decay")] + bucket_numel[("adamw", "no_decay")]),
    }
    return param_groups, stats


class HybridMuonAdamw(torch.optim.Optimizer):
    """Single optimizer wrapping Muon + AdamW with robust parameter routing."""

    def __init__(
        self,
        named_params: Sequence[tuple[str, torch.nn.Parameter]],
        train_cfg: dict,
        learning_rate: float | Tensor,
    ):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "Configured train.optimizer='Hybrid', but torch.optim.Muon is unavailable in this PyTorch build."
            )

        muon_adamw_prefixes = [
            str(x).strip().lower() for x in (train_cfg.get("muon_adamw_prefixes") or []) if str(x).strip()
        ]
        muon_ns_coefficients = train_cfg.get("muon_ns_coefficients") or (3.4445, -4.775, 2.0315)
        param_groups, stats = _build_hybrid_param_groups(
            named_params,
            lr=learning_rate,
            weight_decay=float(train_cfg["adam_weight_decay"]),
            exclude_norm_and_bias_from_wd=bool(train_cfg.get("exclude_norm_and_bias_from_wd", True)),
            exclude_embedding_from_wd=bool(train_cfg.get("exclude_embedding_from_wd", True)),
            muon_adamw_prefixes=muon_adamw_prefixes,
            adam_betas=(float(train_cfg["adam_beta1"]), float(train_cfg["adam_beta2"])),
            adam_eps=float(train_cfg["adam_epsilon"]),
            muon_momentum=float(train_cfg.get("muon_momentum", 0.95)),
            muon_nesterov=bool(train_cfg.get("muon_nesterov", True)),
            muon_ns_coefficients=tuple(float(x) for x in muon_ns_coefficients),
            muon_ns_steps=int(train_cfg.get("muon_ns_steps", 5)),
            muon_adjust_lr_fn=train_cfg.get("muon_adjust_lr_fn"),
            muon_eps=float(train_cfg.get("muon_eps", 1e-7)),
        )
        if stats["muon_tensors"] <= 0:
            raise RuntimeError(
                "Configured train.optimizer='Hybrid' but no eligible Muon parameters were found "
                "(all params were routed to AdamW)."
            )

        defaults = {"lr": _as_0d_tensor(learning_rate)}
        super().__init__(param_groups, defaults)
        self.muon_tensors = int(stats["muon_tensors"])
        self.adamw_tensors = int(stats["adamw_tensors"])
        self.muon_numel = int(stats["muon_numel"])
        self.adamw_numel = int(stats["adamw_numel"])
        raw_muon_shard_across_ranks = train_cfg.get("muon_shard_across_ranks")
        if raw_muon_shard_across_ranks is None:
            raw_muon_shard_across_ranks = train_cfg.get("muon_shard_accross_rank")
        if raw_muon_shard_across_ranks is None:
            raw_muon_shard_across_ranks = train_cfg.get("muon_shard_accross_ranks")
        self.muon_shard_across_ranks = _normalize_muon_shard_across_ranks(raw_muon_shard_across_ranks)
        raw_muon_max_inflight_buckets = train_cfg.get("muon_max_inflight_buckets", 1)
        self.muon_max_inflight_buckets = int(
            1 if raw_muon_max_inflight_buckets is None else raw_muon_max_inflight_buckets
        )
        if self.muon_max_inflight_buckets < -1:
            raise ValueError(
                "train.muon_max_inflight_buckets must be -1 (unbounded) or a non-negative integer, "
                f"got {self.muon_max_inflight_buckets}."
            )
        self._muon_dist_world_size = 1
        self._muon_dist_rank = 0
        self._muon_shard_process_group: Any = None
        self._muon_shard_group_name: str | None = None
        self._muon_shard_world_size = 1
        self._muon_shard_rank = 0
        self._muon_shard_runtime: list[list[dict[str, Any]] | None] = [None for _ in self.param_groups]

        for group in self.param_groups:
            backend = str(group.get("backend", "")).lower().strip()
            if backend not in {"muon", "adamw"}:
                raise ValueError(f"HybridMuonAdamw only supports backends ['muon', 'adamw'], got: {backend!r}")
            if backend == "muon":
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            "Muon param groups must only contain 2D parameters, "
                            f"but found parameter with shape={tuple(p.shape)}"
                        )

        self._normalize_group_tensors()
        self._preinitialize_state()
        self._refresh_muon_shard_runtime()

    def _get_effective_muon_shard_world_size(self) -> int:
        if not _can_shard_muon_across_ranks():
            return 1
        return _resolve_muon_shard_world_size(self.muon_shard_across_ranks, dist.get_world_size())

    def _should_shard_muon(self) -> bool:
        return self._get_effective_muon_shard_world_size() > 1

    def _normalize_group_tensors(self) -> None:
        self.defaults["lr"] = _as_0d_tensor(self.defaults["lr"])
        for group in self.param_groups:
            backend = str(group.get("backend", "")).lower().strip()
            if backend not in {"muon", "adamw"}:
                backend = "muon" if bool(group.get("use_muon", False)) else "adamw"
                group["backend"] = backend
            group["use_muon"] = bool(group.get("use_muon", backend == "muon"))
            group["lr"] = _as_0d_tensor(group["lr"])
            if "initial_lr" in group:
                group["initial_lr"] = _as_0d_tensor(group["initial_lr"])

    def _preinitialize_state(self) -> None:
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    state.setdefault(
                        "momentum_buffer",
                        torch.zeros_like(p, memory_format=torch.preserve_format),
                    )
                continue

            use_param_device_for_step = bool(group["capturable"] or group["fused"])
            for p in group["params"]:
                state = self.state[p]
                if "step" not in state:
                    step_device = p.device if use_param_device_for_step else torch.device("cpu")
                    state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
                elif use_param_device_for_step and state["step"].device != p.device:
                    state["step"] = state["step"].to(device=p.device)

                state.setdefault(
                    "exp_avg",
                    torch.zeros_like(p, memory_format=torch.preserve_format),
                )
                state.setdefault(
                    "exp_avg_sq",
                    torch.zeros_like(p, memory_format=torch.preserve_format),
                )
                if group["amsgrad"]:
                    state.setdefault(
                        "max_exp_avg_sq",
                        torch.zeros_like(p, memory_format=torch.preserve_format),
                    )

    def _refresh_muon_shard_runtime(self) -> None:
        if not _can_shard_muon_across_ranks():
            self._muon_dist_world_size = 1
            self._muon_dist_rank = 0
            self._muon_shard_process_group = None
            self._muon_shard_group_name = None
            self._muon_shard_world_size = 1
            self._muon_shard_rank = 0
            self._muon_shard_runtime = [None for _ in self.param_groups]
            return

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        shard_world_size = _resolve_muon_shard_world_size(self.muon_shard_across_ranks, world_size)
        self._muon_dist_world_size = world_size
        self._muon_dist_rank = rank
        if shard_world_size <= 1:
            self._muon_shard_process_group = None
            self._muon_shard_group_name = None
            self._muon_shard_world_size = 1
            self._muon_shard_rank = 0
            self._muon_shard_runtime = [None for _ in self.param_groups]
            return

        process_group, shard_rank = _get_muon_shard_process_group(shard_world_size)
        self._muon_shard_process_group = process_group
        self._muon_shard_group_name = funcol._resolve_group_name(process_group)
        self._muon_shard_world_size = shard_world_size
        self._muon_shard_rank = shard_rank
        runtime: list[list[dict[str, Any]] | None] = []

        for group in self.param_groups:
            if not group["use_muon"]:
                runtime.append(None)
                continue

            buckets: dict[tuple[tuple[int, ...], str, str], list[Tensor]] = {}
            for p in group["params"]:
                key = (tuple(p.shape), str(p.dtype), str(p.device))
                buckets.setdefault(key, []).append(p)

            group_runtime: list[dict[str, Any]] = []
            for _, params in sorted(buckets.items(), key=_muon_bucket_sort_key):
                if len(params) == 0:
                    continue
                local_slots = (len(params) + self._muon_shard_world_size - 1) // self._muon_shard_world_size
                local_slot_indices: list[int] = []
                local_params: list[Tensor] = []
                local_momentum_buffers: list[Tensor] = []
                for slot_idx in range(local_slots):
                    param_idx = slot_idx * self._muon_shard_world_size + self._muon_shard_rank
                    if param_idx >= len(params):
                        continue
                    p = params[param_idx]
                    local_slot_indices.append(slot_idx)
                    local_params.append(p)
                    local_momentum_buffers.append(self.state[p]["momentum_buffer"])

                param_template = params[0]
                group_runtime.append(
                    {
                        "params": params,
                        "param_count": len(params),
                        "local_slots": local_slots,
                        "local_slot_indices": local_slot_indices,
                        "local_params": local_params,
                        "local_momentum_buffers": local_momentum_buffers,
                        "zero_grad": torch.zeros_like(param_template, memory_format=torch.preserve_format),
                        "local_buffer": torch.zeros(
                            (local_slots, *param_template.shape),
                            dtype=param_template.dtype,
                            device=param_template.device,
                        ),
                        "param_shape": tuple(param_template.shape),
                        "lr_scale": float(_muon_adjust_lr(1.0, group["adjust_lr_fn"], param_template.shape)),
                    }
                )
            runtime.append(group_runtime)

        self._muon_shard_runtime = runtime

    def _ensure_muon_shard_runtime(self) -> None:
        expected_shard_world_size = self._get_effective_muon_shard_world_size()
        if (
            self._muon_shard_world_size != expected_shard_world_size
            or self._muon_dist_world_size != (dist.get_world_size() if _can_shard_muon_across_ranks() else 1)
            or self._muon_dist_rank != (dist.get_rank() if _can_shard_muon_across_ranks() else 0)
            or (expected_shard_world_size > 1 and self._muon_shard_group_name is None)
            or (expected_shard_world_size <= 1 and self._muon_shard_group_name is not None)
            or len(self._muon_shard_runtime) != len(self.param_groups)
        ):
            self._refresh_muon_shard_runtime()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._normalize_group_tensors()
        self._preinitialize_state()
        self._refresh_muon_shard_runtime()

    def _collect_muon_group(self, group: dict[str, Any]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_bufs: list[Tensor] = []

        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]
            momentum_bufs.append(state["momentum_buffer"])

        return params_with_grad, grads, momentum_bufs

    def _finish_muon_shard_bucket(self, bucket: dict[str, Any], gathered: Tensor) -> None:
        gathered = torch.ops._c10d_functional.wait_tensor(gathered)
        ordered = gathered.view(
            self._muon_shard_world_size,
            bucket["local_slots"],
            *bucket["param_shape"],
        )
        ordered = ordered.transpose(0, 1).reshape(
            bucket["local_slots"] * self._muon_shard_world_size,
            *bucket["param_shape"],
        )
        torch._foreach_copy_(
            bucket["params"],
            ordered[: bucket["param_count"]].unbind(0),
        )
    # @torch._dynamo.disable()
    def _step_muon_group_sharded(self, group_idx: int, group: dict[str, Any]) -> None:
        bucket_runtime = self._muon_shard_runtime[group_idx]
        if bucket_runtime is None or len(bucket_runtime) == 0 or self._muon_shard_group_name is None:
            return

        lr = group["lr"]
        weight_decay = float(group["weight_decay"])
        momentum = float(group["momentum"])
        nesterov = bool(group["nesterov"])
        ns_coefficients = tuple(group["ns_coefficients"])
        ns_steps = int(group["ns_steps"])
        eps = float(group["eps"])
        max_inflight_buckets = self.muon_max_inflight_buckets
        pending_bucket_gathers: list[tuple[dict[str, Any], Tensor]] = []
        pending_start = 0
        for bucket in bucket_runtime:
            local_buffer = bucket["local_buffer"]
            local_buffer.zero_()
            adjusted_lr = lr * bucket["lr_scale"]

            for slot_idx, p, momentum_buffer in zip(
                bucket["local_slot_indices"],
                bucket["local_params"],
                bucket["local_momentum_buffers"],
            ):
                grad = p.grad if p.grad is not None else bucket["zero_grad"]
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                update = _muon_update(
                    grad,
                    momentum_buffer,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_coefficients=ns_coefficients,
                    ns_steps=ns_steps,
                    eps=eps,
                )
                p.mul_(1 - lr * weight_decay)
                p.add_(update * adjusted_lr.neg())
                local_buffer[slot_idx].copy_(p)

            gathered = torch.ops._c10d_functional.all_gather_into_tensor(
                local_buffer,
                self._muon_shard_world_size,
                self._muon_shard_group_name,
            )
            # torch._dynamo.graph_break()
            pending_bucket_gathers.append((bucket, gathered))

            if max_inflight_buckets >= 0:
                while len(pending_bucket_gathers) - pending_start > max_inflight_buckets:
                    finished_bucket, finished_gather = pending_bucket_gathers[pending_start]
                    pending_start += 1
                    self._finish_muon_shard_bucket(finished_bucket, finished_gather)

        for pending_idx in range(pending_start, len(pending_bucket_gathers)):
            finished_bucket, finished_gather = pending_bucket_gathers[pending_idx]
            self._finish_muon_shard_bucket(finished_bucket, finished_gather)

    def _collect_adamw_group(
        self, group: dict[str, Any]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor], bool]:
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        max_exp_avg_sqs: list[Tensor] = []
        state_steps: list[Tensor] = []
        has_complex = False

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            state_steps.append(state["step"])

        return params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, has_complex

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            if group["use_muon"]:
                if self._should_shard_muon():
                    self._ensure_muon_shard_runtime()
                    self._step_muon_group_sharded(group_idx, group)
                    continue
                params_with_grad, grads, momentum_bufs = self._collect_muon_group(group)
                if len(params_with_grad) == 0:
                    continue
                torch_muon(
                    params_with_grad,
                    grads,
                    momentum_bufs,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    nesterov=group["nesterov"],
                    ns_coefficients=group["ns_coefficients"],
                    ns_steps=group["ns_steps"],
                    eps=group["eps"],
                    adjust_lr_fn=group["adjust_lr_fn"],
                    has_complex=False,
                )
                continue

            (
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                has_complex,
            ) = self._collect_adamw_group(group)
            if len(params_with_grad) == 0:
                continue

            beta1, beta2 = group["betas"]
            capturable = bool(group["capturable"]) and _can_use_capturable(params_with_grad)
            torch_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                foreach=group["foreach"],
                capturable=capturable,
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

        return loss

    def describe(self) -> str:
        inflight_desc = "inf" if self.muon_max_inflight_buckets < 0 else str(self.muon_max_inflight_buckets)
        shard_desc = "all" if self.muon_shard_across_ranks == 0 else str(self.muon_shard_across_ranks)
        return (
            f"HybridMuonAdamw(Muon={self.muon_tensors} tensors/{self.muon_numel} numel, "
            f"AdamW={self.adamw_tensors} tensors/{self.adamw_numel} numel, "
            f"muon_shard_across_ranks={shard_desc}, "
            f"muon_max_inflight_buckets={inflight_desc})"
        )
