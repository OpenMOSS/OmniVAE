"""Heterogeneous-LR variant of ``omnivae_generation.trainer.optim.HybridMuonAdamw``.

We keep the same Muon / AdamW routing logic as the parent (so 2D
weights still go to Muon and embeddings/heads/biases still go to
AdamW), but additionally bucket each parameter by a *tag* string
provided alongside its name. Each ``(backend, decay, tag)`` combination
becomes its own ``param_group`` with an independent ``lr`` tensor.

This lets us train the bridge module with a higher LR
(``bridge_lr=2e-5``) than the pretrained backbone (``backbone_lr=5e-6``)
without forking the entire optimizer.

Compatible with the existing ``diffusers.optimization.get_scheduler``
LambdaLR path: each group's ``initial_lr`` is set on first scheduler
step so the warmup/cosine schedules walk per-group.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from omnivae_generation.trainer.optim import (
    HybridMuonAdamw,
    _as_0d_tensor,
    _build_hybrid_param_groups,
    _normalize_muon_shard_across_ranks,
)


# Public tags used by the trainer; kept here so trainer/validation/loader
# all share a single source of truth for legal tag values.
TAG_BACKBONE = "backbone"
TAG_BRIDGE = "bridge"
DEFAULT_TAGS: tuple[str, ...] = (TAG_BACKBONE, TAG_BRIDGE)


def _split_named_params_by_tag(
    tagged_named_params: Sequence[tuple[str, str, torch.nn.Parameter]],
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Group ``(tag, name, param)`` triplets into ``{tag: [(name, param), ...]}``."""
    by_tag: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
    for tag, name, param in tagged_named_params:
        if not param.requires_grad:
            continue
        by_tag.setdefault(str(tag), []).append((str(name), param))
    return by_tag


def _build_tagged_param_groups(
    tagged_named_params: Sequence[tuple[str, str, torch.nn.Parameter]],
    *,
    base_lr: float | Tensor,
    per_tag_lr: dict[str, float | Tensor],
    train_cfg: dict,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Reuse ``_build_hybrid_param_groups`` per-tag, then re-tag each
    resulting group's ``lr`` and ``tag`` field. Returns the merged
    groups and per-tag stats for logging."""
    by_tag = _split_named_params_by_tag(tagged_named_params)
    if not by_tag:
        raise ValueError("No tagged trainable parameters provided to optimizer.")

    muon_adamw_prefixes = [
        str(x).strip().lower() for x in (train_cfg.get("muon_adamw_prefixes") or []) if str(x).strip()
    ]
    muon_ns_coefficients = train_cfg.get("muon_ns_coefficients") or (3.4445, -4.775, 2.0315)

    param_groups: list[dict[str, Any]] = []
    per_tag_stats: dict[str, dict[str, float]] = {}

    for tag, named_params in by_tag.items():
        lr_value = per_tag_lr.get(tag, base_lr)
        groups, stats = _build_hybrid_param_groups(
            named_params,
            lr=lr_value,
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
        for group in groups:
            group["tag"] = tag
            param_groups.append(group)
        per_tag_stats[tag] = stats

    return param_groups, per_tag_stats


class HybridMuonAdamwTagged(HybridMuonAdamw):
    """``HybridMuonAdamw`` with per-tag LR groups.

    Inherits the entire ``step`` / ``load_state_dict`` / Muon-shard
    machinery from the parent; only the constructor changes to accept
    ``tagged_named_params`` and a ``per_tag_lr`` dict.

    Backward compatibility: a tag absent from ``per_tag_lr`` falls back
    to ``learning_rate``. Tags listed in ``per_tag_lr`` but missing from
    the parameter list are silently ignored.
    """

    def __init__(
        self,
        tagged_named_params: Sequence[tuple[str, str, torch.nn.Parameter]],
        train_cfg: dict,
        learning_rate: float | Tensor,
        per_tag_lr: dict[str, float | Tensor] | None = None,
    ):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "Configured train.optimizer='hybrid_tagged' but torch.optim.Muon is unavailable."
            )

        per_tag_lr = dict(per_tag_lr or {})
        param_groups, per_tag_stats = _build_tagged_param_groups(
            tagged_named_params,
            base_lr=learning_rate,
            per_tag_lr=per_tag_lr,
            train_cfg=train_cfg,
        )
        if not param_groups:
            raise RuntimeError(
                "HybridMuonAdamwTagged: no parameter groups were built; check that "
                "named_params is non-empty and all entries have requires_grad=True."
            )
        # Verify Muon eligibility across all tags combined; we keep the
        # parent's invariant ("at least one Muon-eligible 2D weight") so
        # the Muon kernel never gets called on empty.
        total_muon_tensors = sum(int(stats["muon_tensors"]) for stats in per_tag_stats.values())
        if total_muon_tensors <= 0:
            raise RuntimeError(
                "HybridMuonAdamwTagged: no Muon-eligible parameters found across any tag. "
                "Either pass at least one 2D weight in the bridge tag, or switch to plain AdamW."
            )

        # Skip HybridMuonAdamw.__init__ (it builds its own untagged groups);
        # call the next-level Optimizer.__init__ directly with our pre-built
        # tagged groups.
        defaults = {"lr": _as_0d_tensor(learning_rate)}
        torch.optim.Optimizer.__init__(self, param_groups, defaults)

        self.muon_tensors = total_muon_tensors
        self.adamw_tensors = sum(int(stats["adamw_tensors"]) for stats in per_tag_stats.values())
        self.muon_numel = sum(int(stats["muon_numel"]) for stats in per_tag_stats.values())
        self.adamw_numel = sum(int(stats["adamw_numel"]) for stats in per_tag_stats.values())
        self._per_tag_stats = per_tag_stats

        raw_muon_shard_across_ranks = (
            train_cfg.get("muon_shard_across_ranks")
            or train_cfg.get("muon_shard_accross_rank")
            or train_cfg.get("muon_shard_accross_ranks")
        )
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

        # Mirror the parent's bookkeeping so the inherited step() works.
        self._muon_dist_world_size = 1
        self._muon_dist_rank = 0
        self._muon_shard_process_group: Any = None
        self._muon_shard_group_name: str | None = None
        self._muon_shard_world_size = 1
        self._muon_shard_rank = 0
        self._muon_shard_runtime: list[Any] = [None for _ in self.param_groups]

        for group in self.param_groups:
            backend = str(group.get("backend", "")).lower().strip()
            if backend not in {"muon", "adamw"}:
                raise ValueError(
                    f"HybridMuonAdamwTagged only supports backends ['muon', 'adamw'], got: {backend!r}"
                )
            if backend == "muon":
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            "Muon param groups must only contain 2D parameters, "
                            f"but found parameter with shape={tuple(p.shape)} (tag={group.get('tag')!r})"
                        )

        self._normalize_group_tensors()
        self._preinitialize_state()
        self._refresh_muon_shard_runtime()

    # -------- introspection helpers used by the trainer for logging -------- #

    def per_tag_lrs(self) -> dict[str, float]:
        """Return ``{tag: current_lr}`` averaged across (backend, kind)
        sub-groups within the tag. Useful for ``train/lr_<tag>`` wandb
        logs."""
        per_tag_sum: dict[str, list[float]] = {}
        for group in self.param_groups:
            tag = str(group.get("tag", "untagged"))
            lr_val = group.get("lr")
            if isinstance(lr_val, torch.Tensor):
                lr_val = float(lr_val.detach().cpu().item())
            else:
                lr_val = float(lr_val)
            per_tag_sum.setdefault(tag, []).append(lr_val)
        return {tag: (sum(v) / max(1, len(v))) for tag, v in per_tag_sum.items()}

    def per_tag_param_counts(self) -> dict[str, dict[str, int]]:
        """Return ``{tag: {"muon_tensors": N, "adamw_tensors": M, "muon_numel": ..., "adamw_numel": ...}}``."""
        return {
            tag: {k: int(v) for k, v in stats.items()}
            for tag, stats in self._per_tag_stats.items()
        }

    def describe(self) -> str:
        inflight_desc = "inf" if self.muon_max_inflight_buckets < 0 else str(self.muon_max_inflight_buckets)
        shard_desc = "all" if self.muon_shard_across_ranks == 0 else str(self.muon_shard_across_ranks)
        per_tag_desc = ", ".join(
            f"{tag}=(muon={int(s['muon_tensors'])}/{int(s['muon_numel'])} numel, "
            f"adamw={int(s['adamw_tensors'])}/{int(s['adamw_numel'])} numel)"
            for tag, s in self._per_tag_stats.items()
        )
        return (
            f"HybridMuonAdamwTagged(per_tag={{{per_tag_desc}}}, "
            f"muon_shard_across_ranks={shard_desc}, "
            f"muon_max_inflight_buckets={inflight_desc})"
        )
