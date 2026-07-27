"""Video discriminator loss primitives (CausalVAE / VQGAN style).

Provides hinge / vanilla discriminator losses and a helper for the
VQGAN-style adaptive generator weight computed from the gradient-norm
ratio on the last decoder layer.

These are intentionally minimal re-implementations so the training pipeline
does not drag in the full LPIPS-with-discriminator module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        torch.mean(F.softplus(-logits_real))
        + torch.mean(F.softplus(logits_fake))
    )


def generator_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: maximize logits_fake."""
    return -torch.mean(logits_fake)


def calculate_adaptive_weight(
    nll_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer: torch.Tensor,
    discriminator_weight: float = 1.0,
    clamp_min: float = 0.0,
    clamp_max: float = 1e4,
) -> torch.Tensor:
    """VQGAN-style adaptive generator weight.

    Returns ``discriminator_weight * ||grad(nll)|| / (||grad(g)|| + eps)``
    computed w.r.t. ``last_layer``. The result is detached so it flows
    into the backward as a fixed scalar.
    """
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, clamp_min, clamp_max).detach()
    return d_weight * discriminator_weight


__all__ = [
    'hinge_d_loss',
    'vanilla_d_loss',
    'generator_loss',
    'calculate_adaptive_weight',
]
