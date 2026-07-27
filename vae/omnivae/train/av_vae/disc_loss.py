"""LSGAN-style audio discriminator losses.

Vendored verbatim from the speechtokenizer codec project:
  OmniVAE-codec_with_disc/speechtokenizer/trainer/loss.py (L136-L161)
"""

import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + (r_loss + g_loss)
    return loss


def adversarial_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss = loss + l
    return loss
