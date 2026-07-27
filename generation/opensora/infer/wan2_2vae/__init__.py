"""Self-contained WanVAE2.2 inference package.

Usage::

    from opensora.infer.wan2_2vae import WanVAE22Model

    model = WanVAE22Model.from_pretrained(
        "/path/to/ckpt_dir",      # dir with config.json + *.pth
    ).eval().cuda()

    posterior = model.encode(video)         # video: (B, 3, T, H, W) in [-1, 1]
    z = posterior.mode()
    rec = model.decode(z)
"""

from .distributions import DiagonalGaussianDistribution
from .model import WanVAE22Model
from .modules import (
    AttentionBlock,
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    ResidualBlock,
    RMS_norm,
    count_conv3d,
    patchify,
    unpatchify,
)

__all__ = [
    "WanVAE22Model",
    "DiagonalGaussianDistribution",
    "Encoder3d",
    "Decoder3d",
    "ResidualBlock",
    "AttentionBlock",
    "CausalConv3d",
    "RMS_norm",
    "patchify",
    "unpatchify",
    "count_conv3d",
]
