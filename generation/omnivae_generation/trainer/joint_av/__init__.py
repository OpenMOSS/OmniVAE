"""Text-to-Audio-Video (T2AV) joint generation extension.

Composes two pretrained Z-Image transformer branches (t2v + t2a) by
inserting Bridge cross-attention modules between every ``bridge_interval``
main transformer blocks. Bridges are zero-initialized so the joint model
behaves identically to the two independent branches at step 0.
"""

from omnivae_generation.trainer.joint_av.bridge import BridgeBlock, CrossModalityAdaLN, build_temporal_rope_freqs
from omnivae_generation.trainer.joint_av.model import BridgedZImageJointModel
from omnivae_generation.trainer.joint_av.sigma import apply_sigma_shift, prepare_dual_diffusion_batch, DualDiffusionBatch
from omnivae_generation.trainer.joint_av.optim import HybridMuonAdamwTagged
from omnivae_generation.trainer.joint_av.dataset import AVPairedJsonlDataset, collate_av_paired_samples
from omnivae_generation.trainer.joint_av.loader import (
    load_pretrained_branches,
    save_split_branches,
    load_bridges_from_dir,
    BRIDGE_PARAM_PREFIX,
)
from omnivae_generation.trainer.joint_av.validation import run_joint_av_validation

__all__ = [
    "BridgeBlock",
    "CrossModalityAdaLN",
    "build_temporal_rope_freqs",
    "BridgedZImageJointModel",
    "apply_sigma_shift",
    "prepare_dual_diffusion_batch",
    "DualDiffusionBatch",
    "HybridMuonAdamwTagged",
    "AVPairedJsonlDataset",
    "collate_av_paired_samples",
    "load_pretrained_branches",
    "save_split_branches",
    "load_bridges_from_dir",
    "BRIDGE_PARAM_PREFIX",
    "run_joint_av_validation",
]
