from typing import List

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class MultiResolutionMelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        weights: List[float] = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None]
    ):
        super().__init__()
        self.mel_transforms = nn.ModuleList([
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=window_length,
                hop_length=window_length // 4,
                n_mels=n_mel,
                power=1.0,
                center=True,
                norm='slaney',
                mel_scale='slaney',
            )
            for n_mel, window_length in zip(n_mels, window_lengths)
        ])
        self.n_mels = n_mels
        self.weights = weights
        assert len(self.weights) == len(self.mel_transforms), \
            "Weights length must match the number of mel scales"
        self.loss_fn = nn.L1Loss(reduction='none')
        self.clamp_eps = clamp_eps
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x, y):
        """
        计算多分辨率梅尔频谱损失。

        参数:
            x (torch.Tensor): 预测波形，形状 [B, 1, T]
            y (torch.Tensor): 目标波形，形状 [B, 1, T]

        返回:
            torch.Tensor: 损失值
        """
        loss = 0.0

        for i, mel_transform in enumerate(self.mel_transforms):
            x_mel = mel_transform(x.squeeze(1))
            y_mel = mel_transform(y.squeeze(1))
            log_x_mel = x_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            log_y_mel = y_mel.clamp(self.clamp_eps).pow(self.pow).log10()

            scale_loss = self.loss_fn(log_x_mel, log_y_mel).mean()
            loss += self.weights[i] * scale_loss

        return loss


class WaveformLoss(nn.Module):
    """波形重建损失"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(pred, target)


@torch.no_grad()
def compute_segment_intra_precision(vfeat: torch.Tensor, afeat: torch.Tensor, B: int, S: int):
    """
    Same-video segment retrieval precision: for each video, use its S audio
    segments to retrieve among S video segments (and vice versa).
    """
    D = vfeat.shape[-1]
    v = vfeat.view(B, S, D)
    a = afeat.view(B, S, D)
    sim = a @ v.mT  # (B, S, S)
    gt = torch.arange(S, device=vfeat.device).unsqueeze(0)  # (1, S)
    prec_a2v = (sim.argmax(dim=-1) == gt).float().mean()
    prec_v2a = (sim.argmax(dim=-2) == gt).float().mean()
    return {
        'seg_intra_prec_a2v': prec_a2v.item(),
        'seg_intra_prec_v2a': prec_v2a.item(),
        'seg_intra_prec_avg': ((prec_a2v + prec_v2a) / 2.0).item(),
    }


@torch.no_grad()
def compute_segment_sampled_precision(
    sim_v2a: torch.Tensor,
    sim_a2v: torch.Tensor,
    num_intra: int,
):
    """
    Segment retrieval precision from pre-sampled similarity matrices.
    Column layout: [pos(col 0) | intra(cols 1..num_intra) | cross(cols num_intra+1..)]
    """
    target = torch.zeros(sim_v2a.shape[0], dtype=torch.long, device=sim_v2a.device)
    results = {}

    results['seg_overall_prec_v2a'] = (sim_v2a.argmax(dim=1) == target).float().mean().item()
    results['seg_overall_prec_a2v'] = (sim_a2v.argmax(dim=1) == target).float().mean().item()
    results['seg_overall_prec_avg'] = (
        results['seg_overall_prec_v2a'] + results['seg_overall_prec_a2v']
    ) / 2.0

    cross_start = 1 + num_intra
    if sim_v2a.shape[1] > cross_start:
        cross_v2a = torch.cat([sim_v2a[:, :1], sim_v2a[:, cross_start:]], dim=1)
        cross_a2v = torch.cat([sim_a2v[:, :1], sim_a2v[:, cross_start:]], dim=1)
        target_cross = torch.zeros(cross_v2a.shape[0], dtype=torch.long, device=sim_v2a.device)
        results['seg_cross_prec_v2a'] = (cross_v2a.argmax(dim=1) == target_cross).float().mean().item()
        results['seg_cross_prec_a2v'] = (cross_a2v.argmax(dim=1) == target_cross).float().mean().item()
        results['seg_cross_prec_avg'] = (
            results['seg_cross_prec_v2a'] + results['seg_cross_prec_a2v']
        ) / 2.0

    return results


@torch.no_grad()
def compute_global_sampled_precision(
    global_vfeat: torch.Tensor,
    global_afeat: torch.Tensor,
    num_negatives: int = 32,
) -> dict:
    """
    Global-level precision by randomly sampling negatives for each query.
    For each query i, sample `num_negatives` random negatives from pool
    (excluding self), prepend positive at index 0, check argmax == 0.
    """
    B = global_vfeat.shape[0]
    device = global_vfeat.device
    K = min(num_negatives, B - 1)
    if K <= 0:
        return {}

    all_indices = torch.arange(B, device=device)
    pos_idx = all_indices.unsqueeze(1)  # (B, 1)

    mask = all_indices.unsqueeze(0) != all_indices.unsqueeze(1)  # (B, B)
    neg_pool_per_query = all_indices.unsqueeze(0).expand(B, -1)[mask].view(B, B - 1)
    perm = torch.rand(B, B - 1, device=device).argsort(dim=1)[:, :K]
    neg_idx = neg_pool_per_query.gather(1, perm)  # (B, K)

    sample_idx = torch.cat([pos_idx, neg_idx], dim=1)  # (B, 1+K)

    afeat_sampled = global_afeat[sample_idx]  # (B, 1+K, D)
    vfeat_sampled = global_vfeat[sample_idx]  # (B, 1+K, D)

    sim_v2a = torch.einsum("bd,bkd->bk", global_vfeat, afeat_sampled)
    sim_a2v = torch.einsum("bd,bkd->bk", global_afeat, vfeat_sampled)
    target = torch.zeros(B, dtype=torch.long, device=device)

    prec_v2a = (sim_v2a.argmax(dim=1) == target).float().mean().item()
    prec_a2v = (sim_a2v.argmax(dim=1) == target).float().mean().item()
    return {
        'global_precision_v2a': prec_v2a,
        'global_precision_a2v': prec_a2v,
        'global_precision_avg': (prec_v2a + prec_a2v) / 2.0,
    }
