"""Smoke test: per-row variable negatives for no-sibling fallback.

Run from the repo root:
    cd /path/to/OmniVAE
    python scripts/tools/smoke_test_variable_negatives.py

Validates:
    1. `_build_neg_indices_sibling_aware` 返回 4-tuple，且 `valid_neg_mask` shape
       = (n_local, C_max)。
    2. 全 sibling pool: 每行 mask 全 True（C_max 有效列）。
    3. 全无 sibling pool: 每行前 C_no_sib 列 True、后 C_max-C_no_sib 列 False；
       neg_idx 中的 padding 列是安全 index（不越界）。
    4. `sample_negatives_for_loss` 跑完整 pipeline，sim 在 padding 列为 -inf，
       CE loss 是有限数（not NaN / not +inf）。
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F

# Make ``omnivae`` importable no matter where the script is launched from:
# scripts/tools/smoke_test_variable_negatives.py -> repo root is two levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omnivae.models.audio_video_vae.contrastive import LatentAVContrastiveHead


def _build_head(device: torch.device) -> LatentAVContrastiveHead:
    head = LatentAVContrastiveHead(
        video_latent_dim=16,
        audio_latent_dim=8,
        embed_dim=32,
        segment_count=None,
        init_scale=0.07,
        clamp_scale_min=0.001,
        clamp_scale_max=0.5,
        gather_for_loss=False,
        num_negatives=96,
        num_negative_videos=None,
        same_long_video_priority=True,
        same_long_video_num_negatives=48,
        num_negatives_no_sibling=24,
    ).to(device)
    return head


def test_build_neg_indices():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _build_head(device)

    B, B_eff, S = 4, 4, 49
    rank_offset = 0
    K_seg = 48
    num_total = 96
    num_total_with_sibling = head._resolve_num_negatives_with_sibling(0, S)
    print(f"[cfg] S={S} B={B} B_eff={B_eff} K_seg={K_seg} "
          f"num_total={num_total} C_max={num_total_with_sibling}")

    C_max = int(num_total_with_sibling)
    C_no_sib = int(num_total)
    n_local = B * S

    # ---- Case 1: all pool share the same long_video_id (全 sibling) ----
    lvid_all_same = torch.zeros(B_eff, dtype=torch.long, device=device)
    neg_idx, num_intra, sib_take, mask = head._build_neg_indices_sibling_aware(
        B=B, B_eff=B_eff, S=S, rank_offset=rank_offset,
        long_video_ids_pool=lvid_all_same,
        K_seg=K_seg, num_total=num_total,
        num_total_with_sibling=num_total_with_sibling,
        device=device,
    )
    assert neg_idx.shape == (n_local, C_max), f"neg_idx shape {neg_idx.shape}"
    assert mask.shape == (n_local, C_max), f"mask shape {mask.shape}"
    assert mask.dtype == torch.bool
    assert mask.all(), "[all-sibling] expected mask to be all True"
    assert neg_idx.max().item() < B_eff * S, "[all-sibling] neg_idx out of range"
    print(f"[case1 all-sibling] OK: mask.all()=True, num_intra={num_intra}, "
          f"sibling_take mean={sib_take.float().mean().item():.2f}")

    # ---- Case 2: all pool have distinct long_video_id (全无 sibling) ----
    lvid_all_diff = torch.arange(B_eff, dtype=torch.long, device=device)
    neg_idx2, num_intra2, sib_take2, mask2 = head._build_neg_indices_sibling_aware(
        B=B, B_eff=B_eff, S=S, rank_offset=rank_offset,
        long_video_ids_pool=lvid_all_diff,
        K_seg=K_seg, num_total=num_total,
        num_total_with_sibling=num_total_with_sibling,
        device=device,
    )
    assert mask2[:, :C_no_sib].all(), "[no-sibling] first C_no_sib cols should be True"
    assert (~mask2[:, C_no_sib:]).all(), "[no-sibling] trailing cols should be False"
    assert neg_idx2.max().item() < B_eff * S, "[no-sibling] neg_idx out of range"
    assert (sib_take2 == 0).all(), "[no-sibling] sibling_take should be 0"
    print(f"[case2 no-sibling ] OK: valid cols per row = {C_no_sib}, "
          f"padded tail = {C_max - C_no_sib}")

    # ---- Case 3: mixed (半 sibling) ----
    lvid_mixed = torch.tensor([0, 0, 1, 2], dtype=torch.long, device=device)
    neg_idx3, _, sib_take3, mask3 = head._build_neg_indices_sibling_aware(
        B=B, B_eff=B_eff, S=S, rank_offset=rank_offset,
        long_video_ids_pool=lvid_mixed,
        K_seg=K_seg, num_total=num_total,
        num_total_with_sibling=num_total_with_sibling,
        device=device,
    )
    row_valid = mask3.sum(dim=1)  # (n_local,)
    unique_counts = torch.unique(row_valid).tolist()
    print(f"[case3 mixed      ] OK: per-row valid counts unique = {unique_counts} "
          f"(expected subset of {{{C_no_sib}, {C_max}}})")
    assert set(unique_counts).issubset({C_no_sib, C_max}), unique_counts
    assert neg_idx3.max().item() < B_eff * S

    return head, device


def test_sample_negatives_for_loss(head: LatentAVContrastiveHead, device: torch.device):
    B, B_eff, S = 4, 4, 49
    D = int(head.embed_dim)
    n_local = B * S
    n_pool = B_eff * S

    torch.manual_seed(0)
    vfeat_local = F.normalize(torch.randn(n_local, D, device=device), dim=-1)
    afeat_local = F.normalize(torch.randn(n_local, D, device=device), dim=-1)
    vfeat_pool = F.normalize(torch.randn(n_pool, D, device=device), dim=-1)
    afeat_pool = F.normalize(torch.randn(n_pool, D, device=device), dim=-1)

    scale = head.logit_scale.detach().clamp(
        head.clamp_scale_min, head.clamp_scale_max
    )

    num_total_with_sibling = head._resolve_num_negatives_with_sibling(0, S)

    for tag, lvid in [
        ("all-sibling", torch.zeros(B_eff, dtype=torch.long, device=device)),
        ("no-sibling ", torch.arange(B_eff, dtype=torch.long, device=device)),
        ("mixed      ", torch.tensor([0, 0, 1, 2], dtype=torch.long, device=device)),
    ]:
        sim_v2a, sim_a2v, targets, _ = head.sample_negatives_for_loss(
            vfeat_local=vfeat_local,
            afeat_local=afeat_local,
            vfeat_pool=vfeat_pool,
            afeat_pool=afeat_pool,
            B=B, B_eff=B_eff, S=S,
            scale=scale, rank_offset=0,
            num_negatives=head.num_negatives_list[0],
            num_negative_videos=head.num_negative_videos_list[0],
            long_video_ids_pool=lvid,
            same_long_video_num_negatives=head.same_long_video_num_negatives_list[0],
            num_negatives_with_sibling=num_total_with_sibling,
        )
        loss = (F.cross_entropy(sim_v2a, targets) + F.cross_entropy(sim_a2v, targets)) / 2
        # Padding columns are filled with ``torch.finfo().min`` (≈ -3.4e38 for
        # float32), not literal ``-inf``. The latter is avoided so that rows
        # whose every negative is masked still yield a finite softmax.
        pad_threshold = -1e30
        n_pad_v2a = (sim_v2a < pad_threshold).sum().item()
        n_pad_a2v = (sim_a2v < pad_threshold).sum().item()
        assert torch.isfinite(loss), f"[{tag}] loss is not finite: {loss.item()}"
        assert sim_v2a[:, 0].isfinite().all(), f"[{tag}] positive column must stay finite"
        # Cross-check: number of padded columns equals the row-summed
        # complement of the valid mask (26 no-sibling rows × 24 pad cols, etc.).
        print(f"[loss {tag}] loss={loss.item():.4f}  "
              f"pad(v2a)={n_pad_v2a}  pad(a2v)={n_pad_a2v}")


def main() -> int:
    head, device = test_build_neg_indices()
    test_sample_negatives_for_loss(head, device)
    print("\nALL SMOKE TESTS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
