"""Minimal smoke test for IntraSegCrossAttnHead.

Builds a tiny fake batch, runs the head forward and backward, and checks:
  * output dict is structurally identical to LatentAVContrastiveHead's
  * loss is finite and backward produces gradients on learnable params
  * with audio_latent_lengths padded, masking still yields a finite loss

Run:
    cd OmniVAE
    python scripts/tools/smoke_intra_seg_xattn_head.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from omnivae.models.audio_video_vae.contrastive import IntraSegCrossAttnHead


def _assert_same_keys(got_keys, exp_keys, label):
    missing = set(exp_keys) - set(got_keys)
    extra = set(got_keys) - set(exp_keys)
    assert not missing, f"{label}: missing keys {missing}"
    # Extra keys are OK but we want at least the contract.


def _run(
    device: torch.device,
    use_itm: bool = False,
    itm_neg_per_direction: int = 1,
    itm_neg_source: str = "near",
    itm_start_step: int = 0,
    current_step: int = 0,
):
    torch.manual_seed(0)

    B, Dv, T, H, W = 2, 16, 7, 8, 8
    Da, La = 24, 300

    head = IntraSegCrossAttnHead(
        video_latent_dim=Dv,
        audio_latent_dim=Da,
        embed_dim=64,
        nhead=4,
        self_attn_layers=1,
        cross_attn_layers=1,
        spatial_merge_factor=2,
        max_spatial_h=16,
        max_spatial_w=16,
        max_audio_tokens_per_seg=16,
        num_negatives=3,
        num_negative_videos=1,
        skip_first_video_latent_frame=True,
        video_temporal_compress_factor=4,
        use_itm=use_itm,
        lambda_itm=0.5,
        itm_neg_per_direction=itm_neg_per_direction,
        itm_sim_temperature=1.0,
        itm_neg_source=itm_neg_source,
        itm_start_step=itm_start_step,
    ).to(device)
    head.current_step = int(current_step)

    # Note: skip_first_video_latent_frame drops frame 0 -> S = T-1 = 6
    video = torch.randn(B, Dv, T, H, W, device=device, requires_grad=False)
    audio = torch.randn(B, Da, La, device=device, requires_grad=False)
    audio_lens = torch.tensor([La, La - 50], device=device, dtype=torch.long)

    out = head(
        video_latent=video,
        audio_latent=audio,
        audio_latent_lengths=audio_lens,
        world_size=1,
    )

    print("[check] top-level keys:", sorted(out.keys()))
    _assert_same_keys(
        out.keys(),
        {"logit_scales", "granularities", "losses"},
        "top-level",
    )
    assert len(out["granularities"]) == 1, "single-granularity head"

    g = out["granularities"][0]
    print("[check] granularity keys:", sorted(g.keys()))
    _assert_same_keys(
        g.keys(),
        {"segment_count", "S", "losses", "segment_vfeat", "segment_afeat",
         "segment_vfeat_pool", "segment_afeat_pool", "B", "B_eff", "rank_offset"},
        "granularity",
    )

    S_expected = T - 1
    assert g["S"] == S_expected, f"expected S={S_expected} (causal drop), got {g['S']}"
    assert g["B"] == B and g["B_eff"] == B and g["rank_offset"] == 0

    seg_loss = g["losses"]["segment_contrastive_loss"]
    print(f"[check] segment_contrastive_loss = {float(seg_loss):.4f}")
    assert torch.isfinite(seg_loss), f"non-finite loss: {seg_loss}"

    itm_active = use_itm and current_step >= itm_start_step
    if itm_active:
        for k in ("itc_loss_raw", "itm_loss_raw", "itm_acc"):
            assert k in g["losses"], f"itm active but '{k}' missing from losses dict"
        itc_raw = g["losses"]["itc_loss_raw"]
        itm_raw = g["losses"]["itm_loss_raw"]
        itm_acc = g["losses"]["itm_acc"]
        print(
            f"[check] itc_raw={float(itc_raw):.4f} itm_raw={float(itm_raw):.4f} "
            f"itm_acc={float(itm_acc):.4f}"
        )
        assert torch.isfinite(itc_raw) and torch.isfinite(itm_raw)
        assert 0.0 <= float(itm_acc) <= 1.0, f"itm_acc out of range: {float(itm_acc)}"
    elif use_itm:
        # Warmup: ITC should still be reported, but ITM-specific keys MUST NOT
        # appear so downstream logging cleanly shows "nothing yet".
        assert "itc_loss_raw" in g["losses"], (
            "use_itm=True warmup: itc_loss_raw should still be logged"
        )
        assert "itm_loss_raw" not in g["losses"], (
            "use_itm=True warmup: itm_loss_raw must be omitted before itm_start_step"
        )
        assert "itm_acc" not in g["losses"], (
            "use_itm=True warmup: itm_acc must be omitted before itm_start_step"
        )
        print(
            f"[check] warmup step={current_step} < start={itm_start_step} "
            f"-> ITC-only, segment_loss={float(seg_loss):.4f}"
        )
    else:
        # ITM-specific keys should be absent
        assert "itm_loss_raw" not in g["losses"], (
            "use_itm=False should not produce itm_loss_raw"
        )

    # Check shapes
    assert g["segment_vfeat"].shape == (B * S_expected, 64)
    assert g["segment_afeat"].shape == (B * S_expected, 64)
    assert g["segment_vfeat_pool"].shape == (B * S_expected, 64)
    assert g["segment_afeat_pool"].shape == (B * S_expected, 64)

    # Normalized
    v_norm = g["segment_vfeat"].norm(dim=-1)
    a_norm = g["segment_afeat"].norm(dim=-1)
    assert torch.allclose(v_norm, torch.ones_like(v_norm), atol=1e-4), "video CLS not L2-normalized"
    assert torch.allclose(a_norm, torch.ones_like(a_norm), atol=1e-4), "audio CLS not L2-normalized"

    # Backward -> params with grad
    seg_loss.backward()
    n_with_grad = 0
    n_total = 0
    names_with_grad = []
    for name, p in head.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_with_grad += 1
            names_with_grad.append(name)
    print(f"[check] params with non-zero grad: {n_with_grad}/{n_total}")
    assert n_with_grad > 0, "no parameters received gradients"

    if itm_active:
        # Verify BOTH self-attn and cross-attn towers (and itm_head) received grads.
        has_self = any("self_blocks" in n for n in names_with_grad)
        has_cross = any("cross_blocks" in n for n in names_with_grad)
        has_itm_head = any(n.startswith("itm_head") for n in names_with_grad)
        assert has_self, "itm active: self-attn tower received no grad"
        assert has_cross, "itm active: cross-attn tower received no grad"
        assert has_itm_head, "itm active: itm_head received no grad"
    elif use_itm:
        # Warmup path: cross-attn + itm_head must NOT see gradient, self-attn still must.
        has_self = any("self_blocks" in n for n in names_with_grad)
        has_cross = any("cross_blocks" in n for n in names_with_grad)
        has_itm_head = any(n.startswith("itm_head") for n in names_with_grad)
        assert has_self, "warmup: self-attn tower (used by ITC) received no grad"
        assert not has_cross, (
            "warmup: cross-attn tower should be skipped before itm_start_step "
            "but received gradient"
        )
        assert not has_itm_head, (
            "warmup: itm_head should be skipped before itm_start_step "
            "but received gradient"
        )

    # Trainer-facing attributes
    for attr in [
        "n_granularities", "segment_count_list", "num_negatives_list",
        "num_negative_videos_list", "same_long_video_num_negatives_list",
        "num_negatives_with_sibling_list", "gather_for_loss",
        "same_long_video_priority",
    ]:
        assert hasattr(head, attr), f"missing trainer-facing attribute: {attr}"
    assert head.n_granularities == 1

    # clamp_logit_scales API
    seg_scale, global_scale = head.clamp_logit_scales()
    assert seg_scale is not None and global_scale is None

    if use_itm:
        mode_tag = (
            f"use_itm=True,src={itm_neg_source},k={itm_neg_per_direction},"
            f"start={itm_start_step},step={current_step},"
            f"active={'Y' if itm_active else 'N'}"
        )
    else:
        mode_tag = "use_itm=False"
    print(f"[smoke] all checks passed on {device} ({mode_tag})")


def main():
    # Base head: no ITM
    _run(torch.device("cpu"), use_itm=False)
    # ITM with "near" (default): intra-only, k=1 and k=2
    _run(torch.device("cpu"), use_itm=True, itm_neg_per_direction=1, itm_neg_source="near")
    _run(torch.device("cpu"), use_itm=True, itm_neg_per_direction=2, itm_neg_source="near")
    # ITM with legacy "hard_itc" source for regression safety
    _run(torch.device("cpu"), use_itm=True, itm_neg_per_direction=1, itm_neg_source="hard_itc")
    # itm_start_step: warmup path (ITM off) and post-start path (ITM on).
    _run(torch.device("cpu"), use_itm=True, itm_neg_per_direction=1,
         itm_neg_source="near", itm_start_step=100, current_step=50)
    _run(torch.device("cpu"), use_itm=True, itm_neg_per_direction=1,
         itm_neg_source="near", itm_start_step=100, current_step=100)
    if torch.cuda.is_available():
        _run(torch.device("cuda:0"), use_itm=False)
        _run(torch.device("cuda:0"), use_itm=True, itm_neg_per_direction=1, itm_neg_source="near")
        _run(torch.device("cuda:0"), use_itm=True, itm_neg_per_direction=1,
             itm_neg_source="near", itm_start_step=100, current_step=50)


if __name__ == "__main__":
    main()
