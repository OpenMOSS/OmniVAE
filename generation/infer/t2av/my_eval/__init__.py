"""Distributed T2AV evaluation toolkit.

Reads (mp4, wav) pairs from the joint_av layout produced by sweep_t2av_ckpts and
evaluates them against a fixed set of metrics (MOVA-aligned where possible plus
Verse-Bench AS/audiobox-aesthetics ports). Each metric writes one per-sample JSON
and one summary JSON; no overall score is computed.
"""
