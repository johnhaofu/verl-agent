"""
Group-Null-Advantage SFT gating (a.k.a. DFG-RL, group-local form).

Premise: in GRPO/GiGPO, a group's contribution to the policy gradient comes
from intra-group advantage variance. When a group is fully homogeneous
(all-pass or all-fail), every trajectory's advantage is exactly zero and that
group contributes no learning signal regardless of the policy.

This file exposes a single decision rule with no hyperparameters:

    is_winner[i]  =  ( all trajectories in i's group passed )

i.e. apply SFT-on-winners loss exactly when (a) GRPO advantage vanishes
because the group is homogeneous AND (b) there is a successful trajectory
to imitate. Mixed groups are left to pure RL; all-fail groups have nothing
to distill from and are skipped automatically.

This contrasts with the original v2-distill ablation, which applied SFT
unconditionally and lost -22.7pp on OOD eval_fixed; and with batch-global
threshold gating, which requires hand-tuned cutoffs. The group-local rule is
threshold-free and structurally aligned with GRPO's own group normalization.

Decidable fraction D = |groups with both pass AND fail| / |all groups| is
still computed and emitted as a metric, but it is no longer a gate.
"""

from __future__ import annotations

from typing import Tuple

import torch


def compute_group_stats(
    episode_rewards: torch.Tensor,
    group_n: int,
    pass_threshold: float = 0.5,
) -> dict:
    """Return per-batch group-structure metrics.

    Args:
        episode_rewards: (B,) tensor where B = num_groups * group_n.
            Trajectories from the same prompt must be contiguous
            (verl-agent's interleave=True default).
        group_n: rollouts per prompt.
        pass_threshold: reward >= threshold counts as a pass.

    Returns dict with:
        decidable_fraction:   |mixed groups|       / |groups|
        all_pass_fraction:    |all-pass groups|    / |groups|   (SFT fires here)
        all_fail_fraction:    |all-fail groups|    / |groups|
        group_pass_rate:      mean per-trajectory pass rate
    """
    if episode_rewards.numel() == 0 or group_n <= 0:
        return {
            "decidable_fraction": 0.0,
            "all_pass_fraction": 0.0,
            "all_fail_fraction": 0.0,
            "group_pass_rate": 0.0,
        }
    if episode_rewards.numel() % group_n != 0:
        raise ValueError(
            f"episode_rewards size {episode_rewards.numel()} not divisible by group_n={group_n}"
        )

    grouped = episode_rewards.view(-1, group_n)               # (G, n)
    passes = (grouped >= pass_threshold)                      # (G, n) bool
    has_pass = passes.any(dim=-1)
    has_fail = (~passes).any(dim=-1)
    all_pass = has_pass & ~has_fail
    all_fail = ~has_pass & has_fail
    decidable = has_pass & has_fail
    return {
        "decidable_fraction": float(decidable.float().mean().item()),
        "all_pass_fraction": float(all_pass.float().mean().item()),
        "all_fail_fraction": float(all_fail.float().mean().item()),
        "group_pass_rate": float(passes.float().mean().item()),
    }


def compute_group_null_winner_mask(
    episode_rewards: torch.Tensor,
    group_n: int,
    pass_threshold: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """Per-trajectory SFT mask for the group-null-advantage rule.

    Returns:
        winner_mask: (B,) float tensor. mask[i] = 1.0 iff trajectory i's group
            has all trajectories passing (so GRPO advantage = 0 AND there is
            a winner to imitate); else 0.0.
        sft_active_fraction: fraction of trajectories the SFT loss is applied
            to this batch (= all_pass_fraction).
    """
    if episode_rewards.numel() == 0 or group_n <= 0:
        return torch.zeros_like(episode_rewards), 0.0

    grouped = episode_rewards.view(-1, group_n)               # (G, n)
    passes = (grouped >= pass_threshold)                      # (G, n)
    all_pass = passes.all(dim=-1, keepdim=True)               # (G, 1)
    winner_mask_grouped = all_pass.expand(-1, group_n).float()
    winner_mask = winner_mask_grouped.reshape(-1)             # (B,)
    sft_active_fraction = float(winner_mask.mean().item())
    return winner_mask, sft_active_fraction
