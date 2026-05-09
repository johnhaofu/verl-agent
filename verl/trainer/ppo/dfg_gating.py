"""
Decidable-Fraction-Gated RL (DFG-RL) helpers.

Per-batch, partition rollouts into groups of size `group_n` (one prompt -> n
sampled trajectories). A group is "decidable" iff it contains at least one
passing trajectory AND at least one failing trajectory; only decidable groups
contribute non-zero advantage in GRPO/GiGPO.

The decidable fraction D = |decidable groups| / |all groups| measures the
batch's effective RL signal density.

Gating policy:
    D  <  threshold_low   -> sparse regime: temporarily turn ON SFT-on-winners
                              (compensate for vanishing RL gradient).
    D  in [low, high]     -> healthy regime: pure RL, SFT OFF.
    D  >  threshold_high  -> saturated regime: emit signal so trainer can
                              raise difficulty / early-stop.

This file exposes pure functions; the trainer calls them and acts on the
returned decisions. dp_actor.py already gates SFT on the *presence* of the
`is_winner` tensor in the batch, so toggling distill ON/OFF per step reduces
to inserting or omitting that tensor.
"""

from __future__ import annotations

from typing import Tuple

import torch


REGIME_SPARSE = "sparse"
REGIME_HEALTHY = "healthy"
REGIME_SATURATED = "saturated"


def compute_decidable_fraction(
    episode_rewards: torch.Tensor,
    group_n: int,
    pass_threshold: float = 0.5,
) -> Tuple[float, float]:
    """Return (decidable_fraction, mean_pass_rate).

    Args:
        episode_rewards: (B,) tensor where B = num_groups * group_n. Trajectories
            from the same prompt must be contiguous (verl-agent's default).
        group_n: rollouts per prompt.
        pass_threshold: reward >= threshold counts as a pass.
    """
    if episode_rewards.numel() == 0 or group_n <= 0:
        return 0.0, 0.0
    if episode_rewards.numel() % group_n != 0:
        raise ValueError(
            f"episode_rewards size {episode_rewards.numel()} not divisible by group_n={group_n}"
        )

    grouped = episode_rewards.view(-1, group_n)               # (G, n)
    passes = (grouped >= pass_threshold)                      # (G, n) bool
    has_pass = passes.any(dim=-1)
    has_fail = (~passes).any(dim=-1)
    decidable = (has_pass & has_fail).float()
    return float(decidable.mean().item()), float(passes.float().mean().item())


def classify_regime(
    decidable_fraction: float,
    threshold_low: float,
    threshold_high: float,
) -> str:
    if decidable_fraction < threshold_low:
        return REGIME_SPARSE
    if decidable_fraction > threshold_high:
        return REGIME_SATURATED
    return REGIME_HEALTHY


def should_apply_distill(regime: str) -> bool:
    """Whether to inject is_winner into the batch this step."""
    return regime == REGIME_SPARSE
