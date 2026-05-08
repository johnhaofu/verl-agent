"""Action projection for Horizon (Shopify Liquid template) agent.

Expected model output format:
    <think>some reasoning ...</think><action>verb[arg]</action>

Verbs:
    list_sections      Read-only: lists available section names
    describe_section   Read-only: returns the section's {% schema %} block
    describe_block     Read-only: same for blocks
    submit             Terminal: compile + reward 1.0/0.0
    fix                Non-terminal compile attempt; errors → next obs

Returns the inner content (e.g. ``submit[{"sections": ...}]``); the
core env parses verb + bracket arg internally.
"""
from __future__ import annotations

import re
from typing import List

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_VERB_RE = re.compile(
    r"^(list_sections|describe_section|describe_block|submit|fix)\[(.*)\]\s*$",
    re.DOTALL,
)


def horizon_projection(actions: List[str]):
    """Parse ``<action>verb[arg]</action>`` for the Horizon template agent.

    Returns ``(parsed_actions, valids)`` where ``valids[i] == 1`` iff the
    output contains a parseable ``<action>verb[arg]</action>``.

    The Spider/webshop projections additionally required ``<think>``, but
    Qwen3-4B-Instruct-2507 emits ``<tool_call>`` reasoning instead — that
    requirement caused 100% of actions to be flagged invalid in v1
    training, stacking ``-invalid_action_penalty`` on every turn and
    polluting the reward signal.
    """
    valids = [0] * len(actions)
    parsed: List[str] = []

    for i, original in enumerate(actions):
        matches = _ACTION_RE.findall(original)
        if not matches:
            parsed.append("")
            continue

        # Take the LAST <action>...</action> (matches parse_action in core_env)
        inner = matches[-1].strip()
        m = _VERB_RE.match(inner)
        if not m:
            parsed.append("")
            continue

        parsed.append(inner)
        valids[i] = 1

    return parsed, valids
