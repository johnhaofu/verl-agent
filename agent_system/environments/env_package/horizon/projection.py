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
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_VERB_RE = re.compile(
    r"^(list_sections|describe_section|describe_block|submit|fix)\[(.*)\]\s*$",
    re.DOTALL,
)


def horizon_projection(actions: List[str]):
    """Parse ``<action>verb[arg]</action>`` for the Horizon template agent.

    Returns ``(parsed_actions, valids)`` where ``valids[i] == 1`` iff the
    output contains a ``<think>`` tag and a parseable ``<action>``.
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

        # Mirror webshop / spider: require <think> for full credit
        if not _THINK_RE.search(original):
            valids[i] = 0

    return parsed, valids
