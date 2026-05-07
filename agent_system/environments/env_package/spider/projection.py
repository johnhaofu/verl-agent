"""Action projection for Spider SQL agent.

Expected model output format:
    <think>some reasoning ...</think><action>sql[SELECT ...]</action>
or
    <think>some reasoning ...</think><action>answer[final SELECT ...]</action>

Returns the inner content (e.g. "sql[SELECT ...]"); the SQL env
parses verb + bracket arg internally.
"""
from __future__ import annotations
import re
from typing import List


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RAW_ACTION_RE = re.compile(
    r"(sql|answer)\[(.+)\](?!.*?(?:sql|answer)\[)",
    re.DOTALL,
)
_SLOP_VERB_RE = re.compile(
    r"^(sql|answer)\s*[>:\-=]\s*(.+?)\s*$", re.DOTALL
)


def _normalise_action(action: str) -> str | None:
    a = action.strip()
    if not a:
        return None
    a = re.sub(r"[;\s]+$", "", a)
    if re.match(r"^(sql|answer)\[.+\]$", a, re.DOTALL):
        return a
    m = _SLOP_VERB_RE.match(a)
    if m:
        verb, arg = m.group(1).lower(), m.group(2).strip()
        if arg:
            return f"{verb}[{arg}]"
    return None


def spider_projection(actions: List[str]):
    """Parse `<action>sql[...]</action>` or `<action>answer[...]</action>`.

    Returns (parsed_actions, valids).
    `valids[i] == 1` iff: <think> present + <action> parses as sql/answer + ASCII only.
    """
    valids = [0] * len(actions)
    parsed: List[str] = []

    for i, original in enumerate(actions):
        # 1. extract <action>...</action>
        m = _ACTION_RE.search(original)
        action_inner = m.group(1).strip() if m else None

        # 2. fallback: raw verb[arg] anywhere in output
        if action_inner is None:
            m2 = _RAW_ACTION_RE.search(original)
            action_inner = f"{m2.group(1)}[{m2.group(2)}]" if m2 else None

        if action_inner is None:
            parsed.append("")
            continue

        normalised = _normalise_action(action_inner)
        if normalised is None:
            parsed.append("")
            continue

        parsed.append(normalised)
        valids[i] = 1

        # 3. require <think> for full credit (mirror webshop)
        if not _THINK_RE.search(original):
            valids[i] = 0

        # 4. ASCII only (mirror webshop's CJK rejection)
        if re.search(r"[\u4e00-\u9fff]", original):
            valids[i] = 0

    return parsed, valids
