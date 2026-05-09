"""Standalone Horizon (Shopify Liquid template) multi-turn agent env.

Ported from compiler-reward-agent-rl/environments/horizon_env_multiturn.py
with verl-agent contract changes:
- reset(idx) -> (obs_text, info_dict)
- step(inner_action) -> (obs_text, reward, done, info_dict)
- info has 'won', 'task_score', 'available_actions' keys

The compiler is the underlying HorizonEnvironment.validate() which calls
shopify theme-check (real compiler, deterministic). Action grammar is
the same as the source env; we just inline parsing because the projection
layer already extracted the inner content.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

# HorizonEnvironment used only for local theme metadata (sections list,
# schema previews). Real compile validation goes through SitemuseValidator
# (Shopify themeFilesUpsert GraphQL) — that's the actual reward signal.
import sys
_possible_lib_paths = [
    "/root/autodl-tmp/datasets/horizon/lib",
    "/root/autodl-tmp/compiler-reward-agent-rl/environments",
    "/tmp/compiler-reward-agent-rl/environments",
]
for _p in _possible_lib_paths:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

from horizon_env import HorizonEnvironment  # type: ignore  # local theme reader
from sitemuse_validator import SitemuseValidator  # type: ignore  # API validator

from .error_anchor import essa_anchor  # ESSA step anchor for inner-group GiGPO


_VERB_RE = re.compile(
    r"^(list_sections|describe_section|describe_block|submit|fix)\[(.*)\]\s*$",
    re.DOTALL,
)


class HorizonAgentEnv:
    """One Horizon-template-generation prompt per episode.

    obs format (init):
        Task: Generate a Shopify Horizon theme template.
        Template type: <type>
        Description: <prompt text>

        You may explore via list_sections[] / describe_section[<name>].
        Submit final answer with submit[<JSON>] or test draft with fix[<JSON>].

    Reward (binary, mirrors mlx-agent-rl Spider lesson):
        submit pass / fix pass → 1.0  (terminal)
        submit fail            → 0.0  (terminal)
        fix fail               → 0.0  (non-terminal, errors in next obs)
        invalid action         → -0.1 (non-terminal)
        list/describe          → 0.0  (non-terminal, info text in next obs)
    """

    SCHEMA_PREVIEW_CHARS = 1500

    def __init__(
        self,
        horizon_path: str,
        prompts: list[dict] | None = None,
        max_steps: int = 6,
        invalid_action_penalty: float = -0.1,
        opd_step_reward: float = 0.0,
        opd_max_credited_validates: int = 2,
        # Cursor-style product-penalty reward shaping (Composer 2 §4.2).
        # Penalize "started but not finished" tool patterns: declared section
        # types with no settings/blocks, schemas described but never used in
        # the submission, repeated identical fix-then-submit. Bonus for
        # describe-then-use (rewards exploration that pays off).
        empty_section_penalty: float = 0.0,    # per empty section in submitted JSON
        unused_describe_penalty: float = 0.0,  # per describe_X without using X type
        used_describe_bonus: float = 0.0,      # per describe_X used in submission
        repeat_submit_penalty: float = 0.0,    # for fix(X) -> submit(X) without iteration
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        api_shop_id: Optional[str] = None,
        api_theme_id: Optional[str] = None,
        api_timeout: int = 30,
    ) -> None:
        """If ``prompts`` is provided, use it directly (skips JSON load).
        Pre-loaded list shared across Ray workers via plasma object store.

        Validation is performed via SitemuseValidator (Shopify
        themeFilesUpsert GraphQL). HorizonEnvironment is only used for
        local theme metadata (list_sections / describe_section).
        """
        self.base = HorizonEnvironment(horizon_path)
        if prompts is None:
            raise ValueError(
                "HorizonAgentEnv requires `prompts` list (loaded by driver)"
            )
        self._prompts = prompts
        self.max_steps = max_steps
        self.invalid_action_penalty = invalid_action_penalty
        # Compiler-OPD: small step reward for first K fix calls
        # (paper §3.1 — "validate" action densifies signal). 0.0 = OPD off.
        self.opd_step_reward = opd_step_reward
        self.opd_max_credited_validates = opd_max_credited_validates
        # Cursor-style product penalties (all default 0 = disabled).
        self.empty_section_penalty = empty_section_penalty
        self.unused_describe_penalty = unused_describe_penalty
        self.used_describe_bonus = used_describe_bonus
        self.repeat_submit_penalty = repeat_submit_penalty

        validator_kwargs: dict[str, Any] = {"timeout": api_timeout}
        if api_url is not None:
            validator_kwargs["api_url"] = api_url
        if api_token is not None:
            validator_kwargs["token"] = api_token
        if api_shop_id is not None:
            validator_kwargs["shop_id"] = api_shop_id
        if api_theme_id is not None:
            validator_kwargs["theme_id"] = api_theme_id
        self.validator = SitemuseValidator(**validator_kwargs)

        self._current: Optional[dict] = None
        self._template_type: str = ""
        self._description: str = ""
        self._steps_done = 0
        self._finished = False
        self._last_won = False
        self._last_obs: str = ""
        # Per-episode tracking for product penalties.
        self._described_sections: set[str] = set()
        self._described_blocks: set[str] = set()
        self._last_fix_arg: str = ""

    @property
    def n_examples(self) -> int:
        return len(self._prompts)

    @staticmethod
    def _extract_description(ex: dict) -> str:
        """Pull the user message from a chat-format prompt entry."""
        if "description" in ex:
            return ex["description"]
        chat = ex.get("prompt") or []
        for msg in chat:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ex.get("question", "")

    def reset(self, idx: int) -> tuple[str, dict]:
        idx = int(idx) % len(self._prompts)
        ex = self._prompts[idx]
        self._current = ex
        self._template_type = ex.get("template_type", "page")
        self._description = self._extract_description(ex)
        self._steps_done = 0
        self._finished = False
        self._last_won = False
        self._fix_count = 0  # OPD: count of fix/submit "validate-like" calls
        # Reset per-episode tracking for product penalties.
        self._described_sections = set()
        self._described_blocks = set()
        self._last_fix_arg = ""

        obs_text = (
            f"Task: Generate a Shopify Horizon theme template.\n"
            f"Template type: {self._template_type}\n"
            f"Description: {self._description}\n\n"
            f"You may explore the theme first using list_sections[] or "
            f"describe_section[<name>] to inspect available pieces.\n"
            f"Submit your final answer with submit[<full JSON>]. "
            f"You can also fix[<JSON>] to test a draft and see compile errors "
            f"without committing."
        )
        self._last_obs = obs_text
        info = {
            "available_actions": [
                "list_sections[]",
                "describe_section[<name>]",
                "describe_block[<name>]",
                "submit[<full template JSON>]",
                "fix[<modified template JSON>]",
            ],
            "won": False,
            "task_score": 0.0,
            "template_type": self._template_type,
            "question_idx": idx,
        }
        return obs_text, info

    def step(self, inner_action: str) -> tuple[str, float, bool, dict]:
        info: dict = {
            "available_actions": [
                "list_sections[]",
                "describe_section[<name>]",
                "describe_block[<name>]",
                "submit[<full template JSON>]",
                "fix[<modified template JSON>]",
            ],
            "won": False,
            "task_score": 0.0,
            "template_type": self._template_type,
        }

        if self._finished:
            info["essa_anchor"] = ("done", "noop")
            return self._last_obs, 0.0, True, info

        self._steps_done += 1
        timeout = self._steps_done >= self.max_steps

        m = _VERB_RE.match(inner_action.strip()) if inner_action else None
        if m is None:
            obs, r, done, info = self._invalid_action(timeout, info)
            info["essa_anchor"] = essa_anchor(verb="", arg="", won=False, invalid=True, error_message="")
            return obs, r, done, info

        verb, arg = m.group(1), m.group(2).strip()

        if verb == "list_sections":
            obs, r, done, info = self._list_sections(timeout, info)
        elif verb == "describe_section":
            obs, r, done, info = self._describe(arg, kind="section", timeout=timeout, info=info)
        elif verb == "describe_block":
            obs, r, done, info = self._describe(arg, kind="block", timeout=timeout, info=info)
        elif verb == "submit":
            obs, r, done, info = self._submit(arg, terminal=True, info=info, verb="submit")
        elif verb == "fix":
            obs, r, done, info = self._submit(arg, terminal=timeout, info=info, verb="fix")
        else:
            obs, r, done, info = self._invalid_action(timeout, info)
            verb = ""

        info["essa_anchor"] = essa_anchor(
            verb=verb,
            arg=arg if verb in ("describe_section", "describe_block") else "",
            won=info.get("won", False),
            invalid=info.get("invalid_action", False),
            error_message=info.get("error_message", ""),
        )
        return obs, r, done, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _list_sections(self, timeout: bool, info: dict):
        sections = sorted(self.base.available_sections)
        if len(sections) > 60:
            sections = sections[:60] + ["... (truncated)"]
        obs = "Available sections:\n" + "\n".join(sections)
        if timeout:
            self._finished = True
        self._last_obs = obs
        return obs, 0.0, timeout, info

    def _describe(self, name: str, kind: str, timeout: bool, info: dict):
        name = name.strip().lstrip("_")
        if kind == "section":
            available = self.base.available_sections
            ext = "sections"
            if name in available:
                self._described_sections.add(name)
        else:
            available = self.base.available_blocks
            ext = "blocks"
            if name in available:
                self._described_blocks.add(name)

        if name not in available:
            obs = (
                f"{kind.capitalize()} '{name}' not found. "
                f"Use list_{ext}[] to see available {ext}."
            )
        else:
            path = Path(self.base.horizon_path) / ext / f"{name}.liquid"
            if not path.exists():
                path = Path(self.base.horizon_path) / ext / f"_{name}.liquid"
            try:
                text = path.read_text()
            except Exception as e:  # noqa: BLE001
                obs = f"Error reading {kind} '{name}': {e}"
                if timeout:
                    self._finished = True
                self._last_obs = obs
                return obs, 0.0, timeout, info

            schema_match = re.search(
                r"\{\%\s*schema\s*\%\}(.+?)\{\%\s*endschema\s*\%\}",
                text, re.DOTALL,
            )
            snippet = schema_match.group(1).strip() if schema_match else text[:2000]
            if len(snippet) > self.SCHEMA_PREVIEW_CHARS:
                snippet = snippet[: self.SCHEMA_PREVIEW_CHARS] + "\n... (truncated)"
            obs = f"{kind} '{name}' schema:\n{snippet}"

        if timeout:
            self._finished = True
        self._last_obs = obs
        return obs, 0.0, timeout, info

    def _submit(self, template_json: str, terminal: bool, info: dict, verb: str = "submit"):
        # Sitemuse expects "templates/<type>.json" filename; the
        # validator's _call_upsert_api handles the prefix internally
        # if no slash is given, but we pass the explicit path for clarity.
        file_path = f"templates/{self._template_type}.json"
        result = self.validator.validate(file_path, template_json)

        # Detect "fix(X) -> submit(X) without iteration" (Cursor-style
        # repeat-submit penalty). Compute BEFORE updating _last_fix_arg.
        repeat_no_iter = (
            verb == "submit"
            and self._last_fix_arg
            and self._normalize_json(template_json) == self._last_fix_arg
        )
        if verb == "fix":
            self._last_fix_arg = self._normalize_json(template_json)

        # Compiler-OPD: +step_reward for first K validate-like calls.
        # Both fix and submit count as "validate" since both hit the API.
        # Cap prevents the spam-validate hack (paper §3.1).
        self._fix_count += 1
        opd_step = (
            self.opd_step_reward
            if self._fix_count <= self.opd_max_credited_validates
            else 0.0
        )

        if result.all_passed:
            obs = "Template compiled successfully. ✓"
            self._finished = True
            self._last_won = True
            info["won"] = True
            info["task_score"] = 1.0
            info["validation"] = result.to_reward_dict()
            info["opd_step_reward"] = opd_step
            self._last_obs = obs
            # Cursor-style product penalties / bonus on compile-pass terminal.
            shape = self._compute_reward_shaping(template_json, repeat_no_iter)
            for k, v in shape.items():
                info[f"shape/{k}"] = v
            reward = 1.0 + opd_step + shape["total"]
            return obs, reward, True, info

        err = result.get_error_message()
        obs = (
            f"Template did not compile.\n"
            f"Error: {err}\n\n"
            f"Refine and try fix[<JSON>] again, or submit[<JSON>] to commit "
            f"as your final answer."
        )
        info["validation"] = result.to_reward_dict()
        info["error_message"] = err
        info["opd_step_reward"] = opd_step

        if terminal:
            self._finished = True
            self._last_won = False
            self._last_obs = obs
            # Terminal fail: only OPD step credit (no terminal reward)
            return obs, opd_step, True, info

        # Non-terminal fail (fix[] within budget): just OPD step credit
        self._last_obs = obs
        return obs, opd_step, False, info

    @staticmethod
    def _normalize_json(s: str) -> str:
        """Canonicalize JSON string for repeat-submit detection.
        Returns parsed-then-dumped form to ignore whitespace/key-order; falls
        back to the raw stripped string if parse fails (treat as opaque)."""
        try:
            import json as _json
            return _json.dumps(_json.loads(s), sort_keys=True, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            return (s or "").strip()

    def _compute_reward_shaping(self, template_json: str, repeat_no_iter: bool) -> dict:
        """Cursor-style product-penalty shaping (Composer 2 §4.2).

        Decomposes "good submission" into structural signals beyond compile:
            - empty_section_penalty: declared section with empty settings AND
              empty blocks (Cursor's "started but didn't finish" analog)
            - unused_describe_penalty: schemas explored via describe_section/
              describe_block but never used in submitted types
            - used_describe_bonus:    explored AND used (rewards
              exploration-pays-off behavior)
            - repeat_submit_penalty:  fix(X) -> submit(X) with identical JSON,
              i.e. "iterate" was theatrical

        All terms default 0 (disabled). On disable, returns total=0 and the
        reward function reduces to v3-essa baseline exactly.
        """
        out = {
            "empty_section": 0,
            "unused_describe": 0,
            "used_describe": 0,
            "repeat_submit": 0,
            "n_described_sections": len(self._described_sections),
            "n_described_blocks": len(self._described_blocks),
            "total": 0.0,
        }
        try:
            import json as _json
            parsed = _json.loads(template_json)
        except Exception:  # noqa: BLE001
            return out

        sections = parsed.get("sections") or {}
        if not isinstance(sections, dict):
            return out

        # Empty section count (declared type but no settings AND no blocks).
        empty_count = 0
        submitted_types: set[str] = set()
        submitted_block_types: set[str] = set()
        for sec in sections.values():
            if not isinstance(sec, dict):
                continue
            sec_type = sec.get("type")
            if isinstance(sec_type, str):
                submitted_types.add(sec_type)
            settings = sec.get("settings") or {}
            blocks = sec.get("blocks") or {}
            if not settings and not blocks:
                empty_count += 1
            if isinstance(blocks, dict):
                for blk in blocks.values():
                    if isinstance(blk, dict):
                        bt = blk.get("type")
                        if isinstance(bt, str):
                            submitted_block_types.add(bt.lstrip("_"))

        # Describe coverage.
        sec_used = self._described_sections & submitted_types
        sec_unused = self._described_sections - submitted_types
        blk_used = self._described_blocks & submitted_block_types
        blk_unused = self._described_blocks - submitted_block_types

        out["empty_section"] = empty_count
        out["used_describe"] = len(sec_used) + len(blk_used)
        out["unused_describe"] = len(sec_unused) + len(blk_unused)
        out["repeat_submit"] = int(repeat_no_iter)

        total = (
            -self.empty_section_penalty * empty_count
            -self.unused_describe_penalty * out["unused_describe"]
            +self.used_describe_bonus * out["used_describe"]
            -self.repeat_submit_penalty * out["repeat_submit"]
        )
        out["total"] = float(total)
        return out

    def _invalid_action(self, timeout: bool, info: dict):
        obs = (
            "Error: could not parse <action>. Respond with one of:\n"
            "  <action>list_sections[]</action>\n"
            "  <action>describe_section[<name>]</action>\n"
            "  <action>describe_block[<name>]</action>\n"
            "  <action>submit[<full template JSON>]</action>\n"
            "  <action>fix[<modified template JSON>]</action>"
        )
        if timeout:
            self._finished = True
        info["invalid_action"] = True
        self._last_obs = obs
        return obs, self.invalid_action_penalty, timeout, info
