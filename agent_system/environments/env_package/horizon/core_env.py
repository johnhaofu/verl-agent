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
            return self._last_obs, 0.0, True, info

        self._steps_done += 1
        timeout = self._steps_done >= self.max_steps

        m = _VERB_RE.match(inner_action.strip()) if inner_action else None
        if m is None:
            return self._invalid_action(timeout, info)

        verb, arg = m.group(1), m.group(2).strip()

        if verb == "list_sections":
            return self._list_sections(timeout, info)
        if verb == "describe_section":
            return self._describe(arg, kind="section", timeout=timeout, info=info)
        if verb == "describe_block":
            return self._describe(arg, kind="block", timeout=timeout, info=info)
        if verb == "submit":
            return self._submit(arg, terminal=True, info=info)
        if verb == "fix":
            return self._submit(arg, terminal=timeout, info=info)
        return self._invalid_action(timeout, info)

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
        else:
            available = self.base.available_blocks
            ext = "blocks"

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

    def _submit(self, template_json: str, terminal: bool, info: dict):
        # Sitemuse expects "templates/<type>.json" filename; the
        # validator's _call_upsert_api handles the prefix internally
        # if no slash is given, but we pass the explicit path for clarity.
        file_path = f"templates/{self._template_type}.json"
        result = self.validator.validate(file_path, template_json)
        if result.all_passed:
            obs = "Template compiled successfully. ✓"
            self._finished = True
            self._last_won = True
            info["won"] = True
            info["task_score"] = 1.0
            info["validation"] = result.to_reward_dict()
            self._last_obs = obs
            return obs, 1.0, True, info

        err = result.get_error_message()
        obs = (
            f"Template did not compile.\n"
            f"Error: {err}\n\n"
            f"Refine and try fix[<JSON>] again, or submit[<JSON>] to commit "
            f"as your final answer."
        )
        info["validation"] = result.to_reward_dict()
        info["error_message"] = err

        if terminal:
            self._finished = True
            self._last_won = False
            self._last_obs = obs
            return obs, 0.0, True, info

        # Non-terminal fail (fix[] within budget)
        self._last_obs = obs
        return obs, 0.0, False, info

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
