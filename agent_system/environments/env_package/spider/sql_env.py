"""Standalone Spider SQL agent (no external deps beyond stdlib).

Ported from mlx-agent-rl with minimal interface changes:
- reset(idx) -> (obs_text, info_dict)         (no Observation wrapper)
- step(action) -> (obs_text, reward, done, info_dict)
- info has 'won', 'task_score', 'available_actions' keys (verl-agent contract)

The SQLite verifier (_exec_match) is unchanged from mlx-agent-rl.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _is_order_sensitive(sql: str) -> bool:
    s = sql.lower()
    return " order by " in s or " limit " in s


def _normalise_rows(rows: list, ordered: bool) -> Any:
    norm_rows = [tuple(repr(c) for c in row) for row in rows]
    if not ordered:
        norm_rows = [tuple(sorted(r)) for r in norm_rows]
        return tuple(sorted(norm_rows))
    return tuple(norm_rows)


def _exec_match(pred_sql: str, gold_sql: str, db_path: Path) -> tuple[bool, str]:
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = str
    except sqlite3.Error as exc:
        return False, f"db connect: {exc}"
    try:
        try:
            pred_rows = conn.execute(pred_sql).fetchall()
        except sqlite3.Error as exc:
            return False, str(exc)
        try:
            gold_rows = conn.execute(gold_sql).fetchall()
        except sqlite3.Error as exc:
            return False, f"gold failed: {exc}"
        ordered = _is_order_sensitive(gold_sql)
        return (
            _normalise_rows(pred_rows, ordered) == _normalise_rows(gold_rows, ordered),
            "",
        )
    finally:
        conn.close()


class SpiderSQLEnv:
    """One Spider question per episode. Stateful, single-process.

    obs format (init):
        Database: {db_id}
        Schema:
        {CREATE TABLE ...; ...}

        Question: {question}

    obs format (after sql[] step):
        SQL ok (N rows shown):
        ...rows...
        OR
        SQL error: ...

    obs format (after answer[] step):
        Answer recorded. EX=1 (match)  / EX=0 (...)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        schema_max_chars: int = 4000,
        rows_per_query: int = 10,
    ) -> None:
        self.data_dir = Path(data_dir)
        if split == "test":
            self.db_dir = self.data_dir / "test_database"
        else:
            self.db_dir = self.data_dir / "database"
        self.split = split
        self.schema_max_chars = schema_max_chars
        self.rows_per_query = rows_per_query

        self._examples = self._load_examples(split)
        self._schema_cache: dict[str, str] = {}

        self._current: Optional[dict] = None
        self._db_path: Optional[Path] = None
        self._gold_sql: Optional[str] = None
        self._finished = False
        self._steps_done = 0
        self._last_match = False
        self._last_obs: str = ""

    @property
    def n_examples(self) -> int:
        return len(self._examples)

    def reset(self, idx: int) -> tuple[str, dict]:
        idx = int(idx) % len(self._examples)
        ex = self._examples[idx]
        db_id = ex["db_id"]
        db_path = self.db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"missing Spider DB: {db_path}")

        self._current = ex
        self._db_path = db_path
        self._gold_sql = ex["query"]
        self._finished = False
        self._steps_done = 0
        self._last_match = False

        question = ex["question"]
        schema_block = self._render_schema(db_id, db_path)
        obs_text = (
            f"Database: {db_id}\n"
            f"Schema:\n{schema_block}\n\n"
            f"Question: {question}"
        )
        self._last_obs = obs_text
        info = {
            "available_actions": ["sql[<query>]", "answer[<final SELECT>]"],
            "won": False,
            "task_score": 0.0,
            "db_id": db_id,
            "question_idx": idx,
        }
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        # Default info
        info: dict = {
            "available_actions": ["sql[<query>]", "answer[<final SELECT>]"],
            "won": False,
            "task_score": 0.0,
        }

        if self._finished:
            return self._last_obs, 0.0, True, info

        m = re.match(r"^(sql|answer)\[(.+)\]\s*$", action, re.DOTALL)
        if not m:
            obs_text = (
                "Error: action must be 'sql[query]' or 'answer[query]'. "
                "Try again."
            )
            self._last_obs = obs_text
            return obs_text, 0.0, False, info

        verb, arg = m.group(1), m.group(2).strip()
        self._steps_done += 1

        if verb == "sql":
            return self._handle_sql(arg, info)
        return self._handle_answer(arg, info)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_examples(self, split: str) -> list[dict]:
        if split == "train":
            files = ["train_spider.json"]
        elif split in ("validation", "dev"):
            files = ["dev.json"]
        elif split == "test":
            files = ["test.json"]
        else:
            raise ValueError(f"unknown split {split!r}")
        items: list[dict] = []
        for fname in files:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"missing Spider split file: {path}")
            with open(path) as fh:
                items.extend(json.load(fh))
        return items

    def _render_schema(self, db_id: str, db_path: Path) -> str:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        schema = "\n\n".join(r[0].strip() + ";" for r in rows if r[0])
        if len(schema) > self.schema_max_chars:
            schema = schema[: self.schema_max_chars] + "\n-- (truncated)"
        self._schema_cache[db_id] = schema
        return schema

    def _handle_sql(self, query: str, info: dict) -> tuple[str, float, bool, dict]:
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.text_factory = str
            rows = conn.execute(query).fetchmany(self.rows_per_query + 1)
        except sqlite3.Error as exc:
            obs_text = f"SQL error: {exc}"
            self._last_obs = obs_text
            return obs_text, 0.0, False, info
        finally:
            try:
                conn.close()
            except Exception:
                pass
        truncated = len(rows) > self.rows_per_query
        rows = rows[: self.rows_per_query]
        if not rows:
            preview = "(0 rows)"
        else:
            preview = "\n".join(repr(r) for r in rows)
            if truncated:
                preview += f"\n… (truncated to {self.rows_per_query} rows)"
        obs_text = f"SQL ok ({len(rows)} rows shown):\n{preview}"
        self._last_obs = obs_text
        return obs_text, 0.0, False, info

    def _handle_answer(self, query: str, info: dict) -> tuple[str, float, bool, dict]:
        match, err = _exec_match(query, self._gold_sql, self._db_path)
        self._finished = True
        self._last_match = match
        if match:
            obs_text = "Answer recorded. EX=1 (match)"
            reward = 1.0
            info["won"] = True
            info["task_score"] = 1.0
        elif err:
            obs_text = f"Answer recorded. EX=0 ({err})"
            reward = 0.0
        else:
            obs_text = "Answer recorded. EX=0 (different result)"
            reward = 0.0
        self._last_obs = obs_text
        return obs_text, reward, True, info
