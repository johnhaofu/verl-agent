"""Ray-parallelised Spider env wrapper for verl-agent.

Mirrors the structure of webshop/envs.py: a Ray actor per env worker,
plus a vectorised wrapper that exposes step()/reset() over a batch.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gym
import numpy as np
import ray

from .sql_env import SpiderSQLEnv


# Default Spider data dir — override via env var SPIDER_DATA_DIR
_DEFAULT_DATA_DIR = os.environ.get(
    "SPIDER_DATA_DIR",
    "/mnt/data/datasets/spider/spider_data",
)


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class SpiderWorker:
    """Ray remote actor hosting one SpiderSQLEnv instance.

    Receives the example list as a Ray ObjectRef so all workers share
    one copy in plasma store (avoids loading 7000-row JSON 100× into
    each Python process — that OOMs the host on 28GB RAM nodes).
    """

    def __init__(self, seed: int, env_kwargs: dict, examples_ref_box: dict):
        """`examples_ref_box` is `{"ref": ObjectRef}` — boxed in a dict
        to prevent Ray's auto-dereference from unpacking the example list
        into separate positional args.
        """
        import ray as _ray
        examples = _ray.get(examples_ref_box["ref"])  # zero-copy from plasma store
        self.env = SpiderSQLEnv(
            data_dir=env_kwargs.get("data_dir", _DEFAULT_DATA_DIR),
            split=env_kwargs.get("split", "train"),
            schema_max_chars=env_kwargs.get("schema_max_chars", 4000),
            rows_per_query=env_kwargs.get("rows_per_query", 10),
            examples=examples,
        )
        self._n_examples = self.env.n_examples

    def step(self, action: str):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, idx: int):
        obs, info = self.env.reset(idx)
        return obs, info

    def n_examples(self) -> int:
        return self._n_examples

    def close(self):
        pass


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class SpiderMultiProcessEnv(gym.Env):
    """Vectorised wrapper around SpiderSQLEnv via Ray actors.

    Group structure mirrors webshop: env_num × group_n total workers,
    where each "group" of group_n workers shares the same reset() idx
    so GiGPO-style group-relative advantage works correctly.
    """

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1, "val should not use group_n > 1"

        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs or {}

        # Load Spider examples ONCE in the driver and put into Ray plasma
        # store so all workers share a single zero-copy reference. Without
        # this, each Python worker process loads ~25 MB JSON × 100+ workers
        # → 2.5+ GB extra RAM, OOMing 28 GB hosts (DSW A10).
        examples = self._load_examples_for_split(
            self._env_kwargs.get("data_dir", _DEFAULT_DATA_DIR),
            self._env_kwargs.get("split", "train"),
        )
        examples_ref = ray.put(examples)
        # Box the ObjectRef in a dict — Ray auto-dereferences positional
        # ObjectRef args, and if the value is a list it unpacks them into
        # separate positional params (Ray quirk). Boxing in dict prevents.
        ref_box = {"ref": examples_ref}

        env_worker = ray.remote(**resources_per_worker)(SpiderWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(
                seed + (i // self.group_n),
                self._env_kwargs,
                ref_box,
            )
            self._workers.append(worker)

        # All workers share the same dataset; query the first one
        self._n_examples = ray.get(self._workers[0].n_examples.remote())

        # Selectable index range. Train uses all train_spider questions;
        # val uses all dev questions (or test if split=test).
        self.question_idxs = list(range(self._n_examples))

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} actions, got {len(actions)}"
            )

        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures)

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Pick env_num distinct question indices, repeat each group_n times
        # so siblings in a group share the same question (GRPO group).
        idx = self._rng.choice(
            self.question_idxs, size=self.env_num, replace=False
        )
        idx = np.repeat(idx, self.group_n).tolist()

        futures = [w.reset.remote(int(i)) for w, i in zip(self._workers, idx)]
        results = ray.get(futures)

        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Helpers -----------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _load_examples_for_split(data_dir, split: str) -> list:
        """Load Spider examples once in the driver. Returns list[dict]."""
        import json
        from pathlib import Path
        if split == "train":
            files = ["train_spider.json"]
        elif split in ("validation", "dev"):
            files = ["dev.json"]
        elif split == "test":
            files = ["test.json"]
        else:
            raise ValueError(f"unknown split {split!r}")
        items: list = []
        for fname in files:
            with open(Path(data_dir) / fname) as fh:
                items.extend(json.load(fh))
        return items

    # ------------------------------------------------------------------
    # Cleanup -----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, "_closed", False):
            return
        close_futures = [w.close.remote() for w in self._workers]
        ray.get(close_futures)
        for w in self._workers:
            ray.kill(w)
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Factory ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_spider_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: Optional[dict] = None,
):
    return SpiderMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
