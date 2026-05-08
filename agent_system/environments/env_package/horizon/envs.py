"""Ray-parallelised Horizon env wrapper for verl-agent.

Mirrors spider/envs.py: a Ray actor per env worker plus a vectorised
wrapper exposing step()/reset() over a batch. The prompt list is shared
via Ray plasma so workers don't each parse the JSONL.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import gym
import numpy as np
import ray

from .core_env import HorizonAgentEnv


# Override via env var HORIZON_DATA_DIR / HORIZON_THEME_PATH
_DEFAULT_DATA_DIR = os.environ.get(
    "HORIZON_DATA_DIR",
    "/root/autodl-tmp/datasets/horizon",
)
_DEFAULT_THEME_PATH = os.environ.get(
    "HORIZON_THEME_PATH",
    "/root/autodl-tmp/datasets/horizon/theme",
)


# ---------------------------------------------------------------------------
# Ray remote worker actor ---------------------------------------------------
# ---------------------------------------------------------------------------

class HorizonWorker:
    """Ray remote actor hosting one HorizonAgentEnv instance.

    Receives the prompt list as a Ray ObjectRef (boxed in dict to dodge
    Ray auto-deref) so all workers share one copy in plasma.
    """

    def __init__(self, seed: int, env_kwargs: dict, prompts_ref_box: dict):
        import ray as _ray
        prompts = _ray.get(prompts_ref_box["ref"])
        self.env = HorizonAgentEnv(
            horizon_path=env_kwargs.get("theme_path", _DEFAULT_THEME_PATH),
            prompts=prompts,
            max_steps=env_kwargs.get("max_steps", 6),
            invalid_action_penalty=env_kwargs.get("invalid_action_penalty", -0.1),
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


# ---------------------------------------------------------------------------
# Vectorised Ray environment ------------------------------------------------
# ---------------------------------------------------------------------------

class HorizonMultiProcessEnv(gym.Env):
    """Vectorised wrapper around HorizonAgentEnv via Ray actors.

    Group structure mirrors spider/webshop: env_num × group_n total
    workers, where each "group" of group_n workers shares the same
    reset() idx so GiGPO-style group-relative advantage works.
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

        # Driver loads prompts once and shares via Ray plasma.
        prompts = self._load_prompts(
            self._env_kwargs.get("data_dir", _DEFAULT_DATA_DIR),
            self._env_kwargs.get("split", "train"),
        )
        prompts_ref = ray.put(prompts)
        ref_box = {"ref": prompts_ref}

        env_worker = ray.remote(**resources_per_worker)(HorizonWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(
                seed + (i // self.group_n),
                self._env_kwargs,
                ref_box,
            )
            self._workers.append(worker)

        self._n_examples = ray.get(self._workers[0].n_examples.remote())
        self.question_idxs = list(range(self._n_examples))
        self._epoch_pool: list = []

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
        """Train: deterministic shuffled-pool draining (full coverage).
        Val: deterministic prefix for stable cross-checkpoint signal.
        """
        if self.is_train:
            if len(self._epoch_pool) < self.env_num:
                fresh = list(self.question_idxs)
                self._rng.shuffle(fresh)
                self._epoch_pool = fresh
            idx = [self._epoch_pool.pop() for _ in range(self.env_num)]
        else:
            idx = self.question_idxs[: self.env_num]

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
    def _load_prompts(data_dir, split: str) -> list:
        """Load prompts JSONL once in the driver. Returns list[dict]."""
        if split == "train":
            fname = "train.jsonl"
        elif split in ("val", "validation", "dev"):
            fname = "val.jsonl"
        elif split in ("eval", "test"):
            fname = "eval_fixed.jsonl"
        else:
            raise ValueError(f"unknown split {split!r}")

        path = Path(data_dir) / "prompts" / fname
        if not path.exists():
            raise FileNotFoundError(f"missing Horizon prompts file: {path}")
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

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


# ---------------------------------------------------------------------------
# Factory -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_horizon_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: Optional[dict] = None,
):
    return HorizonMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
