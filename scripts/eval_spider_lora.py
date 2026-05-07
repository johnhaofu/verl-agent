"""Evaluate Spider LoRA adapters on full dev set.

Loads each LoRA on top of Qwen3-4B base via vLLM, runs the agentic
multi-turn loop (sql exploration + answer commit) on dev questions,
and reports pass@1 (EX match rate).

Usage:
    python scripts/eval_spider_lora.py \
        --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
        --lora-dir /root/autodl-tmp/lora_only \
        --steps 10,20,30,40,50 \
        --spider-data /root/autodl-tmp/datasets/spider/spider_data \
        --split validation \
        --n 1034 \
        --max-turns 6 \
        --max-tokens 512 \
        --history-length 4 \
        --output /root/autodl-tmp/v1_eval_results.json

Architecture:
    - Active set of questions, each with own SpiderSQLEnv state
    - Per-turn: gather prompts of all active episodes, batch through
      vllm.generate, parse actions, step envs
    - Loop until all episodes terminal or max_turns reached
    - Aggregate info["won"] per LoRA
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add verl-agent root to path so we can import spider env + projection
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_system.environments.env_package.spider.sql_env import SpiderSQLEnv
from agent_system.environments.env_package.spider.projection import spider_projection
from agent_system.environments.prompts.spider import (
    SPIDER_TEMPLATE,
    SPIDER_TEMPLATE_NO_HIS,
)


def build_prompt(env_state: dict, history_length: int) -> str:
    """Build prompt for one episode given its history."""
    obs = env_state["current_obs"]
    history = env_state["history"]  # list of {"obs": ..., "action": ...}

    if not history or history_length <= 0:
        return SPIDER_TEMPLATE_NO_HIS.format(current_observation=obs)

    recent = history[-history_length:]
    valid_len = len(recent)
    start_index = len(history) - valid_len
    action_history = ""
    for j, record in enumerate(recent):
        n = start_index + j + 1
        action_history += (
            f"\n[Observation {n}: '{record['obs']}', Action {n}: '{record['action']}']"
        )
    if len(action_history) > 10000:
        action_history = "... " + action_history[-10000:]

    template = SPIDER_TEMPLATE.format(
        step_count=len(history),
        history_length=valid_len,
        action_history=action_history.strip(),
        current_step=len(history) + 1,
        current_observation=obs,
    )
    if len(template) > 13000:
        return SPIDER_TEMPLATE_NO_HIS.format(current_observation=obs)
    return template


def chat_format(prompt: str, tokenizer) -> str:
    """Apply Qwen3 chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_eval(
    llm,
    tokenizer,
    sampling_params,
    lora_request,
    spider_data: str,
    split: str,
    n: int,
    max_turns: int,
    history_length: int,
) -> dict:
    """Run multi-turn agent loop on n dev questions, return stats."""
    # Build per-question env state
    states: list[dict] = []
    envs: list[SpiderSQLEnv] = []
    for i in range(n):
        env = SpiderSQLEnv(data_dir=spider_data, split=split)
        if i == 0:
            print(f"  loaded {env.n_examples} examples for split={split}")
        obs, info = env.reset(i)
        envs.append(env)
        states.append({
            "qid": i,
            "current_obs": obs,
            "history": [],
            "done": False,
            "won": False,
            "n_turns": 0,
            "n_invalid": 0,
        })

    # Multi-turn loop, batched across questions per turn
    for turn in range(max_turns):
        active_idx = [i for i, s in enumerate(states) if not s["done"]]
        if not active_idx:
            break

        # Build prompts batch
        prompts = []
        for i in active_idx:
            p = build_prompt(states[i], history_length)
            prompts.append(chat_format(p, tokenizer))

        # Batch generate
        t0 = time.perf_counter()
        if lora_request is not None:
            outputs = llm.generate(
                prompts, sampling_params, lora_request=lora_request, use_tqdm=False
            )
        else:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        t_gen = time.perf_counter() - t0

        # Parse + step envs
        responses = [out.outputs[0].text for out in outputs]
        actions, valids = spider_projection(responses)

        for k, i in enumerate(active_idx):
            s = states[i]
            action = actions[k]
            valid = valids[k]
            # `valid` flag tracks <think> + ASCII-only quality (mirrors training
            # invalid_action_penalty). Action execution should NOT depend on it
            # — base model often produces prose reasoning without <think> tags
            # but the SQL is still parseable and should run.
            if action == "":
                s["n_invalid"] += 1
                obs_text, reward, done, info = envs[i].step("")
            else:
                if not valid:
                    s["n_invalid"] += 1  # track but still execute
                obs_text, reward, done, info = envs[i].step(action)

            s["history"].append({"obs": s["current_obs"], "action": action})
            s["current_obs"] = obs_text
            s["n_turns"] += 1
            if done:
                s["done"] = True
                s["won"] = bool(info.get("won", False))

        n_active = len(active_idx)
        n_done = sum(1 for s in states if s["done"])
        n_won = sum(1 for s in states if s["won"])
        print(
            f"  turn {turn+1}: gen {t_gen:.1f}s on {n_active} active "
            f"→ {n_done}/{n} done ({n_won} won)"
        )

    # Final tallies
    n_done = sum(1 for s in states if s["done"])
    n_won = sum(1 for s in states if s["won"])
    avg_turns = sum(s["n_turns"] for s in states) / n
    avg_invalid = sum(s["n_invalid"] for s in states) / n

    return {
        "n": n,
        "n_done": n_done,
        "n_won": n_won,
        "pass_at_1": n_won / n,
        "completion_rate": n_done / n,
        "avg_turns": avg_turns,
        "avg_invalid_actions": avg_invalid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="path to Qwen3-4B base model")
    parser.add_argument("--lora-dir", required=True, help="dir containing step_X/ subdirs")
    parser.add_argument("--steps", default="10,20,30,40,50", help="comma-separated step numbers")
    parser.add_argument("--spider-data", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n", type=int, default=1034)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument("--output", default="/root/autodl-tmp/v1_eval_results.json")
    args = parser.parse_args()

    print(f"[setup] base={args.base}")
    print(f"[setup] LoRA dir={args.lora_dir}, steps={args.steps}")
    print(f"[setup] split={args.split}, n={args.n}, max_turns={args.max_turns}")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    llm = LLM(
        model=args.base,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</action>"],
        include_stop_str_in_output=True,
    )

    steps = [int(x) for x in args.steps.split(",")]

    # First: baseline (no LoRA)
    results = {}
    print(f"\n[baseline] no LoRA — evaluating Qwen3-4B-Instruct base")
    t0 = time.perf_counter()
    results["base"] = run_eval(
        llm, tokenizer, sampling_params,
        lora_request=None,
        spider_data=args.spider_data,
        split=args.split,
        n=args.n,
        max_turns=args.max_turns,
        history_length=args.history_length,
    )
    results["base"]["wall_seconds"] = time.perf_counter() - t0
    print(f"  base: pass@1={results['base']['pass_at_1']:.3f} "
          f"({results['base']['n_won']}/{args.n}) "
          f"wall={results['base']['wall_seconds']:.0f}s")

    # Each LoRA
    for step in steps:
        lora_path = Path(args.lora_dir) / f"step_{step}"
        if not lora_path.exists():
            print(f"  ⚠️ skipping step_{step}: not found at {lora_path}")
            continue
        print(f"\n[step_{step}] {lora_path}")
        t0 = time.perf_counter()
        lora_request = LoRARequest(
            lora_name=f"step_{step}",
            lora_int_id=step,
            lora_path=str(lora_path),
        )
        result = run_eval(
            llm, tokenizer, sampling_params,
            lora_request=lora_request,
            spider_data=args.spider_data,
            split=args.split,
            n=args.n,
            max_turns=args.max_turns,
            history_length=args.history_length,
        )
        result["wall_seconds"] = time.perf_counter() - t0
        results[f"step_{step}"] = result
        print(f"  step_{step}: pass@1={result['pass_at_1']:.3f} "
              f"({result['n_won']}/{args.n}) "
              f"wall={result['wall_seconds']:.0f}s")

    # Save + summary
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out_path}")

    print("\n[summary]")
    print(f"{'tag':<12} {'pass@1':<8} {'avg_turns':<10} {'invalid':<8} {'wall':<6}")
    for tag, r in results.items():
        print(f"{tag:<12} {r['pass_at_1']:.3f}    "
              f"{r['avg_turns']:.2f}      {r['avg_invalid_actions']:.2f}    "
              f"{r['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
