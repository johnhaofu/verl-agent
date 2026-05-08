"""Multi-turn Horizon eval — runs HorizonAgentEnv rollouts and reports pass rate.

Mirrors training conditions exactly (same env, same prompt template, same
projection). The only difference is no PPO update — just measurement.

Usage:
    python eval_horizon.py \
        --checkpoint /root/verl-agent/checkpoints/.../global_step_20 \
        --dataset all \
        --output /root/horizon_eval_step_20.json

Datasets handled:
    val           — 50 in-distribution prompts (matches val_split=val)
    test          — 50 held-out same-distribution (seed=1337)
    eval_fixed    — 22 OOD modify/add/create tasks (paper baseline set)
    all           — runs all three

For "base" checkpoint, pass --checkpoint "" or omit (uses base model only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Make our local horizon_env / sitemuse_validator importable
sys.path.insert(0, "/root/verl-agent")
sys.path.insert(0, "/root/autodl-tmp/datasets/horizon/lib")

from agent_system.environments.env_package.horizon.core_env import HorizonAgentEnv
from agent_system.environments.env_package.horizon.projection import horizon_projection
from agent_system.environments.prompts.horizon import (
    HORIZON_TEMPLATE,
    HORIZON_TEMPLATE_NO_HIS,
)


# ---------------------------------------------------------------------------

DATASET_PATHS = {
    "val":        "/root/autodl-tmp/datasets/horizon/prompts/val.jsonl",
    "test":       "/root/autodl-tmp/datasets/horizon/prompts/test.jsonl",
    "eval_fixed": "/root/autodl-tmp/datasets/horizon/prompts/eval_fixed.jsonl",
}

THEME_PATH = "/root/autodl-tmp/datasets/horizon/theme"
DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507"


# ---------------------------------------------------------------------------
# Prompt construction (matches HorizonEnvironmentManager.build_text_obs)
# ---------------------------------------------------------------------------

def build_full_prompt(
    obs: str,
    history: list[dict],
    history_length: int = 4,
) -> str:
    """Wrap the env obs in HORIZON_TEMPLATE / HORIZON_TEMPLATE_NO_HIS.

    Mirrors env_manager.HorizonEnvironmentManager.build_text_obs() logic so
    eval prompts match training prompts exactly.
    """
    if not history:
        return HORIZON_TEMPLATE_NO_HIS.format(current_observation=obs)

    # History is list of {"obs": str, "action": str}
    # Format the most recent N as "[Observation k: '...', Action k: '...']"
    recent = history[-history_length:]
    action_history_str = "\n".join(
        f"[Observation {i+1}: '{h['obs']}', Action {i+1}: '{h['action']}']"
        for i, h in enumerate(recent)
    )
    out = HORIZON_TEMPLATE.format(
        step_count=len(history),
        history_length=len(recent),
        action_history=action_history_str,
        current_step=len(history) + 1,
        current_observation=obs,
    )
    if len(out) > 13000:
        out = HORIZON_TEMPLATE_NO_HIS.format(current_observation=obs)
    return out


def wrap_chat_template(tokenizer, prompt: str) -> str:
    """Apply the model's chat template so generation matches training conditions."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Eval one dataset on one checkpoint
# ---------------------------------------------------------------------------

def run_episode(
    env: HorizonAgentEnv,
    task_idx: int,
    llm,
    sampling_params,
    tokenizer,
    lora_request,
    max_steps: int = 6,
    history_length: int = 4,
) -> dict:
    """Run one full multi-turn episode. Returns trajectory + final reward."""
    obs, info = env.reset(task_idx)
    history = []
    final = {"won": False, "reward": 0.0, "steps": 0, "trajectory": []}

    for step_idx in range(max_steps):
        full_prompt = build_full_prompt(obs, history, history_length=history_length)
        chat_prompt = wrap_chat_template(tokenizer, full_prompt)

        outputs = llm.generate(
            [chat_prompt],
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )
        response = outputs[0].outputs[0].text
        parsed, valids = horizon_projection([response])
        inner_action = parsed[0]
        is_valid = bool(valids[0])

        out_obs, reward, done, out_info = env.step(inner_action)
        history.append({"obs": obs, "action": response})
        final["trajectory"].append({
            "step": step_idx + 1,
            "action_inner": inner_action,
            "valid": is_valid,
            "reward": reward,
            "done": done,
            "obs_next_first200": out_obs[:200],
        })
        final["reward"] += reward
        final["steps"] = step_idx + 1
        if done:
            final["won"] = bool(out_info.get("won"))
            break
        obs = out_obs

    return final


def eval_dataset(
    dataset_name: str,
    llm,
    sampling_params,
    tokenizer,
    lora_request,
    max_tasks: Optional[int] = None,
) -> dict:
    """Run one whole dataset, return aggregate + per-task results."""
    path = DATASET_PATHS[dataset_name]
    with open(path) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    if max_tasks:
        prompts = prompts[:max_tasks]

    env = HorizonAgentEnv(
        horizon_path=THEME_PATH,
        prompts=prompts,
        max_steps=6,
        invalid_action_penalty=-0.1,
    )

    print(f"\n=== eval {dataset_name} (n={len(prompts)}) ===")
    t0 = time.time()
    per_task = []
    for i in range(len(prompts)):
        ep = run_episode(env, i, llm, sampling_params, tokenizer, lora_request)
        ex = prompts[i]
        per_task.append({
            "task_idx": i,
            "id": ex.get("id", str(i)),
            "template_type": ex.get("template_type", ex.get("base_template", "?")),
            "won": ep["won"],
            "reward_total": round(ep["reward"], 3),
            "n_steps": ep["steps"],
            "trajectory": ep["trajectory"],
        })
        if (i + 1) % 5 == 0 or i == len(prompts) - 1:
            won = sum(t["won"] for t in per_task)
            print(f"  {i+1}/{len(prompts)} done | passed {won}/{len(per_task)} "
                  f"({100*won/len(per_task):.1f}%) | "
                  f"elapsed {time.time()-t0:.0f}s")

    won = sum(t["won"] for t in per_task)
    n = len(per_task)
    return {
        "dataset": dataset_name,
        "n": n,
        "n_won": won,
        "pass_at_1": won / n if n else 0.0,
        "avg_reward": sum(t["reward_total"] for t in per_task) / n if n else 0.0,
        "avg_steps": sum(t["n_steps"] for t in per_task) / n if n else 0.0,
        "wall_seconds": time.time() - t0,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to LoRA adapter dir, or empty for base model")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["val", "test", "eval_fixed", "all"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Cap tasks per dataset for quick smoke test")
    # Default sampling matches training val_kwargs (run_horizon_lora.sh)
    # so eval pass@1 numbers are directly comparable with the val/success_rate
    # logged at step 0 / 10 / 20 / ... during training.
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="0.4 matches training val_kwargs.temperature")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Matches training data.max_response_length=2048")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    args = parser.parse_args()

    # ---- vLLM setup ----
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print(f"Loading base model: {args.base_model}")
    llm_kwargs = {
        "model": args.base_model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": 8192 + args.max_new_tokens,
        "dtype": "bfloat16",
    }
    lora_request = None
    if args.checkpoint:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32
        lora_request = LoRARequest("horizon_lora", 1, args.checkpoint)
        print(f"  + LoRA: {args.checkpoint}")

    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        stop=["</action>", "<|im_end|>"],
        include_stop_str_in_output=True,
    )

    # ---- run datasets ----
    datasets = ["val", "test", "eval_fixed"] if args.dataset == "all" else [args.dataset]
    results = {
        "base_model": args.base_model,
        "checkpoint": args.checkpoint or "BASE_ONLY",
        "temperature": args.temperature,
        "datasets": {},
    }
    for ds in datasets:
        results["datasets"][ds] = eval_dataset(
            ds, llm, sampling_params, tokenizer, lora_request,
            max_tasks=args.max_tasks,
        )

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {args.output}")
    print("\n=== SUMMARY ===")
    for ds, r in results["datasets"].items():
        print(f"  {ds:12s} pass@1 = {r['pass_at_1']:.3f}  "
              f"reward_avg = {r['avg_reward']:.3f}  "
              f"steps_avg = {r['avg_steps']:.2f}  "
              f"({r['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
