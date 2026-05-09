#!/bin/bash
# Horizon v3-dfg: GiGPO + LoRA + Decidable-Fraction-Gated SFT distillation.
#
# DFG-RL idea:
#   At each step, compute decidable fraction
#       D = |groups with at least 1 pass AND at least 1 fail| / |all groups|
#   D measures how much actual GRPO/GiGPO advantage signal exists in the batch.
#
#   - D < threshold_low  (sparse)   : turn SFT-on-winners ON for this step
#                                     (compensate for vanishing PG gradient)
#   - D in [low, high]   (healthy)  : pure GiGPO, SFT OFF (avoid forward-KL
#                                     mode-covering damage to pretraining)
#   - D > threshold_high (saturated): log warning; future work: curriculum
#
# Why this beats v2-distill (which lost -22.7pp on OOD eval_fixed):
#   v2-distill ran SFT-on-winners always. With base v1 success ~73% on Horizon,
#   D was already ~0.3-0.5 (healthy) most of the training, so the always-on SFT
#   loss was adding forward-KL pressure when the policy didn't need it. RL's
#   Razor (Shenfeld et al., 2025) predicts forward-KL accumulation breaks
#   pretraining knowledge -> OOD drop.
#
#   DFG-RL only fires SFT when D is genuinely sparse (early sparse-reward
#   regime, or temporary dip), and otherwise lets pure RL preserve OOD.

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH=/root/miniconda3/bin:$PATH

num_cpus_per_env_worker=0.04
train_data_size=8
val_data_size=32
group_size=8
mode="mean_norm"

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-4B-Instruct-2507}
HORIZON_DATA_DIR=${HORIZON_DATA_DIR:-/root/autodl-tmp/datasets/horizon}
HORIZON_THEME_PATH=${HORIZON_THEME_PATH:-/root/autodl-tmp/datasets/horizon/theme}

export SITEMUSE_API=${SITEMUSE_API:-https://api.sitemuse.ai/muse/admin/liquid/theme/files/upsert}
export SITEMUSE_TOKEN=${SITEMUSE_TOKEN:-GqBOT5cvQk3S69e7KL8tjhuC1az20Hsi}
export SITEMUSE_SHOP_ID=${SITEMUSE_SHOP_ID:-cmovac01y0002r601h2f2euqc}
export HORIZON_THEME_ID=${HORIZON_THEME_ID:-gid://shopify/OnlineStoreTheme/156650045637}

# Distill alpha is the *active* weight when DFG decides to inject is_winner.
# Setting >0 is required for the actor to compile the SFT branch; whether it
# actually fires per-step is gated by DFG.
DISTILL_ALPHA=${DISTILL_ALPHA:-0.1}
DFG_LOW=${DFG_LOW:-0.30}
DFG_HIGH=${DFG_HIGH:-0.70}

python -m examples.data_preprocess.prepare_spider_dummy \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    +actor_rollout_ref.actor.distill_alpha=$DISTILL_ALPHA \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.dfg.enable=True \
    algorithm.dfg.threshold_low=$DFG_LOW \
    algorithm.dfg.threshold_high=$DFG_HIGH \
    env.env_name=Horizon \
    env.seed=0 \
    env.max_steps=6 \
    env.history_length=4 \
    env.rollout.n=$group_size \
    +env.horizon.data_dir=$HORIZON_DATA_DIR \
    +env.horizon.theme_path=$HORIZON_THEME_PATH \
    +env.horizon.split=train \
    +env.horizon.val_split=val \
    +env.horizon.max_steps=6 \
    +env.horizon.invalid_action_penalty=-0.1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_horizon' \
    trainer.experiment_name='gigpo_qwen3_4b_lora_horizon_v3_dfg' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=106 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.max_critic_ckpt_to_keep=2 $@
