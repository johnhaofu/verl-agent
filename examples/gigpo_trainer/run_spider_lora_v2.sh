#!/bin/bash
# Spider GiGPO + LoRA v2 — full-coverage training over all 7000 train_spider questions.
#
# Changes from v1:
#   - train_data_size: 8 → 32       (4× more questions per PPO step)
#   - total_epochs:    50 → 220     (220 × 32 = 7040 unique samples ≈ 1 full epoch)
#   - ppo_mini_batch:  32 → 128     (matches new batch)
#   - ppo_micro_batch: 4 → 8        (utilize A800 80GB more)
#   - Deterministic shuffled-pool sampling in SpiderMultiProcessEnv → guaranteed
#     full Spider train coverage (no random duplicates within an epoch)
#   - Val side now uses fixed prefix [0:64] of dev (deterministic) → stable
#     cross-checkpoint val signal (v1 saw 0.49-0.85 swings from random val
#     samples)
#
# Wall-time estimate: ~15-20 hours on A800 (vs v1's 1.75 hr).
# Disk: ~20 GB peak (1 active FSDP + 22 LoRA archives if save_freq=10)

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

num_cpus_per_env_worker=0.05

train_data_size=32      # v1: 8 → 32 (4× batch)
val_data_size=64
group_size=8
mode="mean_norm"
# Total env workers = train_data_size * group_size + val_data_size
#                   = 32 * 8 + 64 = 320

MODEL_PATH=${MODEL_PATH:-/mnt/data/models/Qwen3-4B-Instruct-2507}
SPIDER_DATA_DIR=${SPIDER_DATA_DIR:-/mnt/data/datasets/spider/spider_data}

python3 -m examples.data_preprocess.prepare_spider_dummy \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=Spider \
    env.seed=0 \
    env.max_steps=6 \
    env.history_length=4 \
    env.rollout.n=$group_size \
    +env.spider.data_dir=$SPIDER_DATA_DIR \
    +env.spider.split=train \
    +env.spider.val_split=validation \
    +env.spider.schema_max_chars=4000 \
    +env.spider.rows_per_query=10 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_spider' \
    trainer.experiment_name='gigpo_qwen3_4b_lora_v2_full_train' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=220 \
    trainer.val_before_train=True $@
