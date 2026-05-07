#!/bin/bash
# Spider GiGPO + LoRA on a SINGLE A10 (24 GB) — Qwen3-4B-Instruct-2507.
# Adapted from run_webshop_lora.sh (which uses 7B + 2 GPUs); we shrink:
#   - 7B → 4B model
#   - 2 GPUs → 1 GPU (tensor_model_parallel_size=1)
#   - lora_rank 64 → 32
#   - ppo_micro_batch_size 8 → 4
#   - gpu_memory_utilization 0.6 → 0.5  (more headroom for backward + KV cache)

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.1

train_data_size=4       # 8 → 4: cuts train env workers 64 → 32
val_data_size=16        # 64 → 16: cuts val workers 64 → 16
group_size=8
mode="mean_norm"        # binary reward → mean_norm avoids degenerate std-norm
# Total env workers = train_data_size * group_size + val_data_size = 48
# (was 128, OOMed 28GB host @ ~80MB/worker Python baseline)

MODEL_PATH=${MODEL_PATH:-/mnt/data/models/Qwen3-4B-Instruct-2507}
SPIDER_DATA_DIR=${SPIDER_DATA_DIR:-/mnt/data/datasets/spider/spider_data}

# Data preprocess: generate dummy parquet WITHOUT downloading from HuggingFace
# (DSW often can't reach HF; geometry3k is only used for its row schema).
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
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_spider' \
    trainer.experiment_name='gigpo_qwen3_4b_lora_a10_v1' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=50 \
    trainer.val_before_train=True $@
