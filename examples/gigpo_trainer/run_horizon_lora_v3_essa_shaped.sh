#!/bin/bash
# Horizon v3-essa: GiGPO + LoRA + Error-State Step Anchor (ESSA).
#
# Premise:
#   GiGPO's inner group requires environment states to recur across rollouts
#   (proven on ALFWorld rooms / WebShop pages). Horizon's state space is
#   JSON-parameterized -> states never repeat -> cluster size collapses to 1
#   for fix/submit actions starting from step 2.
#
# ESSA replaces the default observation-string anchor with a canonical
# fingerprint of the validator's error template:
#
#     "Section type 'featured-collection' does not refer..."
#     "Section type 'cta' does not refer..."
#     "Section type 'team-members' does not refer..."
#         -> all canonicalize to:
#            "Section type '<X>' does not refer..."
#
# Two trajectories that wrote totally different JSON but face the same
# outstanding error are at the same problem-instance; their next moves are
# directly comparable. ESSA preserves GiGPO's inner-group machinery while
# making it functional in continuous-state-space environments.
#
# Implementation:
#   - agent_system/environments/env_package/horizon/error_anchor.py
#       : canonicalize_error / essa_anchor helpers
#   - core_env.py emits info["essa_anchor"] per step
#   - env_manager.py HorizonEnvironmentManager.step() now passes
#     essa_anchor as the "anchor" obs key (was next_obs.copy())
#
# Caveat: step-0 anchor is the initial prompt (shared across rollouts of
# the same outer group), so step-0 step_adv double-counts episode_adv.
# Step 1+ is where ESSA's signal lives.

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

# Pure GiGPO with ESSA anchor. No DFG distill (orthogonal experiment).
# Reduce step_advantage_w to dampen step-0 double-counting; 0.5 is a
# pragmatic choice that keeps inner signal while not over-amplifying step 0.
STEP_ADV_W=${STEP_ADV_W:-0.5}

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
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=$STEP_ADV_W \
    algorithm.gigpo.mode=$mode \
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
    +env.horizon.empty_section_penalty=0.05 \
    +env.horizon.unused_describe_penalty=0.03 \
    +env.horizon.used_describe_bonus=0.04 \
    +env.horizon.repeat_submit_penalty=0.10 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_horizon' \
    trainer.experiment_name='gigpo_qwen3_4b_lora_horizon_v3_essa_shaped' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=106 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.max_critic_ckpt_to_keep=2 $@
