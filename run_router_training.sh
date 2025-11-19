#!/bin/bash
# Router Training Script for ToolRL
# Usage: bash run_router_training.sh

# Set these variables before running (adjust as needed)
export DATA_DIR="./dataset/rlla_4k"
export BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # e.g., Qwen/Qwen2.5-7B-Instruct
export EXPERIMENT_NAME="router-grpo-$(date +%Y%m%d-%H%M%S)"
export RAY_memory_usage_threshold=0.99

echo "DATA_DIR=${DATA_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"

# Run GRPO training with router enabled (HuggingFace rollout backend)
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    router.enable=true \
    router.budget_B=0.3 \
    router.cost_weights.any_tool=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=router_grpo \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=200 \
    trainer.save_on_exit=true \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.use_reference_policy=false

