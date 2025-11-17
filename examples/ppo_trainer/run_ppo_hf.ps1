param(
    [string]$DataDir = "dataset/rlla_4k",
    [string]$BaseModel = "",
    [string]$ExperimentName = "router-hf",
    [int]$GPUs = 1
)

if (-not $BaseModel) {
    Write-Error "Please pass -BaseModel pointing to your HF model (e.g. Qwen/Qwen2.5-7B-Instruct)."
    exit 1
}

python -m verl.trainer.main_ppo `
    data.train_files=$DataDir/train.parquet `
    data.val_files=$DataDir/test.parquet `
    data.train_batch_size=512 `
    data.val_batch_size=128 `
    data.max_prompt_length=1024 `
    data.max_response_length=512 `
    actor_rollout_ref.model.path=$BaseModel `
    actor_rollout_ref.actor.optim.lr=1e-6 `
    actor_rollout_ref.model.use_remove_padding=True `
    actor_rollout_ref.actor.ppo_mini_batch_size=128 `
    actor_rollout_ref.actor.ppo_micro_batch_size=8 `
    actor_rollout_ref.actor.fsdp_config.param_offload=False `
    actor_rollout_ref.actor.fsdp_config.grad_offload=False `
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False `
    actor_rollout_ref.rollout.name=hf `
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 `
    actor_rollout_ref.rollout.micro_batch_size=8 `
    actor_rollout_ref.rollout.temperature=1.0 `
    actor_rollout_ref.rollout.top_p=0.95 `
    actor_rollout_ref.rollout.top_k=0 `
    actor_rollout_ref.rollout.do_sample=True `
    actor_rollout_ref.ref.fsdp_config.param_offload=True `
    critic.optim.lr=1e-5 `
    critic.model.use_remove_padding=True `
    critic.model.path=$BaseModel `
    critic.model.enable_gradient_checkpointing=False `
    critic.ppo_mini_batch_size=128 `
    critic.ppo_micro_batch_size=8 `
    critic.model.fsdp_config.param_offload=True `
    critic.model.fsdp_config.grad_offload=True `
    critic.model.fsdp_config.optimizer_offload=True `
    algorithm.kl_ctrl.kl_coef=0.001 `
    trainer.critic_warmup=0 `
    trainer.logger=[\'console\'] `
    trainer.project_name=ToolRL `
    trainer.experiment_name=$ExperimentName `
    trainer.n_gpus_per_node=$GPUs `
    trainer.nnodes=1 `
    trainer.save_freq=-1 `
    trainer.test_freq=-1 `
    trainer.total_epochs=1



