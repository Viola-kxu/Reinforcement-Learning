# Router Training Guide - WSL2/Docker Setup

## 1. Using Your Existing Container

**You do NOT need to relaunch the container!** You can attach to your existing one:

```bash
# From WSL2 or your host terminal
docker exec -it 485c8eba5d32 /bin/bash
```

This will give you a bash shell inside the container where all requirements are already installed.

---

## 2. What is BASE_MODEL?

`BASE_MODEL` is the path to the pre-trained language model that will be used as the starting point for router training. It can be:

1. **A HuggingFace model identifier** (recommended): The code will automatically download it from HuggingFace Hub
   - Example: `"Qwen/Qwen2.5-7B-Instruct"`
   - Example: `"Qwen/Qwen2.5-3B-Instruct"`

2. **A local path**: If you've already downloaded a model to your filesystem
   - Example: `"/workspace/models/Qwen2.5-7B-Instruct"`

---

## 3. What Models Should You Use?

Based on the ToolRL codebase:

### Recommended Models (from examples):
- **`Qwen/Qwen2.5-7B-Instruct`** - Used in the HuggingFace fallback example (good for testing)
- **`Qwen/Qwen2.5-3B-Instruct`** - Smaller, faster model (mentioned in `train_grpo.sh`)

### Pre-trained ToolRL Models:
The authors have published trained models on HuggingFace:
- **Collection**: https://huggingface.co/collections/emrecanacikgoz/toolrl-680706679204ead5a6d44f58
- These are models that have already been trained on the RLLA dataset
- You can use these if you want to start from a model that already knows tool usage

### Other Compatible Models:
Any HuggingFace model that:
- Supports chat templates (most modern instruction-tuned models)
- Is compatible with the veRL framework (Qwen, Llama, Mistral families work well)

---

## 4. Where to Get Models If You Don't Have One?

### Option A: Use HuggingFace Hub (Automatic Download)
**This is the easiest option!** Just use a HuggingFace model identifier:

```bash
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

The code will automatically:
1. Download the model from HuggingFace Hub on first use
2. Cache it in `~/.cache/huggingface/transformers/` (or similar)
3. Use it for training

**Requirements:**
- Internet connection (for first download)
- HuggingFace account (optional, but recommended for some models)
- Sufficient disk space (~14GB for 7B model, ~6GB for 3B model)

### Option B: Download Manually
If you want to pre-download a model:

```bash
# Inside your container
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); \
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

Then use the local path in your config.

### Option C: Use Pre-trained ToolRL Models
If you want to start from a model already trained on tool usage:

1. Visit: https://huggingface.co/collections/emrecanacikgoz/toolrl-680706679204ead5a6d44f58
2. Choose a model from the collection
3. Use its HuggingFace path as `BASE_MODEL`

---

## 5. Quick Start Commands

### Attach to your container:
```bash
docker exec -it 485c8eba5d32 /bin/bash
```

### Inside the container, set up environment:
```bash
cd /workspace  # or wherever your ToolRL code is mounted, it could be /workspace/ToolRL

# Set your model (choose one):
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # 7B model (larger, better quality)
# OR
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"  # 3B model (smaller, faster)

# Set dataset path
export DATA_DIR="dataset/rlla_4k"

# Set experiment name
export EXPERIMENT_NAME="router-test-$(date +%Y%m%d-%H%M%S)"
```

### For Router Training (with router reward function):
If you do not need to override anything, just use:
```base
   chmod +x run_router_training.sh
   bash run_router_training.sh
```

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train_router.parquet \
    data.val_files=${DATA_DIR}/test_router.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    router.enable=true \
    router.budget_B=0.3 \
    router.cost_weights.any_tool=1.0 \
    trainer.save_freq=200 \
    trainer.save_on_exit=true \
    trainer.project_name=router_grpo \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15
```

**Note**: The above uses `rollout.name=hf` (HuggingFace backend) since you're in a Docker container. If you have vLLM working, you can use `rollout.name=vllm` instead.

---

## 6. Model Size Recommendations

| Model | Size | VRAM Needed | Best For |
|-------|------|--------------|----------|
| Qwen2.5-3B-Instruct | ~6GB | ~12GB | Quick experiments, limited GPU memory |
| Qwen2.5-7B-Instruct | ~14GB | ~20GB | Better quality, standard training |
| Qwen2.5-14B-Instruct | ~28GB | ~40GB | Highest quality (if you have the GPU) |

**For router training**, start with **Qwen2.5-3B-Instruct** or **Qwen2.5-7B-Instruct** - they're well-tested in the codebase.

---

## 7. Troubleshooting

### "Model not found" error:
- Check your internet connection (for HuggingFace download)
- Verify the model name is correct: `Qwen/Qwen2.5-7B-Instruct` (note the `/`)
- Try logging into HuggingFace: `huggingface-cli login`

### Out of memory:
- Use a smaller model (3B instead of 7B)
- Reduce `train_batch_size` or `ppo_mini_batch_size`
- Enable parameter offloading in config

### Slow training:
- This is normal with HuggingFace backend (`rollout.name=hf`)
- vLLM is much faster but requires Linux + CUDA setup
- Consider reducing `train_batch_size` for faster iteration

