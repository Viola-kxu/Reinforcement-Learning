## 1. Router definition and action space

- **Action space**: per prompt, the router outputs one of:
  - `ANSWER`: model should answer directly, no tools.
  - `SEARCH-FAMILY`: model may use “search-like” tools (web/API/data retrieval).
  - `CALCULATE-FAMILY`: model may use “calculate-like” tools (math/compute functions).
- **Router head**:
  - Add a small head on top of the *actor* LM (used in ToolRL PPO/GRPO) that takes a **prompt representation** (e.g., last token of the prompt portion, or pooled embedding) and outputs 3 logits.
  - Expose this via the actor’s forward API so that `RayPPOTrainer` can get:
    - `router_logits` (shape `[batch_size, 3]`)
    - `router_action` (sampled index) and `router_log_prob` per prompt.

---

## 2. How the router uses the existing RLLA dataset

From `rlla_training.md` and `dataset_head.md`:

- Each training row has:
  - `prompt`: chat messages `[{"role": "system", "content": instruction}, {"role": "user", "content": input}]`.
  - `reward_model.ground_truth`: the full ground‑truth `<think>/<tool_call>/<response>` string.
  - `extra_info.index`: stable prompt ID used for GRPO grouping.

Router training **does not need new dataset columns** to start:
- It uses the **same prompts** as the actor (the `prompt` column).
- It uses the **same reward function** (`rlla.compute_score`) that compares generated answer vs `reward_model.ground_truth`.
- Optional: you can **enrich** the dataset offline using GPT to add:
  - Per‑tool family labels (`SEARCH` / `CALCULATE` / `OTHER`) from each tool’s name/description in `instruction`.
  - Per‑example weak label `router_target_action ∈ {ANSWER, SEARCH-FAMILY, CALCULATE-FAMILY, MIXED}` based on the ground‑truth `output`.
  - These go into a sidecar JSON/CSV or extra columns and are used only for **supervised shaping**, not required for RL.

---

## 3. Routing‑conditioned rollout inside ToolRL

Integrate router into `RayPPOTrainer.fit` (no changes to how RLLA data is loaded):

1. **Sample router action before generation**

   - After building `batch` and `gen_batch` (the prompt‑only DataProto used for `generate_sequences`):
     - Run the actor’s router head on `gen_batch` to get `router_logits`.
     - Sample an action `a ∈ {ANSWER, SEARCH-FAMILY, CALCULATE-FAMILY}` (or use argmax).
     - Store `router_action` and `router_log_prob` in the batch’s `meta_info` / `non_tensor_batch`.

2. **Modify the prompt according to the action**

   For each prompt in the batch:

   - Start from the original `prompt` (system+user) from Parquet.
   - Build a **router‑conditioned system message**:

     - `ANSWER`:
       - Remove the Available Tools block (or replace with a variant).
       - Append something like:  
         “Do NOT use any tools. Think in `<think>` and then answer directly in `<response>`.”
     - `SEARCH-FAMILY`:
       - Keep tools but **emphasize** search‑type tools and discourage calculation tools, e.g.:  
         “You may use these APIs/tools to search for information. Prefer the search tools listed below; avoid pure calculation helpers.”  
         (If you have GPT‑derived families per tool, you can list only the search tools here.)
     - `CALCULATE-FAMILY`:
       - Similarly, emphasize calculation tools and discourage search.
   - The rest of the rollout (actor generation, vLLM, etc.) stays unchanged: `generate_sequences` just sees a slightly different system message.

3. **Log actual tool usage**

   - After generation, you already have `responses` and can decode them to strings (the reward pipeline does this).
   - Reuse the parsing logic from `rlla.compute_score` / `customize_correctness_reward_tool` to:
     - Check if `<tool_call>` appears at all.
     - Extract all tool JSONs and their `name`s.
     - Using your tool‑family mapping (heuristic or GPT‑labelled), compute:
       - `used_any_tool` (bool)
       - `used_search`, `used_calculate` (bools)
       - Possibly number of calls per family.
   - From this, define a **tool‑cost** per trajectory (see next section).

---

## 4. Router reward with budget (Lagrangian)

For each generated trajectory corresponding to a prompt:

1. **Task reward (reuse existing)**

   - Let `r_task` be the scalar reward from the existing RLLA reward:
     - `score` returned by `rlla.compute_score(solution_str, ground_truth, step)` (sum of format + correctness [+ length]).

2. **Tool cost**

   Define a simple tool cost function based on actual tool usage in the generated answer:

   - Example (scalar per episode):
     - `cost = 0` if `used_any_tool` is false.
     - `cost = 1` if `used_any_tool` is true (ignoring family), or
     - More fine‑grained:
       - `cost = c_search * (used_search ? 1 : 0) + c_calc * (used_calculate ? 1 : 0)`
         (e.g. both 1, or heavier penalty for one family).
   - Optionally, make cost proportional to the number of tool calls.

3. **Budget and Lagrangian**

   - Choose a **budget** `B` = allowed average cost per example (e.g. 0.3 tools per prompt).
   - Maintain a scalar **Lagrange multiplier** `λ ≥ 0` (initially 0 or small).
   - For each episode, define **router reward**:
     \[
     r_{\text{router}} = r_{\text{task}} - \lambda \cdot \text{cost}
     \]
   - Every K steps (or each step), compute the batch empirical mean cost `E_hat[cost]` and update:
     \[
     \lambda \leftarrow \max\big(0,\; \lambda + \eta \cdot (E_{\hat{}}[\text{cost}] - B)\big)
     \]
     where `η` is a small dual learning rate.
   - Log `λ`, `E_hat[cost]`, and `E_hat[r_task]` for monitoring.

4. **Token‑level router rewards**

   - To plug into existing GRPO helpers, write `r_router` as a **token‑level reward tensor**:
     - Same shape as `responses` (`[batch_size, response_length]`), all zeros except the last token where you put `r_router`.
   - Store in the batch as something like `router_token_level_rewards`.

---

## 5. GRPO training of the router head

We reuse the GRPO machinery already used for actor training, but applied to the router’s discrete action instead of token log‑probs:

1. **Multiple router actions per prompt (optional but ideal)**

   - In GRPO mode, ToolRL already supports multiple rollouts per prompt (`actor_rollout_ref.rollout.n`).
   - Extend this so that for each prompt:
     - You may sample multiple router actions (or just reuse the router action per copy, depending on design).
     - Each `(prompt, router_action)` pair leads to one generated answer and one `r_router`.
   - Use the existing `extra_info.index` as **group ID**; within a group, you have multiple candidates with different routes and/or different samples.

2. **Compute GRPO advantages for router**

   - Call `compute_grpo_outcome_advantage` on `router_token_level_rewards`:
     - Use `uid = index` (from `extra_info.index`) so all copies of the same prompt are grouped.
     - This gives you `router_advantages` and `router_returns` (broadcast to all tokens of each response, as in GRPO).
   - You don’t actually need the full sequence mask; you just use the value on the last token or average over response tokens.

3. **Router policy loss**

   - For each sequence, you already have:
     - `router_log_prob` of the sampled action `a`.
     - `router_advantage` (scalar on last token).
   - Define a simple GRPO/PPO‑style loss:
     - `L_router = - E[ router_log_prob * router_advantage ]`
     - If desired, add a small entropy bonus on router logits.
   - This loss is **separate from** the standard token‑level PPO loss for the actor; it can:
     - Share the actor optimizer/schedule, or
     - Use a separate optimizer with its own LR, controlled by config.

4. **Optional GPT-based supervised shaping**

   - Offline, using GPT, generate:
     - Per‑tool families and a per‑example `router_target_action` label (ANSWER/SEARCH/CALCULATE/MIXED).
   - During router training:
     - For examples with non‑MIXED labels, add a small cross‑entropy term between `router_logits` and the label.
     - Weight this term low so RL (the budgeted reward) dominates.

---

## 6. Making the router “work” for ToolRL (inference usage)

Once trained:

1. **Inference pipeline**

   - Given a new prompt (same format as `prompt` in Parquet):
     - Run the actor router head to get `router_action`.
     - Build a route‑specific system message as in §3.2.
     - Run a **single** rollout with the actor LM to produce `<think>/<tool_call>/<response>`.
   - No reference policy, critic, or reward model is needed at inference; only the router + actor LM.

2. **Behavior**

   - For prompts where tools are clearly unnecessary, router should learn to pick `ANSWER`, avoiding `<tool_call>` and speeding things up.
   - For prompts that need external info, router should gravitate toward the appropriate family (`SEARCH` vs `CALCULATE`) while respecting the global average cost budget set by `B`.

---

## 7. Training loop instrumentation (progress bar + intermediate checkpoints)

- **Progress bar**: wrap the main training loop in `tqdm` (or update the existing `Tracking` logs) so each global step shows ETA and throughput. For Ray, print updates from the driver to avoid log spam.
- **Checkpoint cadence**: extend `_save_checkpoint` to run every `trainer.save_freq` steps (even mid-epoch). Each checkpoint should include:
  - Actor weights (including router head parameters).
  - Critic weights (if enabled).
  - Router optimizer state (if separated).
  - Lagrange multiplier and any running averages needed to resume training.
- **Resumability**: store an experiment manifest (YAML/JSON) alongside checkpoints so WSL2/Linux training can resume even if the job is interrupted.

## 8. Running on WSL2 / Docker


1. **Environment prep**:
   - Enable WSL2 and install Ubuntu 22.04 (or similar).
   - Install NVIDIA drivers + `nvidia-container-toolkit` so Docker can access the GPU.
   - Inside WSL2, install Docker and add your user to the `docker` group.
   - Pull or build the ToolRL Docker image (e.g., `docker build -t toolrl .` or use a provided registry image).
2. **Launching the container**:
   ```bash
   docker run --gpus all -it --rm \
     -v /mnt/c/Users/kangq/OneDrive\ -\ Johns\ Hopkins/Documents/ToolRL:/workspace \
     toolrl:latest /bin/bash
   ```
   (Adjust the `-v` path to point to your repo.)
3. **Inside the container**:
   - `cd /workspace`
   - `pip install -e .` (installs ToolRL with router mods)
   - Set env vars: `export DATA_DIR=dataset/rlla_4k`, `export BASE_MODEL=/path/to/model`, etc.
4. **Training command** (example GRPO run with router):

    My commands for training router:
    ```bash
    chmod +x run_router_training.sh
    bash run_router_training.sh
    ```
   
   **Option A: Direct Python call (recommended for router training)**:
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
     trainer.experiment_name=baseline_router \
     trainer.n_gpus_per_node=1 \
     trainer.nnodes=1 \
     trainer.total_epochs=15
   ```
   
   **Option B: Use train_grpo.sh wrapper** (modify it first to include router configs):
   - Edit `train_grpo.sh` to set `DATA_DIR`, `BASE_MODEL`, `EXPERIMENT_NAME`
   - Edit `examples/grpo_trainer/run_grpo.sh` to add router overrides
   - Then run: `bash train_grpo.sh`
   
   **Note**: The direct Python call (Option A) is simpler and more flexible for router experiments.
5. **Monitoring**: use stdout progress bar, plus `wandb` or `tensorboard` logs mounted to `/workspace/runs`.

Following these steps ensures router RL can run entirely inside Linux/WSL2 with visible progress and recoverable checkpoints.