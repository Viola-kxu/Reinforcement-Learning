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

4. **Optional GPT‑based supervised shaping**

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

If you’re happy with this narrower plan, the next step (once you exit plan mode) would be to decide:  
- Do you want to start with **pure RL** router training (no GPT labels) and only add GPT‑based labeling later if needed, or do you want GPT labelling from the beginning for a supervised warm‑start?




## Chat 2
You absolutely can (and should) train the router with RL; in this setup the router is basically a tiny bandit policy sitting on top of ToolRL’s existing environment.

### 1. What is being trained?

- **Router policy** \(π_r(a \mid prompt; θ_r)\):
  - Input: the tokenized `prompt` (system+user) exactly as used now.
  - Output: logits over 3 actions `{ANSWER, SEARCH-FAMILY, CALCULATE-FAMILY}`.
  - Parameters \(θ_r\) live in a small head on top of the actor LM; the LM itself can be:
    - Frozen (simplest first experiment), or
    - Updated with its existing GRPO/PPO loss in parallel.

### 2. One RL episode for the router

For each dataset row (one prompt):

1. **State**: the prompt from `train.parquet` (`prompt` column via `RLHFDataset`).
2. **Action**: sample `a ~ π_r(a | prompt; θ_r)`:
   - 0 = ANSWER, 1 = SEARCH-FAMILY, 2 = CALCULATE-FAMILY.
3. **Conditioned rollout**:
   - Modify the system message according to `a` (as we discussed: no tools vs “prefer search tools” vs “prefer calculation tools”) and call the existing `generate_sequences`.
4. **Reward**:
   - Compute **task reward** `r_task` using the existing RLLA reward:
     - `score` from `rlla.compute_score(solution_str, ground_truth, step)`.
   - Parse the generated text to see **what tools were actually used** (none / search‑family / calculate‑family), and compute a **tool cost** `c` (e.g. 0 if no tools, 1 if any tool, or weighted per family).
   - Maintain a Lagrange multiplier `λ` to enforce a budget `B` on average cost:
     \[
     r_{\text{router}} = r_{\text{task}} - \lambda \cdot c
     \]
5. **Optionally GRPO-style grouping**:
   - If you sample multiple trajectories per prompt (as your GRPO setup already does), you’ll have several \(r_{\text{router},i}\) for the same `extra_info.index`.
   - Use `compute_grpo_outcome_advantage` on these router rewards (grouped by `index`) to get normalized **advantages** \(A_i\) per trajectory.

### 3. The actual RL update

For each trajectory \(i\) in the batch, you have:

- Router log-prob of the chosen action: `log π_r(a_i | prompt_i; θ_r)`.
- Advantage for that trajectory: \(A_i\) (either `r_router` baseline-subtracted, or GRPO-normalized within the prompt group).

Then the router’s loss is the standard policy gradient objective:

\[
L_{\text{router}}(θ_r)
 = - \mathbb{E}_i\big[ \log π_r(a_i \mid prompt_i; θ_r)\, A_i \big]
\]

Practically:

- Implement a small loss head in the actor module:
  - Take `router_logits` from the prompt representation.
  - Use `torch.distributions.Categorical` or `softmax` to sample `a` and compute `log_prob`.
  - Compute `A_i` via your GRPO helper (reusing existing code, but feeding it the router rewards instead of token rewards).
  - Backpropagate `L_router` into only the router head (and optionally higher LM layers if you want).

### 4. Where this fits into ToolRL’s loop

Inside `RayPPOTrainer.fit`, per step:

1. Build the batch from `RLHFDataset` (unchanged).
2. For each prompt, **run the router head**, sample `router_action`, store `router_log_prob`.
3. Run the usual **actor rollout** under the route‑conditioned prompt.
4. Compute:
   - RLLA task reward (already implemented).
   - Tool cost from actual tool use.
   - Router reward `r_router`, and (if GRPO) router advantages.
5. Do **two** updates:
   - Actor/critic update (unchanged, using token‑level rewards and KL).
   - Router update using `L_router` above.

No extra labels are required for this RL training; GPT‑generated labels (if you later add them) are just an optional small supervised term on top of this RL objective.