## RLLA Dataset – How It Is Used for PPO & GRPO Training

This document explains **how the RLLA dataset is structured and how it is used for PPO and GRPO training** in this repository.

The key files are:
- `dataset/rlla_4k_raw/rlla_rl.json`: raw JSON data
- `dataset/rlla_4k_raw/rlla.py`: preprocessing script
- `dataset/rlla_4k/train.parquet` and `dataset/rlla_4k/test.parquet`: processed training/validation data
- `verl/utils/dataset/rl_dataset.py`: RL dataset wrapper (`RLHFDataset`)
- `verl/utils/reward_score/rlla.py`: reward function for RLLA
- `verl/trainer/main_ppo.py`: main entry for PPO/GRPO training
- `examples/grpo_trainer/run_grpo.sh` and `examples/ppo_trainer/run_ppo.sh`: example launch scripts
- `train_grpo.sh` and `train_ppo.sh`: top-level GRPO/PPO launcher scripts

---

## 1. From `instruction` / `input` / `output` to training rows

The raw RLLA data is stored as JSON with three key textual fields:
- **`instruction`**: system-level guidance (how to think, how to format answers, how to use tools)
- **`input`**: user query or problem
- **`output`**: ground-truth solution, including required tool calls and formatting

The preprocessing script `dataset/rlla_4k_raw/rlla.py` converts this JSON into a Parquet dataset:

```28:86:dataset/rlla_4k_raw/rlla.py
dataset = json.load(open("./dataset/rlla_4k_raw/rlla_rl.json", "r"))
...
def process_fn(example, idx, split):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    data = {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": output
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }
    }
    return data

train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)
train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
test_df.to_parquet(os.path.join(local_dir, "test.parquet"))
```

Each row in `train.parquet` / `test.parquet` therefore has:
- `prompt`: a **chat-style prompt** built from `instruction` (system) and `input` (user)
- `reward_model.ground_truth`: the **target output** text, used by the reward function
- `extra_info`: original fields plus a stable `index` that identifies the prompt

This Parquet dataset is what PPO/GRPO actually read during training.

---

## 2. How the RL dataset wrapper (`RLHFDataset`) uses the Parquet data

The RL training code uses `RLHFDataset` from `verl/utils/dataset/rl_dataset.py` to load and tokenize prompts:

```58:83:verl/utils/dataset/rl_dataset.py
class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files,
                 tokenizer,
                 prompt_key="prompt",
                 max_prompt_length=1024,
                 use_chat_template=True,
                 filter_prompts=True,
                 cache_dir="~/.cache/verl/rlhf",
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation="error"):
        ...
        self.parquet_files = parquet_files
        self.prompt_key = prompt_key
        ...
        self._download()
        self._read_files_and_tokenize()
```

During initialization, the dataset:
1. Reads one or more Parquet files (`train.parquet`, `test.parquet`).
2. Keeps the `prompt` column and other metadata in memory.

When a sample is fetched, it:

```123:157:verl/utils/dataset/rl_dataset.py
    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
```

So for each training example:
- The **visible prompt** to the model is:
  - system = `instruction`
  - user = `input`
- The **ground-truth answer** (`output`) is not fed directly as tokens; it is stored as `reward_model.ground_truth` for reward computation.
- The **`index`** field ties all samples of the same prompt together for GRPO.

---

## 3. Reward function: how `output` is used to score responses

PPO/GRPO training uses a `RewardManager` (in `verl/trainer/main_ppo.py`) that selects the correct reward function based on `data_source`:

```24:35:verl/trainer/main_ppo.py
def _select_rm_score_fn(data_source):
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score
    ...
    elif "rlla" in data_source:
        return rlla.compute_score
    else:
        raise NotImplementedError
```

For each generated response:

```61:88:verl/trainer/main_ppo.py
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            ...
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch["data_source"]
            compute_score_fn = _select_rm_score_fn(data_source)

            score, fomrat_score, correctness_score, length_score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                step=step,
            )
            reward_tensor[i, valid_response_length - 1] = score
```

The actual scoring logic for RLLA lives in `verl/utils/reward_score/rlla.py`:

```257:305:verl/utils/reward_score/rlla.py
def compute_score(solution_str, ground_truth, step=0):
    exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    if "llama" in exp_name:
        predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")

    ...
    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    fomrat_score = customize_format_reward_func(completions, answer, step, format_max_possible, format_min_possible)[0]
    correctness_score = customize_correctness_reward_tool(completions, answer, step, tool_max_possible, tool_min_possible)[0]
    ...
    score = fomrat_score + correctness_score + length_score
    return score, fomrat_score, correctness_score, length_score
```

Conceptually:
- The model generates a response to the prompt (`instruction` + `input`).
- The reward function compares the **generated answer** against the **ground-truth `output`**, checking:
  - Formatting (e.g., `<think>`, `<tool_call>`, `<response>` blocks)
  - Tool call correctness (names + arguments)
  - Optionally, length properties
- The resulting scalar reward is written into `reward_tensor` at the final response token and passed into PPO/GRPO core algorithms.

Environment variables (set in `train_grpo.sh`) can change reward behavior:
- `WITHLENGTH`, `SCHEDULELENGTH`, `REFINEDREWARD`, `COARSEREWARD`, `CORRECTMAX1`, `MAX1STEP30MAX3`, `SCHEDULEREWARD`, `INTERMEDIATEREWARD`, etc.

---

## 4. PPO training: how the dataset and rewards drive updates

### 4.1 Launching PPO

PPO training is launched via:
- `train_ppo.sh`: sets environment variables and calls `examples/ppo_trainer/run_ppo.sh`.
- `examples/ppo_trainer/run_ppo.sh`: passes CLI overrides into `python3 -m verl.trainer.main_ppo`.

```1:8:examples/ppo_trainer/run_ppo.sh
python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    ...
```

This uses `verl/trainer/config/ppo_trainer.yaml` as the base Hydra config, with the CLI overrides above.

### 4.2 PPO flow (high level)

For each PPO update:
1. **Sample prompts** from `RLHFDataset` using `train.parquet`.
2. **Roll out** one response per prompt using the actor model.
3. **Compute rewards** by comparing each response to `reward_model.ground_truth` (`output`) with `rlla.compute_score(...)`.
4. **Estimate advantages** using GAE (`compute_gae_advantage_return` in `core_algos.py`) and the learned critic (value head).
5. **Update actor and critic** with the PPO loss, using the advantages and KL penalty against a reference policy.

The important point is that the **dataset supplies prompts and ground-truth outputs**, while **PPO uses the numeric rewards derived from those outputs, not the outputs themselves as labels**.

---

## 5. GRPO training: group-based advantages on the same dataset

GRPO (Group Relative Policy Optimization) is also implemented through `main_ppo.py`, but with:
- `algorithm.adv_estimator=grpo`
- Multiple responses per prompt (e.g., `actor_rollout_ref.rollout.n=4`)

GRPO is launched via:
- `train_grpo.sh`: sets `DATA_DIR`, `BASE_MODEL`, `EXPERIMENT_NAME`, and reward env vars.
- `examples/grpo_trainer/run_grpo.sh`: calls `main_ppo` with GRPO-specific overrides.

```1:9:examples/grpo_trainer/run_grpo.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    ...
    actor_rollout_ref.rollout.n=4 \
    ...
```

### 5.1 GRPO advantage computation

The core difference from PPO is in `compute_grpo_outcome_advantage`:

```110:155:verl/trainer/ppo/core_algos.py
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
```

Key ideas:
- `index` comes from `extra_info.index` in the Parquet data (set in `rlla.py` and carried by `RLHFDataset`).
- GRPO groups responses **by `index`**, i.e., all responses to the same prompt.
- For each group:
  - It collects all reward scores.
  - Computes the **mean** and **standard deviation** of scores in that group.
  - Normalizes each response’s score:  
    \[
    \text{advantage} = \frac{\text{score} - \text{group\_mean}}{\text{group\_std} + \epsilon}
    \]
- This normalized score is then broadcast across all tokens of each response and used as the advantage.

Thus, **GRPO uses the same dataset and reward function as PPO**, but:
- Samples multiple responses per prompt.
- Learns from **relative** quality of responses within a group instead of a critic’s value function.

---

## 6. Summary: how the “database” is used

- The raw JSON "database" (`instruction`, `input`, `output`) is **preprocessed once** into a Parquet dataset.
- PPO/GRPO training use:
  - `instruction` + `input` ⇒ visible prompt (system + user messages).
  - `output` ⇒ ground-truth string for reward computation.
  - `index` ⇒ stable identifier per prompt, critical for GRPO grouping.
- PPO uses **GAE + critic** to compute advantages from scalar rewards.
- GRPO uses **group-normalized rewards** (per prompt) to compute advantages, without relying on a value function.

See also:
- `README.md` (top-level training instructions)
- `examples/rlla_train_grpo_ppo.py` (Python example showing how to launch RLLA training with PPO and GRPO)


