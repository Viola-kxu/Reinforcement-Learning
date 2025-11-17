# Router Labeling Quickstart

This directory contains utility scripts for the ToolRL router project.  
To generate GPT-based evaluation labels for the RLLA dataset:

1. Open `router_trainer/gpt_label_router_dataset.py` and set
   ```python
   DEFAULT_OPENAI_KEY = "sk-REPLACE_ME_WITH_YOUR_OPENAI_KEY"
   ```
   to your actual OpenAI API key (or export `OPENAI_API_KEY` in your shell).

2. Install the dependencies:
   ```bash
   pip install openai pandas pyarrow tqdm
   ```

3. Run the labeling script from the project root:
   ```bash
   python -m router_trainer.gpt_label_router_dataset --train-path dataset/rlla_4k/train.parquet --test-path dataset/rlla_4k/test.parquet --output-train-path dataset/rlla_4k/train_router.parquet --output-test-path dataset/rlla_4k/test_router.parquet --model gpt-4.1-mini
   ```

4. The new parquet files will contain two extra columns:
   - `router_target_action_gt`
   - `router_tool_family_gt`

These labels are **only for evaluation/analysis**; the router RL loop must not use them during training.


