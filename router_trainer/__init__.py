"""
Router trainer utilities for ToolRL.

This package contains scripts and helper modules for:
- Generating GPT-based router evaluation labels for the RLLA dataset
- Defining a lightweight routing head and RL training loop (see other modules)

Note: these tools are designed to be run from the project root, e.g.:

    python -m router_trainer.gpt_label_router_dataset \\
        --train-path dataset/rlla_4k/train.parquet \\
        --test-path dataset/rlla_4k/test.parquet
"""


