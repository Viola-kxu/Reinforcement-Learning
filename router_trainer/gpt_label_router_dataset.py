"""
Generate GPT-based router evaluation labels for the RLLA dataset.

This script:
- Reads the existing RLLA train/test parquet files.
- For each example, extracts:
  - The tool inventory from `extra_info["instruction"]`.
  - The ground-truth solution from `extra_info["output"]`.
- Asks an external LLM (e.g., GPT) to:
  - Classify each tool into {SEARCH, CALCULATE, OTHER}.
  - Classify the example-level router action into
    {ANSWER, SEARCH-FAMILY, CALCULATE-FAMILY, MIXED}.
- Writes new parquet files with added columns:
  - `router_tool_family_gt`: JSON string mapping tool name -> family.
  - `router_target_action_gt`: the example-level router label.

IMPORTANT:
- This script is *evaluation-only* for the router. The training loop for the
  router should NOT depend on these labels; they are for analysis and
  sanity-checking only.
- You must provide your own API key and model name. By default we use the
  `openai` Python client, but you can adapt it to any other LLM provider.

Usage (from project root):

    python -m router_trainer.gpt_label_router_dataset \\
        --train-path dataset/rlla_4k/train.parquet \\
        --test-path dataset/rlla_4k/test.parquet \\
        --output-train-path dataset/rlla_4k/train_router.parquet \\
        --output-test-path dataset/rlla_4k/test_router.parquet \\
        --model gpt-4.1-mini

The script is deliberately conservative: it processes one example per API call
by default. You can batch at a higher level if desired.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
from tqdm import tqdm


DEFAULT_OPENAI_KEY = "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"


try:
    # New-style OpenAI client (>=1.0)
    import openai
    from openai import OpenAI

    _HAS_OPENAI = True
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False


@dataclass
class RouterLabel:
    """Container for GPT-derived labels for a single example."""

    router_target_action_gt: str
    tool_family_map_gt: Dict[str, str]


SYSTEM_PROMPT = """You are an expert at classifying tool-using dialogue examples.

For each example, you will see:
- A tool inventory ("Available Tools" section)
- A ground-truth solution that may contain <think>, <tool_call>, <response>.

Your tasks:
1. For each tool in the inventory, assign it to exactly one of:
   - "SEARCH": tools that primarily fetch, retrieve, query, look up, search, or list external information / resources / content.
   - "CALCULATE": tools that primarily compute, calculate, transform numbers or data (math, statistics, financial calculations, numeric conversions, etc.).
   - "OTHER": anything that doesn't clearly fit SEARCH or CALCULATE.

2. For the example as a whole, assign a router target action:
   - "ANSWER": the ideal solution does NOT require any tool calls in the ground-truth output.
   - "SEARCH-FAMILY": the ground-truth output requires only SEARCH-family tools.
   - "CALCULATE-FAMILY": the ground-truth output requires only CALCULATE-family tools.
   - "MIXED": the ground-truth output uses both SEARCH and CALCULATE tools, or uses only OTHER tools.

Output MUST be a single JSON object with keys:
- "router_target_action_gt": one of "ANSWER", "SEARCH-FAMILY", "CALCULATE-FAMILY", "MIXED".
- "tool_family_map_gt": an object mapping each tool name string to "SEARCH", "CALCULATE", or "OTHER".
"""


USER_PROMPT_TEMPLATE = """You will be given the tool inventory and ground-truth solution for one example.

TOOL INVENTORY (from instruction):
------------------------
{tool_inventory}

GROUND-TRUTH SOLUTION (output):
------------------------
{ground_truth_output}

Remember:
- router_target_action_gt is based ONLY on what the ground-truth solution actually does.
- If there is no <tool_call> in the ground-truth output, router_target_action_gt MUST be "ANSWER".
- If it uses only SEARCH tools, use "SEARCH-FAMILY".
- If it uses only CALCULATE tools, use "CALCULATE-FAMILY".
- Otherwise use "MIXED".

Return ONLY a JSON object."""


def extract_tool_inventory(instruction: str) -> str:
    """
    Very simple heuristic: we just pass the full instruction text to the model.
    The instruction already contains the "Available Tools" section.
    """
    return instruction


def call_gpt_for_example(
    client: Any,
    model: str,
    tool_inventory: str,
    ground_truth_output: str,
) -> RouterLabel:
    """Call the external LLM and parse its JSON response into RouterLabel."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        tool_inventory=tool_inventory,
        ground_truth_output=ground_truth_output,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        if _HAS_OPENAI and isinstance(exc, openai.RateLimitError):
            raise RuntimeError(
                "OpenAI API reported insufficient quota or too many requests. "
                "Please check your plan/billing or reduce --max-rows."
            ) from exc
        raise
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty content.")

    content = content.strip()
    if content.startswith("```"):
        # Remove Markdown code fences such as ```json ... ```
        content = content.strip("`")
        # After stripping backticks, LLMs often prefix with 'json' or similar
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from LLM: {e}\nRaw content: {content}")

    router_target_action_gt = data.get("router_target_action_gt")
    tool_family_map_gt = data.get("tool_family_map_gt")

    if not isinstance(router_target_action_gt, str):
        raise ValueError("router_target_action_gt must be a string.")
    if not isinstance(tool_family_map_gt, dict):
        raise ValueError("tool_family_map_gt must be an object mapping tool names to families.")

    return RouterLabel(
        router_target_action_gt=router_target_action_gt,
        tool_family_map_gt=tool_family_map_gt,
    )


def label_dataframe(
    df: pd.DataFrame,
    client: Any,
    model: str,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Iterate over rows of the RLLA parquet dataframe and attach GPT-derived labels.

    Expected columns:
    - 'extra_info': dict with keys including 'instruction' and 'output'.
    """
    router_actions: list[str] = []
    tool_family_maps: list[str] = []  # store as JSON strings

    iterable = df.itertuples(index=False)
    if max_rows is not None:
        iterable = list(iterable)[:max_rows]

    for row in tqdm(iterable, total=(len(iterable) if isinstance(iterable, list) else None)):
        extra_info = getattr(row, "extra_info", None)
        if not isinstance(extra_info, dict):
            raise ValueError("Each row must have an 'extra_info' dict column.")

        instruction = extra_info.get("instruction", "")
        ground_truth_output = extra_info.get("output", "")

        tool_inventory = extract_tool_inventory(instruction)
        label = call_gpt_for_example(
            client=client,
            model=model,
            tool_inventory=tool_inventory,
            ground_truth_output=ground_truth_output,
        )

        router_actions.append(label.router_target_action_gt)
        tool_family_maps.append(json.dumps(label.tool_family_map_gt, ensure_ascii=False))

    # Align lengths just in case max_rows was used
    if max_rows is not None and len(router_actions) < len(df):
        # For rows we didn't label (beyond max_rows), fill with None
        missing = len(df) - len(router_actions)
        router_actions.extend([None] * missing)
        tool_family_maps.extend([None] * missing)

    df = df.copy()
    df["router_target_action_gt"] = router_actions
    df["router_tool_family_gt"] = tool_family_maps
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPT-based router labels for RLLA dataset.")
    parser.add_argument(
        "--train-path",
        type=str,
        default="dataset/rlla_4k/train.parquet",
        help="Path to the RLLA train parquet file.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="dataset/rlla_4k/test.parquet",
        help="Path to the RLLA test parquet file.",
    )
    parser.add_argument(
        "--output-train-path",
        type=str,
        default="dataset/rlla_4k/train_router.parquet",
        help="Output path for labeled train parquet.",
    )
    parser.add_argument(
        "--output-test-path",
        type=str,
        default="dataset/rlla_4k/test_router.parquet",
        help="Output path for labeled test parquet.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM model name to use for labeling.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional limit on number of train rows to label (for debugging).",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional limit on number of test rows to label (for debugging).",
    )
    args = parser.parse_args()

    if not _HAS_OPENAI:
        raise RuntimeError(
            "The 'openai' package is not installed. Please install it with:\n"
            "  pip install openai\n"
            "and set OPENAI_API_KEY in your environment."
        )

    api_key = os.environ.get("OPENAI_API_KEY", DEFAULT_OPENAI_KEY)
    if not api_key or "REPLACE_ME" in api_key:
        raise RuntimeError(
            "No usable OpenAI API key. Either set OPENAI_API_KEY in your environment "
            "or edit DEFAULT_OPENAI_KEY in router_trainer/gpt_label_router_dataset.py."
        )

    client = OpenAI(api_key=api_key)

    # Load dataframes
    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)

    # Label train
    labeled_train_df = label_dataframe(
        df=train_df,
        client=client,
        model=args.model,
        max_rows=args.max_train_rows,
    )
    os.makedirs(os.path.dirname(args.output_train_path), exist_ok=True)
    labeled_train_df.to_parquet(args.output_train_path)

    # Label test
    labeled_test_df = label_dataframe(
        df=test_df,
        client=client,
        model=args.model,
        max_rows=args.max_test_rows,
    )
    os.makedirs(os.path.dirname(args.output_test_path), exist_ok=True)
    labeled_test_df.to_parquet(args.output_test_path)

    print("Saved labeled train to:", args.output_train_path)
    print("Saved labeled test to:", args.output_test_path)


if __name__ == "__main__":  # pragma: no cover
    main()


