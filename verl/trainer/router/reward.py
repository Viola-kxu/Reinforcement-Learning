from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List

SEARCH_KEYWORDS = [
    "search",
    "retrieve",
    "fetch",
    "lookup",
    "find",
    "query",
    "list",
    "get ",
    "get_",
    "get",
]

CALCULATE_KEYWORDS = [
    "calc",
    "calculate",
    "compute",
    "convert",
    "sum",
    "prob",
    "probability",
    "area",
    "volume",
    "average",
    "ratio",
    "derivative",
    "integral",
    "count",
    "evaluate",
    "solver",
]

TOOL_BLOCK_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


@dataclass
class ToolUsageInfo:
    used_any: bool
    used_search: bool
    used_calculate: bool
    tool_names: List[str]


def infer_family(tool_name: str) -> str:
    """Heuristic family inference based on tool name keywords."""
    name = tool_name.lower()
    if any(keyword in name for keyword in SEARCH_KEYWORDS):
        return "search"
    if any(keyword in name for keyword in CALCULATE_KEYWORDS):
        return "calculate"
    return "other"


def _parse_tool_names(response_text: str) -> List[str]:
    """Extract tool names from `<tool_call>` blocks in the response text."""
    matches = TOOL_BLOCK_PATTERN.findall(response_text)
    tool_names: List[str] = []
    for block in matches:
        for line in block.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if isinstance(name, str):
                tool_names.append(name)
    return tool_names


def analyze_tool_usage(response_text: str) -> ToolUsageInfo:
    tool_names = _parse_tool_names(response_text)
    used_search = False
    used_calculate = False
    for name in tool_names:
        family = infer_family(name)
        if family == "search":
            used_search = True
        elif family == "calculate":
            used_calculate = True
    used_any = len(tool_names) > 0
    return ToolUsageInfo(
        used_any=used_any,
        used_search=used_search,
        used_calculate=used_calculate,
        tool_names=tool_names,
    )


def compute_tool_cost(info: ToolUsageInfo, cost_weights: Dict[str, float]) -> float:
    """Compute the scalar tool cost based on usage info and config weights."""
    cost = 0.0
    if info.used_any:
        cost += float(cost_weights.get("any_tool", 0.0))
    if info.used_search:
        cost += float(cost_weights.get("search", 0.0))
    if info.used_calculate:
        cost += float(cost_weights.get("calculate", 0.0))
    return cost


