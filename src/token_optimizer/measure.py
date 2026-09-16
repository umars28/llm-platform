"""Measuring the three optimisations against a real workload.

The workload is ops-copilot: its ten MCP tool schemas and system prompt form the
prefix that is resent on every turn, and its recorded traces supply real
conversation shapes. Measuring against a synthetic prompt would answer a question
nobody asked.

Every figure is a ratio of two token counts taken with the same tokenizer, so
the proxy tokenizer's systematic error cancels. Dollar figures are computed from
published list prices and carry that error; they are labelled computed, never
billed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .counting import Spend, count, count_messages, count_tools, reduction
from .optimise import plan_cache, prune_duplicate_results, prune_superseded, routing_saving


@dataclass
class Workload:
    """One real request shape: what is stable, what varies, how often it repeats."""

    name: str
    system: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 1
    output_tokens: int = 0

    @property
    def prefix(self) -> str:
        """Tools then system: the render order the cache matches on."""
        return json.dumps(self.tools, sort_keys=True) + "\n" + self.system

    @property
    def prefix_tokens(self) -> int:
        return count_tools(self.tools) + count(self.system)

    @property
    def volatile_tokens(self) -> int:
        return count_messages(self.messages)


def load_workload(path: Path, name: str = "ops-copilot") -> Workload:
    data = json.loads(Path(path).read_text())
    return Workload(
        name=name,
        system=data.get("system", ""),
        tools=data.get("tools", []),
        messages=data.get("messages", []),
        turns=int(data.get("turns", 1)),
        output_tokens=int(data.get("output_tokens", 0)),
    )


@dataclass
class Measurement:
    workload: str
    model: str
    baseline: Spend
    cached: Spend
    pruned_tokens: int
    pruned_removed: list[str]
    cache_reason: str
    breakeven_calls: int

    @property
    def cache_saving_percent(self) -> float:
        return reduction(self.baseline.cost_usd, self.cached.cost_usd)

    @property
    def prune_saving_percent(self) -> float:
        return reduction(self.baseline.input_tokens, self.pruned_tokens)

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("prefix resent per turn", f"{self.baseline.input_tokens // max(self.baseline_turns, 1):,} tokens"),
            ("cache breakpoint", self.cache_reason),
            ("breakeven", f"{self.breakeven_calls} calls"),
            ("cost, no caching (computed)", f"${self.baseline.cost_usd:.4f}"),
            ("cost, cached (computed)", f"${self.cached.cost_usd:.4f}"),
            ("caching saves", f"{self.cache_saving_percent:.0f}%"),
            ("pruning removes", f"{self.prune_saving_percent:.0f}% of input tokens"),
        ]

    baseline_turns: int = 1


def measure(workload: Workload, model: str = "claude-opus-5") -> Measurement:
    plan = plan_cache(workload.prefix, volatile=json.dumps(workload.messages))
    turns = max(workload.turns, 1)

    baseline = Spend(
        model,
        input_tokens=(workload.prefix_tokens + workload.volatile_tokens) * turns,
        output_tokens=workload.output_tokens,
    )
    cached = plan.spend(model, calls=turns, output_tokens=0)
    cached = Spend(
        model,
        input_tokens=cached.input_tokens,
        cache_write_tokens=cached.cache_write_tokens,
        cache_read_tokens=cached.cache_read_tokens,
        output_tokens=workload.output_tokens,
    )

    duplicates = prune_duplicate_results(workload.messages)
    superseded = prune_superseded(duplicates.messages)
    pruned_volatile = count_messages(superseded.messages)

    return Measurement(
        workload=workload.name,
        model=model,
        baseline=baseline,
        cached=cached,
        pruned_tokens=(workload.prefix_tokens + pruned_volatile) * turns,
        pruned_removed=duplicates.removed + superseded.removed,
        cache_reason=plan.reason,
        breakeven_calls=plan.breakeven_calls(model),
        baseline_turns=turns,
    )
