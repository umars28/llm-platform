from __future__ import annotations

import json
from pathlib import Path

import pytest

from token_optimizer.measure import Workload, load_workload, measure

TOOLS = [
    {"name": f"tool_{i}", "description": "does a thing " * 30,
     "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}
    for i in range(10)
]
SYSTEM = "You are an on-call assistant. " * 120


def workload(**kw) -> Workload:
    base = dict(name="test", system=SYSTEM, tools=TOOLS, turns=8, output_tokens=2000,
                messages=[{"role": "user", "content": "a log result " * 120} for _ in range(4)])
    return Workload(**(base | kw))


def test_the_prefix_is_tools_then_system_matching_render_order():
    w = workload()
    assert w.prefix.index(json.dumps(TOOLS, sort_keys=True)) < w.prefix.index(SYSTEM[:20])


def test_caching_saves_over_a_multi_turn_run():
    m = measure(workload())
    assert m.cached.cost_usd < m.baseline.cost_usd
    assert m.cache_saving_percent > 0


def test_the_prefix_is_written_once_and_read_thereafter():
    m = measure(workload(turns=8))
    assert m.cached.cache_write_tokens == m.cached.cache_read_tokens // 7


def test_a_single_turn_does_not_benefit():
    """Below breakeven the write costs more than it saves."""
    m = measure(workload(turns=1))
    assert m.cache_saving_percent <= 0


def test_the_breakeven_point_is_reported():
    assert measure(workload()).breakeven_calls >= 1


def test_pruning_repeated_results_reduces_input_tokens():
    repeated = [{"role": "user", "content": "identical result " * 120} for _ in range(4)]
    m = measure(workload(messages=repeated))
    assert m.prune_saving_percent > 0
    assert m.pruned_removed


def test_distinct_results_are_not_pruned_away():
    distinct = [{"role": "user", "content": f"result {i} " * 5} for i in range(3)]
    assert measure(workload(messages=distinct)).pruned_removed == []


def test_an_uncacheable_prefix_says_why():
    m = measure(workload(system=SYSTEM + "\ngenerated at 2026-09-16T04:12:00"))
    assert "never hit" in m.cache_reason
    assert m.cache_saving_percent <= 0


def test_a_workload_round_trips_through_disk(tmp_path: Path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"system": SYSTEM, "tools": TOOLS, "turns": 3,
                                "messages": [], "output_tokens": 10}))
    loaded = load_workload(path)
    assert loaded.turns == 3 and loaded.prefix_tokens > 0
