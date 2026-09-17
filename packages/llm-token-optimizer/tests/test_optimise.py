from __future__ import annotations

import pytest

from token_optimizer.counting import count
from token_optimizer.optimise import (
    MIN_CACHEABLE_TOKENS,
    find_invalidators,
    plan_cache,
    prune_duplicate_results,
    prune_superseded,
    route,
    routing_saving,
)

BIG = "You are an on-call assistant. " * 300  # comfortably over the minimum


# -- invalidators ------------------------------------------------------

@pytest.mark.parametrize("volatile", [
    "current time is 2026-09-16T04:12:00",
    "request_id: 7f3a9c2e-1b4d-4c8a-9e2f-6d5b1a3c7e9f",
    "generated at 1789541339",
    'trace_id = "abc123xyz"',
])
def test_per_request_values_in_a_prefix_are_found(volatile):
    """These fail silently: the code looks right and the cache never hits."""
    assert find_invalidators(f"{BIG}\n{volatile}")


def test_a_stable_prefix_has_no_invalidators():
    assert find_invalidators(BIG) == []


def test_a_version_number_is_not_mistaken_for_a_timestamp():
    assert find_invalidators(f"{BIG}\nrolled back to v2.31.0") == []


# -- cache planning ----------------------------------------------------

def test_a_stable_prefix_over_the_minimum_is_cacheable():
    plan = plan_cache(BIG, volatile="the alert fired at 02:14")
    assert plan.cacheable
    assert plan.prefix_tokens >= MIN_CACHEABLE_TOKENS


def test_a_short_prefix_is_refused_with_the_reason():
    plan = plan_cache("short system prompt")
    assert not plan.cacheable
    assert "silently ignored" in plan.reason


def test_a_prefix_with_a_timestamp_is_refused_and_says_where():
    plan = plan_cache(f"{BIG}\ngenerated 2026-09-16T04:12:00")
    assert not plan.cacheable
    assert plan.invalidators
    assert "never hit" in plan.reason


def test_caching_charges_one_write_and_the_rest_as_reads():
    plan = plan_cache(BIG)
    spend = plan.spend("claude-opus-5", calls=10)
    assert spend.cache_write_tokens == plan.prefix_tokens
    assert spend.cache_read_tokens == plan.prefix_tokens * 9


def test_caching_beats_not_caching_over_many_calls():
    plan = plan_cache(BIG)
    assert plan.spend("claude-opus-5", 50).cost_usd < plan.spend("claude-opus-5", 50).uncached_cost_usd


def test_a_single_call_is_more_expensive_cached_than_not():
    """A write costs 1.25x fresh input, so caching a one-shot prefix loses money."""
    plan = plan_cache(BIG)
    spend = plan.spend("claude-opus-5", calls=1)
    assert spend.cost_usd > spend.uncached_cost_usd


def test_the_breakeven_point_is_reported():
    assert plan_cache(BIG).breakeven_calls("claude-opus-5") == 2


def test_an_uncacheable_plan_charges_everything_fresh():
    plan = plan_cache("too short")
    spend = plan.spend("claude-opus-5", calls=5)
    assert spend.cache_read_tokens == 0 and spend.input_tokens > 0


# -- pruning -----------------------------------------------------------

def big_result(text: str) -> dict:
    """Comfortably over both pruning thresholds, so a test failure means the
    logic changed rather than that the fixture sat on a boundary."""
    return {"role": "user", "content": text * 120}


def test_a_repeated_tool_result_is_replaced_by_a_pointer():
    messages = [big_result("log line "), {"role": "assistant", "content": "ok"},
                big_result("log line ")]
    pruned = prune_duplicate_results(messages)
    assert pruned.saved_percent > 20
    assert "identical to the result at position 0" in pruned.messages[2]["content"]
    assert pruned.removed


def test_the_first_occurrence_is_kept_intact():
    messages = [big_result("payload "), big_result("payload ")]
    assert prune_duplicate_results(messages).messages[0]["content"] == messages[0]["content"]


def test_distinct_results_are_left_alone():
    messages = [big_result("alpha "), big_result("beta ")]
    pruned = prune_duplicate_results(messages)
    assert pruned.removed == []
    assert pruned.saved_percent == 0


def test_short_repeated_messages_are_not_worth_pruning():
    messages = [{"role": "user", "content": "ok"}, {"role": "user", "content": "ok"}]
    assert prune_duplicate_results(messages).removed == []


def test_superseded_results_are_summarised_and_the_latest_kept():
    messages = [big_result(f"result {i} ") for i in range(5)]
    pruned = prune_superseded(messages, keep_last=2)
    assert len(pruned.removed) == 3
    assert "superseded" in pruned.messages[0]["content"]
    assert pruned.messages[-1]["content"] == messages[-1]["content"]


def test_nothing_is_pruned_when_there_is_little_to_supersede():
    messages = [big_result("only one ")]
    assert prune_superseded(messages, keep_last=2).removed == []


# -- routing -----------------------------------------------------------

def test_reasoning_work_goes_to_the_strong_model():
    assert route("diagnose why checkout latency rose").model == "claude-opus-5"


def test_routine_work_goes_to_the_cheap_model():
    assert route("extract the service name from this log line").model == "claude-haiku-4-5"


def test_the_route_says_why():
    assert "diagnose" in route("diagnose this incident").reason


def test_routing_saves_against_sending_everything_to_the_strong_model():
    tasks = ["classify this log"] * 9 + ["diagnose the root cause"]
    saving = routing_saving(tasks, input_tokens=2000, output_tokens=500)
    assert saving["to_cheap"] == 9 and saving["to_strong"] == 1
    assert saving["saving_percent"] > 50


def test_routing_everything_to_the_strong_model_saves_nothing():
    saving = routing_saving(["diagnose it"] * 5, input_tokens=100, output_tokens=10)
    assert saving["saving_percent"] == 0.0
