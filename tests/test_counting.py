from __future__ import annotations

import pytest

from token_optimizer.counting import (
    Spend,
    count,
    count_messages,
    count_tools,
    pricing_for,
    reduction,
)


def test_counting_uses_a_real_tokenizer_not_a_character_rule():
    """The two disagree most on exactly what this project optimises."""
    code = 'def handle(self, request_id: str) -> dict[str, Any]: return {"ok": True}'
    assert count(code) != len(code) // 4


def test_empty_text_costs_nothing():
    assert count("") == 0


def test_longer_text_counts_higher():
    assert count("word " * 100) > count("word " * 10)


def test_messages_include_the_system_prompt():
    with_system = count_messages([{"role": "user", "content": "hi"}], system="you are a bot")
    without = count_messages([{"role": "user", "content": "hi"}])
    assert with_system > without


def test_structured_content_is_counted_not_skipped():
    blocks = [{"role": "user", "content": [{"type": "text", "text": "hello there"}]}]
    assert count_messages(blocks) > 0


def test_tool_schemas_are_counted():
    tools = [{"name": "query_logs", "description": "search logs",
              "input_schema": {"type": "object", "properties": {"service": {"type": "string"}}}}]
    assert count_tools(tools) > 10


def test_no_tools_costs_nothing():
    assert count_tools([]) == 0


# -- pricing -----------------------------------------------------------

def test_pricing_tolerates_gateway_namespacing():
    assert pricing_for("anthropic/claude-haiku-4-5") == pricing_for("claude-haiku-4-5")


def test_a_free_tier_costs_nothing():
    assert pricing_for("some/model:free") == (0.0, 0.0, 0.0, 0.0)


def test_an_unknown_model_falls_back_to_the_dearest_tier():
    assert pricing_for("model-from-2028") == pricing_for("claude-opus-5")


def test_cache_reads_are_a_tenth_of_fresh_input():
    fresh = Spend("claude-opus-5", input_tokens=1_000_000)
    cached = Spend("claude-opus-5", cache_read_tokens=1_000_000)
    assert cached.cost_usd == pytest.approx(fresh.cost_usd / 10)


def test_cache_writes_cost_more_than_fresh_input():
    """Which is why caching only pays above a reuse threshold."""
    fresh = Spend("claude-opus-5", input_tokens=1_000_000)
    written = Spend("claude-opus-5", cache_write_tokens=1_000_000)
    assert written.cost_usd > fresh.cost_usd


def test_the_counterfactual_prices_every_token_as_fresh_input():
    spend = Spend("claude-opus-5", cache_read_tokens=1_000_000)
    assert spend.uncached_cost_usd == pytest.approx(5.0)
    assert spend.uncached_cost_usd > spend.cost_usd


def test_spends_add_up():
    total = Spend("claude-opus-5", input_tokens=10) + Spend("claude-opus-5", output_tokens=5)
    assert total.input_tokens == 10 and total.output_tokens == 5


# -- reduction ---------------------------------------------------------

def test_reduction_is_a_percentage():
    assert reduction(1000, 620) == pytest.approx(38.0)


def test_growth_reports_as_negative_reduction():
    assert reduction(100, 150) == pytest.approx(-50.0)


def test_reduction_from_nothing_is_zero_rather_than_infinite():
    assert reduction(0, 10) == 0.0
