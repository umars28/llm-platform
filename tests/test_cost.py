from __future__ import annotations

import pytest

from llm_eval.cost import Usage, UsageLedger, compare, pricing_for


def test_pricing_tolerates_gateway_namespacing():
    assert pricing_for("anthropic/claude-haiku-4-5") == pricing_for("claude-haiku-4-5")


def test_an_unknown_model_falls_back_to_the_dearest_tier():
    """Overstating cost prompts a question; understating it gets quoted."""
    assert pricing_for("some-model-from-2027") == pricing_for("claude-opus-5")


def test_cost_follows_the_model():
    opus = Usage("claude-opus-5", input_tokens=100_000, output_tokens=10_000)
    haiku = Usage("claude-haiku-4-5", input_tokens=100_000, output_tokens=10_000)
    assert opus.cost_usd == pytest.approx(0.75)
    assert haiku.cost_usd == pytest.approx(0.15)


def test_cache_reads_are_priced_far_below_fresh_input():
    fresh = Usage("claude-opus-5", input_tokens=100_000)
    cached = Usage("claude-opus-5", cache_read_tokens=100_000)
    assert cached.cost_usd == pytest.approx(fresh.cost_usd / 10)


def test_the_caching_saving_is_stated_against_a_counterfactual():
    """Comparing to another run conflates the cache with whatever else changed."""
    usage = Usage("claude-opus-5", input_tokens=10_000, cache_read_tokens=90_000,
                  output_tokens=5_000)
    assert usage.uncached_cost_usd > usage.cost_usd
    assert usage.cache_saving_usd == pytest.approx(
        usage.uncached_cost_usd - usage.cost_usd
    )


def test_cache_hit_rate_reports_zero_when_caching_is_not_working():
    assert Usage("claude-opus-5", input_tokens=50_000).cache_hit_rate == 0.0


def test_cache_hit_rate_counts_every_kind_of_input_token():
    usage = Usage("claude-opus-5", input_tokens=10, cache_write_tokens=10,
                  cache_read_tokens=80)
    assert usage.cache_hit_rate == pytest.approx(0.8)


def test_usage_is_read_from_a_response_object_defensively():
    class Resp:
        input_tokens = 100
        output_tokens = 20
        cache_read_input_tokens = None  # absent on some providers

    usage = Usage.from_response("claude-opus-5", Resp())
    assert usage.input_tokens == 100
    assert usage.cache_read_tokens == 0
    assert usage.calls == 1


def test_adding_usage_for_a_different_model_is_refused():
    opus = Usage("claude-opus-5", input_tokens=10, calls=1)
    with pytest.raises(ValueError, match="aggregate per model"):
        opus.add(Usage("claude-haiku-4-5", input_tokens=10, calls=1))


# -- ledger ------------------------------------------------------------

def test_a_ledger_totals_across_models():
    ledger = UsageLedger()
    ledger.record(Usage("claude-opus-5", input_tokens=100_000, calls=1))
    ledger.record(Usage("claude-haiku-4-5", input_tokens=100_000, calls=3))
    assert ledger.calls == 4
    assert ledger.cost_usd == pytest.approx(0.5 + 0.1)


def test_the_ledger_shows_where_the_money_went():
    ledger = UsageLedger()
    ledger.record(Usage("claude-opus-5", input_tokens=100_000, calls=1))
    ledger.record(Usage("claude-haiku-4-5", input_tokens=100_000, calls=9))
    breakdown = ledger.by_model()
    assert breakdown["claude-opus-5"]["share"] > breakdown["claude-haiku-4-5"]["share"]
    assert breakdown["claude-haiku-4-5"]["calls"] == 9


def test_repeated_records_for_one_model_accumulate():
    ledger = UsageLedger()
    for _ in range(3):
        ledger.record(Usage("claude-opus-5", input_tokens=1_000, calls=1))
    assert ledger.entries["claude-opus-5"].input_tokens == 3_000


def test_an_empty_ledger_does_not_divide_by_zero():
    ledger = UsageLedger()
    assert ledger.cost_usd == 0.0
    assert ledger.cache_hit_rate == 0.0
    assert ledger.summary()["calls"] == 0


def test_comparison_reports_the_direction_of_change():
    before, after = UsageLedger(), UsageLedger()
    before.record(Usage("claude-opus-5", input_tokens=100_000, calls=1))
    after.record(Usage("claude-opus-5", cache_read_tokens=100_000, calls=1))
    delta = compare(before, after)
    assert delta["change_percent"] == pytest.approx(-90.0)
    assert delta["after_cache_hit_rate"] == 1.0
