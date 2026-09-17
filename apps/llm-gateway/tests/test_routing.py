from __future__ import annotations

import pytest

from llm_gateway.routing import (
    AllProvidersFailed,
    BreakerState,
    CircuitBreaker,
    Provider,
    Router,
)


def provider(name: str, priority: int = 100, **breaker) -> Provider:
    return Provider(
        name=name, base_url=f"https://{name}.example",
        models={"claude-opus-5": "opus"}, priority=priority,
        breaker=CircuitBreaker(now=0.0, **breaker),
    )


# -- circuit breaker ---------------------------------------------------

def test_a_healthy_breaker_lets_traffic_through():
    assert CircuitBreaker(now=0.0).allows(now=0.0)


def test_repeated_failures_open_the_breaker():
    b = CircuitBreaker(failure_threshold=3, now=0.0)
    for _ in range(3):
        b.record_failure(now=0.0)
    assert b.state is BreakerState.OPEN
    assert not b.allows(now=0.0)


def test_a_success_resets_the_failure_count():
    b = CircuitBreaker(failure_threshold=3, now=0.0)
    b.record_failure(now=0.0)
    b.record_failure(now=0.0)
    b.record_success()
    b.record_failure(now=0.0)
    assert b.state is BreakerState.CLOSED


def test_an_open_breaker_probes_without_waiting_for_a_human():
    """A breaker that only closes on restart turns a 12-minute outage into hours."""
    b = CircuitBreaker(failure_threshold=1, probe_after_s=30, now=0.0)
    b.record_failure(now=0.0)
    assert not b.allows(now=10.0)
    assert b.allows(now=31.0)
    assert b.state is BreakerState.HALF_OPEN


def test_half_open_admits_exactly_one_trial_request():
    """A flood of probes would re-break a provider that is only just recovering."""
    b = CircuitBreaker(failure_threshold=1, probe_after_s=1, now=0.0)
    b.record_failure(now=0.0)
    assert b.allows(now=2.0)
    assert not b.allows(now=2.0)


def test_a_failed_probe_reopens_for_another_interval():
    b = CircuitBreaker(failure_threshold=1, probe_after_s=10, now=0.0)
    b.record_failure(now=0.0)
    assert b.allows(now=11.0)
    b.record_failure(now=11.0)
    assert b.state is BreakerState.OPEN
    assert not b.allows(now=12.0)
    assert b.allows(now=22.0)


def test_a_successful_probe_closes_the_breaker():
    b = CircuitBreaker(failure_threshold=1, probe_after_s=1, now=0.0)
    b.record_failure(now=0.0)
    b.allows(now=2.0)
    b.record_success()
    assert b.state is BreakerState.CLOSED
    assert b.allows(now=2.0)


# -- routing -----------------------------------------------------------

def test_the_highest_priority_provider_is_preferred():
    r = Router([provider("secondary", priority=200), provider("primary", priority=10)])
    assert [p.name for p in r.candidates("claude-opus-5", now=0.0)] == ["primary", "secondary"]


def test_a_healthy_fallback_outranks_a_failing_preferred_provider():
    primary, secondary = provider("primary", 10, failure_threshold=1), provider("secondary", 200)
    primary.observe(ok=False, now=0.0)
    r = Router([primary, secondary])
    # Primary is open, so it is not offered at all.
    assert [p.name for p in r.candidates("claude-opus-5", now=0.0)] == ["secondary"]


def test_an_open_provider_is_not_offered_until_its_probe_window():
    p = provider("only", failure_threshold=1, probe_after_s=30)
    r = Router([p])
    p.observe(ok=False, now=0.0)
    assert r.candidates("claude-opus-5", now=5.0) == []
    assert [x.name for x in r.candidates("claude-opus-5", now=31.0)] == ["only"]


def test_a_provider_that_does_not_serve_the_model_is_skipped():
    other = Provider(name="other", base_url="x", models={"claude-haiku-4-5": "h"})
    r = Router([other, provider("opus-capable")])
    assert [p.name for p in r.candidates("claude-opus-5", now=0.0)] == ["opus-capable"]


def test_an_unknown_model_has_no_candidates():
    assert Router([provider("a")]).candidates("gpt-9", now=0.0) == []


def test_the_logical_model_maps_to_the_provider_id():
    p = Provider(name="p", base_url="x", models={"claude-opus-5": "anthropic/claude-opus-5"})
    assert p.upstream_model("claude-opus-5") == "anthropic/claude-opus-5"


def test_an_unmapped_model_passes_through_unchanged():
    assert provider("p").upstream_model("something-else") == "something-else"


# -- config and reporting ----------------------------------------------

def test_a_router_loads_from_config():
    r = Router.from_config({"providers": [
        {"name": "anthropic", "base_url": "https://api.anthropic.com",
         "models": {"claude-opus-5": "claude-opus-5"}, "priority": 10},
        {"name": "gateway", "base_url": "https://openrouter.ai/api",
         "models": {"claude-opus-5": "anthropic/claude-opus-5"}, "priority": 20},
    ]})
    assert r.known_models() == {"claude-opus-5"}
    assert [p.name for p in r.candidates("claude-opus-5", now=0.0)] == ["anthropic", "gateway"]


def test_the_report_shows_breaker_state_and_latency():
    p = provider("primary")
    p.observe(ok=True, latency_ms=120)
    p.observe(ok=True, latency_ms=80)
    row = Router([p]).report()[0]
    assert row["state"] == "closed"
    assert row["mean_latency_ms"] == 100.0


def test_the_failure_names_what_each_provider_did():
    """"The gateway returned 503" is not enough to act on at 3am."""
    error = AllProvidersFailed("claude-opus-5", [("primary", "500"), ("secondary", "timeout")])
    assert "primary: 500" in str(error)
    assert "secondary: timeout" in str(error)


def test_a_model_no_provider_serves_says_so():
    assert "no provider serves it" in str(AllProvidersFailed("gpt-9", []))
