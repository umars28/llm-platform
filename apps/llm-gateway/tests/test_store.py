"""Quota state, and the multi-replica bug it exists to fix."""

from __future__ import annotations

import pytest

from llm_gateway.policy import PolicyEngine, Tenant
from llm_gateway.store import MemoryStore, RedisStore, build_store

from conftest import BrokenRedis


def replica(store) -> PolicyEngine:
    return PolicyEngine(
        {"k": Tenant(name="team", api_key="k", monthly_budget_usd=10.0,
                     requests_per_s=1000, burst=1000)},
        store=store,
    )


def spend(pods, requests: int, each_usd: float = 1.0) -> float:
    total = 0.0
    for i in range(requests):
        decision, _ = pods[i % len(pods)].admit("k", "m", estimated_usd=each_usd)
        if decision.allowed:
            total += each_usd
    return total


# -- the bug ------------------------------------------------------------

def test_per_process_state_lets_each_replica_spend_the_whole_budget():
    """The deployed behaviour before this module existed: $10 budget, $20 spent."""
    pods = [replica(MemoryStore()), replica(MemoryStore())]
    assert spend(pods, 20) == pytest.approx(20.0)


def test_shared_state_holds_the_budget_across_replicas():
    shared = MemoryStore()
    pods = [replica(shared), replica(shared)]
    assert spend(pods, 20) == pytest.approx(10.0)


def test_the_overspend_scales_with_replica_count():
    """Which is why this is not optional for anything that autoscales."""
    for count in (2, 5, 10):
        pods = [replica(MemoryStore()) for _ in range(count)]
        assert spend(pods, count * 15) == pytest.approx(10.0 * count)


def test_a_gateway_reports_whether_its_state_is_shared():
    assert not replica(MemoryStore()).shared_state


# -- memory store semantics --------------------------------------------

def test_a_reservation_below_the_limit_is_accepted():
    assert MemoryStore().reserve("t", 1.0, 10.0).ok


def test_a_reservation_over_the_limit_is_refused_with_the_current_figure():
    store = MemoryStore()
    store.reserve("t", 9.0, 10.0)
    result = store.reserve("t", 2.0, 10.0)
    assert not result.ok
    assert "9.00" in result.reason


def test_settlement_replaces_the_estimate():
    store = MemoryStore()
    store.reserve("t", 0.5, 10.0)
    store.settle("t", 0.5, 0.1)
    assert store.usage("t")["reserved_usd"] == pytest.approx(0.1)
    assert store.usage("t")["settled_usd"] == pytest.approx(0.1)


def test_release_returns_an_unused_reservation():
    store = MemoryStore()
    store.reserve("t", 5.0, 10.0)
    store.release("t", 5.0)
    assert store.usage("t")["reserved_usd"] == 0.0


def test_counters_never_go_negative():
    store = MemoryStore()
    store.release("t", 99.0)
    assert store.usage("t")["reserved_usd"] == 0.0


def test_the_token_bucket_refuses_a_flood():
    store = MemoryStore()
    allowed = sum(store.take_token("t", rate_per_s=1, burst=3)[0] for _ in range(10))
    assert allowed == 3


def test_a_rate_denial_from_the_store_returns_a_wait():
    store = MemoryStore()
    for _ in range(3):
        store.take_token("t", 1, 3)
    ok, wait = store.take_token("t", 1, 3)
    assert not ok and wait > 0


def test_a_rate_denial_gives_the_budget_reservation_back():
    """Otherwise a throttled tenant is charged for requests it never made."""
    shared = MemoryStore()
    engine = PolicyEngine(
        {"k": Tenant(name="team", api_key="k", monthly_budget_usd=10.0,
                     requests_per_s=1, burst=1)},
        store=shared,
    )
    engine.admit("k", "m", estimated_usd=1.0)
    denied, _ = engine.admit("k", "m", estimated_usd=1.0)
    assert not denied.allowed
    assert shared.usage("team")["reserved_usd"] == pytest.approx(1.0)


# -- redis behaviour under failure --------------------------------------

def test_budget_fails_closed_when_the_store_is_unreachable():
    """A budget that lifts itself when its store is down is not a budget."""
    store = RedisStore("", client=BrokenRedis())
    result = store.reserve("t", 1.0, 1000.0)
    assert not result.ok
    assert "unavailable" in result.reason


def test_rate_limiting_fails_open_when_the_store_is_unreachable():
    """Refusing all traffic because Redis blinked is an outage we caused."""
    store = RedisStore("", client=BrokenRedis())
    assert store.take_token("t", 10, 10)[0]


def test_an_unreachable_store_reports_unhealthy():
    assert not RedisStore("", client=BrokenRedis()).healthy()


def test_usage_degrades_to_zero_rather_than_raising():
    assert RedisStore("", client=BrokenRedis()).usage("t")["settled_usd"] == 0.0


# -- selection ----------------------------------------------------------

def test_no_url_gives_the_single_process_store():
    store = build_store(None)
    assert isinstance(store, MemoryStore) and not store.shared


def test_a_client_gives_the_shared_store():
    assert build_store(None, client=BrokenRedis()).shared
