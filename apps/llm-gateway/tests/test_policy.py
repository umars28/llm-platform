from __future__ import annotations

from pathlib import Path

import pytest

from llm_gateway.policy import Decision, Denial, PolicyEngine, Tenant, TokenBucket


def engine(**overrides) -> PolicyEngine:
    base = dict(name="team-a", api_key="key-a", monthly_budget_usd=10.0,
                requests_per_s=100.0, burst=100)
    return PolicyEngine({"key-a": Tenant(**(base | overrides))})


# -- identity ----------------------------------------------------------

def test_an_unknown_key_is_rejected_as_unauthenticated():
    decision, tenant = PolicyEngine().admit("nope", "claude-opus-5")
    assert not decision.allowed
    assert decision.status_code == 401
    assert tenant is None


def test_a_missing_key_is_not_treated_as_a_tenant():
    assert not PolicyEngine().admit(None, "m")[0].allowed


# -- model allow-lists -------------------------------------------------

def test_a_model_outside_the_allow_list_is_refused_explicitly():
    """Silently downgrading produces a quality regression nobody can trace."""
    decision, _ = engine(models=["claude-haiku-4-5"]).admit("key-a", "claude-opus-5")
    assert decision.denial is Denial.MODEL
    assert decision.status_code == 403
    assert "claude-haiku-4-5" in decision.reason


def test_an_empty_allow_list_permits_every_model():
    assert engine(models=[]).admit("key-a", "anything")[0].allowed


def test_allow_lists_support_a_prefix_wildcard():
    e = engine(models=["claude-haiku-*"])
    assert e.admit("key-a", "claude-haiku-4-5")[0].allowed
    assert not e.admit("key-a", "claude-opus-5")[0].allowed


# -- rate ---------------------------------------------------------------

def test_a_burst_is_allowed_and_a_flood_is_not():
    e = engine(requests_per_s=1, burst=3)
    assert all(e.admit("key-a", "m")[0].allowed for _ in range(3))
    decision, _ = e.admit("key-a", "m")
    assert decision.denial is Denial.RATE
    assert decision.status_code == 429


def test_a_rate_denial_says_how_long_to_wait():
    e = engine(requests_per_s=2, burst=1)
    e.admit("key-a", "m")
    decision, _ = e.admit("key-a", "m")
    assert decision.retry_after_s and decision.retry_after_s > 0


def test_the_bucket_refills_over_time():
    """The clock is injected, so this proves a refill without sleeping."""
    bucket = TokenBucket(rate_per_s=10, burst=1, now=0.0)
    assert bucket.take(now=0.0)[0]
    assert not bucket.take(now=0.0)[0]
    assert bucket.take(now=1.0)[0]


def test_a_clock_going_backwards_does_not_mint_tokens():
    bucket = TokenBucket(rate_per_s=10, burst=1, now=100.0)
    assert bucket.take(now=100.0)[0]
    assert not bucket.take(now=50.0)[0]


# -- budget -------------------------------------------------------------

def test_a_request_beyond_the_budget_is_refused():
    decision, _ = engine(monthly_budget_usd=1.0).admit("key-a", "m", estimated_usd=2.0)
    assert decision.denial is Denial.BUDGET


def test_an_exhausted_budget_is_not_a_429():
    """429 invites a retry into a wall and hides the real problem behind backoff."""
    decision, _ = engine(monthly_budget_usd=0.0).admit("key-a", "m", estimated_usd=1.0)
    assert decision.status_code == 402


def test_budget_is_checked_before_rate():
    """A tenant out of money should be told that, not told to slow down."""
    e = engine(monthly_budget_usd=0.0, requests_per_s=0.001, burst=0)
    assert e.admit("key-a", "m", estimated_usd=1.0)[0].denial is Denial.BUDGET


def test_spend_is_reserved_at_admission_not_at_completion():
    """Concurrent calls would otherwise blow through the limit together."""
    e = engine(monthly_budget_usd=1.0)
    for _ in range(5):
        e.admit("key-a", "m", estimated_usd=0.2)
    assert e.store.usage("team-a")["reserved_usd"] == pytest.approx(1.0)
    assert e.admit("key-a", "m", estimated_usd=0.2)[0].denial is Denial.BUDGET


def test_settlement_replaces_the_estimate_with_the_real_cost():
    e = engine()
    _, tenant = e.admit("key-a", "m", estimated_usd=0.50)
    e.settle(tenant, estimated_usd=0.50, actual_usd=0.10)
    usage = e.store.usage("team-a")
    assert usage["reserved_usd"] == pytest.approx(0.10)
    assert usage["settled_usd"] == pytest.approx(0.10)


def test_estimate_error_does_not_accumulate_across_requests():
    """Without settlement a tenant is eventually throttled by our arithmetic."""
    e = engine(monthly_budget_usd=10.0)
    for _ in range(50):
        _, tenant = e.admit("key-a", "m", estimated_usd=0.10)
        e.settle(tenant, 0.10, 0.01)
    assert e.store.usage("team-a")["reserved_usd"] == pytest.approx(0.50)
    assert e.admit("key-a", "m", estimated_usd=0.10)[0].allowed


def test_a_failed_call_releases_its_reservation():
    e = engine(monthly_budget_usd=1.0)
    _, tenant = e.admit("key-a", "m", estimated_usd=0.9)
    e.release(tenant, 0.9)
    assert e.store.usage("team-a")["reserved_usd"] == 0.0
    assert e.admit("key-a", "m", estimated_usd=0.9)[0].allowed


def test_release_and_settle_never_go_negative():
    e = engine()
    tenant = e.tenant_for("key-a")
    e.release(tenant, 5.0)
    e.settle(tenant, 5.0, 0.0)
    assert e.store.usage("team-a")["reserved_usd"] == 0.0


# -- config and reporting ----------------------------------------------

def test_policy_loads_from_a_file(tmp_path: Path):
    path = tmp_path / "tenants.yaml"
    path.write_text(
        "tenants:\n"
        "  - name: platform\n    api_key: k1\n    monthly_budget_usd: 50\n"
        "    models: ['claude-haiku-*']\n"
        "  - name: research\n    api_key: k2\n"
    )
    e = PolicyEngine.from_file(path)
    assert set(e.tenants) == {"k1", "k2"}
    assert e.tenant_for("k1").monthly_budget_usd == 50
    assert e.tenant_for("k2").models == []


def test_the_report_shows_where_the_money_went():
    e = engine()
    _, tenant = e.admit("key-a", "m", estimated_usd=0.5)
    e.settle(tenant, 0.5, 0.4)
    row = e.report()[0]
    assert row["tenant"] == "team-a"
    assert row["settled_usd"] == pytest.approx(0.4)
    assert row["utilisation"] == pytest.approx(0.04)


def test_an_allowed_decision_is_a_200():
    assert Decision(True).status_code == 200
