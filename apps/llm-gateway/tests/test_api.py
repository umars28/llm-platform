from __future__ import annotations

from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from llm_gateway.main import create_app
from llm_gateway.routing import AllProvidersFailed
from llm_gateway.upstream import UpstreamResult, is_provider_fault

CONFIG: dict[str, Any] = {
    "tenants": [
        {"name": "platform", "api_key": "k-plat", "monthly_budget_usd": 1.0,
         "requests_per_s": 100, "burst": 100},
        {"name": "research", "api_key": "k-res", "monthly_budget_usd": 50,
         "models": ["claude-haiku-*"], "requests_per_s": 100, "burst": 100},
    ],
    "providers": [
        {"name": "primary", "base_url": "https://a.example", "priority": 10,
         "models": {"claude-opus-5": "opus", "claude-haiku-4-5": "haiku"}},
        {"name": "fallback", "base_url": "https://b.example", "priority": 20,
         "models": {"claude-opus-5": "opus", "claude-haiku-4-5": "haiku"}},
    ],
}


class FakeUpstream:
    """Stands in for the provider chain, recording what it was asked."""

    def __init__(self, result: UpstreamResult | None = None, raises: Exception | None = None):
        self.result, self.raises = result, raises
        self.calls: list[tuple[str, dict]] = []

    async def send(self, model, payload, now=None):
        self.calls.append((model, payload))
        if self.raises:
            raise self.raises
        return self.result


def ok_result(cost_tokens: int = 1000) -> UpstreamResult:
    return UpstreamResult(
        provider="primary", status=200, latency_ms=120.0,
        body={"id": "msg_1", "content": [{"type": "text", "text": "hi"}],
              "usage": {"input_tokens": cost_tokens, "output_tokens": 10}},
    )


def client(upstream: FakeUpstream) -> TestClient:
    return TestClient(create_app(CONFIG, upstream))


def post(c: TestClient, key: str, model: str = "claude-opus-5", **extra):
    return c.post("/v1/messages", headers={"x-api-key": key},
                  json={"model": model, "max_tokens": 64,
                        "messages": [{"role": "user", "content": "hello"}], **extra})


# -- the happy path ----------------------------------------------------

def test_a_valid_request_is_forwarded_and_returned():
    up = FakeUpstream(ok_result())
    r = post(client(up), "k-plat")
    assert r.status_code == 200
    assert r.json()["content"][0]["text"] == "hi"
    assert up.calls[0][0] == "claude-opus-5"


def test_a_bearer_token_is_accepted_as_well_as_x_api_key():
    c = client(FakeUpstream(ok_result()))
    r = c.post("/v1/messages", headers={"authorization": "Bearer k-plat"},
               json={"model": "claude-opus-5", "max_tokens": 64, "messages": []})
    assert r.status_code == 200


# -- policy denials ----------------------------------------------------

def test_an_unknown_key_is_401_and_never_reaches_a_provider():
    up = FakeUpstream(ok_result())
    assert post(client(up), "nope").status_code == 401
    assert up.calls == []


def test_a_forbidden_model_is_403_not_a_silent_downgrade():
    r = post(client(FakeUpstream(ok_result())), "k-res", model="claude-opus-5")
    assert r.status_code == 403
    assert "claude-haiku-*" in r.json()["error"]["message"]


def test_an_allowed_model_passes_for_the_restricted_tenant():
    assert post(client(FakeUpstream(ok_result())), "k-res",
                model="claude-haiku-4-5").status_code == 200


def test_an_exhausted_budget_is_402_with_a_reason():
    c = client(FakeUpstream(ok_result()))
    for _ in range(40):
        post(c, "k-plat", max_tokens=100000)
    r = post(c, "k-plat", max_tokens=100000)
    assert r.status_code == 402
    assert "committed" in r.json()["error"]["message"]


def test_a_rate_limited_request_carries_retry_after():
    config = {**CONFIG, "tenants": [
        {"name": "slow", "api_key": "k-slow", "requests_per_s": 1, "burst": 1,
         "monthly_budget_usd": 100}]}
    c = TestClient(create_app(config, FakeUpstream(ok_result())))
    post(c, "k-slow")
    r = post(c, "k-slow")
    assert r.status_code == 429
    assert "retry-after" in r.headers


# -- upstream failure --------------------------------------------------

def test_every_provider_failing_is_503_naming_each_one():
    up = FakeUpstream(raises=AllProvidersFailed(
        "claude-opus-5", [("primary", "HTTP 500"), ("fallback", "timeout")]))
    r = post(client(up), "k-plat")
    assert r.status_code == 503
    attempts = r.json()["error"]["attempts"]
    assert {a["provider"] for a in attempts} == {"primary", "fallback"}


def test_a_failed_request_releases_its_budget_reservation():
    """A leaked reservation looks like a tenant losing quota for no reason."""
    up = FakeUpstream(raises=AllProvidersFailed("claude-opus-5", []))
    c = client(up)
    for _ in range(30):
        post(c, "k-plat")
    assert c.get("/v1/usage").json()["tenants"][0]["reserved_usd"] == pytest.approx(0.0)


# -- what kubernetes asks ----------------------------------------------

def test_liveness_never_depends_on_an_upstream():
    assert client(FakeUpstream(ok_result())).get("/healthz").json()["status"] == "ok"


def test_readiness_is_true_while_a_provider_is_healthy():
    r = client(FakeUpstream(ok_result())).get("/readyz")
    assert r.status_code == 200 and r.json()["ready"]


def test_readiness_fails_when_every_breaker_is_open():
    """Liveness and readiness must not be the same check."""
    app = create_app(CONFIG, FakeUpstream(ok_result()))
    c = TestClient(app)
    for provider in app.state.router.providers:
        for _ in range(provider.breaker.failure_threshold):
            provider.observe(ok=False)
    assert c.get("/readyz").status_code == 503
    assert c.get("/healthz").status_code == 200


# -- observability -----------------------------------------------------

def test_metrics_are_labelled_by_tenant():
    """The first incident question is 'everyone, or one team?'"""
    c = client(FakeUpstream(ok_result()))
    post(c, "k-plat")
    body = c.get("/metrics").text
    assert 'tenant="platform"' in body
    assert "gateway_request_duration_seconds" in body


def test_denials_are_counted_by_reason():
    c = client(FakeUpstream(ok_result()))
    post(c, "k-res", model="claude-opus-5")
    assert 'reason="model_not_allowed"' in c.get("/metrics").text


def test_usage_reports_settled_spend_per_tenant():
    c = client(FakeUpstream(ok_result()))
    post(c, "k-plat")
    row = next(t for t in c.get("/v1/usage").json()["tenants"] if t["tenant"] == "platform")
    assert row["settled_usd"] > 0


# -- fault classification ----------------------------------------------

@pytest.mark.parametrize("status", [500, 502, 503, 504, 429, 408, 529])
def test_provider_side_failures_are_worth_failing_over(status):
    assert is_provider_fault(status)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_caller_side_failures_are_not(status):
    """Failing over a 400 turns one clear error into three slow ones."""
    assert not is_provider_fault(status)
