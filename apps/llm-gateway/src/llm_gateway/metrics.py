"""Prometheus metrics, chosen so an SLO can be written against them.

Every series is labelled by tenant, because the first question during an
incident is "is this everyone or one team", and a metric that cannot answer it
sends someone to the logs.

Latency is a histogram rather than a gauge or a summary: an SLO is a quantile
over a window, and you cannot recover a quantile from an average. The buckets
are spread across seconds rather than milliseconds because that is the range LLM
calls actually live in -- default buckets top out at ten seconds and would put
every slow completion in +Inf, which is the same as not measuring it.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.requests = Counter(
            "gateway_requests_total", "Requests served",
            ["tenant", "model", "provider", "status"], registry=self.registry,
        )
        self.denials = Counter(
            "gateway_denied_total", "Requests refused by policy",
            ["tenant", "model", "reason"], registry=self.registry,
        )
        self.upstream_failures = Counter(
            "gateway_upstream_failures_total", "Requests where every provider failed",
            ["tenant", "model"], registry=self.registry,
        )
        self.latency = Histogram(
            "gateway_request_duration_seconds", "End-to-end request latency",
            ["tenant", "model"], buckets=LATENCY_BUCKETS, registry=self.registry,
        )
        self.cost = Counter(
            "gateway_cost_usd_total", "Settled spend",
            ["tenant", "model"], registry=self.registry,
        )
        self.tokens = Counter(
            "gateway_tokens_total", "Tokens billed",
            ["tenant", "model", "kind"], registry=self.registry,
        )
        self.budget_utilisation = Gauge(
            "gateway_budget_utilisation", "Committed spend as a fraction of budget",
            ["tenant"], registry=self.registry,
        )

    def served(
        self, tenant: str, model: str, provider: str, status: int,
        seconds: float, cost_usd: float, usage: dict[str, int],
    ) -> None:
        self.requests.labels(tenant, model, provider, str(status)).inc()
        self.latency.labels(tenant, model).observe(seconds)
        self.cost.labels(tenant, model).inc(cost_usd)
        for kind, value in usage.items():
            if value:
                self.tokens.labels(tenant, model, kind).inc(value)

    def denied(self, tenant: str, model: str, reason: str) -> None:
        self.denials.labels(tenant, model, reason).inc()

    def upstream_failed(self, tenant: str, model: str) -> None:
        self.upstream_failures.labels(tenant, model).inc()

    def observe_budgets(self, report: list[dict[str, Any]]) -> None:
        for row in report:
            self.budget_utilisation.labels(row["tenant"]).set(row["utilisation"])

    def render(self) -> bytes:
        return generate_latest(self.registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST
