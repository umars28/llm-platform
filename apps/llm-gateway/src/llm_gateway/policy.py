"""Tenant policy: who may spend what, on which models, how fast.

This is the part that makes an LLM gateway infrastructure rather than a proxy.
A proxy forwards requests. A gateway decides, per tenant, whether a request is
allowed at all -- and that decision has to be correct when the tenant is
misbehaving, not only when it is not.

Three limits, because they fail differently and want different answers:

**Rate** protects the upstream provider from one tenant's burst. Exceeding it is
transient; the right response is 429 with a retry hint, because the request will
be fine in a second.

**Budget** protects the organisation from a bug. A retry loop that would cost
ten thousand dollars overnight is stopped by a number, not by an alert someone
reads in the morning. Exceeding it is not transient and 429 would be a lie --
retrying will not help until someone raises the limit or the window resets.

**Model allow-lists** keep an experiment off the expensive tier. A tenant asking
for a model it is not entitled to is a configuration error on their side and
should say so, not silently downgrade -- a silent downgrade produces a quality
regression nobody can trace.

Budget is enforced on *reserved* spend, not on settled spend. A request that has
been admitted but not yet answered has already committed money, and a gateway
that only counts completed requests lets a burst of concurrent calls blow
through the limit together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Denial(str, Enum):
    RATE = "rate_limit_exceeded"
    BUDGET = "budget_exhausted"
    MODEL = "model_not_allowed"
    UNKNOWN_TENANT = "unknown_tenant"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    denial: Denial | None = None
    retry_after_s: float | None = None

    @property
    def status_code(self) -> int:
        """429 only where retrying will actually help.

        Returning 429 for an exhausted budget invites the client to retry into a
        wall, and its backoff will hide the real problem from whoever needs to
        raise the limit.
        """
        if self.allowed:
            return 200
        return {
            Denial.RATE: 429,
            Denial.BUDGET: 402,
            Denial.MODEL: 403,
            Denial.UNKNOWN_TENANT: 401,
        }[self.denial or Denial.UNKNOWN_TENANT]


@dataclass
class TokenBucket:
    """Rate limiting that allows a burst without allowing a flood.

    The clock is injectable. Taking `time.monotonic()` internally makes rate
    limiting untestable without sleeping, and a test that sleeps to prove a
    refill is a test nobody runs often enough to catch a regression.
    """

    rate_per_s: float
    burst: float
    now: float | None = None
    tokens: float = field(init=False)
    updated: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.burst
        self.updated = self.now if self.now is not None else time.monotonic()

    def take(self, now: float | None = None) -> tuple[bool, float]:
        now = now if now is not None else time.monotonic()
        # A clock that goes backwards must not mint tokens.
        now = max(now, self.updated)
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate_per_s)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - self.tokens) / self.rate_per_s


@dataclass
class Tenant:
    name: str
    api_key: str
    models: list[str] = field(default_factory=list)
    requests_per_s: float = 10.0
    burst: int = 20
    monthly_budget_usd: float = 100.0

    # Spend already committed this window, including requests still in flight.
    reserved_usd: float = 0.0
    settled_usd: float = 0.0
    _bucket: TokenBucket | None = None

    def bucket(self) -> TokenBucket:
        if self._bucket is None:
            self._bucket = TokenBucket(self.requests_per_s, float(self.burst))
        return self._bucket

    def may_use(self, model: str) -> bool:
        """An empty allow-list means every model, which is the sane default for
        a tenant that has not been restricted."""
        if not self.models:
            return True
        return any(
            model == allowed or (allowed.endswith("*") and model.startswith(allowed[:-1]))
            for allowed in self.models
        )

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.monthly_budget_usd - self.reserved_usd)

    @property
    def utilisation(self) -> float:
        if not self.monthly_budget_usd:
            return 0.0
        return self.reserved_usd / self.monthly_budget_usd


@dataclass
class PolicyEngine:
    tenants: dict[str, Tenant] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "PolicyEngine":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        tenants = {}
        for entry in raw.get("tenants", []):
            tenant = Tenant(
                name=entry["name"],
                api_key=entry["api_key"],
                models=list(entry.get("models", [])),
                requests_per_s=float(entry.get("requests_per_s", 10)),
                burst=int(entry.get("burst", 20)),
                monthly_budget_usd=float(entry.get("monthly_budget_usd", 100)),
            )
            tenants[tenant.api_key] = tenant
        return cls(tenants)

    def tenant_for(self, api_key: str | None) -> Tenant | None:
        return self.tenants.get(api_key or "")

    def admit(
        self, api_key: str | None, model: str, estimated_usd: float = 0.0
    ) -> tuple[Decision, Tenant | None]:
        """Decide before the upstream call, reserving the estimated spend."""
        tenant = self.tenant_for(api_key)
        if tenant is None:
            return Decision(False, "no tenant for this key", Denial.UNKNOWN_TENANT), None

        if not tenant.may_use(model):
            allowed = ", ".join(tenant.models)
            return Decision(
                False,
                f"tenant {tenant.name!r} may not use {model!r}; allowed: {allowed}",
                Denial.MODEL,
            ), tenant

        # Budget before rate: a tenant that is out of money should be told that,
        # not told to slow down and try the same wall again.
        if tenant.reserved_usd + estimated_usd > tenant.monthly_budget_usd:
            return Decision(
                False,
                f"tenant {tenant.name!r} has ${tenant.reserved_usd:.2f} of "
                f"${tenant.monthly_budget_usd:.2f} committed; this request needs "
                f"${estimated_usd:.4f}",
                Denial.BUDGET,
            ), tenant

        ok, wait = tenant.bucket().take()
        if not ok:
            return Decision(
                False,
                f"tenant {tenant.name!r} is over {tenant.requests_per_s}/s",
                Denial.RATE,
                retry_after_s=round(wait, 3),
            ), tenant

        tenant.reserved_usd += estimated_usd
        return Decision(True, "admitted"), tenant

    def settle(self, tenant: Tenant, estimated_usd: float, actual_usd: float) -> None:
        """Replace the reservation with what the call actually cost.

        Estimates are always wrong; without settlement the error accumulates and
        a tenant is eventually throttled by the gateway's arithmetic rather than
        by its own spending.
        """
        tenant.reserved_usd = max(0.0, tenant.reserved_usd - estimated_usd + actual_usd)
        tenant.settled_usd += actual_usd

    def release(self, tenant: Tenant, estimated_usd: float) -> None:
        """Give back the reservation for a call that never happened."""
        tenant.reserved_usd = max(0.0, tenant.reserved_usd - estimated_usd)

    def report(self) -> list[dict[str, Any]]:
        return [
            {
                "tenant": t.name,
                "settled_usd": round(t.settled_usd, 6),
                "reserved_usd": round(t.reserved_usd, 6),
                "budget_usd": t.monthly_budget_usd,
                "utilisation": round(t.utilisation, 4),
            }
            for t in sorted(self.tenants.values(), key=lambda x: x.name)
        ]
