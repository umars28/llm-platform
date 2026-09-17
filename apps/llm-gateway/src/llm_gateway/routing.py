"""Provider selection and failover.

The question an infrastructure interview asks about this layer is not "which
model is best". It is "what happens at three in the morning when the provider
returns 500s", and the answer has to be something other than "every request
fails until someone wakes up".

Three mechanisms, each answering a different failure:

**Circuit breaking** stops sending traffic to a provider that is already
failing. Without it every request pays the full timeout before failing, so one
dead provider turns into a latency incident across every tenant. The breaker
probes periodically rather than waiting for a human, because a breaker that only
closes on restart turns a twelve-minute outage into a three-hour one -- a failure
this project has already seen in `ops-copilot`'s SC-020.

**Failover** sends the request to the next provider in the chain. It is only
safe because these are read-like operations: a completion that was never
returned can be requested again without double-charging anyone.

**Health-aware ordering** prefers the provider that is currently working, rather
than the one that is first in a config file.

What is deliberately *not* here: retrying the same provider immediately. A
provider returning 500 is not helped by being asked again a millisecond later,
and retry-without-backoff is exactly the amplification that turns a blip into an
outage -- `ops-copilot`'s SC-018 is a scenario about that mistake.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Sequence


class BreakerState(str, Enum):
    CLOSED = "closed"      # traffic flows
    OPEN = "open"          # traffic refused, provider presumed down
    HALF_OPEN = "half_open"  # one trial request allowed through


@dataclass
class CircuitBreaker:
    """Opens on repeated failure, and probes without waiting for a human."""

    failure_threshold: int = 5
    probe_after_s: float = 30.0
    now: float | None = None

    failures: int = 0
    opened_at: float | None = None
    state: BreakerState = BreakerState.CLOSED
    half_open_in_flight: bool = False

    def _clock(self, now: float | None = None) -> float:
        if now is not None:
            return now
        return self.now if self.now is not None else time.monotonic()

    def allows(self, now: float | None = None) -> bool:
        now = self._clock(now)
        if self.state is BreakerState.CLOSED:
            return True
        if self.state is BreakerState.OPEN:
            if self.opened_at is not None and now - self.opened_at >= self.probe_after_s:
                self.state = BreakerState.HALF_OPEN
                self.half_open_in_flight = False
            else:
                return False
        # Half-open admits exactly one trial request; a flood of them would
        # re-break a provider that is only just recovering.
        if self.half_open_in_flight:
            return False
        self.half_open_in_flight = True
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self.half_open_in_flight = False
        self.state = BreakerState.CLOSED

    def record_failure(self, now: float | None = None) -> None:
        now = self._clock(now)
        self.half_open_in_flight = False
        if self.state is BreakerState.HALF_OPEN:
            # The probe failed; go back to open and wait another interval.
            self.state = BreakerState.OPEN
            self.opened_at = now
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = now


@dataclass
class Provider:
    name: str
    base_url: str
    models: dict[str, str] = field(default_factory=dict)  # logical -> provider id
    priority: int = 100
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    successes: int = 0
    failures: int = 0
    latency_ms_total: float = 0.0

    def serves(self, model: str) -> bool:
        return model in self.models

    def upstream_model(self, model: str) -> str:
        return self.models.get(model, model)

    @property
    def healthy(self) -> bool:
        return self.breaker.state is not BreakerState.OPEN

    @property
    def mean_latency_ms(self) -> float:
        return self.latency_ms_total / self.successes if self.successes else 0.0

    def observe(self, ok: bool, latency_ms: float = 0.0, now: float | None = None) -> None:
        if ok:
            self.successes += 1
            self.latency_ms_total += latency_ms
            self.breaker.record_success()
        else:
            self.failures += 1
            self.breaker.record_failure(now)


@dataclass
class Router:
    """Chooses providers for a model, healthiest and highest priority first."""

    providers: list[Provider] = field(default_factory=list)

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "Router":
        providers = [
            Provider(
                name=entry["name"],
                base_url=entry["base_url"],
                models=dict(entry.get("models", {})),
                priority=int(entry.get("priority", 100)),
                breaker=CircuitBreaker(
                    failure_threshold=int(entry.get("failure_threshold", 5)),
                    probe_after_s=float(entry.get("probe_after_s", 30)),
                ),
            )
            for entry in raw.get("providers", [])
        ]
        return cls(providers)

    def candidates(self, model: str, now: float | None = None) -> list[Provider]:
        """Providers that serve this model and are willing to take traffic.

        Sorted by health first, then configured priority: a healthy fallback is
        a better choice than a preferred provider that is currently failing.
        """
        eligible = [p for p in self.providers if p.serves(model)]
        allowed = [p for p in eligible if p.breaker.allows(now)]
        return sorted(allowed, key=lambda p: (not p.healthy, p.priority))

    def chain(self, model: str, now: float | None = None) -> Iterator[Provider]:
        yield from self.candidates(model, now)

    def known_models(self) -> set[str]:
        return {m for p in self.providers for m in p.models}

    def report(self) -> list[dict[str, Any]]:
        return [
            {
                "provider": p.name,
                "state": p.breaker.state.value,
                "successes": p.successes,
                "failures": p.failures,
                "mean_latency_ms": round(p.mean_latency_ms, 1),
            }
            for p in sorted(self.providers, key=lambda x: x.priority)
        ]


class AllProvidersFailed(RuntimeError):
    """Every provider for a model refused or failed.

    Carries what each one did, because "the gateway returned 503" is not enough
    to act on at three in the morning.
    """

    def __init__(self, model: str, attempts: Sequence[tuple[str, str]]):
        self.model = model
        self.attempts = list(attempts)
        detail = "; ".join(f"{name}: {why}" for name, why in attempts) or "no provider serves it"
        super().__init__(f"no provider could serve {model!r} ({detail})")
