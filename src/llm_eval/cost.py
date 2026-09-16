"""Token and cost accounting, with cache reads counted separately.

Cost is treated as a correctness property rather than a footnote. A change that
improves quality by two points and triples the bill is a regression in every
sense that matters to whoever pays for it, and a suite that only tracks quality
will pass it.

Cache reads are separated from fresh input because that separation *is* the
optimisation. A prompt caching change shows up as input tokens moving into the
cache-read column at a tenth of the price; totalling them together hides exactly
the effect being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

# USD per million tokens: (input, output, cache write, cache read).
# Cache writes cost more than fresh input and cache reads cost far less, which
# is why a caching change only pays off above a certain reuse rate.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5": (10.00, 50.00, 12.50, 1.00),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-7": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-4-6": (3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}
DEFAULT_PRICING = PRICING["claude-opus-5"]


def pricing_for(model: str) -> tuple[float, float, float, float]:
    """Price lookup tolerant of gateway namespacing such as "anthropic/".

    An unknown model falls back to the most expensive tier: an overstated cost
    prompts a question, an understated one gets quoted in a README.
    """
    return PRICING.get(model.split("/")[-1].lower(), DEFAULT_PRICING)


@dataclass
class Usage:
    """Token counts for one call or one aggregate."""

    model: str = "claude-opus-5"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache. Zero means caching is not working."""
        return self.cache_read_tokens / self.total_input if self.total_input else 0.0

    @property
    def cost_usd(self) -> float:
        price_in, price_out, price_write, price_read = pricing_for(self.model)
        return (
            self.input_tokens * price_in
            + self.output_tokens * price_out
            + self.cache_write_tokens * price_write
            + self.cache_read_tokens * price_read
        ) / 1_000_000

    @property
    def uncached_cost_usd(self) -> float:
        """What the same tokens would have cost with no caching at all.

        The counterfactual is the only honest way to state a caching saving:
        comparing against a different run conflates the cache with whatever else
        changed between them.
        """
        price_in, price_out, _, _ = pricing_for(self.model)
        return (
            (self.input_tokens + self.cache_write_tokens + self.cache_read_tokens) * price_in
            + self.output_tokens * price_out
        ) / 1_000_000

    @property
    def cache_saving_usd(self) -> float:
        return max(0.0, self.uncached_cost_usd - self.cost_usd)

    def add(self, other: "Usage") -> "Usage":
        if other.model != self.model and self.calls:
            raise ValueError(
                f"cannot add usage for {other.model!r} to {self.model!r}; "
                "aggregate per model, or use UsageLedger"
            )
        return Usage(
            model=other.model if not self.calls else self.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            calls=self.calls + other.calls,
        )

    @classmethod
    def from_response(cls, model: str, usage: object) -> "Usage":
        get = lambda name: int(getattr(usage, name, 0) or 0)  # noqa: E731
        return cls(
            model=model,
            input_tokens=get("input_tokens"),
            output_tokens=get("output_tokens"),
            cache_write_tokens=get("cache_creation_input_tokens"),
            cache_read_tokens=get("cache_read_input_tokens"),
            calls=1,
        )


@dataclass
class UsageLedger:
    """Usage across several models, which is what routing produces."""

    entries: dict[str, Usage] = field(default_factory=dict)

    def record(self, usage: Usage) -> None:
        existing = self.entries.get(usage.model)
        self.entries[usage.model] = existing.add(usage) if existing else usage

    def record_many(self, usages: Iterable[Usage]) -> None:
        for usage in usages:
            self.record(usage)

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.entries.values())

    @property
    def uncached_cost_usd(self) -> float:
        return sum(u.uncached_cost_usd for u in self.entries.values())

    @property
    def calls(self) -> int:
        return sum(u.calls for u in self.entries.values())

    @property
    def cache_hit_rate(self) -> float:
        read = sum(u.cache_read_tokens for u in self.entries.values())
        total = sum(u.total_input for u in self.entries.values())
        return read / total if total else 0.0

    def by_model(self) -> dict[str, dict[str, float]]:
        return {
            model: {
                "calls": usage.calls,
                "cost_usd": round(usage.cost_usd, 6),
                "share": round(usage.cost_usd / self.cost_usd, 4) if self.cost_usd else 0.0,
                "cache_hit_rate": round(usage.cache_hit_rate, 4),
            }
            for model, usage in sorted(self.entries.items())
        }

    def summary(self) -> dict[str, float]:
        return {
            "calls": self.calls,
            "cost_usd": round(self.cost_usd, 6),
            "uncached_cost_usd": round(self.uncached_cost_usd, 6),
            "cache_saving_usd": round(max(0.0, self.uncached_cost_usd - self.cost_usd), 6),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
        }


def compare(before: UsageLedger, after: UsageLedger) -> dict[str, float]:
    """Percentage change, for stating an optimisation without overclaiming."""
    baseline = before.cost_usd
    return {
        "before_usd": round(baseline, 6),
        "after_usd": round(after.cost_usd, 6),
        "change_percent": round((after.cost_usd - baseline) / baseline * 100, 1)
        if baseline
        else 0.0,
        "before_cache_hit_rate": round(before.cache_hit_rate, 4),
        "after_cache_hit_rate": round(after.cache_hit_rate, 4),
    }
