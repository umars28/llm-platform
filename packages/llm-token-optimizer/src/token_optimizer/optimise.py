"""Three ways to spend fewer tokens, each measurable on its own.

**Prefix caching** is the largest win and the one people get wrong. A cache is a
prefix match: any byte that changes anywhere before the breakpoint invalidates
everything after it. The saving is therefore not a property of the cache, it is a
property of how stable your prefix is -- and a single timestamp or an unsorted
dict in the system prompt reduces the hit rate to zero while every line of
caching code still looks correct. `find_invalidators` looks for those.

**Context pruning** removes what the model will not use: repeated tool results,
superseded observations, whole blocks of a payload that no later turn refers to.
Unlike caching it changes what the model sees, so it is the one that can cost
quality and the one that has to be paired with an eval.

**Model routing** sends work to the cheapest model that can do it. It is the
biggest saving available and the easiest to overstate -- the number only means
something next to an accuracy measurement on the routed traffic.

Every function here reports what it removed, not just the result, so a saving
can be inspected rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .counting import Spend, count, count_messages, pricing_for, reduction

# Anthropic caches in blocks and will not cache a prefix shorter than this, so a
# breakpoint placed before it silently does nothing.
MIN_CACHEABLE_TOKENS = 1024


# -- prefix caching ------------------------------------------------------

@dataclass(frozen=True)
class Invalidator:
    """Something in a prefix that changes between requests and kills the cache."""

    kind: str
    evidence: str
    why: str


_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?|\b\d{2}:\d{2}:\d{2}\b"
)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_EPOCH = re.compile(r"\b1[6-9]\d{8}\b")
_REQUEST_ID = re.compile(r"\b(req|request|trace|span|run)[-_]?id[\"']?\s*[:=]\s*\S+", re.I)


def find_invalidators(prefix: str) -> list[Invalidator]:
    """Content that varies per request and therefore prevents a cache hit.

    These are found by pattern because they are findable that way, and because
    the failure is silent: caching code that never hits looks exactly like
    caching code that does until you read `cache_read_input_tokens`.
    """
    found: list[Invalidator] = []
    for pattern, kind, why in (
        (_TIMESTAMP, "timestamp", "changes every request, invalidating everything after it"),
        (_UUID, "uuid", "unique per request"),
        (_EPOCH, "epoch", "a unix timestamp changes every second"),
        (_REQUEST_ID, "request-id", "identifies one request and cannot repeat"),
    ):
        for match in pattern.findall(prefix)[:3]:
            text = match if isinstance(match, str) else next((m for m in match if m), "")
            if text:
                found.append(Invalidator(kind, text[:60], why))
    return found


@dataclass
class CachePlan:
    """Where a breakpoint goes, and what it is worth."""

    prefix_tokens: int
    volatile_tokens: int
    cacheable: bool
    reason: str
    invalidators: list[Invalidator] = field(default_factory=list)

    def spend(self, model: str, calls: int, output_tokens: int = 0) -> Spend:
        """What `calls` requests cost with this plan: one write, the rest reads."""
        if not self.cacheable or calls < 1:
            return Spend(model, input_tokens=(self.prefix_tokens + self.volatile_tokens) * calls,
                         output_tokens=output_tokens * calls)
        return Spend(
            model,
            input_tokens=self.volatile_tokens * calls,
            cache_write_tokens=self.prefix_tokens,
            cache_read_tokens=self.prefix_tokens * (calls - 1),
            output_tokens=output_tokens * calls,
        )

    def breakeven_calls(self, model: str) -> int:
        """How many requests before the cache write pays for itself.

        A write costs 1.25x fresh input and a read 0.1x, so a prefix used once
        is more expensive cached than not. Caching a prefix below the breakeven
        loses money while looking like an optimisation.
        """
        if not self.cacheable:
            return 0
        price_in, _, price_write, price_read = pricing_for(model)
        if price_in == 0:
            return 1
        calls = 1
        while calls < 1000:
            cached = price_write + price_read * (calls - 1)
            fresh = price_in * calls
            if cached < fresh:
                return calls
            calls += 1
        return calls


def plan_cache(prefix: str, volatile: str = "") -> CachePlan:
    """Decide whether a prefix is worth a breakpoint, and say why if not."""
    prefix_tokens = count(prefix)
    volatile_tokens = count(volatile)
    invalidators = find_invalidators(prefix)

    if invalidators:
        return CachePlan(
            prefix_tokens, volatile_tokens, cacheable=False,
            reason=f"prefix contains {len(invalidators)} per-request value(s); "
                   f"the cache would never hit",
            invalidators=invalidators,
        )
    if prefix_tokens < MIN_CACHEABLE_TOKENS:
        return CachePlan(
            prefix_tokens, volatile_tokens, cacheable=False,
            reason=f"prefix is {prefix_tokens} tokens, below the {MIN_CACHEABLE_TOKENS} "
                   "minimum; a breakpoint here is silently ignored",
        )
    return CachePlan(
        prefix_tokens, volatile_tokens, cacheable=True,
        reason=f"{prefix_tokens} stable tokens ahead of {volatile_tokens} volatile",
    )


# -- context pruning -----------------------------------------------------

@dataclass
class Pruned:
    messages: list[dict[str, Any]]
    removed: list[str] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def saved_percent(self) -> float:
        return reduction(self.tokens_before, self.tokens_after)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True)


def prune_duplicate_results(messages: Sequence[dict[str, Any]]) -> Pruned:
    """Replace a repeated tool result with a pointer to the first one.

    An agent that queries the same thing twice pays for the answer twice and
    reads nothing new. Keeping the first occurrence preserves the information;
    keeping both preserves only the cost.
    """
    before = count_messages(messages)
    seen: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    removed: list[str] = []

    for index, message in enumerate(messages):
        text = _content_text(message.get("content", ""))
        digest = hashlib.sha1(text.encode()).hexdigest()

        if message.get("role") == "user" and len(text) > 200 and digest in seen:
            out.append({
                **message,
                "content": f"[identical to the result at position {seen[digest]}]",
            })
            removed.append(f"duplicate at {index} (same as {seen[digest]})")
            continue

        seen.setdefault(digest, index)
        out.append(message)

    return Pruned(out, removed, before, count_messages(out))


def prune_superseded(
    messages: Sequence[dict[str, Any]], keep_last: int = 2
) -> Pruned:
    """Summarise older tool results, keeping the most recent intact.

    Early observations in an investigation are usually superseded by later ones.
    This is the pruning that can cost quality, so it is the one that must be run
    against an eval rather than adopted on the strength of the saving alone.
    """
    before = count_messages(messages)
    large = [
        i for i, m in enumerate(messages)
        if m.get("role") == "user" and len(_content_text(m.get("content", ""))) > 400
    ]
    doomed = set(large[:-keep_last]) if len(large) > keep_last else set()

    out: list[dict[str, Any]] = []
    removed: list[str] = []
    for index, message in enumerate(messages):
        if index in doomed:
            text = _content_text(message.get("content", ""))
            out.append({**message, "content": f"[earlier result, {len(text)} chars, superseded]"})
            removed.append(f"superseded result at {index}")
        else:
            out.append(message)

    return Pruned(out, removed, before, count_messages(out))


# -- model routing -------------------------------------------------------

@dataclass(frozen=True)
class Route:
    model: str
    reason: str


def route(
    task: str,
    cheap: str = "claude-haiku-4-5",
    strong: str = "claude-opus-5",
    strong_markers: Sequence[str] = (
        "diagnose", "root cause", "why did", "explain the failure",
        "design", "trade-off", "compare", "decide",
    ),
) -> Route:
    """Send classification and extraction to the cheap model, reasoning to the strong one.

    Routing by surface keywords is crude and stated as such. It is here to make
    the saving measurable, not to be the final policy -- a real router would
    classify the task, and the classifier would need its own evaluation before
    its decisions counted.
    """
    lowered = task.lower()
    hit = next((m for m in strong_markers if m in lowered), None)
    if hit:
        return Route(strong, f"reasoning task (matched {hit!r})")
    return Route(cheap, "no reasoning marker; cheap model is sufficient")


def routing_saving(
    tasks: Sequence[str], input_tokens: int, output_tokens: int,
    cheap: str = "claude-haiku-4-5", strong: str = "claude-opus-5",
) -> dict[str, float]:
    """Cost of routing a workload, against sending all of it to the strong model."""
    routed = Spend(strong)
    by_model: dict[str, int] = {}
    for task in tasks:
        chosen = route(task, cheap, strong).model
        by_model[chosen] = by_model.get(chosen, 0) + 1

    routed_cost = sum(
        Spend(model, input_tokens=input_tokens * n, output_tokens=output_tokens * n).cost_usd
        for model, n in by_model.items()
    )
    baseline = Spend(
        strong, input_tokens=input_tokens * len(tasks), output_tokens=output_tokens * len(tasks)
    ).cost_usd

    return {
        "tasks": len(tasks),
        "to_cheap": by_model.get(cheap, 0),
        "to_strong": by_model.get(strong, 0),
        "baseline_usd": round(baseline, 6),
        "routed_usd": round(routed_cost, 6),
        "saving_percent": round(reduction(baseline, routed_cost), 1) if baseline else 0.0,
    }
