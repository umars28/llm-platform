"""Deterministic assertions: the layer that runs before any model is asked.

Most of what people reach for an LLM judge to check does not need one. Whether
a required concept appears, whether a forbidden claim was made, whether the
output parses, whether a tool was called, whether latency stayed under budget --
all of that is decidable in code, for free, with zero variance between runs.

That matters more than the saving. A judge that varies run to run makes it
impossible to tell whether a score moved because the system changed or because
the judge did, and a judge needs its own evaluation before its verdicts mean
anything. Every check that can be deterministic should be, so the judge is left
holding only the questions that genuinely require reading.

Each assertion returns a `Check` with a verdict and a reason. A failed check
always says what it wanted and what it got, because a regression gate that only
says "failed" sends someone to read the code instead of the diff.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    reason: str
    kind: str = "deterministic"
    weight: float = 1.0

    @property
    def failed(self) -> bool:
        return not self.passed


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def contains_all(text: str, groups: Sequence[Sequence[str]], name: str = "contains_all") -> Check:
    """Every group must be satisfied by at least one of its alternatives.

    Groups are alternatives so that wording does not decide the verdict: a
    concept expressed as "pool exhausted" or "pool saturated" is the same
    concept, and an assertion that only accepts one of them is measuring
    phrasing rather than correctness.
    """
    haystack = _normalise(text)
    missing = [g for g in groups if not any(alt.lower() in haystack for alt in g)]
    if missing:
        return Check(
            name, False,
            f"missing {len(missing)} of {len(groups)} required concepts: "
            + "; ".join(" / ".join(g) for g in missing[:3]),
        )
    return Check(name, True, f"all {len(groups)} required concepts present")


def contains_none(text: str, forbidden: Sequence[str], name: str = "contains_none") -> Check:
    haystack = _normalise(text)
    hits = [phrase for phrase in forbidden if phrase.lower() in haystack]
    if hits:
        return Check(name, False, f"asserted forbidden claims: {', '.join(hits[:3])}")
    return Check(name, True, "no forbidden claims")


def equals(actual: Any, expected: Any, name: str = "equals") -> Check:
    if actual == expected:
        return Check(name, True, f"got {actual!r}")
    return Check(name, False, f"expected {expected!r}, got {actual!r}")


def one_of(actual: Any, allowed: Sequence[Any], name: str = "one_of") -> Check:
    if actual in allowed:
        return Check(name, True, f"got {actual!r}")
    return Check(name, False, f"expected one of {list(allowed)!r}, got {actual!r}")


def is_valid_json(text: str, name: str = "is_valid_json") -> Check:
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return Check(name, False, f"not valid JSON: {exc}")
    return Check(name, True, "parses as JSON")


def matches_schema(payload: Any, required: Sequence[str], name: str = "matches_schema") -> Check:
    if not isinstance(payload, dict):
        return Check(name, False, f"expected an object, got {type(payload).__name__}")
    missing = [key for key in required if key not in payload]
    if missing:
        return Check(name, False, f"missing keys: {', '.join(missing)}")
    return Check(name, True, f"all {len(required)} required keys present")


def called_tools(actual: Sequence[str], expected: Sequence[str], name: str = "called_tools") -> Check:
    missing = [tool for tool in expected if tool not in actual]
    if missing:
        return Check(name, False, f"never called: {', '.join(missing)}")
    return Check(name, True, f"called all of {', '.join(expected)}")


def did_not_call(actual: Sequence[str], forbidden: Sequence[str], name: str = "did_not_call") -> Check:
    hits = [tool for tool in forbidden if tool in actual]
    if hits:
        return Check(name, False, f"called forbidden tools: {', '.join(hits)}")
    return Check(name, True, "no forbidden tool calls")


def under_budget(value: float, limit: float, unit: str, name: str = "under_budget") -> Check:
    """Cost and latency are correctness properties in a system that ships."""
    if value <= limit:
        return Check(name, True, f"{value:.4g}{unit} within {limit:.4g}{unit}")
    over = (value - limit) / limit * 100 if limit else float("inf")
    return Check(name, False, f"{value:.4g}{unit} exceeds {limit:.4g}{unit} by {over:.0f}%")


@dataclass
class Result:
    """The outcome of every check run against one case."""

    case_id: str
    checks: list[Check] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.failed]

    @property
    def score(self) -> float:
        """Weighted pass fraction, for tracking drift short of a hard failure."""
        total = sum(c.weight for c in self.checks)
        if not total:
            return 0.0
        return sum(c.weight for c in self.checks if c.passed) / total

    def deterministic_only(self) -> bool:
        return all(c.kind == "deterministic" for c in self.checks)

    def explain(self) -> str:
        if self.passed:
            return f"{self.case_id}: pass ({len(self.checks)} checks)"
        lines = [f"{self.case_id}: FAIL ({len(self.failures)}/{len(self.checks)} checks)"]
        lines += [f"    {c.name}: {c.reason}" for c in self.failures]
        return "\n".join(lines)


ASSERTIONS: dict[str, Callable[..., Check]] = {
    "contains_all": contains_all,
    "contains_none": contains_none,
    "equals": equals,
    "one_of": one_of,
    "is_valid_json": is_valid_json,
    "matches_schema": matches_schema,
    "called_tools": called_tools,
    "did_not_call": did_not_call,
    "under_budget": under_budget,
}
