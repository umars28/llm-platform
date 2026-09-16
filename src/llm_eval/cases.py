"""Case definitions and the runner.

A case is a subject plus the checks that must hold over it. The subject is
whatever the system under test produced -- text, the tools it called, what it
cost, how long it took -- and it can come from a live call or from a recorded
trace on disk.

Recorded traces are the default on purpose. Re-running an agent to evaluate it
makes every number a function of two things at once, and means a suite cannot be
re-scored after a rubric change without paying for the whole run again. Scoring
a recorded artefact separates producing behaviour from judging it, which is what
lets the deterministic half of this suite run in CI with no credentials at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from .assertions import (
    ASSERTIONS,
    Check,
    Result,
    called_tools,
    contains_all,
    contains_none,
    did_not_call,
    equals,
    is_valid_json,
    matches_schema,
    one_of,
    under_budget,
)
from .judge import Criterion, Judge


@dataclass
class Subject:
    """What the system under test produced for one case."""

    text: str = ""
    tools_called: list[str] = field(default_factory=list)
    payload: Any = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trace(cls, trace: dict[str, Any]) -> "Subject":
        """Read an ops-copilot style run record.

        The field names are that project's; anything absent simply defaults,
        so a partial or older trace still scores rather than raising.
        """
        proposal = trace.get("proposal") or {}
        text = " ".join(
            str(proposal.get(key, ""))
            for key in ("root_cause", "justification")
        ).strip() or trace.get("final_text", "")
        return cls(
            text=text,
            tools_called=[c.get("name", "") for c in trace.get("tool_calls", [])],
            payload=proposal or None,
            cost_usd=float(trace.get("cost_usd", 0.0) or 0.0),
            latency_s=float(trace.get("elapsed_s", 0.0) or 0.0),
            metadata={k: trace.get(k) for k in ("scenario_id", "model", "turns")},
        )


@dataclass
class Case:
    id: str
    description: str = ""
    assertions: list[dict[str, Any]] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)
    cost_budget_usd: float | None = None
    latency_budget_s: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Case":
        return cls(
            id=raw["id"],
            description=raw.get("description", ""),
            assertions=list(raw.get("assertions", [])),
            criteria=[
                Criterion(
                    name=c["name"], question=c["question"], weight=float(c.get("weight", 1.0))
                )
                for c in raw.get("judge", [])
            ],
            cost_budget_usd=raw.get("cost_budget_usd"),
            latency_budget_s=raw.get("latency_budget_s"),
        )


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return [Case.from_dict(entry) for entry in raw.get("cases", [])]


def _run_assertion(spec: dict[str, Any], subject: Subject) -> Check:
    """Dispatch one YAML assertion against the subject."""
    kind = spec.get("type")
    name = spec.get("name", kind or "unnamed")

    if kind not in ASSERTIONS:
        return Check(name, False, f"unknown assertion type {kind!r}; "
                                  f"have {', '.join(sorted(ASSERTIONS))}")

    if kind == "contains_all":
        return contains_all(subject.text, spec["groups"], name)
    if kind == "contains_none":
        return contains_none(subject.text, spec["phrases"], name)
    if kind == "called_tools":
        return called_tools(subject.tools_called, spec["tools"], name)
    if kind == "did_not_call":
        return did_not_call(subject.tools_called, spec["tools"], name)
    if kind == "is_valid_json":
        return is_valid_json(subject.text, name)
    if kind == "matches_schema":
        return matches_schema(subject.payload, spec["required"], name)
    if kind == "equals":
        return equals(_field(subject, spec["field"]), spec["value"], name)
    if kind == "one_of":
        return one_of(_field(subject, spec["field"]), spec["values"], name)
    if kind == "under_budget":
        return under_budget(
            _field(subject, spec["field"]), float(spec["limit"]), spec.get("unit", ""), name
        )
    return Check(name, False, f"assertion {kind!r} has no dispatch")


def _field(subject: Subject, path: str) -> Any:
    """Read a dotted field, so a case can assert on the payload's contents."""
    target: Any = subject
    for part in path.split("."):
        if isinstance(target, dict):
            target = target.get(part)
        else:
            target = getattr(target, part, None)
        if target is None:
            return None
    return target


def run_case(case: Case, subject: Subject, judge: Judge | None = None) -> Result:
    """Deterministic checks first; the judge only if they all held.

    Ordering is not an optimisation detail. If a deterministic check already
    failed, the case fails, and paying a model to elaborate on a decided outcome
    buys nothing. It also keeps the judge's verdicts to the cases where its
    opinion is the deciding factor, which is where its agreement rate matters.
    """
    result = Result(case_id=case.id, metadata=dict(subject.metadata))

    for spec in case.assertions:
        result.add(_run_assertion(spec, subject))

    if case.cost_budget_usd is not None:
        result.add(under_budget(subject.cost_usd, case.cost_budget_usd, "$", "cost_budget"))
    if case.latency_budget_s is not None:
        result.add(under_budget(subject.latency_s, case.latency_budget_s, "s", "latency_budget"))

    if judge is not None and case.criteria and result.passed:
        for criterion in case.criteria:
            result.add(judge.check(criterion, subject.text))

    return result


def run_suite(
    cases: Sequence[Case],
    subjects: dict[str, Subject],
    judge: Judge | None = None,
) -> list[Result]:
    results = []
    for case in cases:
        subject = subjects.get(case.id)
        if subject is None:
            result = Result(case_id=case.id)
            result.add(Check("subject", False, "no recorded subject for this case"))
            results.append(result)
            continue
        results.append(run_case(case, subject, judge))
    return results
