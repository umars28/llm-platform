"""Deterministic scoring of a diagnosis against a scenario's ground truth.

Scoring here is keyword-based on purpose. It is cheap, it has no variance
between runs, and it never needs its own evaluation -- which means a change in
the score is a change in the agent, not in the judge. An LLM judge belongs in
the eval project that builds on this one, not in the harness the agent is
developed against.

The cost of that choice is real and worth stating: substring matching cannot
tell a claim from a mention. It is measured rather than hidden -- `clean` is
reported separately from `root_cause_hit` instead of being folded into one
number, so a run where the agent named a misleading factor while reaching the
right conclusion is visible as exactly that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .agent import AgentRun
from .world import Scenario


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _group_matched(haystack: str, group: list[str]) -> str | None:
    """Return the first alternative in the group that appears, or None."""
    for alternative in group:
        if alternative.lower() in haystack:
            return alternative
    return None


@dataclass
class Score:
    scenario_id: str
    category: str

    root_cause_hit: bool = False
    matched_groups: int = 0
    total_groups: int = 0
    missing_groups: list[list[str]] = field(default_factory=list)

    clean: bool = True
    misleading_claims: list[str] = field(default_factory=list)

    action_match: bool = False
    proposed_action: str | None = None
    expected_actions: list[str] = field(default_factory=list)

    over_reach: bool = False
    confidence: float | None = None

    read_tool_calls: int = 0
    turns: int = 0
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def correct(self) -> bool:
        """The strict bar: right cause, right action, nothing misleading asserted."""
        return self.root_cause_hit and self.action_match and self.clean

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "correct": self.correct,
            "root_cause_hit": self.root_cause_hit,
            "matched_groups": self.matched_groups,
            "total_groups": self.total_groups,
            "missing_groups": self.missing_groups,
            "clean": self.clean,
            "misleading_claims": self.misleading_claims,
            "action_match": self.action_match,
            "proposed_action": self.proposed_action,
            "expected_actions": self.expected_actions,
            "over_reach": self.over_reach,
            "confidence": self.confidence,
            "read_tool_calls": self.read_tool_calls,
            "turns": self.turns,
            "elapsed_s": round(self.elapsed_s, 2),
            "cost_usd": round(self.cost_usd, 4),
            "error": self.error,
        }


def score_run(run: AgentRun, scenario: Scenario) -> Score:
    truth = scenario.ground_truth
    expected = list(truth.get("expected_actions", []))

    score = Score(
        scenario_id=scenario.id,
        category=scenario.category,
        expected_actions=expected,
        total_groups=len(truth.get("must_include", [])),
        read_tool_calls=run.read_tool_calls,
        turns=run.turns,
        elapsed_s=run.elapsed_s,
        cost_usd=run.cost_usd,
        error=run.error,
    )

    if run.proposal is None:
        score.missing_groups = list(truth.get("must_include", []))
        return score

    proposal = run.proposal
    score.proposed_action = proposal.get("action_id")
    score.confidence = proposal.get("confidence")
    score.action_match = score.proposed_action in expected

    # The committed answer is the diagnosis plus the reasoning offered for the
    # action, not any thinking that preceded it.
    haystack = _normalise(
        f"{proposal.get('root_cause', '')} {proposal.get('justification', '')}"
    )

    for group in truth.get("must_include", []):
        if _group_matched(haystack, group):
            score.matched_groups += 1
        else:
            score.missing_groups.append(group)
    score.root_cause_hit = score.matched_groups == score.total_groups

    for group in truth.get("must_not_include", []):
        hit = _group_matched(haystack, group if isinstance(group, list) else [group])
        if hit:
            score.misleading_claims.append(hit)
    score.clean = not score.misleading_claims

    # Queueing a production change against a system that needs none is a
    # distinct failure from getting the cause wrong, so it is counted separately.
    if expected == ["no_action_required"] or (
        "no_action_required" in expected and len(expected) == 1
    ):
        score.over_reach = bool(run.change_requests)
    else:
        score.over_reach = any(
            req.get("action_id") not in expected for req in run.change_requests
        )

    return score


def summarise(scores: list[Score]) -> dict[str, Any]:
    """Aggregate scores into the numbers the README reports."""
    n = len(scores)
    if n == 0:
        return {"scenarios": 0}

    completed = [s for s in scores if s.error is None]
    by_category: dict[str, dict[str, int]] = {}
    for s in scores:
        bucket = by_category.setdefault(s.category, {"n": 0, "correct": 0, "cause": 0})
        bucket["n"] += 1
        bucket["correct"] += int(s.correct)
        bucket["cause"] += int(s.root_cause_hit)

    def rate(predicate) -> float:
        return round(100 * sum(1 for s in scores if predicate(s)) / n, 1)

    return {
        "scenarios": n,
        "completed": len(completed),
        "errored": n - len(completed),
        "root_cause_hit_rate": rate(lambda s: s.root_cause_hit),
        "action_match_rate": rate(lambda s: s.action_match),
        "clean_rate": rate(lambda s: s.clean),
        "strict_correct_rate": rate(lambda s: s.correct),
        "over_reach_count": sum(1 for s in scores if s.over_reach),
        "mean_read_tool_calls": round(
            sum(s.read_tool_calls for s in scores) / n, 1
        ),
        "mean_turns": round(sum(s.turns for s in scores) / n, 1),
        "mean_elapsed_s": round(sum(s.elapsed_s for s in scores) / n, 1),
        "total_cost_usd": round(sum(s.cost_usd for s in scores), 3),
        "mean_cost_usd": round(sum(s.cost_usd for s in scores) / n, 4),
        "by_category": {
            k: {
                **v,
                "correct_rate": round(100 * v["correct"] / v["n"], 1),
                "cause_rate": round(100 * v["cause"] / v["n"], 1),
            }
            for k, v in sorted(by_category.items())
        },
    }
