"""The regression gate: compare a run against a committed baseline.

A suite that prints a score is a dashboard. A suite that fails a build is a
gate, and the difference is entirely in what it does when the number moves.

Four ways to fail, because they are different problems:

**A case that passed now fails.** The strongest signal, and never tolerated by a
threshold -- a named case going from pass to fail is a specific regression with
a specific cause, and averaging it away is how suites stop catching things.

**The aggregate score drops more than the tolerance.** Secondary to the check
above rather than parallel to it: a `Result` passes only when every check
passes, so a case cannot slide from 1.0 to 0.9 and still be passing. What this
catches is an already-failing case losing further ground, which the pass/fail
comparison would otherwise report as unchanged.

**Cost rises more than the tolerance.** A change that improves quality and
triples the bill is a regression to whoever pays for it.

**The baseline is stale.** If the judge version changed, the old numbers were
produced by a different grader and comparing to them is meaningless. That is a
failure to be resolved by re-baselining, not a quality regression, and the gate
says which.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .assertions import Result

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "baselines"


@dataclass
class Baseline:
    """A committed snapshot of what the suite scored at a known good commit."""

    judge_version: str
    cases: dict[str, bool] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    recorded_at: str = ""
    commit: str = ""

    @property
    def pass_rate(self) -> float:
        return sum(self.cases.values()) / len(self.cases) if self.cases else 0.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_version": self.judge_version,
            "recorded_at": self.recorded_at,
            "commit": self.commit,
            "cost_usd": round(self.cost_usd, 6),
            "pass_rate": round(self.pass_rate, 4),
            "cases": self.cases,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }

    @classmethod
    def from_results(
        cls, results: Sequence[Result], judge_version: str,
        cost_usd: float = 0.0, commit: str = "",
    ) -> "Baseline":
        return cls(
            judge_version=judge_version,
            cases={r.case_id: r.passed for r in results},
            scores={r.case_id: r.score for r in results},
            cost_usd=cost_usd,
            recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            commit=commit,
        )

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        data = json.loads(Path(path).read_text())
        return cls(
            judge_version=data["judge_version"],
            cases=dict(data.get("cases", {})),
            scores={k: float(v) for k, v in data.get("scores", {}).items()},
            cost_usd=float(data.get("cost_usd", 0.0)),
            recorded_at=data.get("recorded_at", ""),
            commit=data.get("commit", ""),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    new_cases: list[str] = field(default_factory=list)
    removed_cases: list[str] = field(default_factory=list)
    stale_baseline: bool = False

    def explain(self) -> str:
        head = "gate passed" if self.passed else "GATE FAILED"
        lines = [head]
        lines += [f"  {r}" for r in self.reasons]
        if self.regressions:
            lines.append(f"  regressed: {', '.join(self.regressions)}")
        if self.improvements:
            lines.append(f"  improved: {', '.join(self.improvements)}")
        if self.new_cases:
            lines.append(f"  new (not in baseline): {', '.join(self.new_cases)}")
        if self.removed_cases:
            lines.append(f"  missing from this run: {', '.join(self.removed_cases)}")
        return "\n".join(lines)


def check_gate(
    results: Sequence[Result],
    baseline: Baseline,
    judge_version: str,
    cost_usd: float = 0.0,
    score_tolerance: float = 0.02,
    cost_tolerance: float = 0.15,
) -> GateResult:
    """Compare a run to its baseline and decide whether to fail the build."""
    current = {r.case_id: r for r in results}
    gate = GateResult(passed=True)

    if baseline.judge_version != judge_version:
        gate.passed = False
        gate.stale_baseline = True
        gate.reasons.append(
            f"baseline was graded by {baseline.judge_version!r} but this run used "
            f"{judge_version!r}; the numbers are not comparable. Re-baseline rather "
            "than treating this as a quality regression."
        )
        return gate

    for case_id, was_passing in baseline.cases.items():
        if case_id not in current:
            gate.removed_cases.append(case_id)
            continue
        now_passing = current[case_id].passed
        if was_passing and not now_passing:
            gate.regressions.append(case_id)
        elif not was_passing and now_passing:
            gate.improvements.append(case_id)

    gate.new_cases = [case_id for case_id in current if case_id not in baseline.cases]

    # A named case going from pass to fail is never averaged away.
    if gate.regressions:
        gate.passed = False
        gate.reasons.append(
            f"{len(gate.regressions)} case(s) passed in the baseline and fail now"
        )

    if gate.removed_cases:
        gate.passed = False
        gate.reasons.append(
            f"{len(gate.removed_cases)} baseline case(s) did not run; a suite that "
            "quietly shrinks stops catching things"
        )

    mean_now = sum(r.score for r in results) / len(results) if results else 0.0
    drop = baseline.mean_score - mean_now
    if drop > score_tolerance:
        gate.passed = False
        gate.reasons.append(
            f"mean score fell {drop:.3f} (from {baseline.mean_score:.3f} to "
            f"{mean_now:.3f}), over the {score_tolerance:.3f} tolerance"
        )

    if baseline.cost_usd and cost_usd:
        rise = (cost_usd - baseline.cost_usd) / baseline.cost_usd
        if rise > cost_tolerance:
            gate.passed = False
            gate.reasons.append(
                f"cost rose {rise:.0%} (from ${baseline.cost_usd:.4f} to "
                f"${cost_usd:.4f}), over the {cost_tolerance:.0%} tolerance"
            )

    if gate.passed and not gate.reasons:
        gate.reasons.append(
            f"{len(results)} cases, mean score {mean_now:.3f}, cost ${cost_usd:.4f}"
        )
    return gate
