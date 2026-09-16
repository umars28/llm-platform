"""The regression gate. These decide whether a build fails, so they matter most."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval.assertions import Check, Result
from llm_eval.gate import Baseline, check_gate

JUDGE = "claude-opus-5/v1"


def result(case_id: str, passed: bool, score: float | None = None) -> Result:
    r = Result(case_id)
    r.add(Check("c", passed, "reason"))
    if score is not None and score != (1.0 if passed else 0.0):
        r.checks.clear()
        r.add(Check("a", True, "ok", weight=score))
        r.add(Check("b", False, "no", weight=1.0 - score))
    return r


def baseline(cases: dict[str, bool], cost: float = 1.0, judge: str = JUDGE) -> Baseline:
    return Baseline(
        judge_version=judge,
        cases=cases,
        scores={k: (1.0 if v else 0.0) for k, v in cases.items()},
        cost_usd=cost,
    )


# -- the four ways to fail ---------------------------------------------

def test_a_case_that_regressed_fails_the_gate():
    gate = check_gate([result("C-1", False)], baseline({"C-1": True}), JUDGE)
    assert not gate.passed
    assert gate.regressions == ["C-1"]


def test_a_single_regression_is_never_averaged_away():
    """Ninety-nine passes do not excuse one named case breaking."""
    results = [result(f"C-{i}", True) for i in range(99)] + [result("C-99", False)]
    base = baseline({f"C-{i}": True for i in range(100)})
    assert not check_gate(results, base, JUDGE).passed


def test_a_broad_score_drop_fails_even_with_no_case_flipping():
    results = [result("C-1", True, score=0.90)]
    base = Baseline(judge_version=JUDGE, cases={"C-1": True}, scores={"C-1": 1.0})
    gate = check_gate(results, base, JUDGE, score_tolerance=0.02)
    assert not gate.passed
    assert "mean score fell" in gate.reasons[0]


def test_a_cost_rise_beyond_tolerance_fails():
    gate = check_gate(
        [result("C-1", True)], baseline({"C-1": True}, cost=1.0), JUDGE,
        cost_usd=1.5, cost_tolerance=0.15,
    )
    assert not gate.passed
    assert "cost rose 50%" in gate.reasons[0]


def test_a_changed_judge_is_reported_as_stale_not_as_a_regression():
    """Comparing to numbers a different grader produced is meaningless."""
    gate = check_gate([result("C-1", False)], baseline({"C-1": True}), "claude-opus-5/v2")
    assert not gate.passed
    assert gate.stale_baseline
    assert "Re-baseline" in gate.reasons[0]
    assert gate.regressions == []  # not attributed to quality


# -- what must not fail the gate ---------------------------------------

def test_an_unchanged_run_passes():
    gate = check_gate([result("C-1", True)], baseline({"C-1": True}), JUDGE, cost_usd=1.0)
    assert gate.passed


def test_an_improvement_passes_and_is_reported():
    gate = check_gate([result("C-1", True)], baseline({"C-1": False}), JUDGE)
    assert gate.passed
    assert gate.improvements == ["C-1"]


def test_a_cost_fall_never_fails():
    gate = check_gate(
        [result("C-1", True)], baseline({"C-1": True}, cost=1.0), JUDGE, cost_usd=0.2
    )
    assert gate.passed


def test_a_score_drop_inside_tolerance_passes():
    results = [result("C-1", True, score=0.99)]
    base = Baseline(judge_version=JUDGE, cases={"C-1": True}, scores={"C-1": 1.0})
    assert check_gate(results, base, JUDGE, score_tolerance=0.02).passed


def test_a_new_case_is_reported_but_does_not_fail():
    gate = check_gate(
        [result("C-1", True), result("C-2", True)], baseline({"C-1": True}), JUDGE
    )
    assert gate.passed
    assert gate.new_cases == ["C-2"]


def test_a_baseline_case_that_stopped_running_fails():
    """A suite that quietly shrinks stops catching things."""
    gate = check_gate([result("C-1", True)], baseline({"C-1": True, "C-2": True}), JUDGE)
    assert not gate.passed
    assert gate.removed_cases == ["C-2"]


# -- baseline round trip ------------------------------------------------

def test_a_baseline_round_trips_through_disk(tmp_path: Path):
    original = Baseline.from_results(
        [result("C-1", True), result("C-2", False)],
        judge_version=JUDGE, cost_usd=0.42, commit="abc123",
    )
    path = original.save(tmp_path / "baseline.json")
    loaded = Baseline.load(path)
    assert loaded.cases == original.cases
    assert loaded.judge_version == JUDGE
    assert loaded.cost_usd == pytest.approx(0.42)
    assert loaded.commit == "abc123"


def test_a_baseline_records_which_judge_graded_it(tmp_path: Path):
    path = Baseline.from_results([result("C-1", True)], JUDGE).save(tmp_path / "b.json")
    assert '"judge_version": "claude-opus-5/v1"' in path.read_text()


def test_an_empty_baseline_does_not_divide_by_zero():
    assert Baseline(judge_version=JUDGE).pass_rate == 0.0
    assert Baseline(judge_version=JUDGE).mean_score == 0.0


def test_the_explanation_names_the_failing_cases():
    gate = check_gate([result("C-1", False)], baseline({"C-1": True}), JUDGE)
    assert "GATE FAILED" in gate.explain()
    assert "regressed: C-1" in gate.explain()
