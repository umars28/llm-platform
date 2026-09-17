from __future__ import annotations

import pytest

from llm_eval.agreement import (
    DAMAGES,
    Probe,
    build_probes,
    evaluate_judge,
    strip_causal_link,
    strip_evidence,
)
from llm_eval.judge import Criterion, Verdict

GOOD = ("checkout-api v2.31.0 added a per-request audit write, so the connection "
        "pool saturated at 20/20 and requests timed out after 5000ms")


def test_stripping_the_link_keeps_the_facts_and_removes_the_conclusion():
    """The hardest negative: nothing false, the connection simply not drawn."""
    damaged = strip_causal_link(GOOD)
    assert "v2.31.0" in damaged and "20/20" in damaged
    assert " so " not in damaged


def test_stripping_evidence_leaves_an_unsupported_assertion():
    damaged = strip_evidence(GOOD)
    assert "20/20" not in damaged
    assert "v2.31.0" not in damaged
    assert "connection" in damaged


def test_a_damage_is_only_used_where_it_actually_breaks_the_criterion():
    """Applying every damage to every criterion scored a correct judge 1/4."""
    chain = build_probes(GOOD, "joins_cause_to_symptom")
    evidence = build_probes(GOOD, "evidence_is_named")

    assert "no_causal_link" in {p.damage for p in chain}
    # A confidently wrong answer still states a causal chain, so it is not a
    # negative for this criterion.
    assert "wrong_cause" not in {p.damage for p in chain}
    # Stripping numbers is what breaks an evidence criterion, not a chain one.
    assert "no_evidence" in {p.damage for p in evidence}
    assert "no_evidence" not in {p.damage for p in chain}


def test_every_probe_set_has_exactly_one_positive():
    for name in ("joins_cause_to_symptom", "evidence_is_named", "unknown_criterion"):
        assert sum(p.should_pass for p in build_probes(GOOD, name)) == 1


def test_an_untested_criterion_is_reported_rather_than_padded():
    """One probe means the judge was barely tested, which is information."""
    assert len(build_probes(GOOD, "criterion_no_damage_breaks")) == 1


CRITERION = Criterion(name="joins_cause_to_symptom", question="Does it join cause to symptom?")


class ScriptedJudge:
    """Returns verdicts from a script, so judge behaviour can be simulated."""

    def __init__(self, verdicts, version="fake/v1"):
        self.verdicts = list(verdicts)
        self.version = version
        self.calls = 0

    def judge(self, criterion, subject, use_cache=True):
        assert not use_cache, "the judge evaluation must not read the cache"
        self.calls += 1
        passed = self.verdicts[(self.calls - 1) % len(self.verdicts)]
        return Verdict(passed=passed, reason="scripted", confidence=0.9)


def test_a_perfect_judge_scores_full_marks():
    probes = build_probes(GOOD, "joins_cause_to_symptom")
    expected = [p.should_pass for p in probes]
    # Each probe is asked twice for the consistency check.
    judge = ScriptedJudge([v for e in expected for v in (e, e)])
    report = evaluate_judge(judge, CRITERION, probes)
    assert report.accuracy == 1.0
    assert report.consistent


def test_a_judge_that_passes_everything_is_caught():
    """It would look flawless against a suite of only good outputs."""
    probes = build_probes(GOOD, "joins_cause_to_symptom")
    report = evaluate_judge(ScriptedJudge([True]), CRITERION, probes)
    negatives = [p for p in probes if not p.should_pass]
    assert negatives, "this criterion must have at least one negative to be a real test"
    assert len(report.mistakes) == len(negatives)
    assert report.accuracy < 1.0


def test_a_judge_that_fails_everything_is_caught():
    probes = build_probes(GOOD, "joins_cause_to_symptom")
    report = evaluate_judge(ScriptedJudge([False]), CRITERION, probes)
    assert "should pass" in report.mistakes[0]


def test_an_inconsistent_judge_is_flagged():
    """A verdict that changes between identical runs is not a measurement."""
    report = evaluate_judge(ScriptedJudge([True, False]), CRITERION, build_probes(GOOD, "joins_cause_to_symptom"))
    assert not report.consistent


def test_the_report_names_the_judge_that_produced_it():
    report = evaluate_judge(ScriptedJudge([True]), CRITERION, build_probes(GOOD, "joins_cause_to_symptom"))
    assert report.judge_version == "fake/v1"
    assert "fake/v1" in report.explain()


def test_the_mistake_list_names_the_damage_that_fooled_it():
    report = evaluate_judge(ScriptedJudge([True]), CRITERION, build_probes(GOOD, "joins_cause_to_symptom"))
    assert any("no_causal_link" in m for m in report.mistakes)
