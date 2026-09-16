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


def test_probes_pair_one_positive_with_every_damage():
    probes = build_probes(GOOD)
    assert sum(p.should_pass for p in probes) == 1
    assert {p.damage for p in probes if not p.should_pass} == set(DAMAGES)


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
    probes = build_probes(GOOD)
    expected = [p.should_pass for p in probes]
    # Each probe is asked twice for the consistency check.
    judge = ScriptedJudge([v for e in expected for v in (e, e)])
    report = evaluate_judge(judge, CRITERION, probes)
    assert report.accuracy == 1.0
    assert report.consistent


def test_a_judge_that_passes_everything_is_caught():
    """It would look flawless against a suite of only good outputs."""
    probes = build_probes(GOOD)
    report = evaluate_judge(ScriptedJudge([True]), CRITERION, probes)
    assert report.accuracy < 0.5
    assert len(report.mistakes) == len(DAMAGES)


def test_a_judge_that_fails_everything_is_caught():
    probes = build_probes(GOOD)
    report = evaluate_judge(ScriptedJudge([False]), CRITERION, probes)
    assert "should pass" in report.mistakes[0]


def test_an_inconsistent_judge_is_flagged():
    """A verdict that changes between identical runs is not a measurement."""
    report = evaluate_judge(ScriptedJudge([True, False]), CRITERION, build_probes(GOOD))
    assert not report.consistent


def test_the_report_names_the_judge_that_produced_it():
    report = evaluate_judge(ScriptedJudge([True]), CRITERION, build_probes(GOOD))
    assert report.judge_version == "fake/v1"
    assert "fake/v1" in report.explain()


def test_the_mistake_list_names_the_damage_that_fooled_it():
    report = evaluate_judge(ScriptedJudge([True]), CRITERION, build_probes(GOOD))
    assert any("no_causal_link" in m for m in report.mistakes)
