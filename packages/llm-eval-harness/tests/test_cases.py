from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval.cases import Case, Subject, load_cases, run_case, run_suite
from llm_eval.judge import Criterion

GOOD_TRACE = {
    "scenario_id": "SC-001",
    "model": "claude-opus-5",
    "turns": 4,
    "proposal": {
        "root_cause": "checkout-api v2.31.0 added a per-request audit write, so the "
                      "connection pool saturated at 20/20 and requests timed out",
        "justification": "reverting removes the extra query",
        "action_id": "rollback_deploy",
    },
    "tool_calls": [{"name": "query_logs"}, {"name": "search_runbook"},
                   {"name": "propose_remediation"}],
    "cost_usd": 0.19,
    "elapsed_s": 42.0,
}


def test_a_trace_becomes_a_scorable_subject():
    subject = Subject.from_trace(GOOD_TRACE)
    assert "connection pool" in subject.text
    assert "search_runbook" in subject.tools_called
    assert subject.cost_usd == 0.19
    assert subject.metadata["scenario_id"] == "SC-001"


def test_a_partial_trace_still_scores_rather_than_raising():
    subject = Subject.from_trace({"final_text": "some answer"})
    assert subject.text == "some answer"
    assert subject.cost_usd == 0.0
    assert subject.tools_called == []


def test_a_trace_with_no_proposal_falls_back_to_final_text():
    subject = Subject.from_trace({"proposal": None, "final_text": "fallback"})
    assert subject.text == "fallback"


# -- running -----------------------------------------------------------

from llm_eval.cli import DEFAULT_CASES

CASES = load_cases(DEFAULT_CASES)
SC001 = next(c for c in CASES if c.id == "SC-001")


def test_a_good_trace_passes_every_deterministic_check():
    result = run_case(SC001, Subject.from_trace(GOOD_TRACE))
    assert result.passed, result.explain()
    assert result.deterministic_only()


def test_a_forbidden_claim_fails_the_case():
    trace = {**GOOD_TRACE, "proposal": {**GOOD_TRACE["proposal"],
             "justification": "postgres cpu was the real constraint"}}
    result = run_case(SC001, Subject.from_trace(trace))
    assert not result.passed
    assert any("avoids_the_red_herring" == c.name for c in result.failures)


def test_a_missing_tool_call_fails_the_case():
    trace = {**GOOD_TRACE, "tool_calls": [{"name": "query_logs"}]}
    assert not run_case(SC001, Subject.from_trace(trace)).passed


def test_calling_a_forbidden_tool_fails_the_case():
    trace = {**GOOD_TRACE,
             "tool_calls": GOOD_TRACE["tool_calls"] + [{"name": "apply_remediation"}]}
    result = run_case(SC001, Subject.from_trace(trace))
    assert not result.passed
    assert any(c.name == "respected_the_gate" for c in result.failures)


def test_exceeding_the_cost_budget_fails_the_case():
    result = run_case(SC001, Subject.from_trace({**GOOD_TRACE, "cost_usd": 1.20}))
    assert not result.passed
    assert any(c.name == "cost_budget" for c in result.failures)


def test_a_dotted_field_reaches_into_the_payload():
    trace = {**GOOD_TRACE,
             "proposal": {**GOOD_TRACE["proposal"], "action_id": "failover_database"}}
    result = run_case(SC001, Subject.from_trace(trace))
    assert any(c.name == "chose_a_sane_action" for c in result.failures)


def test_an_unknown_assertion_type_fails_loudly_rather_than_passing():
    case = Case(id="X", assertions=[{"type": "vibes", "name": "vibes"}])
    result = run_case(case, Subject(text="anything"))
    assert not result.passed
    assert "unknown assertion type" in result.failures[0].reason


# -- judge ordering ----------------------------------------------------

class CountingJudge:
    def __init__(self):
        self.calls = 0

    def check(self, criterion, subject):
        from llm_eval.assertions import Check
        self.calls += 1
        return Check(criterion.name, True, "ok", kind="judge")


def test_the_judge_is_not_paid_to_elaborate_on_a_decided_failure():
    judge = CountingJudge()
    trace = {**GOOD_TRACE, "tool_calls": [{"name": "query_logs"}]}  # already fails
    run_case(SC001, Subject.from_trace(trace), judge)
    assert judge.calls == 0


def test_the_judge_runs_when_the_deterministic_checks_all_held():
    judge = CountingJudge()
    result = run_case(SC001, Subject.from_trace(GOOD_TRACE), judge)
    assert judge.calls == len(SC001.criteria)
    assert not result.deterministic_only()


# -- suite -------------------------------------------------------------

def test_a_case_with_no_recorded_subject_fails_rather_than_being_skipped():
    """A suite that silently drops a case stops catching what that case caught."""
    results = run_suite(CASES, subjects={})
    assert len(results) == len(CASES)
    assert all(not r.passed for r in results)
    assert "no recorded subject" in results[0].failures[0].reason


def test_the_suite_scores_every_case_it_has_a_subject_for():
    results = run_suite([SC001], {"SC-001": Subject.from_trace(GOOD_TRACE)})
    assert results[0].passed


def test_case_definitions_parse_into_criteria():
    assert all(isinstance(c, Criterion) for c in SC001.criteria)
    assert SC001.criteria[0].question


def test_shared_flags_work_after_the_subcommand(monkeypatch):
    """`llm-eval gate --no-judge` is how the CI workflow invokes it."""
    from llm_eval.cli import main

    monkeypatch.setattr("sys.argv", ["llm-eval", "gate", "--no-judge", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0  # --help exits cleanly, so parsing succeeded


def test_shared_flags_still_work_before_the_subcommand(monkeypatch):
    from llm_eval.cli import main

    monkeypatch.setattr("sys.argv", ["llm-eval", "--no-judge", "run", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
