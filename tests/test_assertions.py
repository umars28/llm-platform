from __future__ import annotations

import pytest

from llm_eval.assertions import (
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

TEXT = "The connection pool saturated at 20/20 after v2.31.0 added an audit write."


def test_alternatives_within_a_group_mean_wording_does_not_decide_the_verdict():
    assert contains_all(TEXT, [["pool"], ["saturat", "exhaust"], ["v2.31.0"]]).passed
    assert contains_all(TEXT, [["exhausted", "saturated"]]).passed


def test_a_missing_concept_names_what_was_wanted():
    check = contains_all(TEXT, [["pool"], ["postgres cpu"]])
    assert check.failed
    assert "postgres cpu" in check.reason


def test_the_reason_reports_how_many_concepts_were_missing():
    check = contains_all(TEXT, [["absent one"], ["absent two"], ["pool"]])
    assert "missing 2 of 3" in check.reason


def test_matching_is_case_and_whitespace_insensitive():
    assert contains_all("Connection   Pool\nSaturated", [["connection pool"]]).passed


def test_forbidden_claims_are_reported_by_name():
    check = contains_none(TEXT + " Postgres CPU was the cause.", ["postgres cpu"])
    assert check.failed and "postgres cpu" in check.reason


def test_clean_text_passes_the_forbidden_check():
    assert contains_none(TEXT, ["postgres cpu", "disk full"]).passed


# -- structural checks -------------------------------------------------

def test_equals_reports_both_sides():
    check = equals("rollback_deploy", "restart_pods")
    assert check.failed
    assert "rollback_deploy" in check.reason and "restart_pods" in check.reason


def test_one_of_accepts_any_allowed_value():
    assert one_of("rollback_deploy", ["rollback_deploy", "restart_pods"]).passed
    assert one_of("failover_database", ["rollback_deploy"]).failed


def test_json_validity_is_checked_without_raising():
    assert is_valid_json('{"a": 1}').passed
    bad = is_valid_json("{not json")
    assert bad.failed and "not valid JSON" in bad.reason


def test_schema_check_lists_every_missing_key():
    check = matches_schema({"a": 1}, ["a", "b", "c"])
    assert check.failed
    assert "b" in check.reason and "c" in check.reason


def test_schema_check_rejects_non_objects():
    assert matches_schema(["a"], ["a"]).failed


# -- behavioural checks ------------------------------------------------

def test_required_tool_calls_are_checked():
    assert called_tools(["query_logs", "get_metrics"], ["query_logs"]).passed
    missed = called_tools(["query_logs"], ["search_runbook"])
    assert missed.failed and "search_runbook" in missed.reason


def test_forbidden_tool_calls_are_checked():
    assert did_not_call(["query_logs"], ["apply_remediation"]).passed
    assert did_not_call(["apply_remediation"], ["apply_remediation"]).failed


# -- budgets -----------------------------------------------------------

def test_cost_within_budget_passes():
    assert under_budget(0.18, 0.25, "$").passed


def test_exceeding_a_budget_reports_by_how_much():
    check = under_budget(0.50, 0.25, "$")
    assert check.failed and "100%" in check.reason


def test_a_budget_met_exactly_passes():
    assert under_budget(0.25, 0.25, "$").passed


# -- results -----------------------------------------------------------

def test_a_result_passes_only_when_every_check_does():
    result = Result("C-1")
    result.add(Check("a", True, "ok"))
    assert result.passed
    result.add(Check("b", False, "nope"))
    assert not result.passed


def test_failures_are_listed_with_their_reasons():
    result = Result("C-1")
    result.add(Check("a", True, "ok"))
    result.add(Check("b", False, "wanted x, got y"))
    explanation = result.explain()
    assert "C-1: FAIL (1/2 checks)" in explanation
    assert "wanted x, got y" in explanation


def test_score_is_weighted_so_drift_is_visible_before_failure():
    result = Result("C-1")
    result.add(Check("cheap", True, "ok", weight=1.0))
    result.add(Check("important", False, "no", weight=3.0))
    assert result.score == pytest.approx(0.25)


def test_a_result_with_no_checks_scores_zero_rather_than_dividing_by_zero():
    assert Result("C-1").score == 0.0


def test_deterministic_only_detects_a_judge_in_the_mix():
    result = Result("C-1")
    result.add(Check("a", True, "ok"))
    assert result.deterministic_only()
    result.add(Check("b", True, "ok", kind="judge"))
    assert not result.deterministic_only()
