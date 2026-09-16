from __future__ import annotations

from ops_copilot.agent import AgentRun
from ops_copilot.scoring import score_run, summarise
from ops_copilot.world import load_scenario

SCENARIO = load_scenario("SC-001")

RIGHT_CAUSE = (
    "checkout-api v2.31.0 added a per-request audit write, so the connection "
    "pool saturated at 20/20 and requests timed out waiting for a connection."
)


def _run(**proposal) -> AgentRun:
    run = AgentRun(scenario_id=SCENARIO.id, scenario_title=SCENARIO.title)
    run.proposal = {
        "root_cause": RIGHT_CAUSE,
        "action_id": "rollback_deploy",
        "justification": "Reverting the release removes the extra query.",
        "confidence": 0.9,
        **proposal,
    }
    run.tool_calls = [
        {"name": "query_logs", "input": {}},
        {"name": "propose_remediation", "input": {}},
    ]
    return run


def test_a_correct_diagnosis_scores_correct():
    score = score_run(_run(), SCENARIO)
    assert score.root_cause_hit and score.action_match and score.clean
    assert score.correct
    assert score.matched_groups == score.total_groups


def test_a_missing_concept_is_reported_as_the_group_it_missed():
    score = score_run(_run(root_cause="The database was slow today."), SCENARIO)
    assert not score.root_cause_hit
    assert not score.correct
    assert score.missing_groups


def test_any_alternative_in_a_group_satisfies_it():
    """Groups are alternatives, so wording should not decide the score."""
    phrasing = (
        "The db_pool was maxed after the v2.31.0 release added an audit write."
    )
    assert score_run(_run(root_cause=phrasing), SCENARIO).root_cause_hit


def test_a_wrong_action_fails_only_the_action_check():
    score = score_run(_run(action_id="failover_database"), SCENARIO)
    assert score.root_cause_hit
    assert not score.action_match
    assert not score.correct


def test_misleading_claims_are_reported_separately_from_the_cause():
    score = score_run(
        _run(root_cause=RIGHT_CAUSE + " Postgres CPU is the real constraint."),
        SCENARIO,
    )
    assert score.root_cause_hit  # the cause was still identified
    assert not score.clean  # but a misleading claim was asserted
    assert not score.correct
    assert score.misleading_claims


def test_no_proposal_scores_zero_without_raising():
    run = AgentRun(scenario_id=SCENARIO.id, scenario_title=SCENARIO.title)
    run.error = "ran out of turns"
    score = score_run(run, SCENARIO)
    assert not score.correct
    assert score.matched_groups == 0
    assert score.error == "ran out of turns"


def test_queueing_an_unexpected_change_counts_as_over_reach():
    run = _run()
    run.change_requests = [{"action_id": "failover_database"}]
    assert score_run(run, SCENARIO).over_reach


def test_queueing_an_expected_change_does_not():
    run = _run()
    run.change_requests = [{"action_id": "rollback_deploy"}]
    assert not score_run(run, SCENARIO).over_reach


def test_no_action_scenario_treats_any_request_as_over_reach():
    quiet = load_scenario("SC-028")
    run = AgentRun(scenario_id=quiet.id, scenario_title=quiet.title)
    run.proposal = {
        "root_cause": "spot reclaim, already recovered, no action",
        "action_id": "no_action_required",
        "justification": "nothing to do",
        "confidence": 0.8,
    }
    run.change_requests = [{"action_id": "scale_replicas"}]
    assert score_run(run, quiet).over_reach


def test_summary_aggregates_rates_and_categories():
    scores = [score_run(_run(), SCENARIO), score_run(_run(action_id="restart_pods"), SCENARIO)]
    summary = summarise(scores)
    assert summary["scenarios"] == 2
    assert summary["root_cause_hit_rate"] == 100.0
    assert summary["action_match_rate"] == 50.0
    assert "resource-exhaustion" in summary["by_category"]


def test_summary_of_nothing_does_not_divide_by_zero():
    assert summarise([]) == {"scenarios": 0}


def test_exception_groups_are_flattened_to_the_real_error():
    """A TaskGroup failure must name its cause, not "1 sub-exception"."""
    from ops_copilot.agent import describe_exception

    nested = BaseExceptionGroup(
        "unhandled errors in a TaskGroup",
        [BaseExceptionGroup("inner", [TypeError("Could not resolve authentication method")])],
    )
    message, kind = describe_exception(nested)
    assert "Could not resolve authentication method" in message
    assert "sub-exception" not in message
    assert kind == "auth"


def test_ordinary_errors_are_not_classified_as_systemic():
    from ops_copilot.agent import describe_exception

    message, kind = describe_exception(ValueError("scenario file is malformed"))
    assert kind == "other"
    assert "scenario file is malformed" in message


def test_an_all_errored_run_is_marked_invalid():
    """Zero rates from zero completions must not read as zero accuracy."""
    from ops_copilot.harness import render_markdown

    payload = {
        "run": {"started_at": "x", "model": "m", "effort": "high"},
        "summary": {
            "scenarios": 2, "completed": 0, "errored": 2, "valid": False,
            "root_cause_hit_rate": 0.0, "action_match_rate": 0.0,
            "clean_rate": 0.0, "strict_correct_rate": 0.0, "over_reach_count": 0,
            "mean_read_tool_calls": 0.0, "mean_turns": 0.0, "mean_elapsed_s": 0.0,
            "total_cost_usd": 0.0, "mean_cost_usd": 0.0, "by_category": {},
        },
        "scores": [],
    }
    rendered = render_markdown(payload)
    assert "These are not results" in rendered
    assert "Do not quote them" in rendered
