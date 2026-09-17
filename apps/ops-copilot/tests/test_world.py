from __future__ import annotations

import pytest

from ops_copilot.world import World, load_scenario

SCENARIO = load_scenario("SC-001")


@pytest.fixture
def world() -> World:
    return World(SCENARIO)


def test_relative_offsets_resolve_against_the_alert(world):
    assert world.at(0) == "2026-09-12T02:14:00Z"
    assert world.at(-60) == "2026-09-12T01:14:00Z"


def test_since_minutes_is_a_lookback_regardless_of_sign(world):
    """`since_minutes=30` and `-30` both mean "the last thirty minutes"."""
    assert world.logs("checkout-api", since_minutes=30) == world.logs(
        "checkout-api", since_minutes=-30
    )


def test_log_filtering_by_level_and_pattern(world):
    errors = world.logs("checkout-api", level="ERROR", since_minutes=60)
    assert errors and all(e["level"] == "ERROR" for e in errors)

    pooled = world.logs("checkout-api", pattern="pool", since_minutes=60)
    assert pooled and all("pool" in e["message"].lower() for e in pooled)


def test_limit_keeps_the_most_recent_entries(world):
    recent = world.logs("checkout-api", since_minutes=60, limit=2)
    everything = world.logs("checkout-api", since_minutes=60)
    assert recent == everything[-2:]


def test_unknown_service_raises_so_the_tool_can_offer_alternatives(world):
    with pytest.raises(KeyError):
        world.logs("does-not-exist")
    with pytest.raises(KeyError):
        world.metrics("does-not-exist")


def test_metric_summary_is_computed_over_the_window(world):
    series = world.metrics("checkout-api", "db_pool_in_use", since_minutes=60)
    summary = series["db_pool_in_use"]["summary"]
    assert summary["max"] == 20.0
    assert summary["latest"] == 20.0


def test_deploys_are_ordered_oldest_first_with_offsets(world):
    deploys = world.deploys(since_minutes=120)
    assert [d["service"] for d in deploys] == ["inventory-api", "checkout-api"]
    assert deploys[-1]["minutes_before_alert"] == 38.0


def test_runbook_search_spans_shared_and_scenario_books(world):
    ids = {r["id"] for r in world.search_runbooks("connection pool timeout", limit=5)}
    assert "RB-014" in ids  # scenario-specific
    assert any(i.startswith("RB-00") for i in ids)  # shared catalogue


def test_actions_merge_shared_catalogue(world):
    ids = {a["id"] for a in world.actions()}
    assert {"rollback_deploy", "no_action_required", "increase_db_pool_size"} <= ids


def test_k8s_lookup_resolves_event_timestamps(world):
    resource = world.k8s("Deployment", "checkout-api", "prod")
    assert resource["status"]["readyReplicas"] == 6
    assert resource["events"][0]["timestamp"].endswith("Z")

    with pytest.raises(KeyError):
        world.k8s("Deployment", "nope", "prod")
