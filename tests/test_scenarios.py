"""Structural checks on the scenario corpus.

These run without an API key and guard the properties the harness assumes:
every scenario is well formed, its expected actions exist, and the tools can
actually reach the evidence its ground truth relies on.
"""

from __future__ import annotations

import pytest

from ops_copilot.world import World, all_scenarios, common_catalogue

SCENARIOS = all_scenarios()
REQUIRED_TRUTH_KEYS = ["root_cause", "must_include", "expected_actions"]


def test_corpus_is_not_empty():
    assert len(SCENARIOS) >= 30


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_ground_truth_is_complete(scenario):
    for key in REQUIRED_TRUTH_KEYS:
        assert key in scenario.ground_truth, f"{scenario.id} missing {key}"
    assert scenario.ground_truth["must_include"], f"{scenario.id} has no must_include groups"
    assert all(
        isinstance(group, list) and group
        for group in scenario.ground_truth["must_include"]
    ), f"{scenario.id} must_include groups must be non-empty lists of alternatives"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_expected_actions_exist_in_the_catalogue(scenario):
    available = {a["id"] for a in World(scenario).actions()}
    for action in scenario.ground_truth["expected_actions"]:
        assert action in available, f"{scenario.id} expects unknown action {action}"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_every_alerting_service_is_queryable(scenario):
    """The agent's first move is always to look at the alerting service."""
    world = World(scenario)
    service = scenario.alert["service"]
    assert service in world.service_names() or service in scenario.world.get("metrics", {}), (
        f"{scenario.id} alerts on {service}, which is in neither the service "
        "inventory nor the metrics"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_timestamps_resolve(scenario):
    """Relative offsets must resolve without raising, for every entry."""
    world = World(scenario)
    for service in scenario.world.get("logs", {}):
        world.logs(service, since_minutes=100000)
    for service in scenario.world.get("metrics", {}):
        world.metrics(service, since_minutes=100000)
    world.deploys(since_minutes=100000)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_scenario_ids_match_filenames(scenario):
    assert scenario.path.stem == scenario.id


def test_scenario_ids_are_unique():
    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_no_action_required_is_a_real_expected_answer_somewhere():
    """A corpus of only-real-incidents cannot catch an agent that always acts."""
    assert any(
        "no_action_required" in s.ground_truth["expected_actions"] for s in SCENARIOS
    )


def test_catalogue_action_ids_are_unique():
    ids = [a["id"] for a in common_catalogue()["actions"]]
    assert len(ids) == len(set(ids))
