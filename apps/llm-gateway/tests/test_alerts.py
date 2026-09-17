"""The alert rules, checked as data.

`promtool check rules` validates the PromQL. These assert the properties that
make a rule set usable by a human on call, which promtool does not know about.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

RULES = Path(__file__).resolve().parents[3] / "deploy/helm/llm-gateway/files/alerts.yaml"
RUNBOOK = Path(__file__).resolve().parents[3] / "deploy/RUNBOOK.md"

pytestmark = pytest.mark.skipif(not RULES.exists(), reason="rules not present")

GROUP = yaml.safe_load(RULES.read_text())["groups"][0]["rules"]
ALERTS = [r for r in GROUP if "alert" in r]
RECORDS = [r for r in GROUP if "record" in r]


def test_there_are_alerts_and_recording_rules():
    assert ALERTS and RECORDS


def test_every_alert_names_a_severity():
    assert all(r["labels"].get("severity") for r in ALERTS)


def test_severities_are_page_or_ticket_only():
    """A third level is a level nobody agrees on the meaning of."""
    assert {r["labels"]["severity"] for r in ALERTS} <= {"page", "ticket"}


def test_most_alerts_do_not_page():
    """A rotation woken by everything learns to ignore the pager."""
    paging = [r for r in ALERTS if r["labels"]["severity"] == "page"]
    assert len(paging) <= len(ALERTS) / 2


def test_every_alert_points_at_a_runbook():
    missing = [r["alert"] for r in ALERTS if "runbook" not in r.get("annotations", {})]
    assert missing == []


def test_every_runbook_anchor_exists():
    """A link to a section that does not exist is worse than no link."""
    text = RUNBOOK.read_text().lower()
    broken = []
    for rule in ALERTS:
        anchor = rule["annotations"]["runbook"].split("#", 1)[-1]
        if f"## {anchor}" not in text:
            broken.append(f"{rule['alert']} -> #{anchor}")
    assert broken == []


def test_every_alert_waits_before_firing():
    """Without `for`, a single scrape blip pages someone."""
    assert all(r.get("for") for r in ALERTS)


def test_every_alert_has_a_summary():
    assert all(r.get("annotations", {}).get("summary") for r in ALERTS)


def test_burn_rate_alerts_use_two_windows():
    """A single long window keeps paging after the incident has ended."""
    burn = [r for r in ALERTS if "BudgetBurning" in r["alert"]]
    assert burn
    for rule in burn:
        assert rule["expr"].count("and") >= 1


def test_the_availability_sli_is_recorded_not_repeated_inline():
    """Two alerts computing the same ratio separately eventually disagree."""
    names = {r["record"] for r in RECORDS}
    assert any("availability" in n for n in names)
    for rule in ALERTS:
        if "BudgetBurning" in rule["alert"]:
            assert "gateway:availability" in rule["expr"]


def test_ratios_cannot_divide_by_zero():
    """An idle gateway must not produce NaN and fire every alert at once."""
    for rule in RECORDS:
        if "/" in rule["expr"]:
            assert "clamp_min" in rule["expr"], rule["record"]


def test_a_provider_breaker_opening_does_not_page():
    """Failover means one provider failing may have no user-visible effect."""
    for rule in ALERTS:
        if rule["labels"]["severity"] == "page":
            assert "breaker" not in rule["expr"].lower()
