"""Deployment configuration, asserted as data.

These catch the class of mistake that only shows up in a cluster: a comment that
claims a property the configuration does not have.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
TF = ROOT / "deploy/terraform"
VALUES = ROOT / "deploy/helm/llm-gateway/values.yaml"
DEPLOYMENT = ROOT / "deploy/helm/llm-gateway/templates/deployment.yaml"

pytestmark = pytest.mark.skipif(not TF.exists(), reason="deploy tree not present")


def tf_text() -> str:
    """Terraform source with comments stripped.

    An earlier version of this helper grepped the raw file and failed on the
    comment explaining why a data source is not used. A test that cannot tell
    code from prose is testing the prose.
    """
    lines = []
    for path in sorted(TF.glob("*.tf")):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                lines.append(line)
    return "\n".join(lines)


# -- secrets -----------------------------------------------------------

def test_terraform_never_reads_the_provider_secret():
    """A data source persists what it reads: checking the secret existed put it
    in state in plain text and base64. Verified by grepping state, not assumed."""
    assert 'data "kubernetes_secret"' not in tf_text()


def test_terraform_does_not_manage_the_secret_either():
    assert 'resource "kubernetes_secret"' not in tf_text()


def test_the_secret_is_referenced_by_name_only():
    assert "var.provider_secret_name" in tf_text()


def test_no_state_file_is_tracked_by_git():
    tracked = (ROOT / ".gitignore").read_text()
    assert "*.tfstate*" in tracked or "terraform.tfstate" in tracked


# -- guards that refuse a dangerous configuration ----------------------

def test_a_mutable_image_tag_is_refused():
    assert "Refusing 'latest'" in tf_text()


def test_a_single_replica_is_refused():
    assert re.search(r"var\.replicas\s*>=\s*2", tf_text())


def test_the_release_is_atomic_so_a_broken_deploy_rolls_back():
    """This caught a missing dependency in the image before it replaced a
    working release."""
    assert re.search(r"atomic\s*=\s*true", tf_text())
    assert re.search(r"wait\s*=\s*true", tf_text())


# -- shutdown ----------------------------------------------------------

def test_a_prestop_hook_exists():
    """Without it a rolling restart under load dropped 7 of 535 requests."""
    assert "preStop" in DEPLOYMENT.read_text()


def test_the_grace_period_is_derived_from_the_drain_and_grace_windows():
    """A hardcoded value silently shortens below the drain window and the
    graceful shutdown becomes decorative."""
    text = DEPLOYMENT.read_text()
    assert "terminationGracePeriodSeconds" in text
    assert "add .Values.shutdown.drainSeconds .Values.shutdown.graceSeconds" in text


# -- resources ---------------------------------------------------------

def test_memory_request_equals_its_limit():
    values = yaml.safe_load(VALUES.read_text())
    assert values["resources"]["requests"]["memory"] == values["resources"]["limits"]["memory"]


def test_there_is_deliberately_no_cpu_limit():
    """CPU is compressible; a limit throttles a latency-sensitive proxy without
    protecting the node the way a memory limit does."""
    assert "cpu" not in yaml.safe_load(VALUES.read_text())["resources"]["limits"]


def test_the_comment_matches_the_configuration():
    """An earlier version claimed Guaranteed QoS while setting unequal CPU
    values. kubectl said Burstable and the comment was wrong."""
    text = VALUES.read_text()
    assert "Guaranteed" not in text or "not Guaranteed" in text


# -- quota state -------------------------------------------------------

def test_more_than_one_replica_requires_a_shared_quota_store():
    guard = (ROOT / "deploy/helm/llm-gateway/templates/_guard.tpl").read_text()
    assert "fail" in guard
    assert "once per replica" in guard


# -- network policy ----------------------------------------------------

NETPOL = ROOT / "deploy/helm/llm-gateway/templates/networkpolicy.yaml"


def netpol_text() -> str:
    return NETPOL.read_text()


def test_both_pods_default_deny_in_and_out():
    """A policy listing only ingress leaves egress wide open, and vice versa."""
    text = netpol_text()
    assert text.count("policyTypes: [Ingress, Egress]") == 2


def test_redis_accepts_only_the_gateway():
    """Redis has no password here, so the selector is the authentication.

    The first version of this template rendered the gateway's labels directly
    under `podSelector` with no `matchLabels`, which is not the same policy --
    it is a selector Kubernetes cannot read on a rule that looks correct.
    """
    text = netpol_text()
    redis_block = text.split("-redis\n", 1)[1]
    ingress = redis_block.split("ingress:", 1)[1].split("egress:", 1)[0]
    assert "podSelector:" in ingress
    assert "matchLabels:" in ingress
    assert "selectorLabels" in ingress


def test_egress_to_the_cluster_is_not_reopened_by_the_internet_rule():
    """0.0.0.0/0 without the private ranges excluded is a hole, not a rule."""
    text = netpol_text()
    assert "cidr: 0.0.0.0/0" in text
    for private in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"):
        assert private in text, f"{private} not excluded from egress"


def test_dns_is_allowed_or_every_rule_below_it_is_dead():
    """Every other egress rule resolves a name first."""
    text = netpol_text()
    assert text.count("port: 53") == 4  # udp+tcp, for both pods


def test_the_policy_renders_when_its_values_key_is_missing():
    """`helm upgrade --reuse-values` replays values that predate this key.

    Treating the absent case as "off" would let that upgrade succeed and delete
    the policies in front of an unauthenticated Redis. The control fails on.
    """
    text = netpol_text()
    assert "ne (.Values.networkPolicy).enabled false" in text
    assert ".Values.networkPolicy.enabled }}" not in text
