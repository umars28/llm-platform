"""MCP server exposing one incident's observability surface as tools.

The server is started per-run with `OPS_COPILOT_SCENARIO` naming the scenario to
serve, so the agent talks to a fixed, replayable world.

Two things are deliberate here:

* Read tools return *shaped* errors -- when a service or metric name is wrong the
  tool answers with the valid options instead of a stack trace. An agent that can
  self-correct in one turn spends far fewer turns than one that cannot.
* `request_remediation` never executes anything. It writes a pending change
  request to the audit log and returns its id. Execution requires a human running
  `ops-copilot approve <id>`. That split is the whole point of the project: the
  model is trusted to diagnose, never to act on production unattended.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .world import World, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCENARIO_REF = os.environ.get("OPS_COPILOT_SCENARIO", "SC-001")
_AUDIT_DIR = Path(os.environ.get("OPS_COPILOT_AUDIT_DIR", REPO_ROOT / "audit"))

WORLD = World(load_scenario(_SCENARIO_REF))

server = MCPServer(
    name="ops-copilot",
    version="0.1.0",
    instructions=(
        "Observability tools for a single production incident. Read tools are "
        "free to call. propose_remediation records your conclusion. "
        "request_remediation queues a change for human approval and never "
        "applies anything by itself."
    ),
)


def _err(message: str, **hints: Any) -> dict[str, Any]:
    return {"error": message, **hints}


# --------------------------------------------------------------------------
# Read tools
# --------------------------------------------------------------------------


@server.tool()
def list_services() -> dict[str, Any]:
    """List every service in the environment with its tier, owner and dependencies.

    Start here when you do not yet know what the affected service talks to.
    """
    return {"services": WORLD.services()}


@server.tool()
def get_recent_deploys(service: str | None = None, since_minutes: float = 180) -> dict[str, Any]:
    """List releases shipped in a time window, newest last.

    Args:
        service: Restrict to one service. Omit to see every service's releases.
        since_minutes: How far back to look from the alert. Defaults to 3 hours.
    """
    if service and service not in WORLD.service_names():
        return _err(f"unknown service {service!r}", known_services=WORLD.service_names())
    return {"deploys": WORLD.deploys(service, since_minutes)}


@server.tool()
def query_logs(
    service: str,
    pattern: str | None = None,
    level: str | None = None,
    since_minutes: float = 60,
    limit: int = 50,
) -> dict[str, Any]:
    """Search a service's logs.

    Repeated identical lines are collapsed and carry a `count` field, so a single
    returned entry may represent hundreds of occurrences -- read `count` before
    judging how severe something is.

    Args:
        service: Service whose logs to search.
        pattern: Case-insensitive regular expression matched against the message.
        level: Filter to one of INFO, WARN, ERROR.
        since_minutes: How far back to look from the alert.
        limit: Maximum entries to return, keeping the most recent.
    """
    try:
        entries = WORLD.logs(service, pattern, level, since_minutes, limit)
    except KeyError:
        return _err(f"no logs for service {service!r}", known_services=WORLD.service_names())
    return {"service": service, "matches": len(entries), "entries": entries}


@server.tool()
def list_metrics(service: str) -> dict[str, Any]:
    """List the metric names recorded for a service."""
    try:
        return {"service": service, "metrics": WORLD.metric_names(service)}
    except KeyError:
        return _err(f"no metrics for service {service!r}", known_services=WORLD.service_names())


@server.tool()
def get_metrics(
    service: str, metric: str | None = None, since_minutes: float = 60
) -> dict[str, Any]:
    """Fetch metric time series for a service, with min/max/latest already computed.

    Args:
        service: Service to read metrics for.
        metric: One metric name. Omit to fetch every metric for the service.
        since_minutes: How far back to look from the alert.
    """
    try:
        return {"service": service, "series": WORLD.metrics(service, metric, since_minutes)}
    except KeyError as exc:
        if str(exc).strip("'") == service:
            return _err(f"no metrics for service {service!r}", known_services=WORLD.service_names())
        return _err(
            f"unknown metric {metric!r} for {service!r}",
            known_metrics=WORLD.metric_names(service),
        )


@server.tool()
def describe_k8s_resource(kind: str, name: str, namespace: str = "prod") -> dict[str, Any]:
    """Describe a Kubernetes resource: spec, status and recent events.

    Args:
        kind: Resource kind, e.g. Deployment, StatefulSet, HorizontalPodAutoscaler.
        name: Resource name.
        namespace: Namespace. Defaults to prod.
    """
    try:
        return WORLD.k8s(kind, name, namespace)
    except KeyError:
        return _err(
            f"no {kind}/{name} in namespace {namespace}",
            available_resources=WORLD.k8s_index(),
        )


@server.tool()
def search_runbook(query: str, limit: int = 3) -> dict[str, Any]:
    """Search the operational runbooks by keyword.

    Runbooks carry the operating limits you cannot infer from telemetry alone --
    connection budgets, which actions are pre-approved, when rollback is expected.

    Args:
        query: Free-text description of the symptom.
        limit: Maximum runbooks to return.
    """
    return {"query": query, "results": WORLD.search_runbooks(query, limit)}


@server.tool()
def list_remediation_actions() -> dict[str, Any]:
    """List the remediation actions this environment supports, with their risk level.

    `propose_remediation` and `request_remediation` only accept ids from this list.
    """
    return {"actions": WORLD.actions()}


# --------------------------------------------------------------------------
# Conclusion and the approval gate
# --------------------------------------------------------------------------


@server.tool()
def propose_remediation(
    root_cause: str,
    action_id: str,
    justification: str,
    confidence: float,
    action_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record your diagnosis and the single action you would take. Call this exactly once, last.

    This is the answer, not an action -- nothing changes in the environment. State
    the causal chain in `root_cause`: what changed, what it saturated or broke, and
    how that produced the alerting symptom. Name the evidence you actually read.

    Args:
        root_cause: The causal explanation, specific enough to be falsifiable.
        action_id: An id from list_remediation_actions.
        justification: Why this action addresses the cause rather than the symptom.
        confidence: 0.0-1.0. Be honest; a low score on thin evidence is useful.
        action_params: Parameters for the action, e.g. {"service": "x", "new_size": 40}.
    """
    known = [a["id"] for a in WORLD.actions()]
    if action_id not in known:
        return _err(f"unknown action_id {action_id!r}", known_actions=known)

    proposal = {
        "root_cause": root_cause,
        "action_id": action_id,
        "action_params": action_params or {},
        "justification": justification,
        "confidence": max(0.0, min(1.0, float(confidence))),
    }
    WORLD.record_proposal(proposal)
    return {"recorded": True, **proposal}


@server.tool()
def request_remediation(
    action_id: str, reason: str, action_params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Queue a remediation for human approval. This does NOT apply the change.

    The request is written to the audit log in `pending` state. A human approves
    and applies it with `ops-copilot approve <request_id>`. There is no tool that
    applies a change directly, by design -- do not look for one.

    Args:
        action_id: An id from list_remediation_actions.
        reason: What this fixes and why it is safe to apply now.
        action_params: Parameters for the action.
    """
    known = [a["id"] for a in WORLD.actions()]
    if action_id not in known:
        return _err(f"unknown action_id {action_id!r}", known_actions=known)

    action = WORLD.action(action_id)
    request_id = f"CR-{uuid.uuid4().hex[:8]}"
    record = {
        "request_id": request_id,
        "state": "pending",
        "scenario": WORLD.scenario.id,
        "action_id": action_id,
        "action_params": action_params or {},
        "risk": action.get("risk", "unknown"),
        "reason": reason,
        "requested_by": "ops-copilot-agent",
        "requested_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "approved_by": None,
        "approved_at": None,
    }

    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with (_AUDIT_DIR / "change_requests.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    return {
        "request_id": request_id,
        "state": "pending",
        "risk": record["risk"],
        "message": (
            f"Change request {request_id} queued for human approval. Nothing has "
            f"been applied. A human must run `ops-copilot approve {request_id}`."
        ),
    }


def run() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    run()
