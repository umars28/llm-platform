"""The human half of the approval gate.

`request_remediation` in the MCP server can only ever append a `pending` record
here. Moving a request to `approved` requires a person at a terminal, and the
transition is appended to the same log rather than rewriting history, so the
audit trail shows who approved what and when.

Applying a change for real is intentionally left unimplemented: this project
ships the gate and the trail, not a production actuator. `apply` prints the
command a human would run.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def audit_path() -> Path:
    base = Path(os.environ.get("OPS_COPILOT_AUDIT_DIR", REPO_ROOT / "audit"))
    return base / "change_requests.jsonl"


def _read_events() -> list[dict[str, Any]]:
    path = audit_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def current_state() -> dict[str, dict[str, Any]]:
    """Fold the append-only log into the latest state of each request."""
    state: dict[str, dict[str, Any]] = {}
    for event in _read_events():
        rid = event["request_id"]
        state[rid] = {**state.get(rid, {}), **event}
    return state


def pending() -> list[dict[str, Any]]:
    return [r for r in current_state().values() if r.get("state") == "pending"]


def approve(request_id: str, approver: str) -> dict[str, Any]:
    record = current_state().get(request_id)
    if record is None:
        raise KeyError(f"no change request {request_id!r}")
    if record.get("state") != "pending":
        raise ValueError(
            f"{request_id} is already {record.get('state')!r}, not pending"
        )

    event = {
        **record,
        "state": "approved",
        "approved_by": approver,
        "approved_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(event) + "\n")
    return event
