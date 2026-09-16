"""The diagnosis agent: Claude Opus 5 driving the MCP tools over stdio.

One run spawns a dedicated MCP server subprocess pinned to a single scenario,
hands Claude the alert, and lets it investigate until it commits to a diagnosis
via `propose_remediation`.

The run record captures more than the answer -- the full ordered tool-call trace,
token usage and wall time -- because the interesting question is not only whether
the agent was right but how much work it did to get there. The harness turns
those into the numbers in the README.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .world import Scenario, load_scenario

MODEL = "claude-opus-5"

# Anything matching these means every scenario will fail the same way, so the
# harness should stop rather than produce thirty identical failures.
_AUTH_MARKERS = (
    "could not resolve authentication",
    "authentication_error",
    "invalid x-api-key",
    "invalid bearer token",
    "permission_error",
)
_CONFIG_MARKERS = ("not_found_error", "model:", "invalid_request_error")


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten anyio/asyncio TaskGroup ExceptionGroups down to real errors.

    Without this every failure inside the MCP task group is reported as
    "unhandled errors in a TaskGroup (1 sub-exception)", which names nothing.
    """
    if isinstance(exc, BaseExceptionGroup):
        found: list[BaseException] = []
        for sub in exc.exceptions:
            found.extend(_leaves(sub))
        return found or [exc]
    return [exc]


def describe_exception(exc: BaseException) -> tuple[str, str]:
    """Return (human-readable message, error kind) for a caught exception."""
    leaves = _leaves(exc)
    message = "; ".join(f"{type(e).__name__}: {e}" for e in leaves)
    lowered = message.lower()
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return message, "auth"
    if any(marker in lowered for marker in _CONFIG_MARKERS):
        return message, "config"
    return message, "other"


def credentials_available() -> bool:
    """Whether the SDK can resolve any credential without making a request."""
    probe = AsyncAnthropic()
    if probe.api_key or getattr(probe, "auth_token", None):
        return True
    # OAuth profiles and workload identity federation resolve through headers.
    try:
        return bool(probe.auth_headers)
    except Exception:
        return False

# Opus 5 list price, USD per million tokens.
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ = 5.00, 25.00, 0.50

SYSTEM_PROMPT = """\
You are an on-call SRE assistant diagnosing a live production incident.

You have read-only tools over logs, metrics, Kubernetes state, deploy history and \
runbooks. Investigate before you conclude -- an answer that names evidence you \
never actually fetched is worse than no answer.

How to think about this:

- The alert reports a symptom, not a cause. Its framing may be misleading; the \
firing service is often the victim rather than the culprit.
- Incidents that begin shortly after a release usually belong to that release. \
Check what shipped and read the change summary before theorising further.
- Separate a client's own resources saturating from the dependency behind it \
being unhealthy. Pool exhaustion, thread starvation and queue backup all look \
like "the database is slow" from the caller's side. Check the dependency's own \
metrics before blaming it.
- Before committing, ask what else you would expect to see if your theory were \
true, and go look. A theory that survives one attempt to break it is worth far \
more than one that was never tested.
- Runbooks carry operating limits you cannot infer from telemetry -- capacity \
budgets, which actions are pre-approved. Consult them before choosing an action.

Finish by calling propose_remediation exactly once with your causal chain: what \
changed, what it saturated or broke, and how that produced the alerting symptom. \
Then, if an action is clearly indicated, call request_remediation to queue it. \
That queues the change for a human; nothing you can call applies it yourself.
"""


@dataclass
class AgentRun:
    scenario_id: str
    scenario_title: str
    proposal: dict[str, Any] | None = None
    change_requests: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    elapsed_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    turns: int = 0
    error: str | None = None
    error_kind: str | None = None

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def read_tool_calls(self) -> int:
        """Investigation calls only -- excludes the two reporting tools."""
        reporting = {"propose_remediation", "request_remediation"}
        return sum(1 for c in self.tool_calls if c["name"] not in reporting)

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_IN
            + self.output_tokens * PRICE_OUT
            + self.cache_read_tokens * PRICE_CACHE_READ
        ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_title": self.scenario_title,
            "proposal": self.proposal,
            "change_requests": self.change_requests,
            "tool_calls": self.tool_calls,
            "tool_call_count": self.tool_call_count,
            "read_tool_calls": self.read_tool_calls,
            "turns": self.turns,
            "final_text": self.final_text,
            "elapsed_s": round(self.elapsed_s, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "error": self.error,
            "error_kind": self.error_kind,
        }


def _alert_prompt(scenario: Scenario) -> str:
    alert = scenario.alert
    return (
        "An alert has fired. Diagnose it.\n\n"
        f"  alert:    {alert.get('name')}\n"
        f"  severity: {alert.get('severity')}\n"
        f"  service:  {alert.get('service')}\n"
        f"  fired_at: {scenario.fired_at.isoformat().replace('+00:00', 'Z')}\n\n"
        f"{alert.get('description', '').strip()}\n\n"
        "Relative time arguments on the tools (`since_minutes`) are measured "
        "back from the moment this alert fired."
    )


def _server_params(scenario_ref: str, audit_dir: str | None) -> StdioServerParameters:
    env = {**os.environ, "OPS_COPILOT_SCENARIO": scenario_ref}
    if audit_dir:
        env["OPS_COPILOT_AUDIT_DIR"] = audit_dir
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ops_copilot.server"],
        env=env,
    )


async def diagnose(
    scenario_ref: str,
    *,
    model: str = MODEL,
    effort: str = "high",
    max_iterations: int = 20,
    audit_dir: str | None = None,
    on_event: Any = None,
) -> AgentRun:
    """Run one incident end to end and return the trace."""
    scenario = load_scenario(scenario_ref)
    run = AgentRun(scenario_id=scenario.id, scenario_title=scenario.title)
    client = AsyncAnthropic()

    def emit(kind: str, payload: Any) -> None:
        if on_event:
            on_event(kind, payload)

    started = time.perf_counter()
    try:
        async with stdio_client(_server_params(scenario_ref, audit_dir)) as (read, write):
            async with ClientSession(read, write) as mcp_client:
                await mcp_client.initialize()
                listed = await mcp_client.list_tools()
                emit("tools", [t.name for t in listed.tools])

                runner = client.beta.messages.tool_runner(
                    model=model,
                    max_tokens=16000,
                    max_iterations=max_iterations,
                    system=SYSTEM_PROMPT,
                    thinking={"type": "adaptive"},
                    output_config={"effort": effort},
                    tools=[async_mcp_tool(t, mcp_client) for t in listed.tools],
                    messages=[{"role": "user", "content": _alert_prompt(scenario)}],
                )

                async for message in runner:
                    run.turns += 1
                    usage = message.usage
                    run.input_tokens += usage.input_tokens or 0
                    run.output_tokens += usage.output_tokens or 0
                    run.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

                    for block in message.content:
                        if block.type == "text" and block.text.strip():
                            run.final_text = block.text.strip()
                            emit("text", block.text.strip())
                        elif block.type == "tool_use":
                            # Tool inputs may arrive with varied JSON escaping;
                            # the SDK has already parsed them into dicts.
                            call = {"name": block.name, "input": block.input}
                            run.tool_calls.append(call)
                            emit("tool_use", call)
                            if block.name == "propose_remediation":
                                run.proposal = dict(block.input)
                            elif block.name == "request_remediation":
                                run.change_requests.append(dict(block.input))

                if run.proposal is None:
                    run.error = (
                        f"agent finished after {run.turns} turns without calling "
                        "propose_remediation"
                    )
    except BaseException as exc:  # includes ExceptionGroup from the MCP task group
        run.error, run.error_kind = describe_exception(exc)
    finally:
        run.elapsed_s = time.perf_counter() - started

    return run


def _cli_event(kind: str, payload: Any) -> None:
    if kind == "tools":
        print(f"  mcp tools: {', '.join(payload)}\n")
    elif kind == "tool_use":
        args = ", ".join(f"{k}={v!r}" for k, v in payload["input"].items())
        if len(args) > 110:
            args = args[:107] + "..."
        print(f"  \033[36m→ {payload['name']}\033[0m({args})")
    elif kind == "text":
        print(f"\n{payload}\n")


async def main() -> None:
    scenario_ref = sys.argv[1] if len(sys.argv) > 1 else "SC-001"
    run = await diagnose(scenario_ref, on_event=_cli_event)
    print(json.dumps(run.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
