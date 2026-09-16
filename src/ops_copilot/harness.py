"""Run the scenario corpus and write a results set.

Scenarios run concurrently with a bounded semaphore. Each gets its own MCP
server subprocess and its own audit directory, so a change request queued by one
run cannot appear in another's trail.

A failed scenario is recorded and the sweep continues -- one API error should
not cost the other twenty-nine results.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

from .agent import diagnose
from .scoring import Score, score_run, summarise
from .world import Scenario, all_scenarios, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"


async def _run_one(
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
    run_dir: Path,
    effort: str,
    model: str,
    progress: Callable[[Score], None] | None,
) -> tuple[Score, dict[str, Any]]:
    async with semaphore:
        run = await diagnose(
            scenario.id,
            model=model,
            effort=effort,
            audit_dir=str(run_dir / "audit" / scenario.id),
        )
    score = score_run(run, scenario)
    if progress:
        progress(score)
    return score, run.to_dict()


async def run_harness(
    scenario_refs: list[str] | None = None,
    *,
    concurrency: int = 4,
    effort: str = "high",
    model: str = "claude-opus-5",
    label: str | None = None,
    progress: Callable[[Score], None] | None = None,
) -> dict[str, Any]:
    scenarios = (
        [load_scenario(ref) for ref in scenario_refs]
        if scenario_refs
        else all_scenarios()
    )

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / (f"{stamp}-{label}" if label else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(
            _run_one(s, semaphore, run_dir, effort, model, progress)
            for s in scenarios
        )
    )

    scores = [score for score, _ in results]
    payload = {
        "run": {
            "started_at": stamp,
            "model": model,
            "effort": effort,
            "concurrency": concurrency,
            "label": label,
        },
        "summary": summarise(scores),
        "scores": [s.to_dict() for s in scores],
        "traces": [trace for _, trace in results],
    }

    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "results.md").write_text(render_markdown(payload))
    payload["run"]["directory"] = str(run_dir)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    meta = payload["run"]
    lines = [
        f"# Ops Copilot harness run {meta['started_at']}",
        "",
        f"Model `{meta['model']}` at effort `{meta['effort']}`, "
        f"{summary['scenarios']} scenarios.",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| root cause identified | {summary['root_cause_hit_rate']}% |",
        f"| action matched | {summary['action_match_rate']}% |",
        f"| free of misleading claims | {summary['clean_rate']}% |",
        f"| strictly correct (all three) | {summary['strict_correct_rate']}% |",
        f"| unwarranted change requests | {summary['over_reach_count']} |",
        f"| mean investigation tool calls | {summary['mean_read_tool_calls']} |",
        f"| mean wall time | {summary['mean_elapsed_s']}s |",
        f"| mean cost per incident | ${summary['mean_cost_usd']} |",
        f"| total cost | ${summary['total_cost_usd']} |",
        "",
        "## By category",
        "",
        "| category | n | root cause | strictly correct |",
        "| --- | --- | --- | --- |",
    ]
    for cat, stats in summary["by_category"].items():
        lines.append(
            f"| {cat} | {stats['n']} | {stats['cause_rate']}% | {stats['correct_rate']}% |"
        )

    lines += [
        "",
        "## Per scenario",
        "",
        "| scenario | cause | action | proposed | tools | cost |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in payload["scores"]:
        cause = "yes" if s["root_cause_hit"] else f"{s['matched_groups']}/{s['total_groups']}"
        action = "yes" if s["action_match"] else "no"
        proposed = s["proposed_action"] or ("error" if s["error"] else "none")
        lines.append(
            f"| {s['scenario_id']} | {cause} | {action} | `{proposed}` | "
            f"{s['read_tool_calls']} | ${s['cost_usd']} |"
        )

    return "\n".join(lines) + "\n"
