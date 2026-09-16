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
import os
from pathlib import Path
from typing import Any, Callable

from .agent import MODEL, credentials_available, diagnose, summarise_error
from .scoring import Score, score_run, summarise
from .world import Scenario, all_scenarios, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

# Below this share of scenarios completing, the rates describe the survivors
# rather than the suite, and are not reported as accuracy.
MIN_COMPLETION = float(os.environ.get("OPS_COPILOT_MIN_COMPLETION", "0.9"))


class HarnessAborted(RuntimeError):
    """The sweep stopped because every scenario would fail identically.

    A results set of thirty auth failures reads as "0% accuracy" to anyone who
    opens it later. Refusing to produce one is the whole point of this class.
    """


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
    trace = run.to_dict()
    _append_partial(run_dir, score, trace)
    if progress:
        progress(score)
    return score, trace


def _append_partial(run_dir: Path, score: Score, trace: dict[str, Any]) -> None:
    """Write each scenario as it finishes.

    A serial sweep takes over an hour, and writing only at the end means a
    laptop sleeping, a process killed or a machine rebooting loses every
    completed scenario. Appending costs nothing and makes the run resumable.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "partial.jsonl").open("a") as fh:
        fh.write(json.dumps({"score": score.to_dict(), "trace": trace}) + "\n")


def load_partial(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Scenarios already completed in an interrupted run, by id."""
    path = Path(run_dir) / "partial.jsonl"
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["score"]["scenario_id"]] = row
    return done


async def run_harness(
    scenario_refs: list[str] | None = None,
    *,
    concurrency: int = 4,
    effort: str = "high",
    model: str = MODEL,
    label: str | None = None,
    progress: Callable[[Score], None] | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    scenarios = (
        [load_scenario(ref) for ref in scenario_refs]
        if scenario_refs
        else all_scenarios()
    )

    if not credentials_available():
        raise HarnessAborted(
            "No Anthropic credentials could be resolved, so every scenario "
            "would fail at authentication and the results would be thirty "
            "identical errors rather than a score.\n\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "Get a key at https://console.anthropic.com/settings/keys"
        )

    if resume_from:
        run_dir = Path(resume_from)
        if not run_dir.exists():
            raise HarnessAborted(f"nothing to resume at {run_dir}")
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / (f"{stamp}-{label}" if label else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    already = load_partial(run_dir)
    if already:
        scenarios = [s for s in scenarios if s.id not in already]

    semaphore = asyncio.Semaphore(concurrency)

    # Canary: prove one scenario can reach the API before committing to the
    # other twenty-nine. An auth or model-id mistake fails the same way every
    # time, and discovering it once is enough.
    if not scenarios:  # everything was already done in a previous attempt
        results = []
        first = None
    else:
        first = await _run_one(scenarios[0], semaphore, run_dir, effort, model, progress)
    if first and first[1].get("error_kind") in {"auth", "config", "billing"}:
        raise HarnessAborted(
            f"{scenarios[0].id} failed before reaching any tool, and the cause "
            f"is not scenario-specific, so the sweep stopped after one instead "
            f"of repeating it {len(scenarios)} times:\n\n  "
            f"{summarise_error(first[1]['error'])}"
        )

    rest = await asyncio.gather(
        *(
            _run_one(s, semaphore, run_dir, effort, model, progress)
            for s in scenarios[1:]
        )
    )
    results = [first, *rest] if first else []

    # Completed scenarios from an interrupted run count towards this one.
    recovered = [
        (Score(**{k: v for k, v in row["score"].items() if k in Score.__dataclass_fields__}),
         row["trace"])
        for row in already.values()
    ]
    results = [*recovered, *results]

    scores = [score for score, _ in results]
    summary = summarise(scores)

    # Validity is a completion rate, not a list of error kinds.
    #
    # The previous rule was "at least one scenario completed", which passed a
    # run where 25 of 30 hit a rate limit and duly reported "16.7% root cause
    # identified" -- a number about the five that ran, presented as a number
    # about thirty. Enumerating which errors are disqualifying only ever covers
    # the failures already seen; a completion floor covers the next one too.
    completion = summary["completed"] / summary["scenarios"] if summary["scenarios"] else 0.0
    summary["completion_rate"] = round(completion, 4)
    summary["valid"] = completion >= MIN_COMPLETION
    payload = {
        "run": {
            "started_at": stamp,
            "model": model,
            "effort": effort,
            "concurrency": concurrency,
            "label": label,
        },
        "summary": summary,
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
    ]
    if not summary.get("valid", True):
        lines += [
            f"> **These are not results.** Only "
            f"{summary['completed']}/{summary['scenarios']} scenarios completed "
            f"({summary['completion_rate']:.0%}); the rates below describe the "
            "survivors, not the suite. Do not quote them.",
            "",
        ]
    lines += [
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
