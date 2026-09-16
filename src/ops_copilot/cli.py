"""Command line entry point.

    ops-copilot list                     show the scenario corpus
    ops-copilot run SC-001               diagnose one incident, streaming the trace
    ops-copilot eval                     run the corpus and write a results set
    ops-copilot approvals                show change requests awaiting a human
    ops-copilot approve CR-1a2b3c4d      approve one, as yourself
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys

from . import approvals
from .agent import _cli_event, diagnose, summarise_error
from .harness import HarnessAborted, run_harness
from .scoring import score_run
from .world import all_scenarios, load_scenario

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m",
)


def cmd_list(args: argparse.Namespace) -> int:
    scenarios = all_scenarios()
    for s in scenarios:
        expected = ", ".join(s.ground_truth.get("expected_actions", []))
        print(f"{BOLD}{s.id}{RESET}  {DIM}{s.category:20}{RESET} {s.title}")
        if args.verbose:
            print(f"        alert: {s.alert.get('name')} ({s.alert.get('severity')})")
            print(f"        expected: {expected}\n")
    print(f"\n{len(scenarios)} scenarios")
    return 0


async def _run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    print(f"{BOLD}{scenario.id}{RESET}  {scenario.title}")
    print(f"{DIM}alert: {scenario.alert.get('name')} on "
          f"{scenario.alert.get('service')} ({scenario.alert.get('severity')}){RESET}\n")

    run = await diagnose(scenario.id, effort=args.effort, on_event=_cli_event)

    if run.error:
        print(f"{RED}error: {summarise_error(run.error)}{RESET}")
        if run.error_kind == "billing":
            print(f"\n{YELLOW}This is a credit/billing limit, not a wrong "
                  f"diagnosis.{RESET}\n"
                  f"  Add credits, or raise the key's own spend limit if it has one.\n"
                  f"  A full 30-scenario sweep needs roughly $6.")
        if run.error_kind == "auth":
            print(f"\n{YELLOW}This is an authentication failure, not a wrong "
                  f"diagnosis.{RESET}\n"
                  f"  export ANTHROPIC_API_KEY=sk-ant-...\n"
                  f"  key at https://console.anthropic.com/settings/keys")

    if run.proposal:
        score = score_run(run, scenario)
        mark = f"{GREEN}correct{RESET}" if score.correct else f"{RED}incorrect{RESET}"
        print(f"{BOLD}diagnosis{RESET} [{mark}]")
        print(f"  root cause : {run.proposal.get('root_cause')}")
        print(f"  action     : {run.proposal.get('action_id')} "
              f"(expected {', '.join(score.expected_actions)})")
        print(f"  confidence : {run.proposal.get('confidence')}")
        if score.missing_groups:
            print(f"  {YELLOW}missing{RESET}    : "
                  f"{'; '.join(' / '.join(g) for g in score.missing_groups)}")
        if score.misleading_claims:
            print(f"  {YELLOW}misleading{RESET} : {', '.join(score.misleading_claims)}")

    print(f"\n{DIM}{run.read_tool_calls} investigation calls, {run.turns} turns, "
          f"{run.elapsed_s:.1f}s, ${run.cost_usd:.4f}{RESET}")

    pending = approvals.pending()
    if pending:
        print(f"\n{YELLOW}{len(pending)} change request(s) awaiting approval{RESET} "
              f"-- run `ops-copilot approvals`")
    return 0 if run.proposal else 1


async def _eval(args: argparse.Namespace) -> int:
    total = len(args.scenarios) if args.scenarios else len(all_scenarios())
    done = 0

    def progress(score) -> None:
        nonlocal done
        done += 1
        mark = f"{GREEN}ok  {RESET}" if score.correct else f"{RED}miss{RESET}"
        print(f"  [{done:>2}/{total}] {mark} {score.scenario_id}  "
              f"cause={score.matched_groups}/{score.total_groups} "
              f"action={score.proposed_action} "
              f"{DIM}{score.read_tool_calls} calls, {score.elapsed_s:.0f}s{RESET}")

    print(f"{BOLD}running {total} scenarios{RESET} "
          f"(concurrency {args.concurrency}, effort {args.effort})\n")
    try:
        payload = await run_harness(
            args.scenarios or None,
            concurrency=args.concurrency,
            effort=args.effort,
            label=args.label,
            progress=progress,
        )
    except HarnessAborted as exc:
        print(f"\n{RED}sweep aborted{RESET}\n\n{exc}")
        return 2

    s = payload["summary"]
    if not s.get("valid", True):
        print(f"\n{RED}NOT A RESULT{RESET}  only {s['completed']}/{s['scenarios']} "
              f"scenarios completed ({s['completion_rate']:.0%}).")
        print("The rates below describe the survivors, not the suite. "
              "Do not quote them.")
        errored = [x for x in payload["scores"] if x["error"]]
        kinds = {}
        for x in payload["traces"]:
            if x.get("error"):
                kinds[x.get("error_kind") or "?"] = kinds.get(x.get("error_kind") or "?", 0) + 1
        print(f"{DIM}failure kinds: {kinds}{RESET}")
        if "rate_limit" in kinds:
            print(f"{YELLOW}Rate limited. Retry with --concurrency 1.{RESET}")
    print(f"\n{BOLD}summary{RESET}")
    print(f"  root cause identified   {s['root_cause_hit_rate']}%")
    print(f"  action matched          {s['action_match_rate']}%")
    print(f"  strictly correct        {s['strict_correct_rate']}%")
    print(f"  unwarranted requests    {s['over_reach_count']}")
    print(f"  mean investigation      {s['mean_read_tool_calls']} calls, "
          f"{s['mean_elapsed_s']}s")
    print(f"  cost                    ${s['mean_cost_usd']}/incident, "
          f"${s['total_cost_usd']} total")
    print(f"\nwritten to {payload['run']['directory']}")
    return 0


def cmd_approvals(args: argparse.Namespace) -> int:
    rows = approvals.pending() if args.pending_only else list(approvals.current_state().values())
    if not rows:
        print("no change requests")
        return 0
    for r in rows:
        colour = YELLOW if r["state"] == "pending" else GREEN
        print(f"{BOLD}{r['request_id']}{RESET}  {colour}{r['state']:8}{RESET} "
              f"risk={r['risk']:6} {r['action_id']}")
        print(f"  scenario  {r['scenario']}   requested {r['requested_at']}")
        print(f"  params    {json.dumps(r['action_params'])}")
        print(f"  reason    {r['reason']}")
        if r.get("approved_by"):
            print(f"  approved  by {r['approved_by']} at {r['approved_at']}")
        print()
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    approver = args.by or getpass.getuser()
    try:
        event = approvals.approve(args.request_id, approver)
    except (KeyError, ValueError) as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    print(f"{GREEN}approved{RESET} {event['request_id']} "
          f"({event['action_id']}, risk {event['risk']}) by {approver}")
    print(f"{DIM}Applying it is out of scope for this project -- the gate and the "
          f"trail are the deliverable, not a production actuator.{RESET}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="ops-copilot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list the scenario corpus")
    p_list.add_argument("-v", "--verbose", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="diagnose one incident")
    p_run.add_argument("scenario", help="scenario id, e.g. SC-001")
    p_run.add_argument("--effort", default="high",
                       choices=["low", "medium", "high", "xhigh", "max"])
    p_run.set_defaults(func=lambda a: asyncio.run(_run(a)))

    p_eval = sub.add_parser("eval", help="run the corpus and score it")
    p_eval.add_argument("scenarios", nargs="*", help="scenario ids (default: all)")
    p_eval.add_argument("--concurrency", type=int, default=4)
    p_eval.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"])
    p_eval.add_argument("--label", help="suffix for the run directory")
    p_eval.set_defaults(func=lambda a: asyncio.run(_eval(a)))

    p_appr = sub.add_parser("approvals", help="show change requests")
    p_appr.add_argument("--all", dest="pending_only", action="store_false",
                        help="include already-approved requests")
    p_appr.set_defaults(func=cmd_approvals, pending_only=True)

    p_ok = sub.add_parser("approve", help="approve a pending change request")
    p_ok.add_argument("request_id")
    p_ok.add_argument("--by", help="approver name (defaults to your username)")
    p_ok.set_defaults(func=cmd_approve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
