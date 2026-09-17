"""Command line entry point.

    token-optimizer measure WORKLOAD    caching, pruning and routing savings
    token-optimizer audit FILE          find cache invalidators in a prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .counting import count
from .measure import load_workload, measure
from .optimise import find_invalidators, plan_cache, routing_saving

BOLD, DIM, RED, GREEN, RESET = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[0m"


def cmd_measure(args) -> int:
    w = load_workload(args.workload)
    m = measure(w, args.model)

    print(f"{BOLD}{w.name}{RESET}  {len(w.tools)} tools, {w.turns} turns, "
          f"{len(w.messages)} messages  {DIM}({args.model}){RESET}\n")
    print(f"  prefix, resent per turn   {w.prefix_tokens:>8,} tokens")
    print(f"  volatile                  {w.volatile_tokens:>8,} tokens")
    print(f"  cache breakpoint          {m.cache_reason}")
    print(f"  breakeven                 {m.breakeven_calls} calls\n")
    print(f"  no caching                ${m.baseline.cost_usd:>8.4f}  {DIM}computed at list price{RESET}")
    print(f"  with caching              ${m.cached.cost_usd:>8.4f}")
    print(f"  {GREEN}caching saves             {m.cache_saving_percent:>8.0f}%{RESET}")
    print(f"  {GREEN}pruning removes           {m.prune_saving_percent:>8.0f}%{RESET} of input tokens"
          f"  {DIM}({len(m.pruned_removed)} messages){RESET}")

    if args.routing:
        tasks = ["classify this log line"] * args.routing + ["diagnose the root cause"] * max(
            1, args.routing // 4)
        r = routing_saving(tasks, w.prefix_tokens + w.volatile_tokens, w.output_tokens)
        print(f"\n  routing {r['to_cheap']}/{r['tasks']} to the cheap model")
        print(f"  all-strong                ${r['baseline_usd']:>8.4f}")
        print(f"  routed                    ${r['routed_usd']:>8.4f}")
        print(f"  {GREEN}routing saves             {r['saving_percent']:>8.0f}%{RESET}")
    return 0


def cmd_audit(args) -> int:
    text = Path(args.file).read_text()
    plan = plan_cache(text)
    print(f"{BOLD}{args.file}{RESET}  {count(text):,} tokens\n")
    if plan.cacheable:
        print(f"{GREEN}cacheable{RESET}  {plan.reason}")
        return 0
    print(f"{RED}not cacheable{RESET}  {plan.reason}")
    for inv in plan.invalidators:
        print(f"  {RED}{inv.kind}{RESET}  {inv.evidence}")
        print(f"      {DIM}{inv.why}{RESET}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="token-optimizer", description=__doc__)
    parser.add_argument("--model", default="claude-opus-5")
    sub = parser.add_subparsers(dest="command", required=True)

    p_m = sub.add_parser("measure")
    p_m.add_argument("workload", type=Path)
    p_m.add_argument("--model", default="claude-opus-5")
    p_m.add_argument("--routing", type=int, default=0,
                     help="simulate routing over N routine tasks")
    p_m.set_defaults(func=cmd_measure)

    p_a = sub.add_parser("audit")
    p_a.add_argument("file")
    p_a.add_argument("--model", default="claude-opus-5")
    p_a.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
