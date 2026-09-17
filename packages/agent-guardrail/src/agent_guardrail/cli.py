"""Command line entry point.

    agent-guardrail scan "text"     scan one piece of untrusted content
    agent-guardrail eval            score the corpora, print the tables
    agent-guardrail gate TOOL       show the capability decision for a tool
"""

from __future__ import annotations

import argparse
import sys

from .detect import heuristic_scan, normalise
from .evaluate import evaluate_classifier, evaluate_combined, evaluate_heuristic, render
from .policy import Policy, explain
from .trust import Context, Segment, trust_for

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m",
)


def cmd_scan(args) -> int:
    finding = heuristic_scan(args.text)
    trust = trust_for(args.source)
    colour = RED if finding.triggered else GREEN

    print(f"{BOLD}source{RESET}    {args.source} (trust={trust.label})")
    print(f"{BOLD}score{RESET}     {colour}{finding.score:.2f}{RESET}")
    if finding.matched:
        for name, why in zip(finding.matched, finding.reasons):
            print(f"  {RED}{name}{RESET}: {why}")
    else:
        print(f"  {GREEN}no pattern matched{RESET}")
    if args.show_normalised:
        print(f"\n{DIM}normalised: {normalise(args.text)[:300]}{RESET}")
    return 1 if finding.triggered else 0


def cmd_gate(args) -> int:
    context = Context()
    for source in args.read or []:
        context.add(Segment(content="...", source=source, origin=source))

    verdict = Policy().evaluate(args.tool, context, [heuristic_scan(args.text or "")])
    colour = {"allow": GREEN, "needs_approval": YELLOW, "deny": RED}[verdict.decision.value]
    print(f"{colour}{explain(verdict)}{RESET}")
    print(f"{DIM}context tainted: {verdict.tainted}"
          f"{'; sources: ' + ', '.join(context.untrusted_sources) if verdict.tainted else ''}{RESET}")
    return 0 if verdict.allowed else 1


def cmd_eval(args) -> int:
    heuristic = evaluate_heuristic()
    reports = [heuristic]
    if not args.quick:
        reports.append(evaluate_classifier(max_fpr=args.max_fpr))
        reports.append(evaluate_combined(max_fpr=args.max_fpr))

    print(render(reports))
    print(f"\n{DIM}{heuristic.caveat}{RESET}\n")

    print(f"{BOLD}detection by attack family (heuristic){RESET}")
    for family, (hit, total) in heuristic.by_family().items():
        mark = GREEN if hit == total else YELLOW
        print(f"  {family:22} {mark}{hit}/{total}{RESET}")

    fps = heuristic.false_positives_by_trap()
    print(f"\n{BOLD}false positives by trap{RESET}")
    print("  " + (", ".join(f"{k} x{v}" for k, v in fps.items()) if fps else f"{GREEN}none{RESET}"))

    print(f"\n{BOLD}missed attacks{RESET}")
    for prediction in heuristic.missed():
        print(f"  {prediction.sample.id} [{prediction.sample.family}] "
              f"{prediction.sample.text[:70]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent-guardrail", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="scan one piece of content")
    p_scan.add_argument("text")
    p_scan.add_argument("--source", default="tool_result")
    p_scan.add_argument("--show-normalised", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_gate = sub.add_parser("gate", help="capability decision for a tool call")
    p_gate.add_argument("tool")
    p_gate.add_argument("--read", action="append", help="a source already read (repeatable)")
    p_gate.add_argument("--text", help="untrusted content to scan alongside")
    p_gate.set_defaults(func=cmd_gate)

    p_eval = sub.add_parser("eval", help="score the corpora")
    p_eval.add_argument("--max-fpr", type=float, default=0.05)
    p_eval.add_argument("--quick", action="store_true", help="heuristic only, no model")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
