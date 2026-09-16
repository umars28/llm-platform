"""Command line entry point.

    llm-tracing show FILE            read a trace file as a tree
    llm-tracing export FILE --to URL ship spans to an OTLP endpoint
    llm-tracing demo                 trace a worked agent run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .export import LocalReceiver, export
from .spans import ATTR_COST, ATTR_INPUT_TOKENS, ATTR_OUTPUT_TOKENS, Span, Tracer

BOLD, DIM, RED, GREEN, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[36m", "\033[0m",
)


def _load(path: Path) -> list[Span]:
    spans = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            data = json.loads(line)
            spans.append(Span(
                name=data["name"], kind=data["kind"], trace_id=data["trace_id"],
                span_id=data["span_id"], parent_id=data.get("parent_id"),
                start_ns=data["start_ns"], end_ns=data["end_ns"],
                attributes=data.get("attributes", {}), events=data.get("events", []),
                status=data.get("status", "unset"), error=data.get("error"),
            ))
    return spans


def cmd_show(args) -> int:
    spans = _load(args.file)
    by_parent: dict[str | None, list[Span]] = {}
    for span in spans:
        by_parent.setdefault(span.parent_id, []).append(span)

    def render(parent: str | None, depth: int = 0) -> None:
        for span in sorted(by_parent.get(parent, []), key=lambda s: s.start_ns):
            mark = f"{RED}x{RESET}" if span.status == "error" else f"{GREEN}+{RESET}"
            cost = span.attributes.get(ATTR_COST, 0)
            detail = f"{span.duration_ms:.0f}ms"
            if cost:
                detail += f"  ${cost:.4f}"
            if span.attributes.get(ATTR_INPUT_TOKENS):
                detail += (f"  {span.attributes[ATTR_INPUT_TOKENS]:,}in/"
                           f"{span.attributes.get(ATTR_OUTPUT_TOKENS, 0):,}out")
            print(f"{'  ' * depth}{mark} {CYAN}{span.name}{RESET}  {DIM}{detail}{RESET}")
            if span.error:
                print(f"{'  ' * depth}    {RED}{span.error[:100]}{RESET}")
            render(span.span_id, depth + 1)

    roots = [s for s in spans if s.parent_id is None]
    for root in roots:
        render(root.parent_id if root.parent_id else None)
        break
    render(None) if not roots else None

    total_cost = sum(s.attributes.get(ATTR_COST, 0) for s in spans)
    errors = sum(1 for s in spans if s.status == "error")
    print(f"\n{BOLD}{len(spans)} spans{RESET}  ${total_cost:.4f}  "
          f"{errors} error{'s' if errors != 1 else ''}")
    return 0


def cmd_export(args) -> int:
    spans = _load(args.file)
    headers = {}
    if args.header:
        for raw in args.header:
            key, _, value = raw.partition(":")
            headers[key.strip()] = value.strip()
    status = export(spans, args.to, service=args.service, headers=headers)
    print(f"{GREEN}exported {len(spans)} spans{RESET} to {args.to} (HTTP {status})")
    return 0


def cmd_demo(args) -> int:
    """Trace a run shaped like an ops-copilot investigation, including a failure."""
    tracer = Tracer(service="ops-copilot", path=args.file)

    with tracer.span("investigate SC-001", alert="CheckoutAPIHighLatency") as run:
        run.add_event("alert received", severity="P1", service="checkout-api")

        for turn, tools in enumerate(
            [["list_services", "get_recent_deploys"],
             ["query_logs", "get_metrics"],
             ["search_runbook", "propose_remediation"]], start=1
        ):
            with tracer.llm_call("claude-opus-5") as call:
                time.sleep(0.01)
                call.set("gen_ai.turn", turn)
                call.record_usage("claude-opus-5", input_tokens=2543 + turn * 900,
                                  output_tokens=380, cache_read=2543 if turn > 1 else 0,
                                  cache_write=2543 if turn == 1 else 0)
            for tool in tools:
                with tracer.tool_call(tool):
                    time.sleep(0.005)

        try:
            with tracer.tool_call("request_remediation", risk="medium"):
                raise PermissionError("change request queued, awaiting human approval")
        except PermissionError:
            pass

    print(f"{GREEN}wrote {len(tracer.finished)} spans{RESET} to {tracer.path}")
    print(f"  cost ${tracer.total_cost():.4f}   tokens {tracer.total_tokens()}")
    if args.to:
        print(f"  exported: HTTP {export(tracer.finished, args.to, 'ops-copilot')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="llm-tracing", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show")
    p_show.add_argument("file", type=Path)
    p_show.set_defaults(func=cmd_show)

    p_exp = sub.add_parser("export")
    p_exp.add_argument("file", type=Path)
    p_exp.add_argument("--to", required=True, help="OTLP-HTTP endpoint")
    p_exp.add_argument("--service", default="llm-app")
    p_exp.add_argument("--header", action="append", help="Key: value (repeatable)")
    p_exp.set_defaults(func=cmd_export)

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--file", type=Path, default=Path("traces/demo.jsonl"))
    p_demo.add_argument("--to", help="also export to this OTLP endpoint")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
