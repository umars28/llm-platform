"""Command line entry point.

    llm-eval run [--no-judge]        score recorded traces, print the report
    llm-eval gate [--no-judge]       score and compare to the committed baseline
    llm-eval baseline                record the current run as the new baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .cases import Subject, load_cases, run_suite
from .cost import UsageLedger
from .gate import BASELINE_DIR, Baseline, check_gate
from .judge import Judge

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m",
)

def _find_data(local: Path, shared_name: str) -> Path:
    """Locate a data directory, standalone or inside the monorepo.

    Each project keeps working on its own, where its data sits beside the source.
    Inside the platform repository the corpora are centralised under
    `benchmarks/`, so this walks up to find them. Without the fallback the
    subtree merge left six of eight projects unable to find their own fixtures --
    which nobody noticed, because their tests had never been run in the new
    location.
    """
    if local.exists():
        return local
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "benchmarks" / shared_name
        if candidate.exists():
            return candidate
    return local


DEFAULT_CASES = _find_data(Path("cases"), "eval-cases") / "ops-copilot.yaml"
DEFAULT_TRACES = Path("traces")


def _load_subjects(directory: Path) -> dict[str, Subject]:
    """Read ops-copilot run records from a directory of JSON files."""
    subjects: dict[str, Subject] = {}
    if not directory.exists():
        return subjects

    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        traces = data.get("traces", [data])
        for trace in traces:
            case_id = trace.get("scenario_id")
            if case_id:
                subjects[case_id] = Subject.from_trace(trace)
    return subjects


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _run(args) -> tuple[list, UsageLedger, str]:
    cases = load_cases(args.cases)
    subjects = _load_subjects(args.traces)
    judge = None if args.no_judge else Judge()
    version = "deterministic-only" if judge is None else judge.version

    results = run_suite(cases, subjects, judge)
    ledger = UsageLedger()
    return results, ledger, version


def cmd_run(args) -> int:
    results, ledger, version = _run(args)
    failed = [r for r in results if not r.passed]

    for result in results:
        mark = f"{GREEN}pass{RESET}" if result.passed else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {result.case_id}  {DIM}score {result.score:.2f}{RESET}")
        for check in result.failures:
            print(f"        {RED}{check.name}{RESET}: {check.reason}")

    print(f"\n{BOLD}{len(results) - len(failed)}/{len(results)} cases passed{RESET}"
          f"  {DIM}judge: {version}{RESET}")
    return 1 if failed else 0


def cmd_gate(args) -> int:
    results, ledger, version = _run(args)
    path = Path(args.baseline or BASELINE_DIR / "ops-copilot.json")

    if not path.exists():
        print(f"{YELLOW}no baseline at {path}{RESET}")
        print("record one with `llm-eval baseline` once a run is known good")
        return cmd_run(args)

    gate = check_gate(
        results, Baseline.load(path), version,
        cost_usd=ledger.cost_usd,
        score_tolerance=args.score_tolerance,
        cost_tolerance=args.cost_tolerance,
    )
    colour = GREEN if gate.passed else RED
    print(f"{colour}{gate.explain()}{RESET}")
    for result in results:
        if not result.passed:
            print(f"\n{result.explain()}")
    return 0 if gate.passed else 1


def cmd_baseline(args) -> int:
    results, ledger, version = _run(args)
    path = Path(args.baseline or BASELINE_DIR / "ops-copilot.json")
    failed = [r for r in results if not r.passed]

    if failed and not args.force:
        print(f"{RED}refusing to baseline a run with {len(failed)} failing case(s){RESET}")
        print("a baseline is a claim that this state is correct; use --force to override")
        for result in failed:
            print(f"  {result.explain()}")
        return 1

    Baseline.from_results(results, version, ledger.cost_usd, _commit()).save(path)
    print(f"{GREEN}baseline written{RESET} {path}")
    print(f"  {len(results)} cases, judge {version}, cost ${ledger.cost_usd:.4f}")
    return 0


def main() -> int:
    # Shared options are attached to every subcommand as well as the top level,
    # so `llm-eval gate --no-judge` works. argparse otherwise requires global
    # flags before the subcommand, which is not how anyone types it -- and the
    # CI workflow in this repository types it the natural way.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    common.add_argument("--traces", type=Path, default=DEFAULT_TRACES,
                        help="directory of recorded run records")
    common.add_argument("--no-judge", action="store_true",
                        help="deterministic assertions only; needs no credentials")
    common.add_argument("--baseline", type=Path)

    parser = argparse.ArgumentParser(prog="llm-eval", description=__doc__,
                                     parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="score and report",
                   parents=[common]).set_defaults(func=cmd_run)

    p_gate = sub.add_parser("gate", help="score and compare to the baseline",
                            parents=[common])
    p_gate.add_argument("--score-tolerance", type=float, default=0.02)
    p_gate.add_argument("--cost-tolerance", type=float, default=0.15)
    p_gate.set_defaults(func=cmd_gate)

    p_base = sub.add_parser("baseline", help="record the current run as the baseline",
                            parents=[common])
    p_base.add_argument("--force", action="store_true")
    p_base.set_defaults(func=cmd_baseline)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
