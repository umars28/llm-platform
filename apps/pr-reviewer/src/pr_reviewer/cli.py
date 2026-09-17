"""Command line entry point.

    pr-reviewer extract REPO...      rebuild the corpus from git history
    pr-reviewer evaluate             review every sample, report recall and FPR
    pr-reviewer review PATH          review one diff, print findings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .backends import OllamaClient, build_client
from .extract import extract_all, load, save
from .review import Review, Score, found_the_defect, review

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m",
)

CORPUS = Path("corpus/samples.json")


def cmd_extract(args) -> int:
    samples = extract_all([Path(r) for r in args.repos])
    save(samples, args.corpus)
    defective = sum(s.has_defect for s in samples)
    print(f"{len(samples)} samples ({defective} defective, "
          f"{len(samples) - defective} clean) -> {args.corpus}")
    return 0


def cmd_evaluate(args) -> int:
    samples = load(args.corpus)
    if args.limit:
        samples = samples[: args.limit]
    client = build_client(args.backend, args.model)
    model = args.model or getattr(client, "model", "claude-opus-5")

    if isinstance(client, OllamaClient):
        ok, detail = client.available()
        if not ok:
            print(f"{RED}{detail}{RESET}")
            return 2

    print(f"{BOLD}reviewing {len(samples)} diffs{RESET} with {model}\n", flush=True)
    score = Score()
    started = time.time()

    for index, sample in enumerate(samples, start=1):
        result = review(client, sample, model=model)
        score.add(result, sample)

        if result.error:
            mark = f"{RED}err {RESET}"
        elif sample.has_defect:
            mark = f"{GREEN}hit {RESET}" if found_the_defect(result, sample) else f"{YELLOW}miss{RESET}"
        else:
            mark = f"{RED}noise{RESET}" if result.reported_a_defect else f"{GREEN}quiet{RESET}"
        print(f"  [{index:>3}/{len(samples)}] {mark} {sample.id[:52]:54} "
              f"{DIM}{len(result.findings)} findings{RESET}", flush=True)

    summary = score.summary()
    elapsed = time.time() - started
    if not summary["valid"]:
        print(f"\n{RED}NOT A RESULT{RESET}  only "
              f"{summary['completion_rate']:.0%} of reviews completed "
              f"({summary['errors']} errors).")
        print("The rates below describe whichever samples survived. "
              "Do not quote them.")
    print(f"\n{BOLD}summary{RESET}  {DIM}{elapsed/len(samples):.0f}s per diff{RESET}")
    print(f"  recall (named the real defect)   {summary['recall']:.0%}")
    print(f"  detection (flagged something)    {summary['detection_rate']:.0%}")
    print(f"  false positives on clean diffs   {summary['false_positive_rate']:.0%}")
    print(f"  findings per clean diff          {summary['findings_per_clean_diff']}")
    print(f"  errors                           {summary['errors']}")

    out = Path("runs"); out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}-{model.replace('/', '_').replace(':', '-')}.json").write_text(
        json.dumps({
            "model": model, "summary": summary,
            "missed": [s.id for s in score.missed()],
            "noisy": [{"sample": s.id, "findings": [f.summary for f in fs]}
                      for s, fs in score.noisy()],
        }, indent=2)
    )
    print(f"\n{DIM}written to runs/{RESET}")
    return 0


def cmd_review(args) -> int:
    from .extract import Sample

    diff = Path(args.path).read_text()
    client = build_client(args.backend, args.model)
    model = args.model or getattr(client, "model", "claude-opus-5")
    sample = Sample(id=args.path, repo="", commit="", path=args.path,
                    diff=diff, has_defect=False)

    result = review(client, sample, model=model)
    if result.error:
        print(f"{RED}{result.error}{RESET}")
        return 2
    if not result.findings:
        print(f"{GREEN}no defects found{RESET}")
        return 0
    for finding in result.findings:
        print(f"{RED}{finding.summary}{RESET}")
        if finding.trigger:
            print(f"  trigger: {finding.trigger}")
        if finding.line_hint:
            print(f"  {DIM}{finding.line_hint}{RESET}")
    return 1


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", type=Path, default=CORPUS)
    common.add_argument("--backend", default="auto", choices=["auto", "api", "local"])
    common.add_argument("--model")

    parser = argparse.ArgumentParser(prog="pr-reviewer", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ex = sub.add_parser("extract", parents=[common])
    p_ex.add_argument("repos", nargs="+")
    p_ex.set_defaults(func=cmd_extract)

    p_ev = sub.add_parser("evaluate", parents=[common])
    p_ev.add_argument("--limit", type=int)
    p_ev.set_defaults(func=cmd_evaluate)

    p_rv = sub.add_parser("review", parents=[common])
    p_rv.add_argument("path")
    p_rv.set_defaults(func=cmd_review)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
