"""Run the golden set against a retriever and write a results set.

Every run records the configuration alongside the numbers, because a metric
without the setup that produced it cannot be compared to anything.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .config import RUNS_DIR
from .golden import Query, load_golden
from .metrics import QueryResult, aggregate, by_facet, score_query
from .retrieve import Retriever, to_documents

# Retrieve deeper than the reported cutoffs so recall@10 is measurable and a
# reranker downstream has something to reorder.
DEPTH = 30


def run_eval(
    retriever: Retriever,
    queries: Sequence[Query] | None = None,
    depth: int = DEPTH,
) -> list[QueryResult]:
    queries = list(queries if queries is not None else load_golden())

    if hasattr(retriever, "search_many"):
        rankings = retriever.search_many([q.text for q in queries], k=depth)
    else:
        rankings = [retriever.search(q.text, k=depth) for q in queries]

    return [
        score_query(query, to_documents(hits))
        for query, hits in zip(queries, rankings)
    ]


def save_run(
    name: str, results: Sequence[QueryResult], config: dict[str, Any] | None = None
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{stamp}-{name.replace('/', '_')}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "ran_at": stamp,
                "config": config or {},
                "summary": aggregate(results),
                "by_facet": by_facet(results),
                "queries": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    return path


HEADLINE = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_5", "hit_at_5")


def comparison_table(runs: dict[str, dict[str, float]]) -> str:
    """Markdown table of several runs, with the first row as the baseline."""
    names = list(runs)
    header = "| configuration | " + " | ".join(HEADLINE) + " |"
    rule = "| --- |" + " --- |" * len(HEADLINE)
    lines = [header, rule]

    baseline = runs[names[0]] if names else {}
    for name in names:
        row = [name]
        for field in HEADLINE:
            value = runs[name][field]
            cell = f"{value:.3f}"
            if name != names[0] and baseline.get(field):
                delta = (value - baseline[field]) / baseline[field] * 100
                cell += f" ({delta:+.0f}%)"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def worst_queries(results: Sequence[QueryResult], n: int = 5) -> list[QueryResult]:
    """The queries a change should be judged on -- averages hide these."""
    return sorted(results, key=lambda r: (r.mrr, r.recall_at_5))[:n]
