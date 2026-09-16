"""Retrieval metrics, computed at document level.

Document level rather than chunk level, deliberately. Chunking is a variable
here -- `header` produces 4,549 chunks where `fixed` produces 2,875 -- so a
chunk-level metric would reward whichever strategy happened to slice a relevant
page into more pieces. Scoring the document a chunk belongs to makes the
strategies comparable, which is the whole reason for having three.

Three metrics, because each answers a different question:

  recall@k   did the answer make it into the window at all
  MRR        how far down the list was the first correct answer
  nDCG@k     were the better answers ranked above the merely useful ones

recall@k is what matters when the result feeds a language model with a fixed
context budget. MRR is what matters when a human reads the list top-down. nDCG
is the only one of the three that uses the graded labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .golden import Query


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents appearing in the top k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if anything relevant is in the top k. The 'is it usable' metric."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], grades: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain over graded relevance."""
    if not grades:
        return 0.0

    gain = sum(
        grades.get(doc, 0) / math.log2(position + 1)
        for position, doc in enumerate(retrieved[:k], start=1)
    )
    ideal = sum(
        grade / math.log2(position + 1)
        for position, grade in enumerate(sorted(grades.values(), reverse=True)[:k], start=1)
    )
    return gain / ideal if ideal else 0.0


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    facet: str
    retrieved: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float

    @property
    def found_nothing(self) -> bool:
        return self.mrr == 0.0


def score_query(query: Query, retrieved_doc_ids: Sequence[str]) -> QueryResult:
    relevant = query.any_relevant
    return QueryResult(
        query_id=query.id,
        facet=query.facet,
        retrieved=list(retrieved_doc_ids[:10]),
        recall_at_1=recall_at_k(retrieved_doc_ids, relevant, 1),
        recall_at_3=recall_at_k(retrieved_doc_ids, relevant, 3),
        recall_at_5=recall_at_k(retrieved_doc_ids, relevant, 5),
        recall_at_10=recall_at_k(retrieved_doc_ids, relevant, 10),
        hit_at_5=hit_at_k(retrieved_doc_ids, relevant, 5),
        mrr=reciprocal_rank(retrieved_doc_ids, relevant),
        ndcg_at_5=ndcg_at_k(retrieved_doc_ids, query.relevant, 5),
        ndcg_at_10=ndcg_at_k(retrieved_doc_ids, query.relevant, 10),
    )


NUMERIC_FIELDS = (
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "hit_at_5", "mrr", "ndcg_at_5", "ndcg_at_10",
)


def aggregate(results: Sequence[QueryResult]) -> dict[str, float]:
    """Macro-average over queries: every query counts the same."""
    if not results:
        return {field: 0.0 for field in NUMERIC_FIELDS} | {"queries": 0}

    summary = {
        field: round(sum(getattr(r, field) for r in results) / len(results), 4)
        for field in NUMERIC_FIELDS
    }
    summary["queries"] = len(results)
    summary["found_nothing"] = sum(1 for r in results if r.found_nothing)
    return summary


def by_facet(results: Sequence[QueryResult]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[QueryResult]] = {}
    for result in results:
        buckets.setdefault(result.facet, []).append(result)
    return {facet: aggregate(rs) for facet, rs in sorted(buckets.items())}
