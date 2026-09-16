from __future__ import annotations

import pytest

from runbook_rag.evaluate import comparison_table, run_eval, worst_queries
from runbook_rag.golden import Query
from runbook_rag.metrics import score_query
from runbook_rag.retrieve import to_documents
from runbook_rag.store import Hit


def hit(doc: str, score: float = 0.5, chunk: str = "c") -> Hit:
    return Hit(f"{doc}:{chunk}", doc, f"Doc {doc}", None, "", "text", score)


def test_chunk_ranking_collapses_to_a_document_ranking():
    """A document ranks by its best chunk, and only appears once."""
    hits = [hit("a", 0.9), hit("a", 0.8, "c2"), hit("b", 0.7)]
    assert to_documents(hits) == ["a", "b"]


def test_document_order_follows_chunk_order():
    assert to_documents([hit("b"), hit("a")]) == ["b", "a"]


class FakeRetriever:
    name = "fake"

    def __init__(self, ranking):
        self.ranking = ranking
        self.k_seen = None

    def search(self, query, k):
        self.k_seen = k
        return [hit(doc) for doc in self.ranking]


def test_run_eval_scores_every_query():
    queries = [
        Query(id="Q-1", text="first symptom sentence here", facet="f", relevant={"a": 2}),
        Query(id="Q-2", text="second symptom sentence here", facet="g", relevant={"z": 2}),
    ]
    results = run_eval(FakeRetriever(["a", "b"]), queries)
    assert [r.query_id for r in results] == ["Q-1", "Q-2"]
    assert results[0].mrr == 1.0
    assert results[1].mrr == 0.0


def test_run_eval_retrieves_deeper_than_the_reported_cutoffs():
    """Otherwise recall@10 is capped by the retrieval depth, not the retriever."""
    retriever = FakeRetriever(["a"])
    run_eval(retriever, [Query(id="Q", text="a symptom sentence", facet="f", relevant={"a": 2})])
    assert retriever.k_seen >= 10


def test_comparison_table_shows_deltas_against_the_first_row():
    table = comparison_table(
        {
            "baseline": {"recall_at_5": 0.50, "recall_at_10": 0.6, "mrr": 0.5,
                         "ndcg_at_5": 0.4, "hit_at_5": 0.7},
            "improved": {"recall_at_5": 0.60, "recall_at_10": 0.6, "mrr": 0.5,
                         "ndcg_at_5": 0.4, "hit_at_5": 0.7},
        }
    )
    assert "0.600 (+20%)" in table
    assert "| baseline | 0.500 |" in table  # no delta on the baseline row


def test_worst_queries_surfaces_the_failures_averages_hide():
    q = Query(id="Q", text="a symptom sentence", facet="f", relevant={"a": 2})
    good = score_query(q, ["a"])
    bad = score_query(q, ["x", "y"])
    assert worst_queries([good, bad], n=1) == [bad]
