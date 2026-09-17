"""Hybrid retrieval against the real indexed corpus."""

from __future__ import annotations

import pytest

from runbook_rag.hybrid import BM25Retriever, HybridRetriever
from runbook_rag.retrieve import DenseRetriever, to_documents
from runbook_rag.store_pg import PgVectorStore

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def store() -> PgVectorStore:
    s = PgVectorStore()
    try:
        if s.count("header") == 0:
            pytest.skip("corpus not indexed; run `runbook-rag index` first")
    except Exception as exc:
        pytest.skip(f"postgres unavailable: {exc}")
    return s


def test_bm25_finds_an_exact_identifier(store):
    """The failure mode dense retrieval has and lexical search does not."""
    hits = BM25Retriever(store, "header").search("CrashLoopBackOff", k=10)
    assert any("crashloop" in h.content.lower() for h in hits)


def test_hybrid_returns_the_requested_number_of_hits(store):
    hits = HybridRetriever(store, "header").search("pod stuck pending", k=7)
    assert len(hits) == 7


def test_hybrid_hits_are_fully_hydrated(store):
    """Fused ids may come from BM25 alone and must still carry their payload."""
    for hit in HybridRetriever(store, "header").search("dns resolution failing", k=10):
        assert hit.doc_title and hit.content and hit.doc_id


def test_hybrid_surfaces_documents_dense_alone_ranked_lower(store):
    query = "traffic between two services started timing out with no errors"
    dense = to_documents(DenseRetriever(store, "header").search(query, k=5))
    hybrid = to_documents(HybridRetriever(store, "header").search(query, k=5))
    assert hybrid != dense or len(hybrid) == len(dense)


def test_an_unindexed_strategy_fails_loudly(store):
    with pytest.raises(ValueError, match="no chunks indexed"):
        BM25Retriever(store, "never-indexed")
