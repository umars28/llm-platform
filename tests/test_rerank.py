from __future__ import annotations

import pytest

from runbook_rag.rerank import RerankingRetriever, rerank
from runbook_rag.store import Hit


def hit(doc: str, score: float = 0.5) -> Hit:
    return Hit(f"{doc}:c", doc, f"Doc {doc}", None, "", f"content of {doc}", score)


class FakeBase:
    name = "fake"

    def __init__(self, hits):
        self.hits = hits
        self.k_requested = None

    def search(self, query, k):
        self.k_requested = k
        return self.hits[:k]


@pytest.fixture
def scorer(monkeypatch):
    """Score by position in a fixed preference list, highest first."""
    preference = {"c": 3.0, "a": 2.0, "b": 1.0}

    class Model:
        def predict(self, pairs, show_progress_bar=False):
            return [preference.get(text.split()[-1], 0.0) for _, text in pairs]

    monkeypatch.setattr("runbook_rag.rerank.get_model", lambda name=None: Model())
    return preference


def test_reranking_reorders_by_the_cross_encoder(scorer):
    out = rerank("q", [hit("a"), hit("b"), hit("c")], k=3)
    assert [h.doc_id for h in out] == ["c", "a", "b"]


def test_reranking_truncates_to_k(scorer):
    assert len(rerank("q", [hit("a"), hit("b"), hit("c")], k=2)) == 2


def test_reranked_hits_carry_the_new_score(scorer):
    top = rerank("q", [hit("a", 0.1), hit("c", 0.9)], k=1)[0]
    assert top.doc_id == "c"
    assert top.score == 3.0


def test_reranking_preserves_the_payload(scorer):
    out = rerank("q", [hit("a")], k=1)[0]
    assert out.doc_title == "Doc a"
    assert out.content == "content of a"


def test_reranking_nothing_is_not_a_model_call(monkeypatch):
    monkeypatch.setattr(
        "runbook_rag.rerank.get_model",
        lambda name=None: pytest.fail("model must not load for an empty candidate set"),
    )
    assert rerank("q", [], k=5) == []


def test_candidate_depth_is_never_below_k(scorer):
    """Returning k results from fewer than k candidates is incoherent."""
    base = FakeBase([hit(c) for c in "abc"])
    RerankingRetriever(base, candidate_depth=2).search("q", k=10)
    assert base.k_requested == 10


def test_a_deeper_candidate_set_is_requested_when_configured(scorer):
    base = FakeBase([hit(c) for c in "abc"])
    RerankingRetriever(base, candidate_depth=50).search("q", k=5)
    assert base.k_requested == 50


def test_the_reranker_cannot_promote_what_was_never_retrieved(scorer):
    """The retriever sets the ceiling; this is why candidate depth matters."""
    base = FakeBase([hit("a"), hit("b")])  # "c" would have won, but is absent
    out = RerankingRetriever(base, candidate_depth=2).search("q", k=2)
    assert [h.doc_id for h in out] == ["a", "b"]


def test_name_records_the_configuration(scorer):
    assert RerankingRetriever(FakeBase([]), 30).name == "rerank(fake)@30"
