"""Runbook retrieval: source separation, graceful degradation, relevance gate."""

from __future__ import annotations

import pytest

from ops_copilot import retrieval
from ops_copilot.retrieval import ENVIRONMENT, REFERENCE, keyword_search, search

BOOKS = [
    {"id": "RB-014", "title": "Database connection pool exhaustion",
     "body": "pool saturates and requests time out", "tags": ["database", "pool"]},
    {"id": "RB-006", "title": "Retry storms and cascading failure",
     "body": "clients retry without backoff", "tags": ["retry", 429]},
]


@pytest.fixture(autouse=True)
def clear_probe_cache():
    retrieval.available.cache_clear()
    yield
    retrieval.available.cache_clear()


# -- keyword fallback --------------------------------------------------

def test_keyword_search_ranks_by_token_overlap():
    hits = keyword_search("connection pool exhaustion", BOOKS, limit=2)
    assert hits[0]["id"] == "RB-014"


def test_keyword_search_tolerates_non_string_tags():
    """YAML parses a bare 429 as an int; it must not crash the join."""
    assert keyword_search("retry storms", BOOKS, limit=2)


def test_keyword_search_returns_nothing_rather_than_noise():
    assert keyword_search("entirely unrelated vocabulary here", BOOKS, limit=3) == []


# -- degradation -------------------------------------------------------

def test_without_retrieval_installed_the_tool_still_works(monkeypatch):
    monkeypatch.setenv("OPS_COPILOT_NO_RETRIEVAL", "1")
    out = search("connection pool exhaustion", BOOKS)
    assert out["results"][0]["id"] == "RB-014"
    assert out["results"][0]["retrieval"] == "keyword"
    assert "reference" not in out


def test_a_broken_semantic_backend_falls_back_rather_than_failing(monkeypatch):
    """An investigation must not end because a model failed to load."""
    monkeypatch.setattr(
        retrieval, "available",
        lambda: retrieval.RetrievalBackends(semantic=True, reference=False, detail="test"),
    )
    monkeypatch.setattr(
        retrieval, "semantic_search",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model gone")),
    )
    out = search("connection pool exhaustion", BOOKS)
    assert out["results"][0]["retrieval"] == "keyword"


def test_the_note_explains_why_reference_is_absent(monkeypatch):
    monkeypatch.setenv("OPS_COPILOT_NO_RETRIEVAL", "1")
    assert "no reference corpus" in search("anything", BOOKS)["reference_note"]


# -- source separation -------------------------------------------------

def test_environment_and_reference_are_never_merged(monkeypatch):
    monkeypatch.setattr(
        retrieval, "available",
        lambda: retrieval.RetrievalBackends(semantic=False, reference=True, detail="test"),
    )
    monkeypatch.setattr(
        retrieval, "reference_search",
        lambda q, limit: [{"title": "Pod Lifecycle", "source": REFERENCE, "similarity": 0.8}],
    )
    out = search("connection pool exhaustion", BOOKS)
    assert all(r["source"] == ENVIRONMENT for r in out["results"])
    assert all(r["source"] == REFERENCE for r in out["reference"])
    assert "not authoritative" in out["reference_note"]


def test_an_empty_reference_result_is_explained_not_hidden(monkeypatch):
    """"Nothing relevant" and "corpus missing" must not look the same."""
    monkeypatch.setattr(
        retrieval, "available",
        lambda: retrieval.RetrievalBackends(semantic=False, reference=True, detail="test"),
    )
    monkeypatch.setattr(retrieval, "reference_search", lambda q, limit: [])
    note = search("connection pool exhaustion", BOOKS)["reference_note"]
    assert "not close enough" in note or "close enough" in note
    assert "do not read it as the corpus being unavailable" in note


def test_a_reference_failure_does_not_lose_the_environment_results(monkeypatch):
    monkeypatch.setattr(
        retrieval, "available",
        lambda: retrieval.RetrievalBackends(semantic=False, reference=True, detail="test"),
    )
    monkeypatch.setattr(
        retrieval, "reference_search",
        lambda q, limit: (_ for _ in ()).throw(RuntimeError("postgres down")),
    )
    out = search("connection pool exhaustion", BOOKS)
    assert out["results"][0]["id"] == "RB-014"
    assert "failed" in out["reference_note"]


@pytest.mark.slow
def test_out_of_domain_questions_get_no_reference_documents():
    """The corpus does not cover application concerns; saying so beats guessing."""
    if not retrieval.available().reference:
        pytest.skip("reference corpus not indexed")
    out = search("database connection pool exhausted, requests timing out", BOOKS)
    assert not out.get("reference")


@pytest.mark.slow
def test_kubernetes_questions_do_get_reference_documents():
    if not retrieval.available().reference:
        pytest.skip("reference corpus not indexed")
    out = search("kubelet is evicting pods because the node ran out of memory", BOOKS, limit=3)
    assert out["reference"]
    assert len(out["reference"]) <= 3
    assert len({r["title"] for r in out["reference"]}) == len(out["reference"])
