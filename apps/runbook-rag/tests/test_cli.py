from __future__ import annotations

import pytest

from runbook_rag.cli import _retriever, _store, main
from runbook_rag.hybrid import HybridRetriever
from runbook_rag.rerank import RerankingRetriever
from runbook_rag.retrieve import DenseRetriever


class FakeStore:
    name = "fake"

    def all_chunks(self, strategy):
        return [("c1", "d1", "some indexed content")]


def test_store_flag_selects_the_backend():
    assert _store("pgvector").name == "pgvector"


def test_retriever_names_are_resolved(monkeypatch):
    store = FakeStore()
    assert isinstance(_retriever("dense", store, "header", 30), DenseRetriever)
    assert isinstance(_retriever("hybrid", store, "header", 30), HybridRetriever)


def test_rerank_wraps_hybrid_at_the_requested_depth():
    wrapped = _retriever("rerank", FakeStore(), "header", 42)
    assert isinstance(wrapped, RerankingRetriever)
    assert isinstance(wrapped.base, HybridRetriever)
    assert wrapped.candidate_depth == 42


def test_a_missing_subcommand_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["runbook-rag"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0


def test_unknown_strategy_is_rejected_by_argparse(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runbook-rag", "search", "q", "--strategy", "semantic"])
    with pytest.raises(SystemExit):
        main()
