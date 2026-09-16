from __future__ import annotations

import pytest

from runbook_rag.hybrid import RRF_K, reciprocal_rank_fusion, tokenise


# -- tokenisation ------------------------------------------------------

def test_camel_case_identifiers_are_indexed_whole_and_split():
    """Matches whether the engineer pasted the identifier or described it."""
    tokens = tokenise("CrashLoopBackOff")
    assert "crashloopbackoff" in tokens
    assert {"crash", "loop", "back", "off"} <= set(tokens)


def test_ordinary_words_are_lowercased():
    assert tokenise("Pod Is Pending") == ["pod", "is", "pending"]


def test_punctuation_and_symbols_are_dropped():
    assert tokenise("kubectl get pods -o wide") == [
        "kubectl", "get", "pods", "o", "wide",
    ]


def test_digits_survive_tokenisation():
    assert "503" in tokenise("upstream returned 503")


# -- reciprocal rank fusion --------------------------------------------

def test_a_document_both_retrievers_rank_highly_wins():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert fused[0][0] == "a"


def test_agreement_beats_a_single_strong_opinion():
    """The point of fusion: two retrievers agreeing outrank one being certain."""
    fused = dict(reciprocal_rank_fusion([["x", "a"], ["y", "a"]]))
    assert fused["a"] > fused["x"]
    assert fused["a"] > fused["y"]


def test_scores_depend_only_on_position_not_on_the_original_scores():
    """Cosine and BM25 are not comparable; fusion must never assume they are."""
    first = reciprocal_rank_fusion([["a", "b"]])
    second = reciprocal_rank_fusion([["a", "b"]])
    assert first == second
    assert first[0][1] == pytest.approx(1.0 / (RRF_K + 1))


def test_a_document_in_only_one_ranking_still_appears():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]]))
    assert set(fused) == {"a", "b"}


def test_fusing_nothing_returns_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_larger_k_flattens_the_advantage_of_rank_one():
    """rrf_k controls how much the top of each list dominates."""
    sharp = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    flat = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]
