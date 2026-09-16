from __future__ import annotations

import math

import pytest

from runbook_rag.golden import Query
from runbook_rag.metrics import (
    aggregate,
    by_facet,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)


def q(**kw) -> Query:
    base = dict(id="Q-1", text="a symptom sentence", facet="f", relevant={"a": 2, "b": 1})
    return Query(**(base | kw))


# -- recall -----------------------------------------------------------

def test_recall_counts_relevant_documents_inside_the_window():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=3) == 1.0
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5


def test_recall_ignores_duplicates_in_the_ranking():
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5


def test_recall_without_labels_is_zero_not_a_crash():
    assert recall_at_k(["a"], set(), k=5) == 0.0


# -- hit rate ---------------------------------------------------------

def test_hit_is_binary_regardless_of_how_many_matched():
    assert hit_at_k(["x", "a"], {"a", "b"}, k=5) == 1.0
    assert hit_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_hit_respects_the_cutoff():
    assert hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


# -- MRR --------------------------------------------------------------

@pytest.mark.parametrize("rank,expected", [(1, 1.0), (2, 0.5), (3, 1 / 3), (4, 0.25)])
def test_reciprocal_rank_is_one_over_the_first_correct_position(rank, expected):
    ranking = ["x"] * (rank - 1) + ["a"]
    assert reciprocal_rank(ranking, {"a"}) == pytest.approx(expected)


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_found():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_only_the_first_relevant_hit_counts():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


# -- nDCG -------------------------------------------------------------

def test_ndcg_is_one_for_the_ideal_ranking():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, k=5) == pytest.approx(1.0)


def test_ndcg_penalises_putting_the_weaker_answer_first():
    grades = {"a": 2, "b": 1}
    ideal = ndcg_at_k(["a", "b"], grades, k=5)
    swapped = ndcg_at_k(["b", "a"], grades, k=5)
    assert swapped < ideal


def test_ndcg_uses_the_grades_not_just_membership():
    """Two rankings with identical recall must differ when the grades differ."""
    grades = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "x"], grades, k=5) > ndcg_at_k(["b", "x"], grades, k=5)


def test_ndcg_discount_matches_the_log2_definition():
    got = ndcg_at_k(["x", "a"], {"a": 2}, k=5)
    assert got == pytest.approx((2 / math.log2(3)) / (2 / math.log2(2)))


def test_ndcg_without_grades_is_zero():
    assert ndcg_at_k(["a"], {}, k=5) == 0.0


# -- scoring and aggregation ------------------------------------------

def test_score_query_reports_every_cutoff():
    result = score_query(q(), ["a", "z", "b"])
    assert result.recall_at_1 == 0.5
    assert result.recall_at_3 == 1.0
    assert result.mrr == 1.0
    assert result.hit_at_5 == 1.0
    assert not result.found_nothing


def test_a_query_that_found_nothing_is_flagged():
    assert score_query(q(), ["x", "y", "z"]).found_nothing


def test_retrieved_list_is_capped_for_the_record():
    result = score_query(q(), [f"d{i}" for i in range(50)])
    assert len(result.retrieved) == 10


def test_aggregate_macro_averages_so_every_query_counts_equally():
    good = score_query(q(), ["a", "b"])
    bad = score_query(q(), ["x", "y"])
    summary = aggregate([good, bad])
    assert summary["queries"] == 2
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["found_nothing"] == 1


def test_aggregate_of_nothing_does_not_divide_by_zero():
    assert aggregate([])["queries"] == 0


def test_by_facet_splits_results_without_losing_any():
    results = [
        score_query(q(facet="dns"), ["a"]),
        score_query(q(facet="storage"), ["x"]),
    ]
    facets = by_facet(results)
    assert set(facets) == {"dns", "storage"}
    assert sum(f["queries"] for f in facets.values()) == 2
