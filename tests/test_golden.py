"""The golden set is the project's most valuable artefact, so it is guarded.

A broken label does not raise at query time -- it scores zero and looks like a
retrieval failure. These tests turn that into a test failure instead.
"""

from __future__ import annotations

from collections import Counter

import pytest
import yaml

from runbook_rag.config import EVAL_DIR
from runbook_rag.golden import LabelError, Query, load_golden
from runbook_rag.ingest import Document, load_saved

RAW = yaml.safe_load((EVAL_DIR / "golden.yaml").read_text())["queries"]
QUERIES = load_golden()


def test_every_label_resolves_to_a_document_in_the_corpus():
    """load_golden raises on a bad path; reaching here means all 143 resolved."""
    assert sum(len(q.relevant) for q in QUERIES) == sum(
        len(e["relevant"]) for e in RAW
    )


def test_a_typo_in_a_label_is_an_error_not_a_silent_zero(tmp_path):
    bad = tmp_path / "golden.yaml"
    bad.write_text(
        "queries:\n"
        "  - id: Q-999\n"
        "    query: anything\n"
        "    facet: test\n"
        "    relevant:\n"
        "      concepts/does-not-exist.md: 2\n"
    )
    with pytest.raises(LabelError, match="not in the corpus"):
        load_golden(bad)


def test_query_ids_are_unique():
    ids = [q.id for q in QUERIES]
    assert len(ids) == len(set(ids))


def test_every_query_has_at_least_one_primary_answer():
    """A query with only grade-1 labels has no right answer to find."""
    without = [q.id for q in QUERIES if not q.primary]
    assert without == []


def test_grades_are_only_one_or_two():
    grades = {g for e in RAW for g in e["relevant"].values()}
    assert grades <= {1, 2}


def test_queries_do_not_simply_restate_the_document_title():
    """A retriever that only works on title-shaped queries has not solved anything.

    Single-word titles are exempt. "Service", "Jobs" and "Volumes" are ordinary
    English that any symptom sentence may contain, so matching one says nothing
    about whether the query was written lazily.
    """
    titles = {d.path.removeprefix("content/en/docs/"): d.title.lower() for d in load_saved()}
    lazy = []
    for entry in RAW:
        text = entry["query"].lower()
        for path in entry["relevant"]:
            title = titles.get(path, "")
            if len(title.split()) > 1 and title in text:
                lazy.append(f"{entry['id']} restates {title!r}")
    assert lazy == []


def test_the_set_spans_many_facets():
    facets = Counter(q.facet for q in QUERIES)
    assert len(facets) >= 20
    # No single facet should dominate, or the headline number measures one topic.
    assert max(facets.values()) <= len(QUERIES) // 4


def test_queries_are_phrased_as_sentences_not_keywords():
    short = [q.id for q in QUERIES if len(q.text.split()) < 5]
    assert short == []


def test_relevance_helpers_agree_with_the_grades():
    q = Query(id="Q", text="t", facet="f", relevant={"a": 2, "b": 1})
    assert q.primary == {"a"}
    assert q.any_relevant == {"a", "b"}
