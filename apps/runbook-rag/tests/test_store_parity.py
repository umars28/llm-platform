"""Both stores must behave identically through the VectorStore interface.

The parity suite is the point of having two backends: anything the pipeline
relies on is asserted against both, so a backend swap cannot quietly change
retrieval behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from runbook_rag.chunking import Chunk
from runbook_rag.store import load
from runbook_rag.store_pg import PgVectorStore
from runbook_rag.store_qdrant import QdrantStore

STRATEGY = "pytest-parity"
OTHER = "pytest-parity-other"
DIM = 8


def chunk(n: int, doc: str, strategy: str = STRATEGY) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}:{strategy}:{n:04d}", doc_id=doc, doc_title=f"Doc {doc}",
        section=f"Section {n}", url=f"https://example/{doc}/", ordinal=n,
        strategy=strategy, content=f"passage {n} of {doc}",
    )


def vector(*weights: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, w in enumerate(weights):
        v[i] = w
    return v / (np.linalg.norm(v) or 1.0)


@pytest.fixture(params=["pgvector", "qdrant"])
def store(request, tmp_path):
    if request.param == "pgvector":
        s = PgVectorStore(table="chunks_parity")
        try:
            s.reset(STRATEGY, dim=DIM)
        except Exception as exc:
            pytest.skip(f"postgres unavailable: {exc}")
    else:
        s = QdrantStore(path=":memory:", collection="parity")

    chunks = [chunk(0, "docA"), chunk(1, "docB"), chunk(2, "docC")]
    vectors = np.vstack([vector(1), vector(0.9, 0.436), vector(0, 1)])
    load(s, chunks, vectors, STRATEGY)
    return s


def test_both_stores_report_what_they_loaded(store):
    assert store.count(STRATEGY) == 3


def test_both_stores_rank_by_cosine_similarity(store):
    hits = store.search(vector(1), k=3, strategy=STRATEGY)
    assert [h.doc_id for h in hits] == ["docA", "docB", "docC"]
    assert hits[0].score > hits[1].score > hits[2].score


def test_both_stores_round_trip_the_payload(store):
    hit = store.search(vector(1), k=1, strategy=STRATEGY)[0]
    assert hit.doc_title == "Doc docA"
    assert hit.section == "Section 0"
    assert hit.url == "https://example/docA/"
    assert hit.content == "passage 0 of docA"


def test_both_stores_respect_k(store):
    assert len(store.search(vector(1), k=2, strategy=STRATEGY)) == 2


def test_both_stores_isolate_strategies(store):
    load(store, [chunk(0, "docZ", OTHER)], np.vstack([vector(1)]), OTHER)
    assert store.count(STRATEGY) == 3
    assert [h.doc_id for h in store.search(vector(1), k=5, strategy=OTHER)] == ["docZ"]


def test_both_stores_are_idempotent_on_reload(store):
    chunks = [chunk(0, "docA"), chunk(1, "docB"), chunk(2, "docC")]
    vectors = np.vstack([vector(1), vector(0.9, 0.436), vector(0, 1)])
    load(store, chunks, vectors, STRATEGY)
    assert store.count(STRATEGY) == 3


def test_both_stores_return_nothing_for_an_unknown_strategy(store):
    assert store.search(vector(1), k=5, strategy="never-loaded") == []


def test_qdrant_point_ids_are_deterministic():
    """Idempotent upserts without keeping a chunk-id-to-uuid mapping table."""
    assert QdrantStore._point_id("abc:header:0001") == QdrantStore._point_id(
        "abc:header:0001"
    )
    assert QdrantStore._point_id("a") != QdrantStore._point_id("b")
