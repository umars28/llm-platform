"""Round-trip tests against a real Postgres.

These use a dedicated strategy name so they never touch indexed corpus data.
"""

from __future__ import annotations

import numpy as np
import pytest

from runbook_rag.chunking import Chunk
from runbook_rag.store import Hit, load, unique_documents
from runbook_rag.store_pg import PgVectorStore

STRATEGY = "pytest-fixture"
DIM = 4


def chunk(n: int, doc: str = "doc1") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}:{STRATEGY}:{n:04d}", doc_id=doc, doc_title=f"Title {doc}",
        section=f"Section {n}", url=f"https://example/{doc}/", ordinal=n,
        strategy=STRATEGY, content=f"content number {n}",
    )


@pytest.fixture(scope="module")
def store() -> PgVectorStore:
    try:
        s = PgVectorStore()
        s.count()
    except Exception as exc:  # no database available
        pytest.skip(f"postgres unavailable: {exc}")
    return s


@pytest.fixture
def loaded(store: PgVectorStore) -> PgVectorStore:
    chunks = [chunk(0), chunk(1), chunk(2, doc="doc2")]
    vectors = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32
    )
    load(store, chunks, vectors, STRATEGY)
    return store


def test_load_writes_every_chunk(loaded: PgVectorStore):
    assert loaded.count(STRATEGY) == 3


def test_search_orders_by_similarity(loaded: PgVectorStore):
    hits = loaded.search(np.array([1, 0, 0, 0], dtype=np.float32), k=3, strategy=STRATEGY)
    assert [h.ordinal if hasattr(h, "ordinal") else h.chunk_id for h in hits]
    assert hits[0].chunk_id.endswith("0000")
    assert hits[0].score > hits[1].score > hits[2].score


def test_search_is_scoped_to_one_strategy(loaded: PgVectorStore):
    hits = loaded.search(np.ones(DIM, dtype=np.float32), k=50, strategy=STRATEGY)
    assert len(hits) == 3


def test_reset_clears_only_its_own_strategy(loaded: PgVectorStore):
    before_other = loaded.count() - loaded.count(STRATEGY)
    loaded.reset(STRATEGY, dim=DIM)
    assert loaded.count(STRATEGY) == 0
    assert loaded.count() == before_other


def test_reloading_the_same_chunks_does_not_duplicate(store: PgVectorStore):
    chunks = [chunk(0), chunk(1)]
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    load(store, chunks, vectors, STRATEGY)
    load(store, chunks, vectors, STRATEGY)
    assert store.count(STRATEGY) == 2


def test_hydrate_returns_rows_for_ids_from_another_retriever(loaded: PgVectorStore):
    hydrated = loaded.hydrate([chunk(0).chunk_id, chunk(1).chunk_id])
    assert set(hydrated) == {chunk(0).chunk_id, chunk(1).chunk_id}
    assert hydrated[chunk(0).chunk_id].doc_title == "Title doc1"


def test_hydrate_of_nothing_is_not_a_query(loaded: PgVectorStore):
    assert loaded.hydrate([]) == {}


def test_mismatched_chunks_and_vectors_are_rejected(store: PgVectorStore):
    with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
        load(store, [chunk(0)], np.zeros((2, DIM), dtype=np.float32), STRATEGY)


def test_unique_documents_preserves_rank_order():
    hits = [
        Hit("c1", "docB", "B", None, "", "", 0.9),
        Hit("c2", "docA", "A", None, "", "", 0.8),
        Hit("c3", "docB", "B", None, "", "", 0.7),
    ]
    assert unique_documents(hits) == ["docB", "docA"]
