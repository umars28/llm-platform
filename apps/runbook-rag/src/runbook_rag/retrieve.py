"""Retrievers, each exposing the same `search(query, k) -> list[Hit]`.

Hybrid and reranked variants are added in later commits; they compose around
this interface rather than replacing it, so the eval can run any of them without
knowing which is which.

Retrieval returns chunks, but the eval scores documents. `to_documents` does
that collapse in one place: a document's rank is the rank of its best chunk,
duplicates removed, order preserved.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .embedding import embed_queries
from .store import Hit, VectorStore


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def to_documents(hits: Sequence[Hit]) -> list[str]:
    """Collapse a chunk ranking to a document ranking, best chunk first."""
    seen: list[str] = []
    for hit in hits:
        if hit.doc_id not in seen:
            seen.append(hit.doc_id)
    return seen


class DenseRetriever:
    """Plain vector search. The baseline every other retriever is measured against."""

    def __init__(self, store: VectorStore, strategy: str = "header") -> None:
        self.store = store
        self.strategy = strategy
        self.name = f"dense/{store.name}/{strategy}"

    def search(self, query: str, k: int) -> list[Hit]:
        vector = embed_queries([query])[0]
        return self.store.search(vector, k=k, strategy=self.strategy)

    def search_many(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        """Batch the embedding step; it dominates per-query latency."""
        vectors = embed_queries(list(queries))
        return [
            self.store.search(vector, k=k, strategy=self.strategy) for vector in vectors
        ]
