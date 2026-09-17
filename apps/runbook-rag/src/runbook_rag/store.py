"""The retrieval interface both vector stores implement.

Two backends exist because job ads ask for both, but the more useful reason is
that writing the second one is what proves the first was not leaking its
assumptions into the rest of the pipeline. Everything above this line -- the
eval, the hybrid retriever, the reranker -- talks only to `VectorStore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str | None
    url: str
    content: str
    score: float

    def __repr__(self) -> str:  # readable in eval output
        where = f"{self.doc_title}"
        if self.section:
            where += f" > {self.section}"
        return f"Hit({self.score:.3f} {where})"


class VectorStore(Protocol):
    name: str

    def reset(self, strategy: str, dim: int) -> None:
        """Drop anything previously indexed for this chunking strategy."""

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Index chunks with their embeddings. Returns the number written."""

    def finalise(self) -> None:
        """Build indexes. Called once after loading, never per batch."""

    def search(self, vector: np.ndarray, k: int, strategy: str) -> list[Hit]:
        """Nearest neighbours by cosine similarity, highest score first."""

    def count(self, strategy: str | None = None) -> int: ...


def load(
    store: VectorStore,
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    strategy: str,
    batch_size: int = 500,
) -> int:
    """Index a full strategy from scratch, then build the index once."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    store.reset(strategy, dim=vectors.shape[1])
    written = 0
    for start in range(0, len(chunks), batch_size):
        stop = start + batch_size
        written += store.upsert(chunks[start:stop], vectors[start:stop])
    store.finalise()
    return written


def unique_documents(hits: Iterable[Hit]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit.doc_id not in seen:
            seen.append(hit.doc_id)
    return seen
