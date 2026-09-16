"""Qdrant backend, implementing the same interface as the pgvector one.

Runs in embedded mode against a local directory, so there is no server to
operate and the comparison costs nothing to reproduce. The client API is the
same one a hosted cluster would use, so moving to a server is a URL change.

Writing this second backend is what proved the first one was not leaking its
assumptions: everything above `VectorStore` works unchanged against either.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from .chunking import Chunk
from .config import REPO_ROOT
from .store import Hit

COLLECTION = "chunks"


class QdrantStore:
    name = "qdrant"

    def __init__(self, path: str | Path | None = None, collection: str = COLLECTION) -> None:
        from qdrant_client import QdrantClient

        # ":memory:" keeps a run entirely ephemeral, which is what the tests want.
        location = str(path) if path is not None else str(REPO_ROOT / ".qdrant")
        self.client = (
            QdrantClient(location=":memory:")
            if location == ":memory:"
            else QdrantClient(path=location)
        )
        self.collection = collection

    # Qdrant needs a UUID or integer id, but chunk ids are readable strings.
    # A deterministic UUID5 keeps upserts idempotent without a mapping table.
    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _ensure(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            if info.config.params.vectors.size == dim:
                return
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def reset(self, strategy: str, dim: int) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._ensure(dim)
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="strategy", match=MatchValue(value=strategy))]
            ),
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=self._point_id(c.chunk_id),
                vector=vector.tolist(),
                payload={
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                    "doc_title": c.doc_title, "section": c.section,
                    "url": c.url, "ordinal": c.ordinal,
                    "strategy": c.strategy, "content": c.content,
                },
            )
            for c, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def finalise(self) -> None:
        """Qdrant builds its HNSW index as it ingests; nothing to do."""

    def search(self, vector: np.ndarray, k: int, strategy: str) -> list[Hit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=k,
            query_filter=Filter(
                must=[FieldCondition(key="strategy", match=MatchValue(value=strategy))]
            ),
            with_payload=True,
        ).points

        return [
            Hit(
                chunk_id=p.payload["chunk_id"], doc_id=p.payload["doc_id"],
                doc_title=p.payload["doc_title"], section=p.payload["section"],
                url=p.payload["url"], content=p.payload["content"],
                score=float(p.score),
            )
            for p in results
        ]

    def count(self, strategy: str | None = None) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self.client.collection_exists(self.collection):
            return 0
        flt = (
            Filter(must=[FieldCondition(key="strategy", match=MatchValue(value=strategy))])
            if strategy
            else None
        )
        return int(self.client.count(self.collection, count_filter=flt).count)
