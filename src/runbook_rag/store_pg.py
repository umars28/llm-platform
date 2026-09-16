"""pgvector backend.

Chunk text and its embedding live in the same row, so a search returns the
passage itself with no second lookup and nothing to keep in sync. At this corpus
size that is the honest argument for pgvector over a dedicated vector database:
one system to operate, one backup, one transaction.

The HNSW index is built after loading rather than before. Building it on an
empty table and filling row by row is slower and produces a worse graph.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .chunking import Chunk
from .config import DSN
from .db import connect
from .store import Hit


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str = DSN, ef_search: int = 100) -> None:
        self.dsn = dsn
        self.ef_search = ef_search
        self._dim: int | None = None

    def reset(self, strategy: str, dim: int) -> None:
        self._dim = dim
        with connect(self.dsn) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id          bigserial PRIMARY KEY,
                    chunk_id    text UNIQUE NOT NULL,
                    doc_id      text NOT NULL,
                    doc_title   text NOT NULL,
                    section     text,
                    url         text,
                    ordinal     int  NOT NULL,
                    token_count int  NOT NULL,
                    strategy    text NOT NULL,
                    content     text NOT NULL,
                    embedding   vector(%s)
                )
                """
                % dim
            )
            conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        rows = [
            (
                c.chunk_id, c.doc_id, c.doc_title, c.section, c.url,
                c.ordinal, c.token_count, c.strategy, c.content, v,
            )
            for c, v in zip(chunks, vectors)
        ]
        with connect(self.dsn) as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_id, doc_title, section, url,
                                    ordinal, token_count, strategy, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                    SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                rows,
            )
        return len(rows)

    def finalise(self) -> None:
        with connect(self.dsn) as conn:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy)"
            )
            conn.execute("ANALYZE chunks")

    def search(self, vector: np.ndarray, k: int, strategy: str) -> list[Hit]:
        with connect(self.dsn) as conn:
            # ef_search trades recall against latency at query time; the default
            # of 40 is low enough to lose neighbours on a corpus this size.
            conn.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, doc_title, section, url, content,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                WHERE strategy = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vector, strategy, vector, k),
            ).fetchall()

        return [
            Hit(chunk_id=r[0], doc_id=r[1], doc_title=r[2], section=r[3],
                url=r[4], content=r[5], score=float(r[6]))
            for r in rows
        ]

    def all_chunks(self, strategy: str) -> list[tuple[str, str, str]]:
        """(chunk_id, doc_id, content) for every chunk -- used to build BM25."""
        with connect(self.dsn) as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, content FROM chunks WHERE strategy = %s"
                " ORDER BY doc_id, ordinal",
                (strategy,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def hydrate(self, chunk_ids: Sequence[str]) -> dict[str, Hit]:
        """Fetch full rows for chunk ids produced by a non-vector retriever."""
        if not chunk_ids:
            return {}
        with connect(self.dsn) as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, doc_title, section, url, content
                FROM chunks WHERE chunk_id = ANY(%s)
                """,
                (list(chunk_ids),),
            ).fetchall()
        return {
            r[0]: Hit(chunk_id=r[0], doc_id=r[1], doc_title=r[2], section=r[3],
                      url=r[4], content=r[5], score=0.0)
            for r in rows
        }

    def count(self, strategy: str | None = None) -> int:
        with connect(self.dsn) as conn:
            if strategy:
                row = conn.execute(
                    "SELECT count(*) FROM chunks WHERE strategy = %s", (strategy,)
                ).fetchone()
            else:
                row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0])
