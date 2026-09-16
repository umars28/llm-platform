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

    def __init__(
        self, dsn: str = DSN, ef_search: int = 100, table: str = "chunks"
    ) -> None:
        self.dsn = dsn
        self.ef_search = ef_search
        # Configurable so tests get their own table instead of sharing the
        # indexed corpus, which they would otherwise have to destroy.
        if not table.replace("_", "").isalnum():
            raise ValueError(f"unsafe table name {table!r}")
        self.table = table

    def _column_dim(self, conn) -> int | None:
        row = conn.execute(
            """
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s AND a.attname = 'embedding'
            """,
            (self.table,),
        ).fetchone()
        return int(row[0]) if row and row[0] and row[0] > 0 else None

    def reset(self, strategy: str, dim: int) -> None:
        """Clear one strategy, rebuilding the table if the dimension changed.

        CREATE TABLE IF NOT EXISTS silently ignores a new dimension, so without
        this check a change of embedding model fails later at insert time with
        an error that points at the data rather than the schema.
        """
        with connect(self.dsn) as conn:
            existing = self._column_dim(conn)
            if existing is not None and existing != dim:
                conn.execute(f"DROP TABLE {self.table}")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
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
            conn.execute(
                f"DELETE FROM {self.table} WHERE strategy = %s", (strategy,)
            )

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
                INSERT INTO {table} (chunk_id, doc_id, doc_title, section, url,
                                    ordinal, token_count, strategy, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                    SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """.format(table=self.table),
                rows,
            )
        return len(rows)

    def finalise(self) -> None:
        with connect(self.dsn) as conn:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw
                    ON {t} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """.format(t=self.table)
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_strategy_idx"
                f" ON {self.table} (strategy)"
            )
            conn.execute(f"ANALYZE {self.table}")

    def search(self, vector: np.ndarray, k: int, strategy: str) -> list[Hit]:
        with connect(self.dsn) as conn:
            # ef_search trades recall against latency at query time; the default
            # of 40 is low enough to lose neighbours on a corpus this size.
            conn.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, doc_title, section, url, content,
                       1 - (embedding <=> %s) AS score
                FROM {table}
                WHERE strategy = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """.format(table=self.table),
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
                f"SELECT chunk_id, doc_id, content FROM {self.table}"
                f" WHERE strategy = %s ORDER BY doc_id, ordinal",
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
                FROM {table} WHERE chunk_id = ANY(%s)
                """.format(table=self.table),
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
                    f"SELECT count(*) FROM {self.table} WHERE strategy = %s",
                    (strategy,),
                ).fetchone()
            else:
                row = conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()
        return int(row[0])
