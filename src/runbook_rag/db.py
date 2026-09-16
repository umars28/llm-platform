"""Postgres schema and connection handling.

One table holds chunks and their embeddings together. Keeping text and vector
in the same row is the practical argument for pgvector over a dedicated vector
database at this scale: retrieval returns the passage itself, with no second
lookup and nothing to keep in sync.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from .config import DSN

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id            bigserial PRIMARY KEY,
    chunk_id      text UNIQUE NOT NULL,
    doc_id        text NOT NULL,
    doc_title     text NOT NULL,
    section       text,
    url           text,
    ordinal       int  NOT NULL,
    token_count   int  NOT NULL,
    strategy      text NOT NULL,
    content       text NOT NULL,
    embedding     vector(%(dim)s)
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx      ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy);
"""

# Built after loading, not before: HNSW on an empty table then filled row by row
# is markedly slower to build and gives a worse graph.
HNSW_INDEX = """
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


@contextmanager
def connect(dsn: str = DSN) -> Iterator[psycopg.Connection]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        yield conn


def init_schema(dim: int, dsn: str = DSN) -> None:
    with connect(dsn) as conn:
        conn.execute(SCHEMA % {"dim": dim})


def build_index(dsn: str = DSN) -> None:
    with connect(dsn) as conn:
        conn.execute(HNSW_INDEX)


def stats(dsn: str = DSN) -> dict[str, int]:
    with connect(dsn) as conn:
        row = conn.execute(
            "SELECT count(*), count(DISTINCT doc_id), count(embedding) FROM chunks"
        ).fetchone()
    return {"chunks": row[0], "documents": row[1], "embedded": row[2]}
