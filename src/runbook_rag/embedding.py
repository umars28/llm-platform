"""Local sentence embeddings, cached on disk.

Everything runs on the machine: no API key, no per-token cost, and the same
vectors on every run. That reproducibility is the point -- a retrieval metric is
only comparable across runs if the vectors behind it did not move.

BGE models expect an instruction prefix on the *query* side only. Embedding a
query with the passage prefix, or vice versa, measurably costs recall, and it is
a silent mistake, so the two paths are separate functions.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import CACHE_DIR, EMBEDDING_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model(name: str = EMBEDDING_MODEL):
    """Load lazily -- importing this module should not cost a model load."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(name)
    return _model


def dimension(name: str = EMBEDDING_MODEL) -> int:
    return int(get_model(name).get_sentence_embedding_dimension())


def _cache_path(texts: Sequence[str], name: str, kind: str) -> Path:
    digest = hashlib.sha1((" ".join(texts) + name + kind).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"emb-{kind}-{digest}.pkl"


def _embed(
    texts: Sequence[str], name: str, kind: str, batch_size: int, use_cache: bool
) -> np.ndarray:
    if not texts:
        return np.zeros((0, dimension(name)), dtype=np.float32)

    path = _cache_path(texts, name, kind)
    if use_cache and path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)

    vectors = (
        get_model(name)
        .encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,  # cosine becomes a dot product
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
        )
        .astype(np.float32)
    )

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(vectors, fh)
    return vectors


def embed_passages(
    texts: Iterable[str],
    name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    use_cache: bool = True,
) -> np.ndarray:
    """Embed corpus passages. No prefix -- BGE wants one only on queries."""
    return _embed(list(texts), name, "passage", batch_size, use_cache)


def embed_queries(
    texts: Iterable[str],
    name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    use_cache: bool = True,
) -> np.ndarray:
    """Embed search queries, with the instruction prefix BGE was trained with."""
    prefixed = [QUERY_PREFIX + t for t in texts]
    return _embed(prefixed, name, "query", batch_size, use_cache)
