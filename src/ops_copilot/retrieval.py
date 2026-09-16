"""Runbook retrieval, upgraded from keyword overlap to semantic search.

Two sources, deliberately kept apart:

**Environment runbooks** are the scenario's own operating limits -- connection
budgets, which actions are pre-approved, what this cluster's conventions are.
They are authoritative for this environment and are always searched.

**Reference documentation** is real published Kubernetes documentation, indexed
by the runbook-rag project. It is general background knowledge and is *not*
authoritative about this environment's numbers. Merging the two into one ranked
list would invite exactly the wrong inference -- reading a capacity figure out
of upstream docs as if it were this cluster's budget -- so they are returned as
separate fields and labelled by source.

Both upgrades degrade rather than fail. Semantic search over the environment
runbooks needs only the embedding model, no database. Reference documentation
needs the runbook-rag corpus indexed in Postgres; without it, that field is
simply absent and the agent is told so.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

ENVIRONMENT = "environment-runbook"
REFERENCE = "kubernetes-docs"

# Reference results below this cosine similarity are dropped rather than shown.
# RRF scores cannot be thresholded -- they are positional, so the top hit scores
# about the same whether it is a perfect match or the best of a bad set. Cosine
# against the query is calibrated, and this corpus genuinely does not cover
# application concerns like database pools or retry policy; returning its best
# guess anyway spends the agent's context and invites a wrong inference.
#
# 0.70 was chosen from the measured separation, not picked round: on this corpus
# genuinely relevant passages score 0.78-0.84 and the best available match for
# an out-of-domain question scores 0.63-0.64. Cosine on this model is compressed
# into the upper range, so a floor that looks high is not.
REFERENCE_FLOOR = float(os.environ.get("OPS_COPILOT_REFERENCE_FLOOR", "0.70"))


@dataclass
class RetrievalBackends:
    """What is actually available in this process."""

    semantic: bool
    reference: bool
    detail: str


def _keyword_score(query: str, haystack: str) -> int:
    terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
    tokens = set(re.findall(r"[a-z0-9]+", haystack.lower()))
    return len(terms & tokens)


def _book_text(book: dict[str, Any]) -> str:
    tags = " ".join(str(t) for t in book.get("tags", []))
    return " ".join([book.get("title", ""), book.get("body", ""), tags])


def keyword_search(
    query: str, books: Sequence[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Token overlap. Always available, and the fallback when nothing else is."""
    scored = [
        (score, book)
        for book in books
        if (score := _keyword_score(query, _book_text(book)))
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "id": b.get("id"),
            "title": b.get("title"),
            "body": b.get("body"),
            "score": float(score),
            "source": ENVIRONMENT,
            "retrieval": "keyword",
        }
        for score, b in scored[:limit]
    ]


@lru_cache(maxsize=1)
def _embedder():
    """The embedding functions from runbook-rag, if it is installed."""
    from runbook_rag.embedding import embed_passages, embed_queries

    return embed_queries, embed_passages


def semantic_search(
    query: str, books: Sequence[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Cosine similarity over the environment runbooks.

    The corpus here is a handful of documents, so this runs in memory with no
    index and no database -- the vectors are normalised, which makes cosine a
    single dot product.
    """
    import numpy as np

    embed_queries, embed_passages = _embedder()
    texts = [_book_text(b) for b in books]
    if not texts:
        return []

    scores = embed_passages(texts) @ embed_queries([query])[0]
    order = np.argsort(scores)[::-1][:limit]
    return [
        {
            "id": books[i].get("id"),
            "title": books[i].get("title"),
            "body": books[i].get("body"),
            "score": round(float(scores[i]), 4),
            "source": ENVIRONMENT,
            "retrieval": "semantic",
        }
        for i in order
    ]


def reference_search(query: str, limit: int) -> list[dict[str, Any]]:
    """Query the indexed Kubernetes documentation through runbook-rag.

    Uses the hybrid retriever, which that project measured as the best
    configuration that is fast enough to sit inside an agent loop: reranking
    scores better still but costs about 200x the latency per call.
    """
    from runbook_rag.hybrid import HybridRetriever
    from runbook_rag.store_pg import PgVectorStore

    import numpy as np

    store = PgVectorStore()
    hits = HybridRetriever(store, "header").search(query, k=limit * 4)
    if not hits:
        return []

    # Rank by RRF, gate by cosine: keep the better ordering, get a number that
    # actually means something on its own.
    embed_queries, embed_passages = _embedder()
    query_vector = embed_queries([query])[0]
    similarity = embed_passages([h.content for h in hits]) @ query_vector

    # One entry per document. Several chunks of the same page tell the agent
    # nothing new and crowd out a second perspective on the problem.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit, sim in zip(hits, similarity):
        if float(sim) < REFERENCE_FLOOR or hit.doc_id in seen:
            continue
        if len(out) >= limit:
            break
        seen.add(hit.doc_id)
        out.append(
            {
                "title": hit.doc_title,
                "section": hit.section,
                "url": hit.url,
                "excerpt": " ".join(hit.content.split())[:600],
                "similarity": round(float(sim), 4),
                "source": REFERENCE,
                "retrieval": "hybrid",
            }
        )
    return out


@lru_cache(maxsize=1)
def available() -> RetrievalBackends:
    """Probe once. Disabled wholesale by OPS_COPILOT_NO_RETRIEVAL."""
    if os.environ.get("OPS_COPILOT_NO_RETRIEVAL"):
        return RetrievalBackends(False, False, "disabled by OPS_COPILOT_NO_RETRIEVAL")

    try:
        _embedder()
    except Exception as exc:
        return RetrievalBackends(False, False, f"runbook-rag not installed: {exc}")

    try:
        from runbook_rag.store_pg import PgVectorStore

        indexed = PgVectorStore().count("header")
    except Exception as exc:
        return RetrievalBackends(True, False, f"reference corpus unavailable: {exc}")

    if indexed == 0:
        return RetrievalBackends(True, False, "reference corpus not indexed")
    return RetrievalBackends(True, True, f"{indexed} reference chunks indexed")


def search(
    query: str, books: Sequence[dict[str, Any]], limit: int = 3
) -> dict[str, Any]:
    """Search environment runbooks, and reference documentation when available."""
    backends = available()

    try:
        results = (
            semantic_search(query, books, limit)
            if backends.semantic
            else keyword_search(query, books, limit)
        )
    except Exception:
        # A retrieval failure must not end an investigation; degrade instead.
        results = keyword_search(query, books, limit)

    payload: dict[str, Any] = {"query": query, "results": results}

    if backends.reference:
        try:
            reference = reference_search(query, limit)
            if reference:
                payload["reference"] = reference
                payload["reference_note"] = (
                    "Published Kubernetes documentation, for background only. It "
                    "is not authoritative about this environment's limits or "
                    "conventions -- prefer `results` for anything specific here."
                )
            else:
                payload["reference_note"] = (
                    "No Kubernetes documentation was close enough to this query "
                    "to be worth reading. That is a normal answer for an "
                    "application-level question; do not read it as the corpus "
                    "being unavailable."
                )
        except Exception as exc:
            payload["reference_note"] = f"reference lookup failed: {exc}"
    else:
        payload["reference_note"] = f"no reference corpus ({backends.detail})"

    return payload
