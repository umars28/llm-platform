"""Cross-encoder reranking of a candidate set.

A bi-encoder embeds query and passage separately, which is what makes searching
millions of vectors possible but also means the two never actually meet -- the
model scores a similarity between two independent summaries. A cross-encoder
reads the pair together and can tell that a passage mentioning the right words
does not answer the question.

That is far too slow to run over a corpus, so it runs over the top candidates a
cheap retriever already found. The retriever decides what is reachable; the
reranker decides what is on top. Recall@k for large k is the ceiling this can
work within -- a reranker cannot promote a document that was never retrieved,
which is why the candidate depth matters more than it looks.

Runs locally on CPU, so it costs latency rather than money.
"""

from __future__ import annotations

from typing import Sequence

from .config import RERANK_MODEL
from .store import Hit

_model = None


def get_model(name: str = RERANK_MODEL):
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(name)
    return _model


def rerank(
    query: str, hits: Sequence[Hit], k: int, model_name: str = RERANK_MODEL
) -> list[Hit]:
    """Rescore candidates by reading each against the query, best first."""
    if not hits:
        return []

    scores = get_model(model_name).predict(
        [(query, hit.content) for hit in hits], show_progress_bar=False
    )
    ordered = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [
        Hit(h.chunk_id, h.doc_id, h.doc_title, h.section, h.url, h.content, float(s))
        for h, s in ordered[:k]
    ]


class RerankingRetriever:
    """Wraps any retriever: fetch a deep candidate set, then reorder it.

    `candidate_depth` is the knob that matters. Too shallow and the reranker
    has nothing to find; too deep and latency grows linearly for diminishing
    returns, because the cross-encoder runs once per candidate.
    """

    def __init__(self, base, candidate_depth: int = 30, model_name: str = RERANK_MODEL) -> None:
        self.base = base
        self.candidate_depth = candidate_depth
        self.model_name = model_name
        self.name = f"rerank({base.name})@{candidate_depth}"

    def search(self, query: str, k: int) -> list[Hit]:
        candidates = self.base.search(query, k=max(self.candidate_depth, k))
        return rerank(query, candidates, k, self.model_name)

    def search_many(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        depth = max(self.candidate_depth, k)
        if hasattr(self.base, "search_many"):
            batches = self.base.search_many(list(queries), k=depth)
        else:
            batches = [self.base.search(q, k=depth) for q in queries]
        return [
            rerank(query, hits, k, self.model_name)
            for query, hits in zip(queries, batches)
        ]
