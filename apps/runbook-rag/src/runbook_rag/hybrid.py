"""Lexical search and hybrid fusion.

Dense retrieval fails in a specific, predictable way: it is good at meaning and
bad at exact tokens. An engineer who types `CrashLoopBackOff`, `ImagePullBackOff`
or a flag name wants the document containing that literal string, and an
embedding happily returns something merely topically adjacent instead. BM25 has
the opposite profile.

Fusion is by reciprocal rank rather than by score. The two retrievers produce
scores on incompatible scales -- cosine similarity is bounded, BM25 is not and
varies with corpus statistics -- so any weighted sum of the raw scores is
tuning a constant that has no meaning. RRF only uses positions, which both
rankings agree on.
"""

from __future__ import annotations

import re
from typing import Sequence

from .embedding import embed_queries
from .store import Hit

RRF_K = 60  # standard smoothing constant; damps the influence of the top rank

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, with camel-case split as well as kept.

    `CrashLoopBackOff` is indexed both whole and as crash/loop/back/off, so it
    matches whether the engineer typed the exact identifier or described it.
    """
    lowered = text.lower()
    tokens = _TOKEN.findall(lowered)
    for match in re.findall(r"[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+", text):
        tokens.extend(p.lower() for p in re.findall(r"[A-Z]?[a-z0-9]+", match))
    return tokens


class BM25Retriever:
    """Lexical ranking over the same chunks the vector store holds."""

    def __init__(self, store, strategy: str = "header") -> None:
        from rank_bm25 import BM25Okapi

        self.store = store
        self.strategy = strategy
        self.name = f"bm25/{strategy}"

        rows = store.all_chunks(strategy)
        if not rows:
            raise ValueError(f"no chunks indexed for strategy {strategy!r}")
        self.chunk_ids = [r[0] for r in rows]
        self.doc_ids = [r[1] for r in rows]
        self.index = BM25Okapi([tokenise(r[2]) for r in rows])

    def rank(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.index.get_scores(tokenise(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]

    def search(self, query: str, k: int) -> list[Hit]:
        ranked = self.rank(query, k)
        hydrated = self.store.hydrate([chunk_id for chunk_id, _ in ranked])
        hits = []
        for chunk_id, score in ranked:
            hit = hydrated.get(chunk_id)
            if hit is not None:
                hits.append(
                    Hit(hit.chunk_id, hit.doc_id, hit.doc_title, hit.section,
                        hit.url, hit.content, score)
                )
        return hits


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Combine rankings by position, highest fused score first.

    Position-based fusion is what makes this safe to apply to retrievers whose
    scores are not comparable.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    """Dense and BM25 fused by reciprocal rank."""

    def __init__(self, store, strategy: str = "header", rrf_k: int = RRF_K,
                 candidate_depth: int = 50) -> None:
        self.store = store
        self.strategy = strategy
        self.rrf_k = rrf_k
        # Fuse over more candidates than are returned; a document the dense side
        # ranks 40th can still win once BM25 agrees with it.
        self.candidate_depth = candidate_depth
        self.bm25 = BM25Retriever(store, strategy)
        self.name = f"hybrid/{strategy}"

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        vectors = embed_queries(list(queries))
        out: list[list[Hit]] = []

        for query, vector in zip(queries, vectors):
            dense = self.store.search(vector, k=self.candidate_depth, strategy=self.strategy)
            lexical = self.bm25.rank(query, self.candidate_depth)

            fused = reciprocal_rank_fusion(
                [[h.chunk_id for h in dense], [chunk_id for chunk_id, _ in lexical]],
                k=self.rrf_k,
            )[:k]

            known = {h.chunk_id: h for h in dense}
            missing = [cid for cid, _ in fused if cid not in known]
            known.update(self.store.hydrate(missing))

            out.append(
                [
                    Hit(h.chunk_id, h.doc_id, h.doc_title, h.section, h.url, h.content, score)
                    for cid, score in fused
                    if (h := known.get(cid)) is not None
                ]
            )
        return out
