from __future__ import annotations

import numpy as np
import pytest

from runbook_rag import embedding as emb


def test_query_and_passage_paths_are_not_interchangeable(monkeypatch):
    """BGE prefixes queries only; mixing the two silently costs recall."""
    seen = {}

    def fake(texts, name, kind, batch_size, use_cache):
        seen[kind] = list(texts)
        return np.zeros((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(emb, "_embed", fake)
    emb.embed_queries(["crashloop"])
    emb.embed_passages(["crashloop"])

    assert seen["query"] == [emb.QUERY_PREFIX + "crashloop"]
    assert seen["passage"] == ["crashloop"]


def test_cache_key_separates_queries_from_passages():
    same_text = ["identical text"]
    assert emb._cache_path(same_text, "m", "query") != emb._cache_path(
        same_text, "m", "passage"
    )


def test_cache_key_changes_with_the_model():
    texts = ["text"]
    assert emb._cache_path(texts, "model-a", "passage") != emb._cache_path(
        texts, "model-b", "passage"
    )


def test_empty_input_returns_a_correctly_shaped_array(monkeypatch):
    monkeypatch.setattr(emb, "dimension", lambda name=None: 384)
    out = emb.embed_passages([])
    assert out.shape == (0, 384)


@pytest.mark.slow
def test_embeddings_separate_relevant_from_irrelevant_text():
    query = emb.embed_queries(["how do I debug a crashlooping pod"])
    passages = emb.embed_passages(
        [
            "To debug a CrashLoopBackOff pod, inspect its logs with kubectl logs.",
            "Ingress exposes HTTP routes from outside the cluster to services.",
        ]
    )
    relevant, unrelated = (query @ passages.T)[0]
    assert relevant > unrelated + 0.2


@pytest.mark.slow
def test_vectors_are_normalised_so_cosine_is_a_dot_product():
    vectors = emb.embed_passages(["one", "two"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
