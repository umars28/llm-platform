from __future__ import annotations

import pytest

from runbook_rag.chunking import (
    STRATEGIES,
    chunk_documents,
    count_tokens,
    header_chunks,
    _sections,
)
from runbook_rag.ingest import Document


def doc(content: str, title: str = "Debugging Pods") -> Document:
    return Document(
        doc_id="abc123", title=title, url="https://example/docs/x/",
        path="x.md", content=content,
    )


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_covers_the_whole_document(strategy):
    """No strategy may silently drop content off the end."""
    body = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_documents([doc(body)], strategy)
    assert chunks
    assert "word999" in chunks[-1].content


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_chunk_ids_are_unique_and_ordinals_are_dense(strategy):
    body = "\n\n".join(f"Paragraph {i}. " + "filler " * 60 for i in range(12))
    chunks = chunk_documents([doc(body)], strategy)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_no_chunk_is_empty(strategy):
    body = "\n\n".join(["# Heading", "", "text " * 300, "", "## Another", "more " * 300])
    for chunk in chunk_documents([doc(body)], strategy):
        assert chunk.content.strip()


def test_headings_inside_code_fences_do_not_start_a_section():
    """A shell comment would otherwise be read as a markdown heading."""
    content = "# Real Heading\n\nprose\n\n```bash\n# restart the pod\nkubectl delete pod x\n```\n\nmore prose"
    headings = [h for h, _ in _sections(content)]
    assert "restart the pod" not in headings
    assert "Real Heading" in headings


def test_header_chunks_carry_document_and_section_context():
    """The prefix is why this strategy retrieves well; assert it is really there."""
    content = "## Connection pool exhaustion\n\n" + "detail " * 120
    chunk = header_chunks(doc(content))[0]
    assert chunk.content.startswith("Debugging Pods — Connection pool exhaustion")
    assert chunk.section == "Connection pool exhaustion"


def test_oversized_code_block_is_split_rather_than_emitted_whole():
    giant = "\n\n" + " ".join(f"tok{i}" for i in range(900)) + "\n\n"
    chunks = chunk_documents([doc(giant)], "recursive")
    assert len(chunks) > 1
    assert all(c.token_count <= 240 for c in chunks)


def test_recursive_prefers_paragraph_boundaries():
    paragraphs = [f"Paragraph number {i} with some words in it." for i in range(6)]
    chunks = chunk_documents([doc("\n\n".join(paragraphs))], "recursive")
    assert len(chunks) == 1
    assert chunks[0].content.count("Paragraph number") == 6


def test_unknown_strategy_is_rejected_by_name():
    with pytest.raises(KeyError, match="unknown strategy"):
        chunk_documents([doc("text")], "semantic")


def test_token_counting_is_whitespace_based():
    assert count_tokens("one two  three\nfour") == 4
    assert count_tokens("") == 0
