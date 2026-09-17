"""Chunking strategies, kept swappable so the choice can be measured.

Chunking decides more of a retriever's quality than the embedding model does,
and the right size is a property of the corpus rather than a constant to copy
from a blog post. Three strategies are implemented so the eval can answer the
question instead of assuming it.

Documentation carries structure that plain text does not -- headings state the
topic, and code blocks are meaningless when split down the middle. `header`
uses both; `fixed` and `recursive` deliberately ignore them, which is what makes
the comparison worth running.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .ingest import Document

# Token counts here are the usual whitespace approximation. The embedding model
# has its own tokenizer, but the ranking between strategies does not change and
# the approximation costs nothing to compute.
_WORD = re.compile(r"\S+")
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.M)
# Hugo heading anchors: "## Pod phase {#pod-phase}". The anchor is markup, and
# it ends up inside the embedded text if it is not removed here.
_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")
_FENCE = re.compile(r"^```", re.M)


def count_tokens(text: str) -> int:
    return len(_WORD.findall(text))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str | None
    url: str
    ordinal: int
    strategy: str
    content: str

    @property
    def token_count(self) -> int:
        return count_tokens(self.content)


def _make(
    doc: Document, strategy: str, ordinal: int, content: str, section: str | None
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:{strategy}:{ordinal:04d}",
        doc_id=doc.doc_id,
        doc_title=doc.title,
        section=section,
        url=doc.url,
        ordinal=ordinal,
        strategy=strategy,
        content=content.strip(),
    )


def _split_into_words(text: str) -> list[str]:
    return _WORD.findall(text)


def fixed_chunks(doc: Document, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Fixed windows over the word stream, ignoring all structure.

    The baseline. Cheap, predictable, and splits mid-sentence and mid-code-block
    without noticing -- which is precisely what the other two try to avoid.
    """
    words = _split_into_words(doc.content)
    step = max(1, size - overlap)
    chunks = []
    for ordinal, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        if len(window) < 20 and chunks:  # trailing scrap
            break
        chunks.append(_make(doc, "fixed", ordinal, " ".join(window), None))
    return chunks


def recursive_chunks(doc: Document, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Split on progressively finer separators, preferring paragraph boundaries.

    Keeps sentences intact where it can, but still has no idea which section a
    passage belongs to.
    """
    paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffered = 0

    def flush() -> None:
        nonlocal buffer, buffered
        if not buffer:
            return
        chunks.append(_make(doc, "recursive", len(chunks), "\n\n".join(buffer), None))
        # Carry the tail forward so a boundary does not orphan its context.
        tail, carried = [], 0
        for para in reversed(buffer):
            tokens = count_tokens(para)
            if carried + tokens > overlap:
                break
            tail.insert(0, para)
            carried += tokens
        buffer, buffered = tail, carried

    for para in paragraphs:
        tokens = count_tokens(para)
        if tokens > size:  # a single oversized paragraph, usually a code block
            flush()
            words = _split_into_words(para)
            for start in range(0, len(words), size):
                chunks.append(
                    _make(doc, "recursive", len(chunks), " ".join(words[start : start + size]), None)
                )
            continue
        if buffered + tokens > size:
            flush()
        buffer.append(para)
        buffered += tokens

    flush()
    return chunks


def _sections(content: str) -> list[tuple[str | None, str]]:
    """Split a markdown document into (heading, body) pairs.

    Headings inside fenced code blocks are not headings; a shell comment of
    `# restart the pod` would otherwise start a new section.
    """
    lines = content.splitlines()
    in_fence = False
    sections: list[tuple[str | None, list[str]]] = [(None, [])]

    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        match = _HEADING.match(line) if not in_fence else None
        if match:
            sections.append((_ANCHOR.sub("", match.group(2)).strip(), []))
        else:
            sections[-1][1].append(line)

    return [(head, "\n".join(body).strip()) for head, body in sections if "".join(body).strip()]


def header_chunks(doc: Document, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Split on markdown headings, then pack sections up to the size budget.

    Each chunk is prefixed with its document title and heading, so an embedding
    of a passage carries the topic it belongs to even when the passage itself
    never names it.

    That prefix was expected to win outright. Measured, it does not: it lifts
    recall@10 by about 8% over fixed windows while costing a little recall@5 and
    MRR. Smaller, more numerous chunks surface more distinct documents deeper in
    the list but dilute the top of it. Which trade is right depends on how many
    passages the consumer can actually read.
    """
    chunks: list[Chunk] = []
    for heading, body in _sections(doc.content):
        label = f"{doc.title} — {heading}" if heading else doc.title
        words = _split_into_words(body)
        if not words:
            continue
        step = max(1, size - overlap)
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if len(window) < 20 and chunks:
                break
            text = f"{label}\n\n{' '.join(window)}"
            chunks.append(_make(doc, "header", len(chunks), text, heading))
    return chunks


STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": fixed_chunks,
    "recursive": recursive_chunks,
    "header": header_chunks,
}


def chunk_documents(
    docs: Iterable[Document], strategy: str = "header", **kwargs
) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; have {sorted(STRATEGIES)}")
    fn = STRATEGIES[strategy]
    return [chunk for doc in docs for chunk in fn(doc, **kwargs)]
