"""Provenance labelling: where a piece of context came from.

The core claim of this project is that a language model reads its context but
not its provenance, so provenance has to be tracked outside the model. Every
segment entering the context carries a `Trust` level, assigned by the caller
based on where the bytes came from -- never inferred from the content, because
inferring trust from content is precisely the thing an attacker attacks.

The capability gate in `policy.py` reads these labels. Detection scores never
raise a trust level; they can only lower it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable


class Trust(enum.IntEnum):
    """Ordered so comparisons read naturally: OPERATOR > USER > UNTRUSTED."""

    UNTRUSTED = 0
    USER = 1
    OPERATOR = 2

    @property
    def label(self) -> str:
        return self.name.lower()


# Where content of each kind comes from. The mapping is deliberately
# conservative: anything an attacker could plausibly influence is UNTRUSTED,
# including sources that feel internal. A log line is written by whoever could
# get a string into a log, which is usually anyone who can send a request.
SOURCE_TRUST: dict[str, Trust] = {
    "system_prompt": Trust.OPERATOR,
    "operator_policy": Trust.OPERATOR,
    "user_message": Trust.USER,
    # Everything below is text the operator did not write.
    "tool_result": Trust.UNTRUSTED,
    "tool_description": Trust.UNTRUSTED,
    "retrieved_document": Trust.UNTRUSTED,
    "log_line": Trust.UNTRUSTED,
    "k8s_object": Trust.UNTRUSTED,
    "http_response": Trust.UNTRUSTED,
    "file_content": Trust.UNTRUSTED,
}


def trust_for(source: str) -> Trust:
    """Trust level for a named source. Unknown sources are untrusted.

    Defaulting an unrecognised source to UNTRUSTED means adding a new channel
    fails closed: it is scanned and it taints the context until someone
    deliberately classifies it otherwise.
    """
    return SOURCE_TRUST.get(source, Trust.UNTRUSTED)


@dataclass(frozen=True)
class Segment:
    """One piece of text entering the context, with where it came from."""

    content: str
    source: str
    origin: str | None = None  # e.g. the service, document or tool name

    @property
    def trust(self) -> Trust:
        return trust_for(self.source)

    @property
    def is_untrusted(self) -> bool:
        return self.trust == Trust.UNTRUSTED

    def describe(self) -> str:
        where = f"{self.source}"
        if self.origin:
            where += f":{self.origin}"
        return f"<{where} trust={self.trust.label}>"


@dataclass
class Context:
    """The accumulating context of one agent turn, with its taint state.

    Taint is monotonic. Once untrusted content has been read, the turn stays
    tainted -- there is no way to unread it, and a model that has seen an
    injection stays influenced by it for the rest of the conversation.
    """

    segments: list[Segment] = field(default_factory=list)
    findings: list[object] = field(default_factory=list)

    def add(self, segment: Segment) -> Segment:
        self.segments.append(segment)
        return segment

    def extend(self, segments: Iterable[Segment]) -> None:
        for segment in segments:
            self.add(segment)

    @property
    def tainted(self) -> bool:
        """Whether any untrusted content has entered this context."""
        return any(s.is_untrusted for s in self.segments)

    @property
    def lowest_trust(self) -> Trust:
        return min((s.trust for s in self.segments), default=Trust.OPERATOR)

    @property
    def untrusted_sources(self) -> list[str]:
        seen: list[str] = []
        for segment in self.segments:
            if segment.is_untrusted:
                key = segment.origin or segment.source
                if key not in seen:
                    seen.append(key)
        return seen

    def record(self, finding: object) -> None:
        self.findings.append(finding)
