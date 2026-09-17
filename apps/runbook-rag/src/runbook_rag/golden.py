"""The labelled query set, resolved against the ingested corpus.

Labels are written as repository paths because a path is readable and reviewable
in a diff; doc ids are hashes and nobody can check those by eye. Resolution to
ids happens here, and a path that no longer exists is an error rather than a
silent zero -- upstream documentation moves, and a moved page would otherwise
quietly depress every metric with no indication why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import EVAL_DIR
from .ingest import Document, load_saved

PATH_PREFIX = "content/en/docs/"


class LabelError(ValueError):
    """A golden label points at a document that is not in the corpus."""


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    facet: str
    # doc_id -> grade (2 answers it, 1 supports it)
    relevant: dict[str, int] = field(default_factory=dict)
    paths: dict[str, int] = field(default_factory=dict)

    @property
    def primary(self) -> set[str]:
        return {doc for doc, grade in self.relevant.items() if grade >= 2}

    @property
    def any_relevant(self) -> set[str]:
        return set(self.relevant)


def _index_by_path(docs: list[Document]) -> dict[str, str]:
    return {d.path.removeprefix(PATH_PREFIX): d.doc_id for d in docs}


def load_golden(
    path: Path | None = None, docs: list[Document] | None = None
) -> list[Query]:
    path = path or EVAL_DIR / "golden.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    by_path = _index_by_path(docs if docs is not None else load_saved())

    queries: list[Query] = []
    missing: list[str] = []
    for entry in raw["queries"]:
        resolved: dict[str, int] = {}
        for doc_path, grade in (entry.get("relevant") or {}).items():
            doc_id = by_path.get(doc_path)
            if doc_id is None:
                missing.append(f"{entry['id']} -> {doc_path}")
                continue
            resolved[doc_id] = int(grade)
        queries.append(
            Query(
                id=entry["id"],
                text=entry["query"],
                facet=entry.get("facet", "general"),
                relevant=resolved,
                paths=dict(entry.get("relevant") or {}),
            )
        )

    if missing:
        raise LabelError(
            "golden labels reference documents not in the corpus:\n  "
            + "\n  ".join(missing)
        )
    return queries
