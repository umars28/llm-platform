"""Connection and path settings, all overridable by environment variable.

The DSN is the only thing that differs between a local Homebrew Postgres and
the docker-compose container, so it is the only thing worth configuring.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = Path(os.environ.get("RUNBOOK_RAG_CORPUS", REPO_ROOT / "corpus"))
RAW_DIR = CORPUS_DIR / "raw"
CHUNK_DIR = CORPUS_DIR / "chunks"
EVAL_DIR = Path(os.environ.get("RUNBOOK_RAG_EVAL", REPO_ROOT / "eval"))
RUNS_DIR = REPO_ROOT / "runs"
CACHE_DIR = REPO_ROOT / ".cache"

# Homebrew Postgres trusts the local user; the container wants credentials.
DSN = os.environ.get(
    "RUNBOOK_RAG_DSN",
    f"postgresql:///runbook_rag?host=/tmp&user={os.environ.get('USER', 'postgres')}",
)

EMBEDDING_MODEL = os.environ.get("RUNBOOK_RAG_EMBEDDING", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RUNBOOK_RAG_RERANKER", "BAAI/bge-reranker-base")
