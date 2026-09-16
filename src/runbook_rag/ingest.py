"""Turn a checkout of the Kubernetes documentation into normalised documents.

The corpus is real published documentation rather than synthetic text, which
matters for retrieval evaluation: real docs repeat themselves, contradict older
pages, bury the answer mid-page and share vocabulary across unrelated topics.
Those are the conditions a retriever actually has to work under.

Hugo shortcodes, front matter and code fences are stripped or normalised here
so that chunking downstream sees prose, not template syntax.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import frontmatter

from .config import RAW_DIR

SOURCE_REPO = "https://github.com/kubernetes/website"
SOURCE_LICENCE = "CC BY 4.0"
DOC_URL_BASE = "https://kubernetes.io/docs"

# Hugo templating that carries no meaning once rendered.
_SHORTCODE = re.compile(r"\{\{[%<].*?[%>]\}\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CAPTION = re.compile(r"^\s*\{\{.*$", re.M)
_MULTI_BLANK = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    url: str
    path: str
    content: str

    @property
    def char_count(self) -> int:
        return len(self.content)


def _clean(text: str) -> str:
    text = _HTML_COMMENT.sub("", text)
    text = _SHORTCODE.sub("", text)
    text = _CAPTION.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _doc_url(relative: Path) -> str:
    slug = relative.with_suffix("").as_posix()
    slug = slug.removeprefix("content/en/docs/")
    slug = slug.removesuffix("/_index")
    return f"{DOC_URL_BASE}/{slug}/"


# Pinned so the corpus, and therefore every metric derived from it, is the same
# for anyone who reproduces this. An unpinned clone silently changes the corpus
# under you and makes two runs incomparable.
SOURCE_COMMIT = "main"
SOURCE_PATHS = ["content/en/docs/tasks", "content/en/docs/concepts"]


def fetch_source(target: Path, commit: str = SOURCE_COMMIT) -> Path:
    """Sparse-clone just the documentation directories we index."""
    import subprocess

    target = Path(target)
    if (target / ".git").exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "--branch", commit, SOURCE_REPO, str(target)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "sparse-checkout", "set", *SOURCE_PATHS],
        check=True,
    )
    return target


def load_documents(source_root: Path) -> list[Document]:
    """Read every markdown page under a kubernetes/website checkout."""
    docs: list[Document] = []
    root = Path(source_root)
    for path in sorted(root.rglob("*.md")):
        post = frontmatter.load(path)
        title = str(post.get("title") or "").strip()
        body = _clean(post.content)

        # Section landing pages are navigation, not content worth retrieving.
        if not title or len(body) < 400:
            continue

        relative = path.relative_to(root)
        doc_id = hashlib.sha1(relative.as_posix().encode()).hexdigest()[:12]
        docs.append(
            Document(
                doc_id=doc_id,
                title=title,
                url=_doc_url(relative),
                path=relative.as_posix(),
                content=body,
            )
        )
    return docs


def save_documents(docs: list[Document], target: Path = RAW_DIR) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    out = target / "documents.jsonl"
    with out.open("w") as fh:
        for doc in docs:
            fh.write(json.dumps(asdict(doc)) + "\n")
    (target / "SOURCE.txt").write_text(
        f"source: {SOURCE_REPO}\nlicence: {SOURCE_LICENCE}\ndocuments: {len(docs)}\n"
    )
    return out


def load_saved(target: Path = RAW_DIR) -> list[Document]:
    path = target / "documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `runbook-rag ingest <path-to-kubernetes-website>` first"
        )
    return [Document(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
