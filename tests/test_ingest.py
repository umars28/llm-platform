from __future__ import annotations

from pathlib import Path

import pytest

from runbook_rag.ingest import Document, _clean, _doc_url, load_documents


def test_hugo_shortcodes_and_comments_are_stripped():
    raw = """Before.

{{< note >}}
This is a Hugo note shortcode.
{{< /note >}}

<!-- an editorial comment -->

After."""
    cleaned = _clean(raw)
    assert "{{<" not in cleaned and "note >}}" not in cleaned
    assert "<!--" not in cleaned
    assert "Before." in cleaned and "After." in cleaned


def test_blank_line_runs_are_collapsed():
    assert _clean("a\n\n\n\n\nb") == "a\n\nb"


def test_doc_url_maps_a_repo_path_to_the_published_page():
    url = _doc_url(Path("content/en/docs/tasks/debug/debug-application.md"))
    assert url == "https://kubernetes.io/docs/tasks/debug/debug-application/"


def test_index_pages_lose_their_index_segment():
    url = _doc_url(Path("content/en/docs/concepts/storage/_index.md"))
    assert url == "https://kubernetes.io/docs/concepts/storage/"


def test_short_and_untitled_pages_are_skipped(tmp_path: Path):
    """Section landing pages are navigation, not retrievable content."""
    (tmp_path / "stub.md").write_text("---\ntitle: Stub\n---\n\ntoo short\n")
    (tmp_path / "untitled.md").write_text("---\nweight: 10\n---\n\n" + "x" * 900)
    (tmp_path / "real.md").write_text("---\ntitle: Real Page\n---\n\n" + "prose. " * 200)

    docs = load_documents(tmp_path)
    assert [d.title for d in docs] == ["Real Page"]


def test_doc_ids_are_stable_and_unique(tmp_path: Path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(f"---\ntitle: {name}\n---\n\n" + "prose. " * 200)

    first = load_documents(tmp_path)
    second = load_documents(tmp_path)
    assert [d.doc_id for d in first] == [d.doc_id for d in second]
    assert len({d.doc_id for d in first}) == 2
