from __future__ import annotations

from pathlib import Path

import pytest

from pr_reviewer.extract import CONVENTIONAL, Sample, _keywords, load

CORPUS = Path("corpus/samples.json")
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not extracted")
SAMPLES = load(CORPUS) if CORPUS.exists() else []
DEFECTIVE = [s for s in SAMPLES if s.has_defect]
CLEAN = [s for s in SAMPLES if not s.has_defect]


def test_the_corpus_has_both_classes():
    assert DEFECTIVE and CLEAN


def test_sample_ids_are_unique():
    """One commit can introduce several defects, each fixed separately."""
    ids = [s.id for s in SAMPLES]
    assert len(ids) == len(set(ids))


def test_every_defect_is_actually_present_in_the_diff_under_review():
    """Otherwise recall measures nothing -- the bug would not be on screen."""
    invisible = []
    for s in DEFECTIVE:
        removed = [
            line[1:].strip() for line in s.fix_diff.splitlines()
            if line.startswith("-") and not line.startswith("---") and line[1:].strip()
        ]
        if removed and not any(r in s.diff for r in removed):
            invisible.append(s.id)
    assert invisible == []


def test_defective_samples_carry_their_ground_truth():
    for s in DEFECTIVE:
        assert s.defect_summary and s.fix_commit and s.keywords


def test_clean_samples_carry_no_defect_label():
    for s in CLEAN:
        assert not s.defect_summary and not s.fix_commit and not s.keywords


def test_clean_samples_come_from_files_no_fix_ever_touched():
    fixed = {s.path for s in DEFECTIVE}
    assert not ({s.path for s in CLEAN} & fixed)


def test_negatives_outnumber_positives():
    """Most code is fine; a corpus that forgets this rewards over-reporting."""
    assert len(CLEAN) > len(DEFECTIVE)


def test_tests_are_excluded_from_review():
    assert not any(s.path.startswith("tests/") for s in SAMPLES)


# -- helpers -----------------------------------------------------------

@pytest.mark.parametrize("subject,kind", [
    ("feat: add a thing", "feat"),
    ("fix: correct a thing", "fix"),
    ("fix(scope): correct a thing", "fix"),
])
def test_conventional_commits_are_parsed(subject, kind):
    assert CONVENTIONAL.match(subject).group("type") == kind


def test_unconventional_commits_are_ignored():
    assert CONVENTIONAL.match("updated stuff") is None


def test_keywords_drop_filler_and_keep_content():
    words = _keywords("fix: the sweep is a result only when most of it completed")
    assert "sweep" in words and "completed" in words
    assert "the" not in words and "fix" not in words


def test_keywords_keep_identifiers():
    assert "max_tokens" in _keywords("fix: max_tokens is configurable")


def test_lines_changed_counts_only_real_changes():
    sample = Sample(id="x", repo="r", commit="c", path="p", has_defect=False,
                    diff="--- a/f\n+++ b/f\n@@\n+added\n-removed\n context")
    assert sample.lines_changed == 2
