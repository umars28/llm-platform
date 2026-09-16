"""Building a review corpus out of real git history.

The usual way to evaluate a code reviewer is to write some buggy snippets and
check it finds them. That measures whether it can spot a bug someone planted
knowing what they wanted found, which is not the job.

This takes defects from history instead. Every `fix:` commit is a defect that was
real enough for someone to ship a correction. The sample shown to the reviewer is
the diff that *introduced* the lines the fix later removed, found by blaming those
lines -- so the defect is genuinely present in the code under review, in its
natural habitat, with no marker saying "the bug is here".

Locating it that way is not fussiness. Showing the commit immediately before the
fix seems equivalent and is not: that commit often never touched the offending
lines, so the reviewer would be asked to find a defect that is not in front of
it, and the resulting recall figure would measure nothing.

Fixes that only add code are excluded. The defect there is an omission, and
"this diff is missing a case it does not mention" is a different and much harder
task than reviewing code that is present; mixing the two into one recall number
would hide which of them the reviewer can actually do.

Negatives come from `feat:` commits that no later fix ever touched. They matter
more than the positives: a reviewer that comments on everything finds every
defect and is useless, and only clean samples reveal that.

The corpus is extracted rather than committed, so it regenerates as history
grows and never goes stale against the repositories it describes.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

CONVENTIONAL = re.compile(r"^(?P<type>feat|fix|docs|test|refactor|chore)(\(.+?\))?:\s*(?P<subject>.+)$")
# Reviewing a test or a README teaches nothing about finding defects in code.
CODE_SUFFIXES = (".py",)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


@dataclass
class Sample:
    """One region of code to review, and what is known about it."""

    id: str
    repo: str
    commit: str
    path: str
    diff: str
    has_defect: bool
    # Only set for defective samples: what the later fix said it was.
    defect_summary: str = ""
    fix_commit: str = ""
    fix_diff: str = ""
    keywords: list[str] = field(default_factory=list)

    @property
    def lines_changed(self) -> int:
        return sum(
            1 for line in self.diff.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )


def _commits(repo: Path) -> list[tuple[str, str, str]]:
    """(sha, type, subject) for every conventional commit, oldest first."""
    out = []
    for line in _git(repo, "log", "--reverse", "--format=%H%x00%s").splitlines():
        if "\x00" not in line:
            continue
        sha, subject = line.split("\x00", 1)
        match = CONVENTIONAL.match(subject)
        if match:
            out.append((sha, match.group("type"), match.group("subject")))
    return out


def _changed_files(repo: Path, sha: str) -> list[str]:
    return [
        p for p in _git(repo, "show", "--name-only", "--format=", sha).split()
        if p.endswith(CODE_SUFFIXES) and not p.startswith("tests/")
    ]


def _file_diff(repo: Path, sha: str, path: str) -> str:
    return _git(repo, "show", "--format=", "--unified=8", sha, "--", path)


def _blame_origin(repo: Path, sha: str, path: str, fix_diff: str) -> str | None:
    """The commit that introduced the lines this fix removed.

    Blames the parent of the fix, then matches the fix's removed lines against
    it. The commit that contributed the most of them is the one that put the
    defect there.
    """
    removed = {
        line[1:].strip()
        for line in fix_diff.splitlines()
        if line.startswith("-") and not line.startswith("---") and line[1:].strip()
    }
    if not removed:
        return None

    try:
        blame = _git(repo, "blame", "--line-porcelain", f"{sha}^", "--", path)
    except subprocess.CalledProcessError:
        return None

    counts: dict[str, int] = {}
    current: str | None = None
    for line in blame.splitlines():
        if re.match(r"^[0-9a-f]{40} ", line):
            current = line.split()[0]
        elif line.startswith("\t") and current:
            if line[1:].strip() in removed:
                counts[current] = counts.get(current, 0) + 1

    if not counts:
        return None
    return max(counts, key=counts.__getitem__)


def _keywords(subject: str) -> list[str]:
    """Content words from the fix subject, used to score whether a finding matched.

    Deliberately crude. A finding is credited when it names the same thing the
    fix named; judging semantic equivalence would need a model, and a scorer
    that needs a model has the same trust problem as the reviewer it scores.
    """
    # Generic verbs and connectives are dropped as well as articles. A correct
    # finding phrased in its own words would otherwise be penalised for not
    # reusing the fix author's vocabulary -- which measures wording rather than
    # whether the same defect was identified.
    stop = {
        "fix", "the", "a", "an", "is", "are", "was", "were", "not", "no", "and",
        "or", "of", "to", "in", "on", "for", "with", "that", "than", "it", "its",
        "now", "so", "when", "only", "but", "as", "at", "by", "be", "from",
        "does", "do", "did", "this", "these", "those", "if", "into", "they",
        "treat", "make", "makes", "use", "uses", "used", "run", "running", "runs",
        "out", "more", "less", "also", "still", "even", "just", "very", "too",
        "should", "must", "can", "may", "will", "would", "actually", "really",
        "how", "what", "which", "where", "why", "own", "their", "them", "one",
        "another", "other", "some", "any", "all", "both", "each", "most",
    }
    words = re.findall(r"[a-z_][a-z0-9_]{2,}", subject.lower())
    return [w for w in words if w not in stop]


def extract(repo: Path, max_lines: int = 400) -> list[Sample]:
    """Pair each fix with the state it corrected, and collect clean features."""
    repo = Path(repo)
    name = repo.name
    commits = _commits(repo)

    fixed_paths: dict[str, list[tuple[str, str]]] = {}
    for sha, kind, subject in commits:
        if kind != "fix":
            continue
        for path in _changed_files(repo, sha):
            fixed_paths.setdefault(path, []).append((sha, subject))

    samples: list[Sample] = []

    for sha, kind, subject in commits:
        if kind != "fix":
            continue
        for path in _changed_files(repo, sha):
            fix_diff = _file_diff(repo, sha, path)
            origin = _blame_origin(repo, sha, path, fix_diff)
            if origin is None:
                # A pure addition: the defect is an omission, not present code.
                continue

            introduced = _file_diff(repo, origin, path)
            if not introduced.strip() or len(introduced.splitlines()) > max_lines:
                continue
            samples.append(
                Sample(
                    # The fix is part of the id: one commit can introduce
                    # several distinct defects, each corrected separately.
                    id=f"{name}:{origin[:8]}:{Path(path).name}:{sha[:8]}",
                    repo=name, commit=origin[:8], path=path,
                    diff=introduced, has_defect=True,
                    defect_summary=subject, fix_commit=sha[:8],
                    fix_diff=fix_diff,
                    keywords=_keywords(subject),
                )
            )

    for sha, kind, subject in commits:
        if kind != "feat":
            continue
        for path in _changed_files(repo, sha):
            # A feature whose file a later fix touched may well contain that
            # defect already, so it cannot serve as a clean sample.
            if path in fixed_paths:
                continue
            diff = _file_diff(repo, sha, path)
            if not diff.strip() or len(diff.splitlines()) > max_lines:
                continue
            samples.append(
                Sample(
                    id=f"{name}:{sha[:8]}:{Path(path).name}",
                    repo=name, commit=sha, path=path,
                    diff=diff, has_defect=False,
                )
            )

    return samples


def extract_all(repos: Sequence[Path]) -> list[Sample]:
    out: list[Sample] = []
    for repo in repos:
        if (Path(repo) / ".git").exists():
            out.extend(extract(Path(repo)))
    return out


def save(samples: Iterable[Sample], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in samples]
    path.write_text(json.dumps({"samples": rows}, indent=2))
    return path


def load(path: Path) -> list[Sample]:
    data = json.loads(Path(path).read_text())
    return [Sample(**row) for row in data["samples"]]
