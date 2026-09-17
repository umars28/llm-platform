"""Loading the labelled red-team and benign corpora.

Two corpora, deliberately the same size. The benign set is not padding: without
hard negatives a detector that flags everything scores a perfect detection rate,
and the first thing it does in production is alert on every stack trace until
someone turns it off.

Each benign sample records the surface feature a naive detector keys on -- an
imperative verb, a base64 blob, the word "system". That label is never shown to
the detector; it exists so a false positive report says *why* something fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"


@dataclass(frozen=True)
class Sample:
    id: str
    text: str
    channel: str
    is_attack: bool
    family: str | None = None  # attack technique
    trap: str | None = None  # benign surface feature

    @property
    def group(self) -> str:
        """Family for attacks, trap for benign. What results are broken down by."""
        return self.family or self.trap or "unlabelled"


def load_attacks(path: Path | None = None) -> list[Sample]:
    raw = yaml.safe_load((path or CORPUS_DIR / "attacks.yaml").read_text())
    return [
        Sample(
            id=entry["id"], text=entry["payload"], channel=entry["channel"],
            is_attack=True, family=entry["family"],
        )
        for entry in raw["attacks"]
    ]


def load_benign(path: Path | None = None) -> list[Sample]:
    raw = yaml.safe_load((path or CORPUS_DIR / "benign.yaml").read_text())
    return [
        Sample(
            id=entry["id"], text=entry["text"], channel=entry["channel"],
            is_attack=False, trap=entry["trap"],
        )
        for entry in raw["benign"]
    ]


def load_all() -> list[Sample]:
    return load_attacks() + load_benign()


def split(samples: list[Sample], fold: int = 0, folds: int = 4) -> tuple[list[Sample], list[Sample]]:
    """Deterministic stratified split for held-out evaluation.

    Stratified by class first, then ordered by group, and assigned round-robin
    from a per-class counter that carries across groups.

    Assigning by position *within* each group is the obvious approach and it is
    wrong here: benign traps have two samples each, so `index % 4` put every
    benign sample in folds 0 and 1 and left folds 2 and 3 with no negatives at
    all. Metrics computed on those folds were undefined, and a false positive
    rate measured against zero negatives is not a low number -- it is not a
    number. Carrying the counter across groups keeps small groups spread.

    The assignment is positional rather than hashed, so it is stable across runs
    and machines without seeding anything.
    """
    by_class: dict[bool, list[Sample]] = {True: [], False: []}
    for sample in sorted(samples, key=lambda s: (s.group, s.id)):
        by_class[sample.is_attack].append(sample)

    train: list[Sample] = []
    test: list[Sample] = []
    for members in by_class.values():
        for position, sample in enumerate(members):
            (test if position % folds == fold else train).append(sample)
    return train, test
