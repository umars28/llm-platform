"""Evaluation: detection rate, false positive rate, and latency overhead.

Three rules this harness follows, because breaking any of them produces a
flattering number rather than an informative one.

**The threshold is chosen on training folds, never on the test fold.** Sweeping
a threshold on the data you report is how a mediocre detector is made to look
good, and on eighty samples it is very easy to do by accident.

**The heuristic layer is reported with a caveat rather than a number alone.**
The patterns were written by someone who had read the corpus, so its score is
optimistic in a way cross-validation cannot correct. It is shown for shape, not
as a claim about unseen attacks.

**False positives are broken down by the trap that fired them.** A rate says how
often the detector is wrong; the trap says whether it is fixable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .classifier import train
from .corpus import Sample, load_all, split
from .detect import heuristic_scan


@dataclass
class Prediction:
    sample: Sample
    score: float
    fired: bool
    layer: str
    matched: list[str] = field(default_factory=list)


@dataclass
class Report:
    name: str
    predictions: list[Prediction]
    seconds_per_call: float = 0.0
    caveat: str | None = None

    @property
    def attacks(self) -> list[Prediction]:
        return [p for p in self.predictions if p.sample.is_attack]

    @property
    def benign(self) -> list[Prediction]:
        return [p for p in self.predictions if not p.sample.is_attack]

    @property
    def detection_rate(self) -> float:
        hits = [p for p in self.attacks if p.fired]
        return len(hits) / len(self.attacks) if self.attacks else 0.0

    @property
    def false_positive_rate(self) -> float:
        misfires = [p for p in self.benign if p.fired]
        return len(misfires) / len(self.benign) if self.benign else 0.0

    def by_family(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for p in self.attacks:
            hit, total = out.get(p.sample.family or "?", (0, 0))
            out[p.sample.family or "?"] = (hit + int(p.fired), total + 1)
        return dict(sorted(out.items()))

    def false_positives_by_trap(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.benign:
            if p.fired:
                out[p.sample.trap or "?"] = out.get(p.sample.trap or "?", 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def missed(self) -> list[Prediction]:
        return [p for p in self.attacks if not p.fired]


def _pick_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float) -> float:
    """Highest detection rate subject to a false positive ceiling, on train data.

    An operational constraint rather than a statistical one: a detector that
    alerts on more than `max_fpr` of ordinary operational text gets switched off,
    and a detector that is switched off has a detection rate of zero.
    """
    best, best_detection = 0.5, -1.0
    for candidate in np.unique(np.round(scores, 3)):
        fired = scores >= candidate
        fpr = float((fired & (labels == 0)).sum() / max((labels == 0).sum(), 1))
        if fpr > max_fpr:
            continue
        detection = float((fired & (labels == 1)).sum() / max((labels == 1).sum(), 1))
        if detection > best_detection:
            best, best_detection = float(candidate), detection
    return best


def evaluate_heuristic(samples: Sequence[Sample] | None = None) -> Report:
    samples = list(samples or load_all())
    start = time.perf_counter()
    findings = [heuristic_scan(s.text) for s in samples]
    elapsed = (time.perf_counter() - start) / len(samples)

    return Report(
        name="heuristic",
        predictions=[
            Prediction(s, f.score, f.score >= 0.5, "heuristic", f.matched)
            for s, f in zip(samples, findings)
        ],
        seconds_per_call=elapsed,
        caveat=(
            "Patterns were written with the corpus in view, so this is an "
            "upper bound on unseen attacks, not an estimate of them."
        ),
    )


def evaluate_classifier(
    samples: Sequence[Sample] | None = None, folds: int = 4, max_fpr: float = 0.10
) -> Report:
    """Cross-validated, with the threshold fitted inside each training fold."""
    samples = list(samples or load_all())
    predictions: list[Prediction] = []
    durations: list[float] = []

    for fold in range(folds):
        train_set, test_set = split(samples, fold=fold, folds=folds)
        model = train(train_set)

        train_scores = model.score([s.text for s in train_set])
        train_labels = np.array([int(s.is_attack) for s in train_set])
        threshold = _pick_threshold(train_scores, train_labels, max_fpr)

        start = time.perf_counter()
        test_scores = model.score([s.text for s in test_set])
        durations.append((time.perf_counter() - start) / max(len(test_set), 1))

        predictions.extend(
            Prediction(s, float(score), float(score) >= threshold, "classifier")
            for s, score in zip(test_set, test_scores)
        )

    return Report(
        name="classifier",
        predictions=predictions,
        seconds_per_call=float(np.mean(durations)),
    )


def evaluate_combined(
    samples: Sequence[Sample] | None = None, folds: int = 4, max_fpr: float = 0.10
) -> Report:
    """Both layers, taking the stronger signal. The configuration that ships."""
    samples = list(samples or load_all())
    predictions: list[Prediction] = []
    durations: list[float] = []

    for fold in range(folds):
        train_set, test_set = split(samples, fold=fold, folds=folds)
        model = train(train_set)

        combined_train = np.maximum(
            model.score([s.text for s in train_set]),
            np.array([heuristic_scan(s.text).score for s in train_set]),
        )
        threshold = _pick_threshold(
            combined_train, np.array([int(s.is_attack) for s in train_set]), max_fpr
        )

        start = time.perf_counter()
        heuristics = [heuristic_scan(s.text) for s in test_set]
        classifier_scores = model.score([s.text for s in test_set])
        durations.append((time.perf_counter() - start) / max(len(test_set), 1))

        for sample, finding, probability in zip(test_set, heuristics, classifier_scores):
            score = max(finding.score, float(probability))
            predictions.append(
                Prediction(
                    sample, score, score >= threshold,
                    "heuristic" if finding.score >= float(probability) else "classifier",
                    finding.matched,
                )
            )

    return Report(name="combined", predictions=predictions,
                  seconds_per_call=float(np.mean(durations)))


def render(reports: Sequence[Report]) -> str:
    lines = [
        "| detector | detection rate | false positive rate | ms/call |",
        "| --- | --- | --- | --- |",
    ]
    for report in reports:
        lines.append(
            f"| {report.name} | {report.detection_rate:.0%} | "
            f"{report.false_positive_rate:.0%} | "
            f"{report.seconds_per_call * 1000:.1f} |"
        )
    return "\n".join(lines)
