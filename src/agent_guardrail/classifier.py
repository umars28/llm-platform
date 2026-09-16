"""A learned detector over normalised-text embeddings.

Logistic regression, not a fine-tuned transformer. With eighty labelled samples
a larger model would memorise the corpus and report a number that means nothing;
a linear model on frozen embeddings has few enough parameters that held-out
performance is informative. It is also fast enough to sit in a request path and
small enough to ship in the repository.

Everything reported here is cross-validated on held-out folds. Training and
scoring on the same samples would produce a near-perfect number and no
information, which is the usual way these projects overstate themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .corpus import Sample, split
from .detect import Finding, embed


@dataclass
class TrainedClassifier:
    model: object
    threshold: float

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Probability that each text is an injection attempt."""
        return self.model.predict_proba(embed(list(texts)))[:, 1]

    def scan(self, text: str) -> Finding:
        probability = float(self.score([text])[0])
        if probability < self.threshold:
            return Finding(score=0.0, layer="classifier")
        return Finding(
            score=probability,
            reasons=[f"classifier probability {probability:.2f}"],
            matched=["learned-classifier"],
            layer="classifier",
        )


def train(samples: Sequence[Sample], threshold: float = 0.5) -> TrainedClassifier:
    from sklearn.linear_model import LogisticRegression

    vectors = embed([s.text for s in samples])
    labels = np.array([int(s.is_attack) for s in samples])
    model = LogisticRegression(
        # Swept over 0.01, 0.1, 1 and 10 by held-out AUC; 0.1 and 1.0 tie at
        # 0.890, so the more regularised of the two is kept.
        C=0.1,
        max_iter=2000,
        # The corpora are balanced by construction, but this keeps the model
        # honest if someone adds samples unevenly later.
        class_weight="balanced",
    )
    model.fit(vectors, labels)
    return TrainedClassifier(model=model, threshold=threshold)


@dataclass
class FoldResult:
    fold: int
    detected: int
    attacks: int
    false_positives: int
    benign: int

    @property
    def detection_rate(self) -> float:
        return self.detected / self.attacks if self.attacks else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.benign if self.benign else 0.0


def cross_validate(
    samples: Sequence[Sample], folds: int = 4, threshold: float = 0.5
) -> list[FoldResult]:
    """Train on three quarters, score the held-out quarter, four times over."""
    results = []
    for fold in range(folds):
        train_set, test_set = split(list(samples), fold=fold, folds=folds)
        classifier = train(train_set, threshold)
        probabilities = classifier.score([s.text for s in test_set])

        attacks = [i for i, s in enumerate(test_set) if s.is_attack]
        benign = [i for i, s in enumerate(test_set) if not s.is_attack]
        results.append(
            FoldResult(
                fold=fold,
                detected=sum(1 for i in attacks if probabilities[i] >= threshold),
                attacks=len(attacks),
                false_positives=sum(1 for i in benign if probabilities[i] >= threshold),
                benign=len(benign),
            )
        )
    return results


def summarise(results: Sequence[FoldResult]) -> dict[str, float]:
    detected = sum(r.detected for r in results)
    attacks = sum(r.attacks for r in results)
    false_positives = sum(r.false_positives for r in results)
    benign = sum(r.benign for r in results)
    return {
        "detection_rate": round(detected / attacks, 4) if attacks else 0.0,
        "false_positive_rate": round(false_positives / benign, 4) if benign else 0.0,
        "attacks": attacks,
        "benign": benign,
        "folds": len(results),
    }
