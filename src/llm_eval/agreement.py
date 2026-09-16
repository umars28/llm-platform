"""Measuring the judge, before trusting it to gate anything.

A judge is a component, and an unmeasured component is a guess. Two properties
are worth knowing before a verdict is allowed to fail a build:

**Discrimination.** Given an answer known to satisfy a criterion and one known
not to, does the judge tell them apart? A judge that passes everything looks
excellent against a suite of good outputs and catches nothing.

**Consistency.** Asked the identical question twice, does it return the identical
verdict? A judge that varies makes every score it produces incomparable to every
other, which is the failure this whole project is organised against.

The negatives are built by damaging a known-good answer in a specific, named way
-- removing the causal link, stripping the evidence, swapping in a plausible but
wrong cause. That is deliberate: a judge that fails only on gibberish has told
you nothing about the failures you will actually see.

**Each damage declares which criterion it actually breaks.** The first version
applied every damage to every criterion and scored a judge 1/4 for verdicts that
were mostly correct: asked "does this state a causal chain", a confidently wrong
answer genuinely does state one, and an answer stripped of numbers still states
one too. The judge was right and the probe set was wrong. A probe whose expected
answer does not follow from the criterion measures the person who wrote it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .judge import Criterion, Judge


@dataclass(frozen=True)
class Probe:
    """One subject with a known expected verdict for one criterion."""

    id: str
    subject: str
    should_pass: bool
    damage: str = "none"


def strip_causal_link(text: str) -> str:
    """Keep every fact, remove the words that join them.

    The hardest negative: nothing is false, the conclusion simply is not drawn.
    """
    for joiner in (" so ", " because ", " which caused ", " causing ", " therefore ",
                   " as a result ", " leading to ", " resulting in "):
        text = text.replace(joiner, ". ")
    return text


def strip_evidence(text: str) -> str:
    """Remove numbers and identifiers, leaving an unsupported assertion."""
    text = re.sub(r"\b\d+(\.\d+)?\s*(ms|s|%|/\d+|gib?|mib?)\b", "some amount", text, flags=re.I)
    text = re.sub(r"\bv\d+\.\d+\.\d+\b", "a recent release", text)
    return re.sub(r"\b\d{2,}\b", "several", text)


def swap_in_wrong_cause(text: str) -> str:
    """Fluent, confident, and about the wrong thing."""
    return (
        "The database was overloaded: Postgres CPU saturation caused every query "
        "to slow down, which is why latency rose across the board. Scaling the "
        "database will resolve it."
    )


@dataclass(frozen=True)
class Damage:
    """A way to break an answer, and what it actually breaks.

    `breaks` names the criteria this damage is a valid negative for. Applying a
    damage to a criterion it does not break produces a probe whose expected
    answer is simply wrong.
    """

    name: str
    apply: Callable[[str], str]
    breaks: frozenset[str]


DAMAGES: tuple[Damage, ...] = (
    Damage("no_causal_link", strip_causal_link,
           frozenset({"joins_cause_to_symptom", "identifies_the_metastable_state",
                      "explains_the_alert_lag", "explains_the_relay_throughput_drop",
                      "separates_volume_from_slowness",
                      "distinguishes_leak_from_undersizing"})),
    Damage("no_evidence", strip_evidence, frozenset({"evidence_is_named"})),
    Damage("wrong_cause", swap_in_wrong_cause,
           frozenset({"evidence_is_named", "identifies_the_metastable_state",
                      "explains_the_alert_lag", "explains_the_relay_throughput_drop",
                      "separates_volume_from_slowness",
                      "distinguishes_leak_from_undersizing"})),
)


def build_probes(
    good_subject: str, criterion_name: str, case_id: str = "probe"
) -> list[Probe]:
    """One positive, plus a negative for every damage that breaks this criterion.

    A criterion with no applicable damage gets only the positive, and that is
    reported honestly rather than padded -- a single-probe result says the judge
    was barely tested, which is information.
    """
    probes = [Probe(f"{case_id}:good", good_subject, should_pass=True)]
    probes += [
        Probe(f"{case_id}:{d.name}", d.apply(good_subject), should_pass=False, damage=d.name)
        for d in DAMAGES
        if criterion_name in d.breaks
    ]
    return probes


@dataclass
class JudgeReport:
    judge_version: str
    criterion: str
    correct: int = 0
    total: int = 0
    mistakes: list[str] = field(default_factory=list)
    inconsistent: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def consistent(self) -> bool:
        return not self.inconsistent

    def explain(self) -> str:
        lines = [
            f"{self.judge_version} on {self.criterion}: "
            f"{self.correct}/{self.total} correct"
        ]
        for mistake in self.mistakes:
            lines.append(f"    wrong: {mistake}")
        for flip in self.inconsistent:
            lines.append(f"    inconsistent: {flip}")
        return "\n".join(lines)


def evaluate_judge(
    judge: Judge,
    criterion: Criterion,
    probes: Sequence[Probe],
    consistency_repeats: int = 2,
) -> JudgeReport:
    """Score the judge against probes whose answers are known.

    Caching is disabled throughout. A cached verdict would make the consistency
    check trivially pass, which would measure the cache rather than the judge.
    """
    report = JudgeReport(judge_version=judge.version, criterion=criterion.name)

    for probe in probes:
        verdict = judge.judge(criterion, probe.subject, use_cache=False)
        report.total += 1
        if verdict.passed == probe.should_pass:
            report.correct += 1
        else:
            expected = "pass" if probe.should_pass else "fail"
            report.mistakes.append(
                f"{probe.id} (damage={probe.damage}) should {expected}: {verdict.reason[:90]}"
            )

        for _ in range(consistency_repeats - 1):
            again = judge.judge(criterion, probe.subject, use_cache=False)
            if again.passed != verdict.passed:
                report.inconsistent.append(probe.id)
                break

    return report
