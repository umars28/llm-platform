"""The reviewer, and how its findings are scored.

The prompt is written against one failure mode: a reviewer that comments on
everything. Style notes, speculative refactors and "consider adding a test" are
free to produce, impossible to disagree with, and they bury the one comment that
mattered. Precision is the property that decides whether anyone keeps it turned
on, so the instruction is to report a defect or report nothing.

Scoring is keyword overlap against what the real fix said, which is crude and
deliberately so. A model judging whether a finding "means the same thing" as a
fix message would need its own evaluation before its verdicts counted -- the
same regress this whole set of projects keeps running into. Crude and stable
beats subtle and unaccountable.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .extract import Sample

SYSTEM_PROMPT = """\
You are reviewing a diff for defects. A defect is something that will produce \
wrong behaviour, lose data, or fail in a way the author did not intend.

Report a defect or report nothing. Do not report style, naming, formatting, \
missing tests, or speculative refactors. Do not report that something "could be \
improved" or "might be worth considering". If the diff is correct, say so and \
return an empty list -- that is a complete and useful review.

For each defect, say what input or state triggers it and what goes wrong. A \
finding that cannot name a way to reach the problem is a guess, and guesses cost \
the reader more than they are worth.

Prefer one specific finding to three vague ones. You are being measured on \
whether your findings are real, not on how many you produce.
"""

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "trigger": {"type": "string"},
                    "line_hint": {"type": "string"},
                },
                "required": ["summary", "trigger"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


@dataclass
class Finding:
    summary: str
    trigger: str = ""
    line_hint: str = ""

    @property
    def text(self) -> str:
        return f"{self.summary} {self.trigger} {self.line_hint}".lower()


@dataclass
class Review:
    sample_id: str
    findings: list[Finding] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def reported_a_defect(self) -> bool:
        return bool(self.findings)


_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _json_objects(text: str):
    for match in _FENCED.finditer(text):
        yield match.group(1)
    depth, start = 0, -1
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]


def _parse(text: str) -> list[Finding]:
    """Parse findings, tolerating prose around the object."""
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for candidate in _json_objects(text):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "findings" in parsed:
                data = parsed

    if not isinstance(data, dict) or "findings" not in data:
        raise ValueError(f"no JSON object with findings in review: {text[:160]!r}")

    return [
        Finding(
            summary=str(f.get("summary", "")),
            trigger=str(f.get("trigger", "")),
            line_hint=str(f.get("line_hint", "")),
        )
        for f in data.get("findings", [])
        if f.get("summary")
    ]


def review(client: Any, sample: Sample, model: str, max_tokens: int = 8192) -> Review:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": FINDINGS_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Review this diff from `{sample.path}`.\n\n"
                        f"```diff\n{sample.diff}\n```"
                    ),
                }
            ],
        )
    except Exception as exc:
        return Review(sample_id=sample.id, model=model, error=f"{type(exc).__name__}: {exc}")

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text.strip():
        return Review(
            sample_id=sample.id, model=model,
            error="empty response: the model produced no output, usually a token "
                  "budget consumed by a thinking block before the answer began",
        )
    try:
        findings = _parse(text)
    except ValueError as exc:
        return Review(sample_id=sample.id, model=model, error=str(exc))

    return Review(
        sample_id=sample.id, findings=findings, model=model,
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
    )


_RESET_AT = re.compile(r"resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def seconds_until_quota_reset(error: str) -> float:
    match = _RESET_AT.search(error or "")
    if not match:
        return 0.0
    reset = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.timezone.utc
    )
    return max(0.0, (reset - dt.datetime.now(dt.timezone.utc)).total_seconds())


def review_waiting_for_quota(
    client: Any,
    sample: Sample,
    model: str,
    max_tokens: int = 8192,
    attempts: int = 4,
    on_wait: Callable[[float], None] | None = None,
) -> Review:
    """Review, treating a gateway quota window as something to wait out.

    An SDK retry backs off by seconds; a per-minute quota resets at a stated
    wall-clock time, and the error says when. Retrying before that point spends
    a request to be told the same thing, so this sleeps until the window turns
    over instead. Only rate-limit errors are retried -- everything else is the
    model's answer and is returned as it stands.
    """
    result = review(client, sample, model=model, max_tokens=max_tokens)
    for _ in range(max(0, attempts - 1)):
        if not result.error or "RateLimitError" not in result.error:
            return result
        wait = seconds_until_quota_reset(result.error)
        if wait <= 0:
            return result
        if on_wait is not None:
            on_wait(wait)
        time.sleep(wait + 2.0)
        result = review(client, sample, model=model, max_tokens=max_tokens)
    return result


# -- scoring ------------------------------------------------------------

MATCH_THRESHOLD = 0.5

# Below this share of reviews completing, the rates describe whichever samples
# happened to survive, and are not reported as quality.
MIN_COMPLETION = 0.9


def finding_matches(finding: Finding, keywords: Sequence[str]) -> float:
    """Share of the fix's content words this finding names."""
    if not keywords:
        return 0.0
    text = finding.text
    hit = sum(1 for word in keywords if word in text)
    return hit / len(keywords)


def found_the_defect(review: Review, sample: Sample, threshold: float = MATCH_THRESHOLD) -> bool:
    return any(
        finding_matches(f, sample.keywords) >= threshold for f in review.findings
    )


@dataclass
class Score:
    reviews: list[tuple[Review, Sample]] = field(default_factory=list)

    def add(self, review: Review, sample: Sample) -> None:
        self.reviews.append((review, sample))

    @property
    def defective(self) -> list[tuple[Review, Sample]]:
        return [(r, s) for r, s in self.reviews if s.has_defect]

    @property
    def clean(self) -> list[tuple[Review, Sample]]:
        return [(r, s) for r, s in self.reviews if not s.has_defect]

    @property
    def recall(self) -> float:
        """Share of real defects the reviewer actually named."""
        rows = self.defective
        if not rows:
            return 0.0
        return sum(found_the_defect(r, s) for r, s in rows) / len(rows)

    @property
    def detection_rate(self) -> float:
        """Share of defective diffs where it flagged *something*.

        Reported next to recall because the gap between them is the reviewer
        noticing that a diff is wrong without being able to say why.
        """
        rows = self.defective
        if not rows:
            return 0.0
        return sum(r.reported_a_defect for r, _ in rows) / len(rows)

    @property
    def false_positive_rate(self) -> float:
        """Share of clean diffs that drew a comment. The number that decides adoption."""
        rows = self.clean
        if not rows:
            return 0.0
        return sum(r.reported_a_defect for r, _ in rows) / len(rows)

    @property
    def findings_per_clean_diff(self) -> float:
        rows = self.clean
        if not rows:
            return 0.0
        return sum(len(r.findings) for r, _ in rows) / len(rows)

    @property
    def errors(self) -> int:
        return sum(1 for r, _ in self.reviews if r.error)

    @property
    def completion_rate(self) -> float:
        rows = self.reviews
        if not rows:
            return 0.0
        return sum(1 for r, _ in rows if not r.error) / len(rows)

    @property
    def valid(self) -> bool:
        """Whether the rates describe the corpus rather than the survivors.

        The same check the other projects in this family grew after reporting
        confident percentages for runs that mostly errored. It belongs here too,
        and its absence let a run with 26 failures report "0% recall".
        """
        return self.completion_rate >= MIN_COMPLETION

    def summary(self) -> dict[str, float]:
        return {
            "valid": self.valid,
            "completion_rate": round(self.completion_rate, 4),
            "defective": len(self.defective),
            "clean": len(self.clean),
            "recall": round(self.recall, 4),
            "detection_rate": round(self.detection_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "findings_per_clean_diff": round(self.findings_per_clean_diff, 2),
            "errors": self.errors,
        }

    def missed(self) -> list[Sample]:
        return [s for r, s in self.defective if not found_the_defect(r, s)]

    def noisy(self) -> list[tuple[Sample, list[Finding]]]:
        return [(s, r.findings) for r, s in self.clean if r.reported_a_defect]
