"""The LLM judge: used only for what deterministic checks cannot decide.

Three constraints make a judge trustworthy enough to gate a merge on.

**It is versioned.** The prompt, the model and the rubric form a version string
that goes into every cached verdict and every report. A judge whose prompt
changed silently is a judge whose scores cannot be compared across runs, and
that is the failure mode that quietly ruins an eval suite.

**It is cached by content.** The cache key is the hash of (judge version, the
question, the thing being judged). Re-running an unchanged case costs nothing
and returns the identical verdict, so a suite stays cheap and its numbers stay
stable between runs of the same commit.

**It is asked a narrow question.** Not "is this good" but "does this answer
name the causal chain, yes or no, and which part is missing". A judge asked to
rate quality on a scale produces a number that drifts; a judge asked a closed
question about presence produces one that can be checked against a human.

The judge is also *measured*: `agreement.py` scores its verdicts against the
deterministic checks on the cases where both apply. A judge that disagrees with
a decidable check is wrong in a knowable way, and that number belongs in the
report next to everything else.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assertions import Check

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("LLM_EVAL_CACHE", REPO_ROOT / ".judge-cache"))

DEFAULT_MODEL = os.environ.get("LLM_EVAL_JUDGE_MODEL", "claude-opus-5")

# Bump when the rubric changes. Every cached verdict carries it, so an old
# verdict can never be silently reused under a new rubric.
RUBRIC_VERSION = "v1"

SYSTEM_PROMPT = """\
You are grading one output against one specific criterion. You are not rating \
quality, helpfulness or style, and you are not being asked whether you would \
have written it differently.

Answer the criterion as asked. If it asks whether something is present, the \
only question is whether it is present -- an output can be clumsy and still \
satisfy it, and it can be fluent and still fail.

Be strict about evidence. If the criterion asks whether a claim is supported, \
an assertion without support does not satisfy it, however confident it sounds.

Return your verdict in the required JSON shape. `reason` must name the specific \
text you based the verdict on, or name what was missing. A reason that merely \
restates the verdict is useless to the person reading the failure.
"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["passed", "reason", "confidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Criterion:
    """One closed question for the judge."""

    name: str
    question: str
    weight: float = 1.0

    def cache_key(self, subject: str, version: str) -> str:
        digest = hashlib.sha256(
            "\x00".join([version, self.name, self.question, subject]).encode()
        ).hexdigest()
        return digest[:32]


@dataclass
class Verdict:
    passed: bool
    reason: str
    confidence: float
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    def to_check(self, name: str, weight: float) -> Check:
        return Check(
            name=name, passed=self.passed, reason=self.reason,
            kind="judge", weight=weight,
        )


class Judge:
    """Asks a model closed questions, caches by content, records its version."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        rubric_version: str = RUBRIC_VERSION,
        cache_dir: Path | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.rubric_version = rubric_version
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self._client = client

    @property
    def version(self) -> str:
        """What every cached verdict and every report is stamped with."""
        return f"{self.model}/{self.rubric_version}"

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    # -- caching -------------------------------------------------------

    def _cache_path(self, criterion: Criterion, subject: str) -> Path:
        return self.cache_dir / f"{criterion.cache_key(subject, self.version)}.json"

    def _read_cache(self, path: Path) -> Verdict | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return Verdict(
            passed=bool(data["passed"]), reason=str(data["reason"]),
            confidence=float(data.get("confidence", 0.0)), cached=True,
        )

    def _write_cache(self, path: Path, verdict: Verdict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "passed": verdict.passed, "reason": verdict.reason,
                    "confidence": verdict.confidence, "judge_version": self.version,
                },
                indent=2,
            )
        )

    # -- judging -------------------------------------------------------

    def _parse(self, text: str) -> dict[str, Any]:
        """Parse the verdict, tolerating a gateway that ignores output_config.

        Structured output is requested, but not every gateway in front of the
        API honours it. Falling back to extracting the first JSON object keeps
        the judge working through a proxy rather than failing on formatting.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"judge returned no JSON object: {text[:200]!r}")
        return json.loads(text[start : end + 1])

    def judge(self, criterion: Criterion, subject: str, use_cache: bool = True) -> Verdict:
        path = self._cache_path(criterion, subject)
        if use_cache:
            cached = self._read_cache(path)
            if cached is not None:
                return cached

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Criterion: {criterion.question}\n\n"
                        f"--- output under test ---\n{subject}\n--- end ---"
                    ),
                }
            ],
        )

        text = next(b.text for b in response.content if b.type == "text")
        data = self._parse(text)
        verdict = Verdict(
            passed=bool(data["passed"]),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        )

        if use_cache:
            self._write_cache(path, verdict)
        return verdict

    def check(self, criterion: Criterion, subject: str, use_cache: bool = True) -> Check:
        return self.judge(criterion, subject, use_cache).to_check(
            criterion.name, criterion.weight
        )
