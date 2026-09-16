"""Detection: heuristics, then a learned classifier over embeddings.

Two layers because they fail differently. Patterns catch the literal, cheap and
explainable cases and are trivially evaded by rewording. An embedding classifier
generalises across phrasing and is opaque about why it fired. Running both and
taking the stronger signal costs one model call and covers more than either.

Normalisation happens before both. Obfuscation -- homoglyphs, zero-width joiners,
leetspeak, unicode escapes, base64 -- is not a separate attack family so much as
a wrapper around the others, so decoding first means every later layer sees the
payload rather than its costume.

Neither layer blocks anything on its own. They produce a score and a reason; the
capability gate in `policy.py` is what actually holds when they are wrong.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

# Zero-width and bidirectional control characters: invisible in a log viewer,
# and enough to break a naive substring match.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_UNICODE_ESCAPE = re.compile(r"\\u00([0-9a-fA-F]{2})")
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_WS = re.compile(r"\s+")


def _decode_base64_runs(text: str) -> str:
    """Append the plaintext of any base64 run that decodes to readable text."""
    found = []
    for match in _B64.findall(text):
        try:
            decoded = base64.b64decode(match + "=" * (-len(match) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            readable = decoded.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # Only keep it if it looks like prose rather than binary noise.
        if sum(c.isalpha() or c.isspace() for c in readable) / max(len(readable), 1) > 0.8:
            found.append(readable)
    return " ".join(found)


def normalise(text: str) -> str:
    """Strip the costume: unicode tricks, escapes, leetspeak, base64.

    NFKC folds full-width and other compatibility forms onto their plain
    equivalents, which handles the homoglyph family without a lookup table.
    """
    text = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)

    decoded = _decode_base64_runs(text)
    if decoded:
        text = f"{text} {decoded}"

    lowered = text.lower()
    # Keep both spellings: leet folding helps "1gn0r3" but would corrupt a
    # legitimate "v2.31.0" or "10.4.3.88" if it were the only form kept.
    return _WS.sub(" ", f"{lowered} {lowered.translate(_LEET)}").strip()


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    weight: float
    why: str


def _p(name: str, source: str, weight: float, why: str) -> Pattern:
    return Pattern(name, re.compile(source, re.I | re.S), weight, why)


# Weights are additive and capped at 1.0. They encode how much a phrase is worth
# on its own: "ignore previous instructions" is close to conclusive, while a URL
# with a query parameter is only suspicious next to something else.
PATTERNS: tuple[Pattern, ...] = (
    _p("override-instructions",
       r"(ignore|disregard|forget|override)\b[^.]{0,40}\b(previous|prior|earlier|above|all)\b[^.]{0,30}\b(instruction|directive|prompt|rule|constraint)",
       0.85, "asks the model to discard earlier instructions"),
    _p("new-system-directive",
       r"(new|updated)\s+(system|operator)\s+(directive|instruction|message|prompt)",
       0.8, "claims to be a fresh operator instruction"),
    _p("instruction-to-ai",
       r"(instruction|note|message)\s+(for|to)\s+(any\s+)?(ai|assistant|agent|model|llm)\b",
       0.75, "addresses the model directly from inside data"),
    _p("chat-role-marker",
       r"(<\|im_(start|end)\|>|\[/?INST\]|^\s*(human|assistant|system)\s*:|\\n\\nhuman:)",
       0.8, "forges a conversation role marker"),
    _p("pseudo-system-block",
       r"\[\[?\s*(begin|end)\s+(of\s+)?(system|tool|operator)[^\]]*\]\]?|</?(system|tool_result|tool_output)>",
       0.7, "forges a section delimiter to escape its own block"),
    _p("privilege-escalation",
       r"(you\s+(are\s+now|have\s+been)|permissions?\s+(have\s+been|are))\s+[^.]{0,40}(unrestricted|escalat|elevat|admin|root|full\s+(cluster\s+)?privile)",
       0.8, "asserts the model has gained privileges"),
    _p("approval-bypass",
       r"(without|skip|bypass|suspend|disable|no\s+need\s+for)\s+[^.]{0,30}(human\s+)?(approval|confirmation|review|change\s+request|gate)",
       0.8, "tells the model to bypass the approval gate"),
    _p("act-autonomously",
       r"(act|proceed|execute|apply|run)\s+(autonomously|directly|immediately|without\s+asking)",
       0.6, "pushes the model to act without checking"),
    _p("pre-authorised-claim",
       r"(pre.?(approved|authoris|authoriz)|already\s+(been\s+)?approved|authorised\s+to\s+skip)",
       0.55, "claims an approval that did not come from the approval service"),
    _p("exfiltrate-prompt",
       r"(system\s+prompt|operator\s+policy|conversation\s+so\s+far|your\s+context)\b[^.]{0,60}(append|include|echo|reveal|send|summaris|summariz|verbatim|post)",
       0.85, "asks the model to disclose its own context"),
    _p("exfiltrate-secrets",
       r"(credential|api\s+key|secret|token|password)\w*\b[^.]{0,60}(include|send|post|echo|reveal|summaris|summariz|collect)",
       0.75, "asks the model to disclose credentials"),
    _p("callback-url",
       r"https?://[^\s\"'<>]*[?&][^\s\"'<>]*=\s*(your|the)?\s*(system|prompt|data|context|conversation)",
       0.8, "URL shaped to carry context out"),
    _p("later-turn-trigger",
       r"(from\s+now\s+on|for\s+the\s+(remainder|rest)\s+of|at\s+any\s+later\s+point|when(ever)?\s+you\s+(are\s+)?(next|later))",
       0.55, "plants an instruction for a later turn"),
    _p("conceal",
       r"(do\s+not|don't|never)\s+(mention|reveal|disclose|report|tell|include)\s+[^.]{0,40}(this|it)\b",
       0.7, "asks the model to conceal something"),
    _p("trust-elevation",
       r"treat\s+(the\s+)?(following|this|every|all)\b[^.]{0,50}(as|like)\s+[^.]{0,30}(trusted|operator|system|higher.priority|authoritative)",
       0.8, "tries to relabel untrusted content as trusted"),
)


@dataclass
class Finding:
    score: float
    reasons: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    layer: str = "heuristic"

    @property
    def triggered(self) -> bool:
        return self.score > 0.0

    def merge(self, other: "Finding") -> "Finding":
        """Strongest signal wins; both explanations are kept."""
        if other.score <= self.score:
            stronger, weaker = self, other
        else:
            stronger, weaker = other, self
        return Finding(
            score=stronger.score,
            reasons=stronger.reasons + weaker.reasons,
            matched=stronger.matched + weaker.matched,
            layer=stronger.layer if stronger.score > weaker.score else "combined",
        )


def heuristic_scan(text: str) -> Finding:
    """Pattern match over normalised text. Cheap, explainable, evadable."""
    normalised = normalise(text)
    score = 0.0
    reasons: list[str] = []
    matched: list[str] = []

    for pattern in PATTERNS:
        if pattern.regex.search(normalised):
            score += pattern.weight
            matched.append(pattern.name)
            reasons.append(pattern.why)

    return Finding(score=min(score, 1.0), reasons=reasons, matched=matched)


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed(texts: Sequence[str]):
    """Embed normalised text, so the classifier never sees the obfuscation."""
    return _embedder().encode(
        [normalise(t) for t in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
