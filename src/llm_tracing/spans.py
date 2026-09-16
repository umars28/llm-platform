"""Spans for LLM work, with the attributes an LLM trace is actually read for.

Generic tracing answers "what was slow". An LLM trace is opened for different
questions: what did this cost, how many tokens went in, did the cache hit, which
model answered, how many tool calls did the agent make before it committed. Those
are attributes, and a tracer that does not carry them produces a timeline nobody
returns to.

Three decisions that differ from a general-purpose tracer:

**A span is written when it ends, even if it ends badly.** The traces worth
having are of runs that crashed, and a buffered exporter that flushes on clean
shutdown loses exactly those. Spans append to disk as they close.

**Prompts and completions are redacted before they leave the process.** A trace
of an LLM call contains whatever the user typed and whatever the model read,
which on an ops agent is logs and Kubernetes state. Sending that to a tracing
backend moves a security boundary quietly, so content is opt-in and redacted by
default.

**Cost is computed at span close, not left for a dashboard to derive.** A trace
whose cost depends on a query that someone has to write correctly is a trace
whose cost nobody knows.

The semantic conventions follow OpenTelemetry's `gen_ai.*` names so an OTLP
backend -- Langfuse, Jaeger, anything that speaks the protocol -- groups these
without per-field configuration.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# OpenTelemetry semantic conventions for generative AI.
ATTR_SYSTEM = "gen_ai.system"
ATTR_MODEL = "gen_ai.request.model"
ATTR_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_CACHE_READ = "gen_ai.usage.cache_read_input_tokens"
ATTR_CACHE_WRITE = "gen_ai.usage.cache_creation_input_tokens"
ATTR_FINISH = "gen_ai.response.finish_reasons"
ATTR_TOOL = "gen_ai.tool.name"
ATTR_COST = "gen_ai.usage.cost_usd"

PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5": (10.00, 50.00, 12.50, 1.00),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}
DEFAULT_PRICING = PRICING["claude-opus-5"]


def pricing_for(model: str) -> tuple[float, float, float, float]:
    name = (model or "").split("/")[-1].lower()
    if name.endswith(":free"):
        return (0.0, 0.0, 0.0, 0.0)
    return PRICING.get(name, DEFAULT_PRICING)


# Patterns redacted from any content that is recorded. Deliberately broad: a
# missed secret in a trace is permanent, and an over-redacted prompt is merely
# annoying.
_SECRETS = (
    (re.compile(r"\b(sk-ant-|sk-or-v1-|ghp_|gho_|github_pat_)[A-Za-z0-9_\-]{8,}"), "<api-key>"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "<jwt>"),
    (re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\b\s*[:=]\s*\S+"), r"\1=<redacted>"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
)


def redact(text: str) -> str:
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


@dataclass
class Span:
    name: str
    kind: str = "internal"
    trace_id: str = ""
    span_id: str = ""
    parent_id: str | None = None
    start_ns: int = 0
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "unset"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if not self.end_ns:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000_000

    def set(self, key: str, value: Any) -> "Span":
        self.attributes[key] = value
        return self

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(
            {"name": name, "timestamp_ns": time.time_ns(), "attributes": attributes}
        )

    def record_usage(
        self, model: str, input_tokens: int = 0, output_tokens: int = 0,
        cache_read: int = 0, cache_write: int = 0,
    ) -> None:
        """Record tokens and compute cost now, rather than leaving it to a query."""
        price_in, price_out, price_write, price_read = pricing_for(model)
        self.attributes.update({
            ATTR_RESPONSE_MODEL: model,
            ATTR_INPUT_TOKENS: input_tokens,
            ATTR_OUTPUT_TOKENS: output_tokens,
            ATTR_CACHE_READ: cache_read,
            ATTR_CACHE_WRITE: cache_write,
            ATTR_COST: round(
                (input_tokens * price_in + output_tokens * price_out
                 + cache_write * price_write + cache_read * price_read) / 1_000_000,
                8,
            ),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind,
            "trace_id": self.trace_id, "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ns": self.start_ns, "end_ns": self.end_ns,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status, "error": self.error,
            "attributes": self.attributes, "events": self.events,
        }


class Tracer:
    """Writes spans to a JSONL file as they close.

    Append-on-close rather than buffer-and-flush, because the runs worth tracing
    are the ones that die, and a buffer that flushes on clean shutdown loses
    precisely those.
    """

    def __init__(
        self,
        service: str = "llm-app",
        path: str | Path | None = None,
        record_content: bool | None = None,
    ) -> None:
        self.service = service
        self.path = Path(path or os.environ.get("LLM_TRACING_FILE", "traces/spans.jsonl"))
        # Opt-in: an LLM trace carries whatever the user typed and whatever the
        # model read, and exporting that moves a security boundary quietly.
        if record_content is None:
            record_content = os.environ.get("LLM_TRACING_CONTENT", "").lower() in {"1", "true", "yes"}
        self.record_content = record_content
        self._stack: list[Span] = []
        self.finished: list[Span] = []

    # -- ids -----------------------------------------------------------

    @staticmethod
    def _new_id(length: int) -> str:
        return uuid.uuid4().hex[:length]

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    # -- spans ---------------------------------------------------------

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attributes: Any) -> Iterator[Span]:
        parent = self.current
        span = Span(
            name=name, kind=kind,
            trace_id=parent.trace_id if parent else self._new_id(32),
            span_id=self._new_id(16),
            parent_id=parent.span_id if parent else None,
            start_ns=time.time_ns(),
            attributes={"service.name": self.service, **attributes},
        )
        self._stack.append(span)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            if span.status == "unset":
                span.status = "ok"
        finally:
            span.end_ns = time.time_ns()
            self._stack.pop()
            self._write(span)

    @contextmanager
    def llm_call(self, model: str, system: str = "", prompt: str = "") -> Iterator[Span]:
        """A span for one model call, with content redacted unless opted in."""
        attributes: dict[str, Any] = {ATTR_SYSTEM: "anthropic", ATTR_MODEL: model}
        if self.record_content:
            attributes["gen_ai.prompt"] = redact(prompt)
            if system:
                attributes["gen_ai.system_instructions"] = redact(system)
        with self.span(f"chat {model}", kind="client", **attributes) as span:
            yield span

    @contextmanager
    def tool_call(self, name: str, **attributes: Any) -> Iterator[Span]:
        with self.span(f"tool {name}", kind="internal", **{ATTR_TOOL: name, **attributes}) as span:
            yield span

    # -- output --------------------------------------------------------

    def _write(self, span: Span) -> None:
        self.finished.append(span)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(span.to_dict()) + "\n")

    def total_cost(self) -> float:
        return sum(s.attributes.get(ATTR_COST, 0.0) for s in self.finished)

    def total_tokens(self) -> dict[str, int]:
        keys = (ATTR_INPUT_TOKENS, ATTR_OUTPUT_TOKENS, ATTR_CACHE_READ, ATTR_CACHE_WRITE)
        return {k.split(".")[-1]: sum(s.attributes.get(k, 0) for s in self.finished) for k in keys}
