"""Judge backends: the hosted API, or a model running locally.

`Judge` takes a client by injection, so a backend only has to expose
`messages.create(...)` and return something with `.content` and `.usage`. That
is the whole contract, and it is why adding a local judge is a small change
rather than a second judge implementation.

A local judge is not equivalent to a frontier one and this module does not
pretend otherwise. What it buys is the ability to develop, debug and regression
test the judged half of a suite with no API key and no per-call cost, and to
keep grading working when a credit balance runs out. Which judge produced a
verdict is recorded in the judge version, so a local score can never be quietly
compared against a hosted one -- the gate fails those as stale.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LOCAL_MODEL = os.environ.get("LLM_EVAL_LOCAL_MODEL", "qwen3:8b")


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _Response:
    content: list[_Block]
    usage: _Usage


class _Messages:
    def __init__(self, outer: "OllamaClient") -> None:
        self._outer = outer

    def create(self, **kwargs: Any) -> _Response:
        return self._outer._create(**kwargs)


class OllamaClient:
    """Minimal adapter presenting the Messages surface `Judge` expects.

    Ollama speaks its own chat API rather than the Messages protocol, so rather
    than pulling in a translation proxy this maps the three fields the judge
    actually uses: system, a single user message, and a JSON-shaped response.

    `format` is set from the requested JSON schema where the server supports it,
    which is a far stronger guarantee than asking a small model to emit JSON and
    hoping. The judge's parser still tolerates prose around the object, because
    not every local model honours the constraint.
    """

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, host: str = OLLAMA_HOST,
                 timeout: float = 180.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.messages = _Messages(self)

    # -- availability ---------------------------------------------------

    def available(self) -> tuple[bool, str]:
        """Whether the server is up and the model is pulled."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            return False, f"no ollama at {self.host}: {exc}"

        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return True, f"{self.model} ready"
        return False, (
            f"{self.model} not pulled; have {', '.join(sorted(names)) or 'nothing'}. "
            f"Run: ollama pull {self.model}"
        )

    # -- the one call the judge makes ------------------------------------

    def _create(self, **kwargs: Any) -> _Response:
        system = kwargs.get("system", "")
        messages = kwargs.get("messages", [])
        prompt = "\n\n".join(
            m["content"] if isinstance(m.get("content"), str) else str(m.get("content"))
            for m in messages
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]
            ),
            "options": {
                # A judge should be as close to deterministic as the runtime
                # allows; a verdict that changes between identical runs is not a
                # measurement.
                "temperature": 0.0,
                "num_predict": int(kwargs.get("max_tokens", 1024)),
            },
        }

        schema = (
            (kwargs.get("output_config") or {}).get("format", {}).get("schema")
        )
        if schema:
            payload["format"] = schema

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"ollama returned {exc.code}: {exc.read()[:200].decode(errors='replace')}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"cannot reach ollama at {self.host}: {exc}") from exc

        text = (body.get("message") or {}).get("content", "")
        return _Response(
            content=[_Block(text=text)],
            usage=_Usage(
                input_tokens=int(body.get("prompt_eval_count", 0) or 0),
                output_tokens=int(body.get("eval_count", 0) or 0),
            ),
        )


def build_client(backend: str = "auto", model: str | None = None):
    """Pick a judge backend.

    "auto" prefers the hosted API and falls back to a local model, because a
    hosted judge is the one whose verdicts are worth baselining. The fallback
    exists so that losing API access degrades the suite rather than stopping it.
    """
    if backend == "local":
        return OllamaClient(model or DEFAULT_LOCAL_MODEL)

    if backend == "api":
        import anthropic

        return anthropic.Anthropic()

    try:
        import anthropic

        client = anthropic.Anthropic()
        if client.api_key or getattr(client, "auth_token", None) or client.auth_headers:
            return client
    except Exception:
        pass
    return OllamaClient(model or DEFAULT_LOCAL_MODEL)
