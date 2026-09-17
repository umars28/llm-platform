"""Exporting spans over OTLP, and a receiver to prove they arrive.

The spans this produces are OTLP-shaped, which is what makes them readable by
Langfuse, Jaeger, Grafana Tempo or anything else that speaks the protocol
without per-field configuration.

`LocalReceiver` exists because "it exports correctly" is a claim, and a claim
about a wire format needs something on the other end. It is a few dozen lines of
HTTP server that accepts OTLP-JSON and records what it got, so the export path is
covered by a test rather than by a screenshot of a dashboard.

Conversion is done by hand rather than through the OpenTelemetry SDK. The SDK is
the right dependency for a production service and the wrong one for showing what
the protocol actually is -- and a reader who wants to know whether attributes are
mapped correctly can see it here in twenty lines.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Sequence

from .spans import Span


def _attribute(key: str, value: Any) -> dict[str, Any]:
    """One OTLP key-value. The type tag is part of the wire format."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, (list, tuple)):
        return {"key": key, "value": {"arrayValue": {
            "values": [_attribute("", v)["value"] for v in value]}}}
    return {"key": key, "value": {"stringValue": str(value)}}


SPAN_KINDS = {
    "internal": 1, "server": 2, "client": 3, "producer": 4, "consumer": 5,
}


def to_otlp(spans: Sequence[Span], service: str = "llm-app") -> dict[str, Any]:
    """Convert spans to the OTLP-JSON trace payload."""
    return {
        "resourceSpans": [{
            "resource": {"attributes": [_attribute("service.name", service)]},
            "scopeSpans": [{
                "scope": {"name": "llm-tracing", "version": "0.1.0"},
                "spans": [{
                    "traceId": span.trace_id,
                    "spanId": span.span_id,
                    **({"parentSpanId": span.parent_id} if span.parent_id else {}),
                    "name": span.name,
                    "kind": SPAN_KINDS.get(span.kind, 1),
                    "startTimeUnixNano": str(span.start_ns),
                    "endTimeUnixNano": str(span.end_ns),
                    "attributes": [_attribute(k, v) for k, v in span.attributes.items()],
                    "events": [{
                        "name": e["name"],
                        "timeUnixNano": str(e["timestamp_ns"]),
                        "attributes": [_attribute(k, v) for k, v in e["attributes"].items()],
                    } for e in span.events],
                    # 0 unset, 1 ok, 2 error -- the codes a backend colours on.
                    "status": {
                        "code": {"unset": 0, "ok": 1, "error": 2}.get(span.status, 0),
                        **({"message": span.error} if span.error else {}),
                    },
                } for span in spans],
            }],
        }]
    }


def export(
    spans: Sequence[Span], endpoint: str, service: str = "llm-app",
    headers: dict[str, str] | None = None, timeout: float = 10.0,
) -> int:
    """POST spans to an OTLP-HTTP endpoint. Returns the status code.

    Failure raises rather than warns. A tracing exporter that swallows its own
    errors leaves you believing you have observability that you do not, which is
    worse than having none and knowing it.
    """
    if not spans:
        return 0

    payload = json.dumps(to_otlp(spans, service)).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/traces",
        data=payload,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"otlp endpoint returned {exc.code}: {exc.read()[:200].decode(errors='replace')}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"cannot reach otlp endpoint {endpoint}: {exc}") from exc


# -- a receiver, so the export path is testable -------------------------

@dataclass
class LocalReceiver:
    """An OTLP-JSON endpoint that records what it is sent.

    Enough to prove spans serialise, transmit and arrive intact. Not a tracing
    backend, and not pretending to be one.
    """

    port: int = 0
    received: list[dict[str, Any]] = field(default_factory=list)
    _server: HTTPServer | None = None
    _thread: threading.Thread | None = None

    def __enter__(self) -> "LocalReceiver":
        received = self.received

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    received.append(json.loads(body))
                    self.send_response(200)
                except json.JSONDecodeError:
                    self.send_response(400)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args: Any) -> None:
                return  # keep the test output readable

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def spans(self) -> list[dict[str, Any]]:
        return [
            span
            for payload in self.received
            for resource in payload.get("resourceSpans", [])
            for scope in resource.get("scopeSpans", [])
            for span in scope.get("spans", [])
        ]
