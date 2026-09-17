"""Request identity, structured logs, and shutting down without dropping work.

Three things that are invisible until an incident, and then are the only things
that matter.

**A request id on everything.** Without one, "a tenant reported a failure at
14:32" is answered by reading two replicas' logs and guessing. The id is taken
from the caller's header when they send one, so a trace that starts in their
service continues through ours rather than restarting at our edge.

**Logs as JSON, one object per line.** Grep works on prose exactly until the
moment someone needs to group by tenant, and then it does not. The fields are
fixed so a query written during one incident still works during the next.

**Draining before exit.** A rolling update sends SIGTERM and Kubernetes removes
the pod from the Service *concurrently*, not before -- so a pod that exits the
instant it is signalled drops requests that were already routed to it. The delay
here is not politeness; it is the window in which endpoint propagation catches
up. Long LLM calls make this worse than usual: a request in flight may have
thirty seconds left to run, and killing it wastes tokens the tenant has already
been charged for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

# Carried through the request without threading it as an argument, so a log line
# written deep in the call stack still knows which request it belongs to.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
tenant_var: ContextVar[str] = ContextVar("tenant", default="-")

REQUEST_ID_HEADERS = ("x-request-id", "x-correlation-id", "traceparent")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with fixed field names."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "tenant": tenant_var.get(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel((level or os.environ.get("LOG_LEVEL", "info")).upper())

    # uvicorn's own access log duplicates what the middleware records, in a
    # different shape. Two formats for the same event is worse than one.
    for name in ("uvicorn.access",):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = False

    return logging.getLogger("gateway")


def log(logger: logging.Logger, level: str, message: str, **fields: Any) -> None:
    logger.log(getattr(logging, level.upper()), message, extra={"fields": fields})


def extract_request_id(headers: Any) -> str:
    """Continue the caller's trace where there is one, or start a new one."""
    for header in REQUEST_ID_HEADERS:
        value = headers.get(header)
        if value:
            # A W3C traceparent is `version-traceid-spanid-flags`; the trace id
            # is the part worth carrying.
            if header == "traceparent" and value.count("-") >= 3:
                return value.split("-")[1]
            return value[:128]
    return uuid.uuid4().hex


class Lifecycle:
    """Tracks in-flight work so shutdown can wait for it.

    `drain_seconds` covers the gap between SIGTERM and the pod leaving the
    Service's endpoints. `grace_seconds` then bounds how long to wait for work
    that is already running, because a request that hangs must not hold the
    rollout open forever.
    """

    def __init__(self, drain_seconds: float | None = None, grace_seconds: float | None = None) -> None:
        self.drain_seconds = (
            drain_seconds if drain_seconds is not None
            else float(os.environ.get("SHUTDOWN_DRAIN_SECONDS", "5"))
        )
        self.grace_seconds = (
            grace_seconds if grace_seconds is not None
            else float(os.environ.get("SHUTDOWN_GRACE_SECONDS", "30"))
        )
        self.shutting_down = False
        self.in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    def enter(self) -> None:
        self.in_flight += 1
        self._idle.clear()

    def leave(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        if self.in_flight == 0:
            self._idle.set()

    async def drain(self, logger: logging.Logger | None = None) -> None:
        self.shutting_down = True
        if logger:
            log(logger, "info", "draining", drain_seconds=self.drain_seconds,
                in_flight=self.in_flight)

        # Readiness is already failing by now; this is the window for the
        # endpoint removal to propagate to every kube-proxy.
        await asyncio.sleep(self.drain_seconds)

        try:
            await asyncio.wait_for(self._idle.wait(), timeout=self.grace_seconds)
        except asyncio.TimeoutError:
            if logger:
                log(logger, "warning", "shutdown grace expired with work in flight",
                    in_flight=self.in_flight)
        if logger:
            log(logger, "info", "drained", in_flight=self.in_flight)

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: setattr(self, "shutting_down", True))
            except NotImplementedError:  # not available on every platform
                pass


async def observed(
    lifecycle: Lifecycle,
    logger: logging.Logger,
    request_id: str,
    handler: Callable[[], Awaitable[Any]],
    **fields: Any,
) -> Any:
    """Run one request with its id bound, counted, and logged once at the end."""
    token = request_id_var.set(request_id)
    lifecycle.enter()
    started = time.perf_counter()
    status = 500
    try:
        response = await handler()
        status = getattr(response, "status_code", 200)
        return response
    finally:
        lifecycle.leave()
        request_id_var.reset(token)
        log(
            logger, "info", "request",
            status=status,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            **fields,
        )
