"""Request identity, log shape, and draining without dropping work."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest

from llm_gateway.observability import (
    JsonFormatter,
    Lifecycle,
    configure_logging,
    extract_request_id,
    log,
    request_id_var,
    tenant_var,
)


class Headers(dict):
    def get(self, key, default=None):
        return dict.get(self, key.lower(), default)


# -- request identity --------------------------------------------------

def test_a_callers_request_id_is_continued_not_replaced():
    """A trace that starts in their service should not restart at our edge."""
    assert extract_request_id(Headers({"x-request-id": "abc-123"})) == "abc-123"


def test_a_correlation_id_is_accepted_too():
    assert extract_request_id(Headers({"x-correlation-id": "corr-9"})) == "corr-9"


def test_a_traceparent_yields_its_trace_id():
    header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    assert extract_request_id(Headers({"traceparent": header})) == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_a_request_with_no_id_gets_one():
    generated = extract_request_id(Headers({}))
    assert generated and generated != extract_request_id(Headers({}))


def test_an_absurd_id_is_truncated_rather_than_logged_whole():
    assert len(extract_request_id(Headers({"x-request-id": "x" * 5000}))) == 128


# -- log shape ---------------------------------------------------------

def record(message: str = "hello", **fields) -> dict:
    rec = logging.LogRecord("gateway", logging.INFO, __file__, 1, message, (), None)
    if fields:
        rec.fields = fields
    return json.loads(JsonFormatter().format(rec))


def test_every_line_is_one_json_object():
    assert record()["msg"] == "hello"


def test_the_request_id_and_tenant_ride_along_without_being_passed():
    token_r = request_id_var.set("req-7")
    token_t = tenant_var.set("platform")
    try:
        entry = record()
        assert entry["request_id"] == "req-7"
        assert entry["tenant"] == "platform"
    finally:
        request_id_var.reset(token_r)
        tenant_var.reset(token_t)


def test_extra_fields_are_merged_at_the_top_level():
    """So a query can group by them without digging into a nested object."""
    entry = record("served", status=200, cost_usd=0.01)
    assert entry["status"] == 200 and entry["cost_usd"] == 0.01


def test_field_names_are_stable():
    assert set(record()) >= {"ts", "level", "logger", "msg", "request_id", "tenant"}


def test_an_exception_is_recorded_as_a_field_not_a_traceback_on_stderr():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = logging.LogRecord("gateway", logging.ERROR, __file__, 1, "failed", (),
                                sys.exc_info())
        entry = json.loads(JsonFormatter().format(rec))
    assert "boom" in entry["error"]


def test_uvicorns_duplicate_access_log_is_silenced():
    """Two formats for the same event is worse than one."""
    configure_logging()
    assert logging.getLogger("uvicorn.access").propagate is False


# -- draining ----------------------------------------------------------

async def test_draining_waits_for_work_already_in_flight():
    life = Lifecycle(drain_seconds=0.01, grace_seconds=2.0)
    life.enter()

    async def finish_later():
        await asyncio.sleep(0.05)
        life.leave()

    asyncio.create_task(finish_later())
    started = time.perf_counter()
    await life.drain()
    assert life.in_flight == 0
    assert time.perf_counter() - started >= 0.05


async def test_draining_pauses_before_waiting_so_endpoints_can_propagate():
    """Kubernetes removes the pod from the Service concurrently with SIGTERM,
    not before it, so exiting immediately drops requests already routed here."""
    life = Lifecycle(drain_seconds=0.08, grace_seconds=1.0)
    started = time.perf_counter()
    await life.drain()
    assert time.perf_counter() - started >= 0.08


async def test_a_hung_request_does_not_hold_the_rollout_open_forever():
    life = Lifecycle(drain_seconds=0.01, grace_seconds=0.05)
    life.enter()  # never left
    started = time.perf_counter()
    await life.drain()
    assert time.perf_counter() - started < 1.0


async def test_shutdown_is_flagged_so_readiness_can_fail_first():
    life = Lifecycle(drain_seconds=0.01, grace_seconds=0.1)
    assert not life.shutting_down
    await life.drain()
    assert life.shutting_down


def test_in_flight_never_goes_negative():
    life = Lifecycle()
    life.leave()
    assert life.in_flight == 0
