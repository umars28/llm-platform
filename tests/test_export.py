from __future__ import annotations

from pathlib import Path

import pytest

from llm_tracing.export import LocalReceiver, export, to_otlp
from llm_tracing.spans import ATTR_COST, Tracer


@pytest.fixture
def spans(tmp_path: Path):
    tracer = Tracer(service="test-app", path=tmp_path / "s.jsonl")
    with tracer.span("investigate") as outer:
        outer.add_event("alert received", severity="P1")
        with tracer.llm_call("claude-opus-5") as call:
            call.record_usage("claude-opus-5", input_tokens=2000, output_tokens=400)
        with tracer.tool_call("query_logs"):
            pass
    return tracer.finished


# -- serialisation -----------------------------------------------------

def test_every_span_appears_in_the_payload(spans):
    payload = to_otlp(spans)
    assert len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"]) == len(spans)


def test_the_service_name_is_a_resource_attribute(spans):
    attrs = to_otlp(spans, "my-service")["resourceSpans"][0]["resource"]["attributes"]
    assert {"key": "service.name", "value": {"stringValue": "my-service"}} in attrs


def test_integers_and_floats_keep_their_wire_types(spans):
    otlp = to_otlp(spans)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    call = next(s for s in otlp if s["name"].startswith("chat"))
    by_key = {a["key"]: a["value"] for a in call["attributes"]}
    assert "intValue" in by_key["gen_ai.usage.input_tokens"]
    assert "doubleValue" in by_key[ATTR_COST]


def test_parenthood_survives_conversion(spans):
    otlp = to_otlp(spans)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    parent = next(s for s in otlp if s["name"] == "investigate")
    child = next(s for s in otlp if s["name"].startswith("tool "))
    assert child["parentSpanId"] == parent["spanId"]
    assert "parentSpanId" not in parent


def test_events_survive_conversion(spans):
    otlp = to_otlp(spans)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    root = next(s for s in otlp if s["name"] == "investigate")
    assert root["events"][0]["name"] == "alert received"


def test_an_error_span_carries_the_error_status_code(tmp_path: Path):
    tracer = Tracer(path=tmp_path / "s.jsonl")
    with pytest.raises(RuntimeError):
        with tracer.span("doomed"):
            raise RuntimeError("boom")
    span = to_otlp(tracer.finished)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["status"]["code"] == 2
    assert "boom" in span["status"]["message"]


def test_span_kinds_map_to_otlp_numbers(spans):
    otlp = to_otlp(spans)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert next(s for s in otlp if s["name"].startswith("chat"))["kind"] == 3  # client


# -- the wire ----------------------------------------------------------

def test_spans_arrive_intact_at_an_otlp_endpoint(spans):
    """A claim about a wire format needs something on the other end."""
    with LocalReceiver() as receiver:
        status = export(spans, receiver.endpoint, service="test-app")
        assert status == 200
        arrived = receiver.spans()
    assert len(arrived) == len(spans)
    assert {s["name"] for s in arrived} == {s.name for s in spans}


def test_the_token_attributes_survive_the_round_trip(spans):
    with LocalReceiver() as receiver:
        export(spans, receiver.endpoint)
        call = next(s for s in receiver.spans() if s["name"].startswith("chat"))
    by_key = {a["key"]: a["value"] for a in call["attributes"]}
    assert by_key["gen_ai.usage.input_tokens"]["intValue"] == "2000"


def test_custom_headers_reach_the_endpoint(spans):
    with LocalReceiver() as receiver:
        assert export(spans, receiver.endpoint, headers={"Authorization": "Basic x"}) == 200


def test_exporting_nothing_makes_no_request():
    assert export([], "http://127.0.0.1:1") == 0


def test_an_unreachable_endpoint_raises_rather_than_warning(spans):
    """Silently swallowing this leaves you believing you have observability."""
    with pytest.raises(RuntimeError, match="cannot reach otlp endpoint"):
        export(spans, "http://127.0.0.1:1", timeout=1)
