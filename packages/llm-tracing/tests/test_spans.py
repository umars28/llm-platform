from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_tracing.spans import (
    ATTR_COST,
    ATTR_INPUT_TOKENS,
    ATTR_MODEL,
    ATTR_TOOL,
    Tracer,
    pricing_for,
    redact,
)


@pytest.fixture
def tracer(tmp_path: Path) -> Tracer:
    return Tracer(service="test", path=tmp_path / "spans.jsonl")


# -- the crash case ----------------------------------------------------

def test_a_span_is_written_even_when_the_body_raises():
    """The runs worth tracing are the ones that die."""
    import tempfile

    path = Path(tempfile.mkdtemp()) / "spans.jsonl"
    tracer = Tracer(path=path)
    with pytest.raises(RuntimeError):
        with tracer.span("doomed"):
            raise RuntimeError("boom")

    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert written[0]["status"] == "error"
    assert "boom" in written[0]["error"]


def test_a_successful_span_is_marked_ok(tracer):
    with tracer.span("fine"):
        pass
    assert tracer.finished[0].status == "ok"


def test_spans_are_on_disk_before_the_process_ends(tracer):
    with tracer.span("one"):
        pass
    assert tracer.path.read_text().strip()


# -- structure ---------------------------------------------------------

def test_nested_spans_share_a_trace_and_link_to_the_parent(tracer):
    with tracer.span("outer") as outer:
        with tracer.span("inner") as inner:
            assert inner.trace_id == outer.trace_id
            assert inner.parent_id == outer.span_id


def test_sibling_traces_are_separate(tracer):
    with tracer.span("first") as a:
        pass
    with tracer.span("second") as b:
        pass
    assert a.trace_id != b.trace_id


def test_the_stack_unwinds_even_on_error(tracer):
    with pytest.raises(ValueError):
        with tracer.span("outer"):
            raise ValueError("x")
    assert tracer.current is None


def test_duration_is_recorded(tracer):
    with tracer.span("timed"):
        pass
    assert tracer.finished[0].duration_ms >= 0


# -- llm attributes ----------------------------------------------------

def test_an_llm_span_carries_the_semantic_convention_names(tracer):
    with tracer.llm_call("claude-opus-5") as span:
        span.record_usage("claude-opus-5", input_tokens=1000, output_tokens=200)
    attrs = tracer.finished[0].attributes
    assert attrs[ATTR_MODEL] == "claude-opus-5"
    assert attrs[ATTR_INPUT_TOKENS] == 1000
    assert attrs[ATTR_COST] > 0


def test_cost_is_computed_at_close_not_left_for_a_query(tracer):
    with tracer.llm_call("claude-opus-5") as span:
        span.record_usage("claude-opus-5", input_tokens=1_000_000)
    assert tracer.finished[0].attributes[ATTR_COST] == pytest.approx(5.0)


def test_cache_reads_are_priced_below_fresh_input(tracer):
    with tracer.llm_call("claude-opus-5") as a:
        a.record_usage("claude-opus-5", input_tokens=100_000)
    with tracer.llm_call("claude-opus-5") as b:
        b.record_usage("claude-opus-5", cache_read=100_000)
    fresh, cached = (s.attributes[ATTR_COST] for s in tracer.finished)
    assert cached == pytest.approx(fresh / 10)


def test_a_free_tier_costs_nothing(tracer):
    with tracer.llm_call("some/model:free") as span:
        span.record_usage("some/model:free", input_tokens=500_000)
    assert tracer.finished[0].attributes[ATTR_COST] == 0.0


def test_an_unknown_model_is_priced_at_the_dearest_tier():
    assert pricing_for("model-from-2028") == pricing_for("claude-opus-5")


def test_a_tool_span_names_the_tool(tracer):
    with tracer.tool_call("query_logs"):
        pass
    assert tracer.finished[0].attributes[ATTR_TOOL] == "query_logs"


def test_totals_aggregate_across_a_run(tracer):
    for _ in range(3):
        with tracer.llm_call("claude-haiku-4-5") as span:
            span.record_usage("claude-haiku-4-5", input_tokens=1000, output_tokens=100)
    assert tracer.total_tokens()["input_tokens"] == 3000
    assert tracer.total_cost() > 0


# -- redaction ---------------------------------------------------------

@pytest.mark.parametrize("secret", [
    "sk-ant-api03-abcdefghijklmnop",
    "sk-or-v1-0000000000000000000000000000",
    "ghp_abcdefghijklmnopqrstuvwxyz",
])
def test_api_keys_are_redacted(secret):
    assert secret not in redact(f"the key is {secret} ok")


def test_jwts_are_redacted():
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"
    assert jwt not in redact(jwt)


def test_labelled_secrets_are_redacted():
    assert "hunter2" not in redact("password: hunter2")


def test_emails_and_addresses_are_redacted():
    out = redact("contact umar@example.com from 10.4.3.88")
    assert "umar@example.com" not in out and "10.4.3.88" not in out


def test_ordinary_text_survives_redaction():
    text = "the connection pool saturated at 20/20 after v2.31.0"
    assert redact(text) == text


def test_content_is_not_recorded_unless_asked_for(tracer):
    """A trace carries whatever the user typed and whatever the model read."""
    with tracer.llm_call("claude-opus-5", prompt="my password: hunter2"):
        pass
    assert "gen_ai.prompt" not in tracer.finished[0].attributes


def test_recorded_content_is_redacted_on_the_way_out(tmp_path: Path):
    tracer = Tracer(path=tmp_path / "s.jsonl", record_content=True)
    with tracer.llm_call("claude-opus-5", prompt="key sk-ant-api03-abcdefghijk here"):
        pass
    assert "sk-ant-api03" not in tracer.finished[0].attributes["gen_ai.prompt"]
