from __future__ import annotations

import json

import pytest

from llm_eval.backends import OllamaClient, build_client


class FakeHTTP:
    """Captures the request body so the mapping can be asserted."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.sent = None

    def __call__(self, request, timeout=None):
        self.sent = json.loads(request.data)
        import io
        return _CM(io.BytesIO(json.dumps(self.reply).encode()))


class _CM:
    def __init__(self, fh): self.fh = fh
    def __enter__(self): return self.fh
    def __exit__(self, *a): return False


REPLY = {
    "message": {"content": '{"passed": true, "reason": "names the release", "confidence": 0.8}'},
    "prompt_eval_count": 120,
    "eval_count": 25,
}


@pytest.fixture
def client(monkeypatch):
    http = FakeHTTP(REPLY)
    monkeypatch.setattr("urllib.request.urlopen", http)
    c = OllamaClient(model="qwen3:8b")
    c._http = http
    return c


def test_the_response_matches_the_shape_the_judge_expects(client):
    response = client.messages.create(
        max_tokens=512, system="sys", messages=[{"role": "user", "content": "q"}]
    )
    assert response.content[0].type == "text"
    assert json.loads(response.content[0].text)["passed"] is True
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 25


def test_the_system_prompt_is_sent_as_a_system_message(client):
    client.messages.create(max_tokens=10, system="be strict",
                           messages=[{"role": "user", "content": "q"}])
    roles = [m["role"] for m in client._http.sent["messages"]]
    assert roles == ["system", "user"]


def test_no_system_message_is_sent_when_there_is_none(client):
    client.messages.create(max_tokens=10, messages=[{"role": "user", "content": "q"}])
    assert [m["role"] for m in client._http.sent["messages"]] == ["user"]


def test_temperature_is_zero_because_a_varying_judge_is_not_a_measurement(client):
    client.messages.create(max_tokens=10, messages=[{"role": "user", "content": "q"}])
    assert client._http.sent["options"]["temperature"] == 0.0


def test_the_json_schema_is_passed_through_as_a_format_constraint(client):
    schema = {"type": "object", "properties": {"passed": {"type": "boolean"}}}
    client.messages.create(
        max_tokens=10, messages=[{"role": "user", "content": "q"}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    assert client._http.sent["format"] == schema


def test_max_tokens_reaches_the_server(client):
    client.messages.create(max_tokens=333, messages=[{"role": "user", "content": "q"}])
    assert client._http.sent["options"]["num_predict"] == 333


def test_an_unreachable_server_raises_something_readable(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="cannot reach ollama"):
        OllamaClient().messages.create(max_tokens=10, messages=[{"role": "user", "content": "q"}])


def test_availability_reports_a_missing_model_with_the_fix(monkeypatch):
    import io

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _CM(io.BytesIO(json.dumps({"models": [{"name": "llama3:8b"}]}).encode())),
    )
    ok, detail = OllamaClient(model="qwen3:8b").available()
    assert not ok
    assert "ollama pull qwen3:8b" in detail


def test_availability_reports_a_down_server_rather_than_raising(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    ok, detail = OllamaClient().available()
    assert not ok and "no ollama" in detail


def test_explicit_backend_selection_is_honoured():
    assert isinstance(build_client("local"), OllamaClient)
