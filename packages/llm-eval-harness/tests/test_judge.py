"""The judge, exercised against a fake client so the suite needs no API."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from llm_eval.judge import Criterion, Judge, Verdict


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 20


@dataclass
class FakeResponse:
    content: list
    usage: FakeUsage


class FakeClient:
    """Records calls so tests can assert the judge did not ask twice."""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return FakeResponse([FakeBlock(self.payload)], FakeUsage())


CRITERION = Criterion(
    name="names_causal_chain",
    question="Does the answer name what changed, what it saturated, and how that caused the symptom?",
)
SUBJECT = "The pool saturated after v2.31.0 added an audit write, so requests timed out."
GOOD = json.dumps({"passed": True, "reason": "names the release and the saturation", "confidence": 0.9})


@pytest.fixture
def judge(tmp_path):
    return Judge(client=FakeClient(GOOD), cache_dir=tmp_path)


def test_a_verdict_becomes_a_check_marked_as_judged(judge):
    check = judge.check(CRITERION, SUBJECT)
    assert check.passed
    assert check.kind == "judge"
    assert check.name == "names_causal_chain"


def test_the_reason_survives_into_the_check(judge):
    assert "saturation" in judge.check(CRITERION, SUBJECT).reason


# -- caching -----------------------------------------------------------

def test_an_unchanged_case_is_judged_once(judge):
    judge.judge(CRITERION, SUBJECT)
    judge.judge(CRITERION, SUBJECT)
    assert judge._client.calls == 1


def test_a_cached_verdict_is_marked_as_cached(judge):
    judge.judge(CRITERION, SUBJECT)
    assert judge.judge(CRITERION, SUBJECT).cached


def test_changing_the_subject_invalidates_the_cache(judge):
    judge.judge(CRITERION, SUBJECT)
    judge.judge(CRITERION, "an entirely different answer about disk pressure")
    assert judge._client.calls == 2


def test_changing_the_rubric_version_invalidates_every_verdict(tmp_path):
    """The failure mode this prevents: old scores reused under a new rubric."""
    first = Judge(client=FakeClient(GOOD), cache_dir=tmp_path, rubric_version="v1")
    first.judge(CRITERION, SUBJECT)

    second = Judge(client=FakeClient(GOOD), cache_dir=tmp_path, rubric_version="v2")
    second.judge(CRITERION, SUBJECT)
    assert second._client.calls == 1


def test_changing_the_model_invalidates_every_verdict(tmp_path):
    first = Judge(client=FakeClient(GOOD), cache_dir=tmp_path, model="model-a")
    first.judge(CRITERION, SUBJECT)
    second = Judge(client=FakeClient(GOOD), cache_dir=tmp_path, model="model-b")
    second.judge(CRITERION, SUBJECT)
    assert second._client.calls == 1


def test_the_cache_file_records_which_judge_produced_it(judge, tmp_path):
    judge.judge(CRITERION, SUBJECT)
    written = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert written["judge_version"] == judge.version


def test_caching_can_be_switched_off(judge):
    judge.judge(CRITERION, SUBJECT, use_cache=False)
    judge.judge(CRITERION, SUBJECT, use_cache=False)
    assert judge._client.calls == 2


def test_a_corrupt_cache_entry_is_ignored_rather_than_raising(judge, tmp_path):
    judge.judge(CRITERION, SUBJECT)
    next(tmp_path.glob("*.json")).write_text("{ truncated")
    assert judge.judge(CRITERION, SUBJECT).passed


# -- parsing through a gateway -----------------------------------------

def test_json_wrapped_in_prose_is_still_parsed(tmp_path):
    """Not every gateway in front of the API honours output_config."""
    wrapped = f"Here is my verdict:\n\n{GOOD}\n\nHope that helps."
    judge = Judge(client=FakeClient(wrapped), cache_dir=tmp_path)
    assert judge.judge(CRITERION, SUBJECT).passed


def test_a_response_with_no_json_fails_loudly(tmp_path):
    judge = Judge(client=FakeClient("I think it is fine, honestly."), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="no JSON object"):
        judge.judge(CRITERION, SUBJECT)


def test_structured_output_is_requested(judge):
    judge.judge(CRITERION, SUBJECT)
    assert judge._client.kwargs["output_config"]["format"]["type"] == "json_schema"


def test_the_criterion_and_subject_both_reach_the_model(judge):
    judge.judge(CRITERION, SUBJECT)
    sent = judge._client.kwargs["messages"][0]["content"]
    assert CRITERION.question in sent
    assert SUBJECT in sent


# -- version -----------------------------------------------------------

def test_the_version_names_both_model_and_rubric(tmp_path):
    judge = Judge(client=FakeClient(GOOD), cache_dir=tmp_path,
                  model="claude-opus-5", rubric_version="v3")
    assert judge.version == "claude-opus-5/v3"


def test_token_usage_is_recorded_for_cost_accounting(judge):
    verdict = judge.judge(CRITERION, SUBJECT)
    assert verdict.input_tokens == 100 and verdict.output_tokens == 20
