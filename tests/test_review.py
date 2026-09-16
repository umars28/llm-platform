from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from pr_reviewer.extract import Sample
from pr_reviewer.review import (
    Finding,
    Review,
    Score,
    _parse,
    finding_matches,
    found_the_defect,
    review,
)


def sample(has_defect=True, keywords=("sweep", "result", "completed")) -> Sample:
    return Sample(id="s1", repo="r", commit="c", path="src/x.py",
                  diff="--- a\n+++ b\n+code", has_defect=has_defect,
                  defect_summary="a sweep is a result only when most of it completed"
                  if has_defect else "",
                  keywords=list(keywords) if has_defect else [])


# -- parsing -----------------------------------------------------------

def test_findings_are_parsed():
    text = json.dumps({"findings": [{"summary": "off by one", "trigger": "empty list"}]})
    assert _parse(text)[0].summary == "off by one"


def test_an_empty_review_is_valid_and_means_no_defect():
    assert _parse(json.dumps({"findings": []})) == []


def test_json_wrapped_in_prose_is_still_parsed():
    text = 'Here you go:\n{"findings": []}\nhope that helps'
    assert _parse(text) == []


def test_a_finding_without_a_summary_is_dropped():
    text = json.dumps({"findings": [{"trigger": "something"}, {"summary": "real"}]})
    assert [f.summary for f in _parse(text)] == ["real"]


def test_a_response_with_no_json_fails_loudly():
    with pytest.raises(ValueError, match="no JSON object"):
        _parse("I think it looks fine")


# -- matching ----------------------------------------------------------

def test_a_finding_naming_the_same_thing_as_the_fix_matches():
    f = Finding(summary="the sweep reports a result even when most scenarios errored",
                trigger="25 of 30 fail")
    assert finding_matches(f, ["sweep", "result", "completed"]) >= 0.5


def test_an_unrelated_finding_does_not_match():
    f = Finding(summary="variable name could be clearer", trigger="")
    assert finding_matches(f, ["sweep", "result", "completed"]) < 0.5


def test_matching_looks_at_trigger_and_hint_too():
    f = Finding(summary="wrong", trigger="sweep result when not completed")
    assert finding_matches(f, ["sweep", "result", "completed"]) >= 0.5


def test_a_sample_with_no_keywords_cannot_be_matched():
    assert finding_matches(Finding("anything"), []) == 0.0


# -- scoring -----------------------------------------------------------

def hit() -> Review:
    return Review("s1", [Finding("the sweep reports a result when nothing completed")])


def miss() -> Review:
    return Review("s1", [Finding("consider renaming this variable")])


def silent() -> Review:
    return Review("s1", [])


def test_recall_counts_only_defects_actually_named():
    score = Score()
    score.add(hit(), sample())
    score.add(miss(), sample())
    assert score.recall == 0.5


def test_detection_rate_counts_flagging_something_at_all():
    """The gap from recall is noticing a diff is wrong without saying why."""
    score = Score()
    score.add(miss(), sample())
    assert score.detection_rate == 1.0
    assert score.recall == 0.0


def test_a_silent_review_of_a_defect_counts_as_neither():
    score = Score()
    score.add(silent(), sample())
    assert score.detection_rate == 0.0 and score.recall == 0.0


def test_false_positives_are_measured_on_clean_diffs_only():
    score = Score()
    score.add(miss(), sample(has_defect=False))
    score.add(silent(), sample(has_defect=False))
    assert score.false_positive_rate == 0.5


def test_a_reviewer_that_comments_on_everything_scores_perfect_recall_and_is_caught():
    score = Score()
    for _ in range(3):
        score.add(hit(), sample())
    for _ in range(3):
        score.add(hit(), sample(has_defect=False))
    assert score.recall == 1.0
    assert score.false_positive_rate == 1.0  # the number that stops adoption


def test_findings_per_clean_diff_measures_volume_not_just_incidence():
    score = Score()
    score.add(Review("s", [Finding("a"), Finding("b"), Finding("c")]), sample(has_defect=False))
    assert score.findings_per_clean_diff == 3.0


def test_missed_defects_are_listed_for_inspection():
    score = Score()
    score.add(miss(), sample())
    assert [s.id for s in score.missed()] == ["s1"]


def test_noisy_reviews_are_listed_with_what_they_said():
    score = Score()
    score.add(miss(), sample(has_defect=False))
    assert score.noisy()[0][1][0].summary == "consider renaming this variable"


def test_an_empty_score_does_not_divide_by_zero():
    empty = Score()
    assert empty.recall == 0.0 and empty.false_positive_rate == 0.0


# -- the call ----------------------------------------------------------

@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 50
    output_tokens: int = 10


@dataclass
class _Resp:
    content: list
    usage: _Usage


class FakeClient:
    def __init__(self, payload, fail=False):
        self.payload, self.fail = payload, fail
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("model unreachable")
        return _Resp([_Block(self.payload)], _Usage())


def test_a_review_records_usage_and_findings():
    client = FakeClient(json.dumps({"findings": [{"summary": "bug", "trigger": "t"}]}))
    result = review(client, sample(), model="m")
    assert result.findings[0].summary == "bug"
    assert result.input_tokens == 50


def test_a_model_failure_becomes_a_recorded_error_not_a_crash():
    result = review(FakeClient("", fail=True), sample(), model="m")
    assert result.error and "model unreachable" in result.error
    assert not result.reported_a_defect


def test_unparseable_output_becomes_a_recorded_error():
    result = review(FakeClient("no json here"), sample(), model="m")
    assert result.error and "no JSON object" in result.error


def test_the_diff_and_path_reach_the_model():
    client = FakeClient(json.dumps({"findings": []}))
    review(client, sample(), model="m")
    sent = client.kwargs["messages"][0]["content"]
    assert "src/x.py" in sent and "+code" in sent


def test_an_empty_model_response_is_a_recorded_error_not_an_empty_review():
    """A thinking block can eat the whole token budget; that is not 'no defects'."""
    result = review(FakeClient("   "), sample(), model="m")
    assert result.error and "empty response" in result.error
    assert not result.reported_a_defect


def test_a_mostly_errored_run_is_not_a_measurement():
    """26 of 55 failing once reported '0% recall' as though it meant something."""
    score = Score()
    for _ in range(3):
        score.add(hit(), sample())
    for _ in range(7):
        score.add(Review("s", [], error="boom"), sample())
    assert not score.valid
    assert score.summary()["valid"] is False


def test_a_clean_run_is_valid():
    score = Score()
    for _ in range(10):
        score.add(hit(), sample())
    assert score.valid
