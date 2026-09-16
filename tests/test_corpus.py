from __future__ import annotations

from collections import Counter

import pytest

from agent_guardrail.corpus import load_all, load_attacks, load_benign, split
from agent_guardrail.trust import trust_for, Trust

ATTACKS = load_attacks()
BENIGN = load_benign()


def test_both_corpora_are_the_same_size():
    """A detector is only as honest as its negative set."""
    assert len(ATTACKS) == len(BENIGN)


def test_ids_are_unique_across_both_corpora():
    ids = [s.id for s in load_all()]
    assert len(ids) == len(set(ids))


def test_attacks_span_several_families():
    families = Counter(s.family for s in ATTACKS)
    assert len(families) >= 6
    assert max(families.values()) <= len(ATTACKS) // 4


def test_benign_samples_all_name_the_trap_they_set():
    assert all(s.trap for s in BENIGN)
    assert len({s.trap for s in BENIGN}) >= 15


def test_attacks_arrive_through_channels_an_attacker_can_actually_reach():
    """Almost all of them must be indirect; direct injection is the easy case."""
    indirect = [s for s in ATTACKS if s.channel != "user_message"]
    assert len(indirect) / len(ATTACKS) > 0.9


def test_every_attack_channel_is_untrusted_provenance():
    for sample in ATTACKS:
        if sample.channel == "user_message":
            continue
        assert trust_for(sample.channel) == Trust.UNTRUSTED, sample.id


def test_benign_samples_come_from_the_same_channels_as_attacks():
    """Otherwise the detector can cheat by learning the channel."""
    assert {s.channel for s in BENIGN} <= {s.channel for s in ATTACKS}


def test_no_sample_is_trivially_short():
    """Measured in characters, not words.

    Machine text is dense: a base64 blob or `component=system-cluster-critical`
    is one whitespace-separated token but carries plenty for a detector to work
    with. Counting words marked four perfectly good samples as too short.
    """
    short = [(s.id, len(s.text)) for s in load_all() if len(s.text) < 40]
    assert short == []


def test_group_falls_back_to_family_then_trap():
    assert ATTACKS[0].group == ATTACKS[0].family
    assert BENIGN[0].group == BENIGN[0].trap


# -- splitting ---------------------------------------------------------

def test_split_is_deterministic():
    assert [s.id for s in split(load_all())[1]] == [s.id for s in split(load_all())[1]]


def test_split_loses_nothing():
    train, test = split(load_all())
    assert len(train) + len(test) == len(load_all())
    assert not {s.id for s in train} & {s.id for s in test}


def test_split_is_stratified_so_no_group_vanishes():
    train, test = split(load_all())
    assert {s.group for s in test} == {s.group for s in load_all()}
    assert {s.group for s in train} == {s.group for s in load_all()}


def test_every_fold_is_a_different_test_set():
    folds = [{s.id for s in split(load_all(), fold=f)[1]} for f in range(4)]
    assert len(set(map(frozenset, folds))) == 4
    assert set().union(*folds) == {s.id for s in load_all()}


def test_both_classes_appear_in_each_half():
    train, test = split(load_all())
    for half in (train, test):
        assert any(s.is_attack for s in half)
        assert any(not s.is_attack for s in half)
