"""Tests for the argumentative substrate (research 4.12)."""
import math

from tessera.argumentative import (
    Claim,
    CounterArgument,
    downweight_confidence,
    decide_with_critic,
)


def test_stronger_critic_drives_confidence_below_half():
    """A moderately-confident proposer (0.6) facing a stronger critic
    (0.9) should land below 0.5 after downweighting."""
    cf = downweight_confidence(0.6, 0.9)
    assert cf < 0.5


def test_weak_critic_barely_dents_confidence():
    """A confident proposer (0.95) facing a weak critic (0.1) keeps
    most of its confidence."""
    cf = downweight_confidence(0.95, 0.1)
    assert cf > 0.9


def test_equal_proposer_and_critic_is_undecided():
    """When both sides have equal log-odds, combined ≈ 0.5."""
    cf = downweight_confidence(0.7, 0.7)
    assert math.isclose(cf, 0.5, abs_tol=1e-6)


def test_extreme_values_are_clamped():
    """0 and 1 inputs don't produce inf/nan."""
    assert math.isfinite(downweight_confidence(0.0, 0.5))
    assert math.isfinite(downweight_confidence(1.0, 0.5))
    assert math.isfinite(downweight_confidence(0.5, 0.0))
    assert math.isfinite(downweight_confidence(0.5, 1.0))


def test_decide_with_critic_accepts_when_proposer_dominates():
    p = Claim(content="x is true", confidence=0.95)
    c = CounterArgument(content="x might be false", strength=0.2)
    d = decide_with_critic(p, c)
    assert d.accept is True
    assert d.final_confidence > 0.5
    assert "above threshold" in d.rationale


def test_decide_with_critic_refuses_when_critic_dominates():
    p = Claim(content="x is true", confidence=0.6)
    c = CounterArgument(content="evidence against x", strength=0.9)
    d = decide_with_critic(p, c)
    assert d.accept is False
    assert d.final_confidence < 0.5
    assert "refuse" in d.rationale


def test_decide_with_critic_threshold_is_configurable():
    """A higher accept_threshold makes the agent more cautious."""
    p = Claim(content="x", confidence=0.6)
    c = CounterArgument(content="counter", strength=0.4)
    relaxed = decide_with_critic(p, c, accept_threshold=0.5)
    strict = decide_with_critic(p, c, accept_threshold=0.75)
    assert relaxed.accept and not strict.accept
