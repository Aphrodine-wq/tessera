"""Tests for the dual-process router (research 4.1)."""
from tessera.dual_process import route


def test_irreversible_forces_slow_even_with_fast_preference():
    d = route(preferred="fast", confidence=1.0, budget=1.0, irreversible=True)
    assert d.mode == "slow"
    assert d.forced_slow is True
    assert "irreversible" in d.rationale


def test_preferred_slow_runs_slow():
    d = route(preferred="slow", confidence=1.0, budget=1.0)
    assert d.mode == "slow"
    assert d.rationale == "preferred_slow"


def test_low_confidence_escalates_to_slow():
    d = route(preferred="fast", confidence=0.4, budget=1.0,
              confidence_threshold=0.7)
    assert d.mode == "slow"
    assert "low_confidence" in d.rationale


def test_low_budget_demotes_to_fast_when_preferred_fast():
    d = route(preferred="fast", confidence=1.0, budget=0.1,
              budget_threshold=0.2)
    assert d.mode == "fast"
    assert "low_budget" in d.rationale


def test_high_confidence_high_budget_honors_fast():
    d = route(preferred="fast", confidence=0.9, budget=0.9)
    assert d.mode == "fast"
    assert "preferred_fast" in d.rationale


def test_irreversible_beats_low_budget():
    """Even when budget is exhausted, irreversible action demands
    deliberation."""
    d = route(preferred="fast", confidence=1.0, budget=0.05,
              irreversible=True)
    assert d.mode == "slow"
    assert d.forced_slow is True
