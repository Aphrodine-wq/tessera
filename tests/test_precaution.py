"""Tests for the precautionary substrate (research 4.7)."""
import math
import pytest

from tessera.precaution import (
    HarmThreshold,
    precaution_gate,
    posterior_tail_from_bayesian,
)


def test_allow_when_tail_below_max():
    t = HarmThreshold(action_class="payment", harm_magnitude=100,
                      irreversible=False, max_tail_probability=0.05)
    d = precaution_gate(t, tail_probability=0.02)
    assert d.verdict == "allow"


def test_refuse_when_tail_above_max():
    t = HarmThreshold(action_class="payment", harm_magnitude=100,
                      irreversible=False, max_tail_probability=0.05)
    d = precaution_gate(t, tail_probability=0.10)
    assert d.verdict == "refuse"
    assert "tail probability" in d.rationale


def test_irreversible_overrides_tail_max():
    """An irreversible action with non-trivial tail risk refuses regardless
    of the configured tail max."""
    t = HarmThreshold(action_class="deletion", harm_magnitude=1000,
                      irreversible=True, max_tail_probability=0.50)
    d = precaution_gate(t, tail_probability=0.02)
    assert d.verdict == "refuse"
    assert "irreversible" in d.rationale
    assert "Hansson" in d.rationale


def test_irreversible_with_negligible_tail_allows():
    """Tail probability below the irreversibility floor (0.001) still allows."""
    t = HarmThreshold(action_class="deletion", harm_magnitude=1000,
                      irreversible=True, max_tail_probability=0.5)
    d = precaution_gate(t, tail_probability=0.0005)
    assert d.verdict == "allow"


def test_posterior_tail_from_bayesian_correctness():
    """Standard 3-outcome example."""
    posterior = [0.6, 0.3, 0.1]
    values = ["low", "medium", "high"]
    tail = posterior_tail_from_bayesian(posterior, values, ["high", "medium"])
    assert math.isclose(tail, 0.4, abs_tol=1e-9)


def test_posterior_tail_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        posterior_tail_from_bayesian([0.5, 0.5], ["a", "b", "c"], ["a"])


def test_posterior_tail_zero_when_no_harm_match():
    posterior = [0.5, 0.5]
    values = ["safe1", "safe2"]
    tail = posterior_tail_from_bayesian(posterior, values, ["danger"])
    assert math.isclose(tail, 0.0)


def test_full_composition_with_bayesian_posterior():
    """Smoke: posterior from a bayesian model → tail → precaution gate."""
    posterior = [0.8, 0.15, 0.05]   # safe, risky, catastrophe
    values = ["safe", "risky", "catastrophe"]
    tail = posterior_tail_from_bayesian(posterior, values, ["catastrophe"])
    t = HarmThreshold(
        action_class="deploy",
        harm_magnitude=1e6,
        irreversible=True,
        max_tail_probability=0.5,
    )
    d = precaution_gate(t, tail)
    assert d.verdict == "refuse"
    assert d.tail_probability == 0.05
