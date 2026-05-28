"""Tests for the moral_foundations substrate (research 4.9)."""
import math

from tessera.moral_foundations import (
    FoundationWeights,
    ActionMoralScore,
    score_action,
    axes_with_weight_zero,
    CANONICAL_FOUNDATIONS,
)


def test_canonical_six_foundations():
    """Haidt/Graham canonical list."""
    assert set(CANONICAL_FOUNDATIONS) == {
        "care", "fairness", "loyalty", "authority", "sanctity", "liberty",
    }


def test_action_with_all_positive_axes_accepted():
    w = FoundationWeights(care=1.0, fairness=1.0)
    a = ActionMoralScore(care=0.8, fairness=0.5)
    d = score_action(a, w)
    assert d.accept
    assert d.weighted_total > 0
    assert d.refuse_axes == []


def test_action_negative_on_weighted_axis_refused():
    """Care=1.0, action care=-0.5 → refused on care axis."""
    w = FoundationWeights(care=1.0, fairness=0.5)
    a = ActionMoralScore(care=-0.5, fairness=0.5)
    d = score_action(a, w)
    assert "care" in d.refuse_axes
    assert not d.accept


def test_negative_on_off_axis_does_not_refuse():
    """If sanctity_weight=0.05 (below tolerance), action's negative
    sanctity score doesn't trigger refusal."""
    w = FoundationWeights(care=0.8, sanctity=0.05)
    a = ActionMoralScore(care=0.5, sanctity=-0.8)
    d = score_action(a, w)
    assert "sanctity" not in d.refuse_axes


def test_per_axis_breakdown_visible():
    w = FoundationWeights(care=0.8, fairness=0.5)
    a = ActionMoralScore(care=0.5, fairness=0.2)
    d = score_action(a, w)
    assert math.isclose(d.per_axis["care"], 0.4)
    assert math.isclose(d.per_axis["fairness"], 0.1)


def test_axes_with_weight_zero_reports_off_axes():
    w = FoundationWeights(loyalty=0.0, authority=0.0,
                          care=0.8, fairness=0.7,
                          sanctity=0.3, liberty=0.5)
    off = axes_with_weight_zero(w)
    assert set(off) == {"loyalty", "authority"}


def test_contractor_vs_regulator_profile():
    """Two different value profiles → different decisions on the same action."""
    contractor = FoundationWeights(
        care=0.9, fairness=0.9, loyalty=0.3,
        authority=0.05, sanctity=0.05, liberty=0.7,
    )
    regulator = FoundationWeights(
        care=0.5, fairness=0.5, loyalty=0.5,
        authority=0.9, sanctity=0.7, liberty=0.4,
    )
    # An action with positive liberty, negative authority
    rebel_action = ActionMoralScore(care=0.2, liberty=0.8, authority=-0.6)
    c_dec = score_action(rebel_action, contractor)
    r_dec = score_action(rebel_action, regulator)
    # Contractor (low authority weight) allows it; regulator refuses
    assert c_dec.accept
    assert not r_dec.accept
    assert "authority" in r_dec.refuse_axes
