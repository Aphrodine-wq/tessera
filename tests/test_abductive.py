"""Tests for abductive reasoning (research 4.3)."""
import math

from tessera.abductive import (
    Hypothesis,
    rank_hypotheses,
    best_explanation,
)


def test_rank_picks_higher_likelihood_when_prior_equal():
    """Two hypotheses, same prior + complexity, but h1 explains obs better."""
    h1 = Hypothesis(name="h1", prior=0.5, likelihood=lambda o: 0.9, complexity=1)
    h2 = Hypothesis(name="h2", prior=0.5, likelihood=lambda o: 0.1, complexity=1)
    ranked = rank_hypotheses([h1, h2], ["obs1"])
    assert ranked[0].name == "h1"
    assert ranked[0].posterior > ranked[1].posterior


def test_rank_penalizes_complexity():
    """Two equally-explaining hypotheses; the simpler one wins."""
    h_simple = Hypothesis(name="simple", prior=0.5, likelihood=lambda o: 0.8, complexity=1)
    h_complex = Hypothesis(name="complex", prior=0.5, likelihood=lambda o: 0.8, complexity=10)
    ranked = rank_hypotheses([h_simple, h_complex], ["obs"])
    assert ranked[0].name == "simple"


def test_rank_uses_prior():
    """Two equally-explaining-and-complex hypotheses; the higher-prior wins."""
    h_a = Hypothesis(name="a", prior=0.9, likelihood=lambda o: 0.5, complexity=1)
    h_b = Hypothesis(name="b", prior=0.1, likelihood=lambda o: 0.5, complexity=1)
    ranked = rank_hypotheses([h_a, h_b], ["obs"])
    assert ranked[0].name == "a"


def test_rank_accumulates_evidence_across_observations():
    """Multiple obs that h1 explains well should compound."""
    h1 = Hypothesis(name="h1", prior=0.5, likelihood=lambda o: 0.9, complexity=1)
    h2 = Hypothesis(name="h2", prior=0.5, likelihood=lambda o: 0.5, complexity=1)
    single = rank_hypotheses([h1, h2], ["obs1"])
    triple = rank_hypotheses([h1, h2], ["obs1", "obs2", "obs3"])
    # h1's relative posterior should grow with more obs
    h1_single = next(r for r in single if r.name == "h1").posterior
    h1_triple = next(r for r in triple if r.name == "h1").posterior
    assert h1_triple > h1_single


def test_best_explanation_returns_none_below_threshold():
    """Three equally-plausible hypotheses → top posterior is ~1/3 < 0.5
    default threshold → best returns None."""
    hs = [Hypothesis(name=str(i), prior=0.5, likelihood=lambda o: 0.5, complexity=1)
          for i in range(3)]
    best, ranked = best_explanation(hs, ["o"])
    assert best is None
    assert len(ranked) == 3


def test_best_explanation_returns_winner_above_threshold():
    h_strong = Hypothesis(name="strong", prior=0.9, likelihood=lambda o: 0.95, complexity=1)
    h_weak = Hypothesis(name="weak", prior=0.1, likelihood=lambda o: 0.05, complexity=5)
    best, ranked = best_explanation([h_strong, h_weak], ["o"], confidence_threshold=0.5)
    assert best is not None
    assert best.name == "strong"
    assert best.posterior > 0.5


def test_empty_hypothesis_space_returns_none():
    best, ranked = best_explanation([], ["o"])
    assert best is None
    assert ranked == []


def test_posteriors_sum_to_one():
    hs = [Hypothesis(name=str(i), prior=0.3 + i * 0.1, likelihood=lambda o: 0.5, complexity=1)
          for i in range(4)]
    ranked = rank_hypotheses(hs, ["o"])
    total = sum(r.posterior for r in ranked)
    assert math.isclose(total, 1.0, abs_tol=1e-9)
