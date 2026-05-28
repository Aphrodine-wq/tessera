"""Tests for the hindsight_review substrate (research 4.10)."""
import math

from tessera.hindsight import (
    HindsightReview,
    review,
    fitness_from_reviews,
)


def test_clean_run_reports_clean():
    r = review(
        plan_name="p",
        intended_outcome="ok",
        actual_outcome="ok",
        declared_ethics=["honesty"],
        applied_ethics=["honesty"],
    )
    assert r.outcome_matched
    assert r.intended_ethics_missed == []
    assert r.unexpected_ethics_applied == []
    assert r.notes == "clean run"


def test_outcome_divergence_recorded():
    r = review(
        plan_name="p",
        intended_outcome="ok",
        actual_outcome="error",
        declared_ethics=[],
        applied_ethics=[],
    )
    assert not r.outcome_matched
    assert "diverged" in r.notes


def test_missed_declared_ethic_recorded():
    r = review(
        plan_name="p",
        intended_outcome="ok",
        actual_outcome="ok",
        declared_ethics=["honesty", "care"],
        applied_ethics=["honesty"],
    )
    assert "care" in r.intended_ethics_missed
    assert "not applied" in r.notes


def test_unexpected_ethic_recorded():
    r = review(
        plan_name="p",
        intended_outcome="ok",
        actual_outcome="ok",
        declared_ethics=["honesty"],
        applied_ethics=["honesty", "loyalty"],
    )
    assert "loyalty" in r.unexpected_ethics_applied


def test_custom_match_predicate_supports_fuzzy_compare():
    r = review(
        plan_name="p",
        intended_outcome=100,
        actual_outcome=105,
        match_predicate=lambda i, a: abs(i - a) <= 10,
    )
    assert r.outcome_matched


def test_to_audit_emits_dict_with_expected_keys():
    r = review(plan_name="p", intended_outcome="x", actual_outcome="x")
    d = r.to_audit()
    assert d["action"] == "hindsight:learning:p"
    assert "intended_outcome" in d
    assert "actual_outcome" in d
    assert "outcome_matched" in d


def test_fitness_perfect_run_scores_one():
    reviews = [
        review("p", "x", "x", declared_ethics=["honesty"],
               applied_ethics=["honesty"]),
        review("q", 1, 1, declared_ethics=["care"],
               applied_ethics=["care"]),
    ]
    assert math.isclose(fitness_from_reviews(reviews), 1.0)


def test_fitness_outcome_miss_penalizes():
    reviews = [
        review("p", "x", "y"),    # missed
        review("q", "x", "x"),   # hit
    ]
    score = fitness_from_reviews(reviews)
    assert math.isclose(score, 0.5)


def test_fitness_empty_returns_zero():
    assert fitness_from_reviews([]) == 0.0
