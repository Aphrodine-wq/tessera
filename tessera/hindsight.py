"""Hindsight review — after-action review pass (research 4.10).

Primary references:
- Bjork, R. A. (1994). Memory and metamemory considerations in the
  training of human beings. In J. Metcalfe & A. P. Shimamura (eds.),
  Metacognition: Knowing About Knowing. MIT Press.
- US Army Combined Arms Center (1993). A Leader's Guide to
  After-Action Reviews. TC 25-20.
- Argyris, C., Schön, D. (1978). Organizational Learning: A Theory
  of Action Perspective. Addison-Wesley.

After a plan completes, the hindsight pass compares:
  - INTENDED outcome (from the plan's declared intent / expected
    return)
  - ACTUAL outcome (what the plan actually returned)
  - DECLARED ETHICS (which ethics principles were supposedly applied)
  - APPLIED ETHICS (from the audit trail)

Discrepancies become hindsight:learning events. These feed
tsr:evolve as a fitness signal — variants whose hindsight reviews
land closer to their declared intents get higher fitness.

CAVEAT (hindsight bias, Fischhoff 1975): "I knew it all along"
distorts retrospective judgment. The discrepancy report shows the
PRIOR (what was expected) AND the POSTERIOR (what happened)
SEPARATELY so a reader (or downstream learning loop) can compare
fairly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HindsightReview:
    """One after-action review report."""
    plan_name: str
    intended_outcome: Any
    actual_outcome: Any
    declared_ethics: list[str]
    applied_ethics: list[str]
    intended_ethics_missed: list[str]
    unexpected_ethics_applied: list[str]
    outcome_matched: bool
    notes: str = ""

    def to_audit(self) -> dict:
        """Return a dict suitable for record_event."""
        return {
            "action": f"hindsight:learning:{self.plan_name}",
            "plan": self.plan_name,
            "intended_outcome": repr(self.intended_outcome),
            "actual_outcome": repr(self.actual_outcome),
            "declared_ethics": list(self.declared_ethics),
            "applied_ethics": list(self.applied_ethics),
            "intended_ethics_missed": list(self.intended_ethics_missed),
            "unexpected_ethics_applied": list(self.unexpected_ethics_applied),
            "outcome_matched": self.outcome_matched,
            "notes": self.notes,
        }


def review(
    plan_name: str,
    intended_outcome: Any,
    actual_outcome: Any,
    *,
    declared_ethics: list[str] | None = None,
    applied_ethics: list[str] | None = None,
    match_predicate=None,
) -> HindsightReview:
    """Run the after-action comparison.

    `match_predicate(intended, actual) -> bool` defaults to
    equality; authors can pass a custom comparator (e.g. fuzzy
    string match, numeric tolerance).
    """
    declared = list(declared_ethics or [])
    applied = list(applied_ethics or [])

    intended_missed = [e for e in declared if e not in applied]
    unexpected = [e for e in applied if e not in declared]

    pred = match_predicate or (lambda i, a: i == a)
    matched = pred(intended_outcome, actual_outcome)

    notes_parts = []
    if not matched:
        notes_parts.append("outcome diverged from intent")
    if intended_missed:
        notes_parts.append(f"ethics declared but not applied: {intended_missed}")
    if unexpected:
        notes_parts.append(f"ethics applied without declaration: {unexpected}")

    return HindsightReview(
        plan_name=plan_name,
        intended_outcome=intended_outcome,
        actual_outcome=actual_outcome,
        declared_ethics=declared,
        applied_ethics=applied,
        intended_ethics_missed=intended_missed,
        unexpected_ethics_applied=unexpected,
        outcome_matched=matched,
        notes="; ".join(notes_parts) or "clean run",
    )


def fitness_from_reviews(reviews: list[HindsightReview]) -> float:
    """Aggregate fitness signal from many hindsight reviews.

    score in [0, 1]:
      - 1.0 = every review matched outcome AND applied every declared
        ethic AND no unexpected ethics fires.
      - 0.0 = no review matched.

    Used by the shipped tsr:evolve substrate as a fitness function
    alternative to eval_pass_rate.
    """
    if not reviews:
        return 0.0
    total = 0.0
    for r in reviews:
        s = 1.0 if r.outcome_matched else 0.0
        if r.intended_ethics_missed:
            s *= max(0.0, 1 - 0.5 * len(r.intended_ethics_missed) / max(1, len(r.declared_ethics)))
        if r.unexpected_ethics_applied:
            s *= max(0.0, 1 - 0.25 * len(r.unexpected_ethics_applied) / max(1, len(r.applied_ethics)))
        total += s
    return total / len(reviews)
