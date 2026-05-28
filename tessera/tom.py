"""Theory of Mind substrate (research C3).

Primary references:
- Premack, D., Woodruff, G. (1978). Does the chimpanzee have a theory
  of mind? Behavioral and Brain Sciences, 1(4), 515-526.
- Baker, C. L., Saxe, R., Tenenbaum, J. B. (2009). Action understanding
  as inverse planning. Cognition, 113(3), 329-349.
- Rabinowitz, N. C. et al. (2018). Machine theory of mind. ICML.
  arXiv:1802.07740.

A `tsr:tom` block declares an agent maintains models of OTHER agents'
beliefs. The substrate ships two operations:

  1. Sally-Anne false-belief detection. Given an event the OTHER agent
     missed, the substrate correctly maintains a SEPARATE belief tuple
     for that agent's view, even when the current agent's world model
     has updated.
  2. Inverse planning. From a sequence of observed actions, score
     candidate (goal, belief-state) pairs by likelihood under
     rational-action assumptions. Returns ranked hypotheses.

This module is pure Python — symbolic ToM, not LLM-augmented. The
verifiable spine is the symbolic representation; LLM-augmented
listener modeling is a follow-up that wants the embedding substrate.

Caveat (per PHILOSOPHY.md): ToM is BEHAVIORAL modeling of other
agents. It says nothing about whether THOSE agents have phenomenal
experience. ToM is a structural primitive for listener-aware
communication and adversarial robustness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BeliefProposition:
    """One belief: 'X is at location Y' with confidence."""
    subject: str        # e.g. "marble"
    predicate: str      # e.g. "in_basket"
    confidence: float = 1.0

    def matches(self, other: "BeliefProposition") -> bool:
        return self.subject == other.subject and self.predicate == other.predicate


@dataclass
class BeliefAboutAgent:
    """An agent's model of OTHER_AGENT's beliefs.

    `propositions` is the set of beliefs we think OTHER_AGENT holds.
    `observed_events` is the list of events we know OTHER_AGENT
    witnessed; we use this to keep their belief state coherent under
    Sally-Anne-style scenarios.
    """
    other_agent: str
    propositions: list[BeliefProposition] = field(default_factory=list)
    observed_events: list[str] = field(default_factory=list)

    def believes(self, prop: BeliefProposition) -> bool:
        return any(p.matches(prop) for p in self.propositions)


def sally_anne_update(
    world_view: list[BeliefProposition],
    agent_views: dict[str, BeliefAboutAgent],
    event_id: str,
    witnesses: list[str],
    belief_change: tuple[BeliefProposition, BeliefProposition | None],
) -> None:
    """A canonical false-belief update.

    Args:
      world_view: the ground-truth current world state (in place).
      agent_views: per-agent belief tracker (in place).
      event_id: a unique identifier for this event.
      witnesses: agents that observed the event.
      belief_change: (old_belief, new_belief). new_belief=None means
        the old belief is just removed (e.g. marble removed entirely).

    For each tracked agent: if they witnessed, update their model.
    If they DID NOT witness, their model retains the OLD belief — even
    though the world has changed. That asymmetry is the Sally-Anne
    test signature.
    """
    old, new = belief_change

    # Update the world view (ground truth)
    world_view[:] = [p for p in world_view if not p.matches(old)]
    if new is not None:
        world_view.append(new)

    # Update each tracked agent's model
    for name, view in agent_views.items():
        if name in witnesses:
            view.propositions = [p for p in view.propositions if not p.matches(old)]
            if new is not None:
                view.propositions.append(new)
            view.observed_events.append(event_id)
        # If the agent did NOT witness, their propositions stay the same.
        # That's the false-belief: their model lags the world.


def has_false_belief(
    world_view: list[BeliefProposition],
    view: BeliefAboutAgent,
) -> list[BeliefProposition]:
    """Return the list of view.propositions that are FALSE in the current
    world. Non-empty list means this agent holds at least one false
    belief — the canonical Sally-Anne diagnostic.
    """
    false_beliefs: list[BeliefProposition] = []
    for p in view.propositions:
        # A view-proposition is false when no world-view proposition
        # matches its subject AND predicate.
        if not any(w.matches(p) for w in world_view):
            false_beliefs.append(p)
    return false_beliefs


# ----- Inverse planning -----


@dataclass
class GoalBeliefHypothesis:
    """A candidate explanation for an agent's behavior."""
    goal: str
    belief_state: list[BeliefProposition]
    score: float = 0.0


def score_hypothesis(
    actions_observed: list[str],
    hypothesis: GoalBeliefHypothesis,
    action_likelihood: dict[tuple[str, str], float],
) -> float:
    """Rational-action assumption (Baker, Saxe, Tenenbaum 2009): an
    agent acts to maximize expected utility under its beliefs and
    goal. The likelihood of an observed action sequence is the product
    of P(action | goal, belief_state) for each step.

    `action_likelihood[(action, goal)]` is the author-declared
    probability that the agent would take `action` to achieve `goal`.
    Beliefs are present in the hypothesis for completeness but the
    MVP uses goal-conditioned likelihoods only.
    """
    if not actions_observed:
        return 0.0
    log_score = 0.0
    for a in actions_observed:
        p = action_likelihood.get((a, hypothesis.goal), 0.01)  # smoothing
        log_score += _log(p)
    return log_score


def rank_hypotheses(
    actions_observed: list[str],
    hypotheses: list[GoalBeliefHypothesis],
    action_likelihood: dict[tuple[str, str], float],
) -> list[GoalBeliefHypothesis]:
    """Return hypotheses sorted by inverse-planning score descending."""
    scored = []
    for h in hypotheses:
        s = score_hypothesis(actions_observed, h, action_likelihood)
        scored.append(GoalBeliefHypothesis(goal=h.goal, belief_state=h.belief_state, score=s))
    scored.sort(key=lambda x: -x.score)
    return scored


def _log(x: float) -> float:
    """Log with floor to avoid -inf on impossible events."""
    import math
    if x <= 0:
        return -1e9
    return math.log(x)
