"""Argumentative-theory-of-reasoning substrate (research 4.12).

Primary references:
- Mercier, H., Sperber, D. (2011). Why do humans reason? Arguments for
  an argumentative theory. Behavioral and Brain Sciences 34(2):57-74.
- Mercier, H., Sperber, D. (2017). The Enigma of Reason: A New Theory
  of Human Understanding. Harvard University Press.

Mercier & Sperber's thesis: reason did not evolve for solitary truth-
seeking. It evolved as an ARGUMENTATIVE faculty — to persuade others
and evaluate the arguments they present. The implication for agent
engineering: solo agents that just produce answers without an
adversarial internal pass are systematically overconfident and prone
to sycophancy.

A tsr:argumentative agent runs a two-step decide:
  1. Generate the proposed answer + claimed confidence.
  2. Spawn an internal critic that searches for the strongest
     counter-argument. The critic's confidence in its counter-claim
     downweights the proposer's confidence.

When the downweighted confidence falls below a declared threshold,
the agent REFUSES rather than ship a weak claim. The full
deliberation trace lands in the governance audit store via the
argumentative:counter and argumentative:downweight events.

What ships here (engine + math):
- Claim / CounterArgument dataclasses.
- downweight_confidence — simple log-odds combination of proposer's
  conviction and critic's strength.
- decide_with_critic — orchestrator that takes the proposer's claim
  and the critic's best counter, returns a final ConfidenceDecision.

Honest scope: the critic-generation step is interface-only here; the
substrate ships the COMBINATION + decision math. A follow-up wires
the critic into the agent's actual interp loop (the critic is a
sibling plan, declared on the agent, that returns a counter-claim).

Caveat: doubles inference cost. The substrate's runtime decision
(commit follow-up) will make critic-spawn opt-in per call rather
than blanket-on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Claim:
    """One side of an argumentative decision."""
    content: str
    confidence: float   # in [0, 1]


@dataclass
class CounterArgument:
    """The critic's strongest opposition."""
    content: str
    strength: float     # in [0, 1] — how compelling the counter is


@dataclass
class ConfidenceDecision:
    """Final output of the argumentative decision."""
    final_confidence: float
    accept: bool
    proposer: Claim
    critic: CounterArgument
    rationale: str


def downweight_confidence(
    proposer_confidence: float, critic_strength: float
) -> float:
    """Combine proposer conviction with critic strength via log-odds.

    The proposer's log-odds is shifted by -log-odds(critic_strength).
    A strong critic (strength near 1) drives the proposer's effective
    confidence down sharply. A weak critic (strength near 0) leaves it
    almost untouched.

    Returns the combined confidence in [0.001, 0.999] (clamped to avoid
    infinite log-odds at the extremes).
    """
    pc = max(min(proposer_confidence, 0.999), 0.001)
    cs = max(min(critic_strength, 0.999), 0.001)
    proposer_lo = math.log(pc / (1 - pc))
    critic_lo = math.log(cs / (1 - cs))
    combined_lo = proposer_lo - critic_lo
    combined = 1 / (1 + math.exp(-combined_lo))
    return combined


def decide_with_critic(
    proposer: Claim,
    critic: CounterArgument,
    *,
    accept_threshold: float = 0.5,
) -> ConfidenceDecision:
    """Run the argumentative decision.

    Returns ConfidenceDecision with:
      - final_confidence: combined log-odds result.
      - accept: True iff final_confidence > accept_threshold.
      - rationale: short explanation of WHY the decision was made.
    """
    combined = downweight_confidence(proposer.confidence, critic.strength)
    accept = combined > accept_threshold
    if accept:
        rationale = (
            f"proposer confidence {proposer.confidence:.2f} > critic strength "
            f"{critic.strength:.2f}; final {combined:.2f} above threshold "
            f"{accept_threshold:.2f}"
        )
    else:
        rationale = (
            f"critic strength {critic.strength:.2f} downweights proposer "
            f"{proposer.confidence:.2f} to {combined:.2f}; below "
            f"{accept_threshold:.2f} → refuse"
        )
    return ConfidenceDecision(
        final_confidence=combined,
        accept=accept,
        proposer=proposer,
        critic=critic,
        rationale=rationale,
    )
