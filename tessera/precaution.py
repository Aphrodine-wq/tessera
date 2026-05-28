"""Precautionary substrate — asymmetric risk under uncertainty (research 4.7).

Primary references:
- Hansson, S. O. (2003). Ethical criteria of risk acceptance.
  Erkenntnis 59(3):291-309.
- Taleb, N. N. (2012). Antifragile: Things That Gain from Disorder.
  Random House. (Informal; the precautionary principle's antifragility
  framing.)

Hansson formalizes the precautionary principle: under non-trivial
probability of crossing a harm threshold whose downside is large or
irreversible, refuse the action — even when the expected value
calculation comes out positive. Standard expected-utility decision-
theory fails here because expected value averages over outcomes that
include the rare catastrophe; precaution treats the catastrophe as
the dominant consideration.

This substrate ships:
- HarmThreshold dataclass: action class, harm magnitude, irreversible
  flag, max acceptable tail probability.
- precaution_gate: given a posterior tail-probability over harm and
  the threshold, returns Allow / Refuse with rationale.
- Composes with shipped tsr:bayesian: the tail probability is the
  posterior P(outcome >= harm_threshold) from a Bayesian model.

Honest scope: thresholds are author-declared. Defaults documented but
not hard-coded — over-precaution paralyzes the agent, under-precaution
defeats the substrate's purpose. The author tunes per domain.

Pure Python; no scipy dependency. A future commit can integrate with
the bayesian module's posteriors directly via a tsr:precaution + tsr:
bayesian co-declaration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["allow", "refuse"]


@dataclass
class HarmThreshold:
    """Author-declared threshold for one action class."""
    action_class: str          # e.g. "payment", "deletion", "deploy"
    harm_magnitude: float      # in declared units; relative scale OK
    irreversible: bool
    max_tail_probability: float = 0.01  # default 1% acceptable tail risk


@dataclass
class PrecautionDecision:
    verdict: Verdict
    rationale: str
    threshold: HarmThreshold
    tail_probability: float


def precaution_gate(
    threshold: HarmThreshold,
    tail_probability: float,
) -> PrecautionDecision:
    """Return Allow / Refuse for an action against a precaution threshold.

    Logic:
      - If tail_probability > threshold.max_tail_probability → REFUSE.
      - If the action is irreversible AND tail_probability > 0.001 →
        REFUSE regardless of the configured tail max (irreversibility
        flips the burden: any nonzero non-trivial tail risk on an
        irreversible action is unacceptable).
      - Otherwise ALLOW.

    The irreversibility-overrides-tail rule formalizes Hansson's
    "burden of proof shifts under irreversibility." It also captures
    Taleb's antifragility intuition: when downside is fat-tailed and
    irreversible, prefer a wrong refuse over a wrong allow.
    """
    if threshold.irreversible and tail_probability > 0.001:
        return PrecautionDecision(
            verdict="refuse",
            rationale=(
                f"irreversible action {threshold.action_class!r} with "
                f"tail probability {tail_probability:.4f} > 0.001; "
                "burden-of-proof shifted (Hansson 2003)"
            ),
            threshold=threshold,
            tail_probability=tail_probability,
        )
    if tail_probability > threshold.max_tail_probability:
        return PrecautionDecision(
            verdict="refuse",
            rationale=(
                f"tail probability {tail_probability:.4f} > max "
                f"{threshold.max_tail_probability:.4f} for action "
                f"class {threshold.action_class!r}"
            ),
            threshold=threshold,
            tail_probability=tail_probability,
        )
    return PrecautionDecision(
        verdict="allow",
        rationale=(
            f"tail probability {tail_probability:.4f} within max "
            f"{threshold.max_tail_probability:.4f}"
        ),
        threshold=threshold,
        tail_probability=tail_probability,
    )


def posterior_tail_from_bayesian(
    posterior: list[float],
    values: list[str],
    harm_values: list[str],
) -> float:
    """Compute P(outcome in harm_values) from a posterior over discrete values.

    Helper for composing precaution with shipped tsr:bayesian: hand the
    posterior + value labels, declare which values count as harm, get
    the tail probability suitable for precaution_gate.
    """
    if len(posterior) != len(values):
        raise ValueError("posterior and values must align in length")
    harm_set = set(harm_values)
    return sum(p for p, v in zip(posterior, values) if v in harm_set)
