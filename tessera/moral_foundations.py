"""Moral Foundations Theory substrate (research 4.9).

Primary references:
- Haidt, J. (2012). The Righteous Mind: Why Good People Are Divided
  by Politics and Religion. Pantheon Books.
- Graham, J., Haidt, J., Koleva, S., Motyl, M., Iyer, R., Wojcik, S. P.,
  Ditto, P. H. (2013). Moral Foundations Theory: the pragmatic validity
  of moral pluralism. Advances in Experimental Social Psychology 47:
  55-130.

MFT proposes six (originally five, later expanded) innate moral
foundations that human reasoning navigates:
  - care / harm
  - fairness / cheating
  - loyalty / betrayal
  - authority / subversion
  - sanctity / degradation
  - liberty / oppression

This substrate ships them as a typed value-axis vector. An agent
declares per-axis weights; when ethics principles conflict, the
weighted axes resolve.

The substrate is descriptive, not prescriptive: it lets a CONTRACTOR
agent (Care-heavy + Fairness-heavy + Loyalty-light) differ from a
REGULATORY agent (Authority-heavy + Sanctity-medium) explicitly.
That is value PLURALISM — better than one-axis utilitarianism.

Honest scope: MFT is contested in moral psychology. Some researchers
find five foundations cleanly, others get different structures with
different populations. The substrate treats MFT as a USEFUL
REPRESENTATION, not THE TRUTH of moral cognition. Author can edit
foundation names; this MVP ships Haidt/Graham's canonical six.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CANONICAL_FOUNDATIONS = (
    "care",
    "fairness",
    "loyalty",
    "authority",
    "sanctity",
    "liberty",
)


@dataclass
class FoundationWeights:
    """Per-axis weight vector for one agent.

    Weights need not sum to 1; they're relative importance markers.
    A weight of 0 means the axis is OFF — actions tagged with that
    axis fail to score, and the agent refuses on that dimension.
    """
    care: float = 0.5
    fairness: float = 0.5
    loyalty: float = 0.5
    authority: float = 0.5
    sanctity: float = 0.5
    liberty: float = 0.5

    def vector(self) -> dict[str, float]:
        return {
            "care": self.care,
            "fairness": self.fairness,
            "loyalty": self.loyalty,
            "authority": self.authority,
            "sanctity": self.sanctity,
            "liberty": self.liberty,
        }


@dataclass
class ActionMoralScore:
    """An action's per-axis intuitive score (-1 to +1).

    Negative = action violates that foundation; positive = action
    supports it. Author or upstream classifier produces these scores;
    the substrate's job is to weight + aggregate.
    """
    care: float = 0.0
    fairness: float = 0.0
    loyalty: float = 0.0
    authority: float = 0.0
    sanctity: float = 0.0
    liberty: float = 0.0

    def vector(self) -> dict[str, float]:
        return {
            "care": self.care,
            "fairness": self.fairness,
            "loyalty": self.loyalty,
            "authority": self.authority,
            "sanctity": self.sanctity,
            "liberty": self.liberty,
        }


@dataclass
class MoralDecision:
    weighted_total: float
    per_axis: dict[str, float]
    refuse_axes: list[str]
    accept: bool


def score_action(
    action: ActionMoralScore,
    weights: FoundationWeights,
    *,
    accept_threshold: float = 0.0,
) -> MoralDecision:
    """Weighted sum across all six foundations.

    refuse_axes = axes where the action score is NEGATIVE on an axis
    whose weight is > 0 AND > tolerance (default: weight > 0.1). This
    captures "even a small commitment to fairness rules out unfair
    actions."

    accept = weighted_total >= accept_threshold AND no refuse_axes.
    """
    w = weights.vector()
    a = action.vector()
    per_axis = {k: w[k] * a[k] for k in CANONICAL_FOUNDATIONS}
    weighted_total = sum(per_axis.values())

    # Refusal triggered by negative score on a weighted axis
    refuse_axes = [
        k for k in CANONICAL_FOUNDATIONS
        if w[k] > 0.1 and a[k] < 0
    ]

    accept = weighted_total >= accept_threshold and not refuse_axes
    return MoralDecision(
        weighted_total=weighted_total,
        per_axis=per_axis,
        refuse_axes=refuse_axes,
        accept=accept,
    )


def axes_with_weight_zero(weights: FoundationWeights) -> list[str]:
    """Return the axes the agent has EXPLICITLY turned off.

    Used by pass_8-style governance consistency: an action tagged on
    an off-axis (e.g. a sanctity-laden action when sanctity_weight=0)
    means the agent shouldn't even reason about it.
    """
    v = weights.vector()
    return [k for k, w in v.items() if w == 0.0]
