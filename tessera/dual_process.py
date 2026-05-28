"""Dual-process router — System 1 / System 2 (research 4.1).

Primary references:
- Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and
  Giroux.
- Evans, J. St. B. T., Stanovich, K. E. (2013). Dual-process theories
  of higher cognition: advancing the debate. Perspectives on
  Psychological Science 8(3):223-241.

A plan can be tagged with a `mode`:
  - fast: pattern-match, cached, low-cost. Used when budget is tight
    and the agent's confidence in the cached pattern is high.
  - slow: deliberative, full evaluation. Used on novel inputs, low-
    confidence patterns, or when an irreversible-action gate
    (composes with tsr:autonomy) forces deliberation.

The router takes (mode_preference, budget, confidence, irreversible)
and returns the actual mode to run plus a rationale string. The
rationale is audit-emitted as `dual_process:route`.

Honest scope: dual-process is a useful abstraction, not a precise
neuroscience claim. Evans & Stanovich (2013) themselves acknowledge
the contested edges (especially around whether the two systems are
discrete or a continuum). The substrate uses the abstraction
operationally — it doesn't claim cognitive-architecture truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["fast", "slow"]


@dataclass
class RoutingDecision:
    mode: Mode
    rationale: str
    confidence_used: float
    budget_used: float
    forced_slow: bool = False


def route(
    *,
    preferred: Mode = "fast",
    confidence: float = 1.0,
    budget: float = 1.0,
    confidence_threshold: float = 0.7,
    budget_threshold: float = 0.2,
    irreversible: bool = False,
) -> RoutingDecision:
    """Pick the actual mode based on conditions.

    Rules (in order):
      1. If irreversible → force slow.
      2. If preferred=slow → run slow.
      3. If confidence < threshold → escalate to slow.
      4. If budget < threshold → demote to fast (regardless of preference).
      5. Otherwise honor preferred.
    """
    if irreversible:
        return RoutingDecision(
            mode="slow",
            rationale="forced_slow: irreversible_action",
            confidence_used=confidence,
            budget_used=budget,
            forced_slow=True,
        )
    if preferred == "slow":
        return RoutingDecision(
            mode="slow",
            rationale="preferred_slow",
            confidence_used=confidence,
            budget_used=budget,
        )
    if confidence < confidence_threshold:
        return RoutingDecision(
            mode="slow",
            rationale=f"low_confidence ({confidence:.2f} < {confidence_threshold:.2f})",
            confidence_used=confidence,
            budget_used=budget,
        )
    if budget < budget_threshold:
        return RoutingDecision(
            mode="fast",
            rationale=f"low_budget ({budget:.2f} < {budget_threshold:.2f})",
            confidence_used=confidence,
            budget_used=budget,
        )
    return RoutingDecision(
        mode=preferred,
        rationale=f"preferred_{preferred}",
        confidence_used=confidence,
        budget_used=budget,
    )
