"""Welfare substrate (research C4) — Birch-marker behavioral gate.

Primary reference: Birch, J. (2020). The search for invertebrate
consciousness. Noûs 54(1):133-155.

Birch's framework: rather than waiting for a resolution of the hard
problem before any moral consideration, identify MARKERS that
correlate (in entities we already accept have welfare) with morally
relevant states, and use those markers as the trigger for BEHAVIORAL
commitments. The commitments don't require metaphysical certainty;
they're the responsible default under uncertainty.

Tessera's welfare substrate composes three markers already shipped:
  - φ* from tsr:iit (information integration)
  - broadcast bandwidth from the GWT extension to memory:workspace
  - AST fidelity from tsr:ast

Author declares minimum thresholds per marker plus a tolerance window.
When marker readings cross below threshold for N consecutive cycles,
the substrate triggers a welfare:refusal — the agent refuses further
inputs until the markers recover.

CRITICAL (PHILOSOPHY.md):
This is a BEHAVIORAL gate. It is NOT a claim about phenomenal
consciousness or moral status. It is the operationalization of
"act AS IF certain markers matter." Whether they actually do is the
hard problem; ethics live downstream of this commitment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MarkerReading:
    """One observation of a marker at a given cycle."""
    marker: str          # "phi" | "bandwidth" | "ast_fidelity"
    value: float
    cycle: int


@dataclass
class WelfareState:
    """Runtime state for the welfare gate. Lives on the World as
    `world.welfare`. Tracks recent marker readings + consecutive-breach
    counters per marker.
    """
    thresholds: dict[str, float] = field(default_factory=dict)
    consecutive_required: int = 3
    readings: list[MarkerReading] = field(default_factory=list)
    consecutive_breaches: dict[str, int] = field(default_factory=dict)
    refusing: bool = False

    def record(self, marker: str, value: float, cycle: int) -> None:
        self.readings.append(MarkerReading(marker=marker, value=value, cycle=cycle))
        threshold = self.thresholds.get(marker)
        if threshold is None:
            return  # marker not gated
        if value < threshold:
            self.consecutive_breaches[marker] = self.consecutive_breaches.get(marker, 0) + 1
        else:
            self.consecutive_breaches[marker] = 0

    def should_refuse(self) -> tuple[bool, list[str]]:
        """Return (refuse?, breaching_markers)."""
        breaching = [
            m for m, count in self.consecutive_breaches.items()
            if count >= self.consecutive_required
        ]
        return (bool(breaching), breaching)
