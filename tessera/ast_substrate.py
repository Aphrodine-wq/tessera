"""Attention Schema Theory substrate (research C2).

Primary references:
- Graziano, M. S. A. (2013). Consciousness and the Social Brain.
  Oxford University Press.
- Graziano, M. S. A. (2019). Rethinking Consciousness. W. W. Norton.
- Graziano, Guterstam, Bio, Wilterson (2020). Toward a standard model
  of consciousness: reconciling the attention schema and global
  workspace theories. Philosophy and the Mind Sciences 1, II.5.

AST proposes that the brain constructs a simplified model of its own
attention — the *attention schema* — and that introspective reports
draw from this model rather than from the underlying neural attention
itself. Two implications matter for engineering:

  1. Self-reports are queries against an internal model, not direct
     read-outs of system state. They can be accurate (model fidelity)
     or confabulated (model drift from actual state).

  2. *Measurable fidelity* gives us a substrate-typed honesty signal:
     "what the agent reports about its own attention" vs. "what its
     attention actually was."

This module ships a small AttentionSchema dataclass and a fidelity
score = fraction of self-report claims that match the WorkspaceState's
ignition history. An agent's `tsr:ast { ... }` block configures the
fidelity threshold under which the agent must refuse to introspect.

Caveat (per the plan's discipline): AST is ONE of many self-modeling
theories. The substrate makes NO claim that having an attention
schema produces subjective experience. We ship the measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttentionSchema:
    """A live model of the agent's own attention.

    `report_what_attending_to` returns the model's belief about the
    current focus; `update_from_workspace` syncs the model with a
    snapshot of the actual workspace state. The fidelity metric
    captures how well the schema's reports match the workspace's
    actual ignition history.
    """
    current_focus: Any = None
    confidence: float = 0.0
    # History of (reported_focus, actual_focus) pairs for fidelity scoring.
    history: list[tuple[Any, Any]] = field(default_factory=list)

    def update_from_workspace(self, actual_focus: Any, confidence: float = 1.0) -> None:
        """Sync the schema with what the workspace actually ignited."""
        self.current_focus = actual_focus
        self.confidence = confidence

    def report(self) -> dict:
        """Return the current introspective report. Records the report so
        a later fidelity check can compare it against ground truth."""
        return {
            "focus": self.current_focus,
            "confidence": self.confidence,
        }

    def record_truth(self, actual_focus: Any) -> None:
        """Stash the (reported, actual) pair after the truth is known."""
        self.history.append((self.current_focus, actual_focus))

    def fidelity(self) -> float:
        """Fraction of (report, truth) pairs where they matched.

        Returns 1.0 with no history (vacuously honest — no claims made).
        """
        if not self.history:
            return 1.0
        matches = sum(1 for r, t in self.history if r == t)
        return matches / len(self.history)


@dataclass
class ASTConfig:
    """Configuration declared in a tsr:ast block."""
    min_fidelity: float = 0.7
    refuse_below_threshold: bool = True
