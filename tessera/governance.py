"""Ethics + autonomy enforcement helpers.

These make the `tsr:ethics` and `tsr:autonomy` substrates real rather than
decorative: ethics principles are injected into prompts as a values frame, and
autonomy gates classify whether an action needs a human before it runs. Both
are deterministic and reuse the trait trigger-term matcher for classification.
"""
from __future__ import annotations

from .sir.nodes import AutonomyDecl, EthicsDecl
from .traits import TriggerContext, _match_term

ETHICS_OPEN = "<ethics>"
ETHICS_CLOSE = "</ethics>"


def ethics_preamble(decl: EthicsDecl) -> str:
    """Render the ethical frame as a single-line, marker-wrapped preamble.

    Principles are ordered by weight desc (the same order `on_conflict:
    highest_weight` resolves by). Injected outermost — values before posture.
    """
    ordered = sorted(decl.principles, key=lambda p: -p.weight)
    parts = [f"[{p.name}] {p.rule}" for p in ordered if p.rule]
    if not parts:
        return ""
    return f"{ETHICS_OPEN} " + " ".join(parts) + f" {ETHICS_CLOSE}\n"


def approval_term(decl: AutonomyDecl, ctx: TriggerContext, action_label: str) -> str | None:
    """Return the first `require_approval` class this action matches, else None.

    A class matches if it's a known trait term that fires on the context
    (e.g. `payments`, `auth`), or a plain keyword appearing in the rendered
    text / action label (e.g. `irreversible`).
    """
    hay = f"{ctx.text} {action_label}".lower()
    for term in decl.require_approval:
        if _match_term(term, ctx) or term.lower() in hay:
            return term
    return None
