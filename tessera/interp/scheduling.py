"""The orchestration spine — one ordering currency for every scheduling decision.

Tessera's runtime makes three kinds of orchestration choice, and historically
each used its own ad-hoc rule. This module unifies them behind a single scalar,
**salience ∈ [0, 1]** (called *priority* when it lives on a plan). Higher always
wins. The three choices and the one currency that resolves them:

    1. Which plan runs first?           plan `priority`      (a salience)
    2. Which contender wins the board?  workspace `arbiter`  (reduces saliences)
    3. What happens when a child fails? `supervise` retries  (re-drive, then abstain)

Because they share a currency, the whole orchestration layer reads as one idea:
*the most salient thing wins, and you can see why in the audit trail.* The plan
that ran first, the contender that won the blackboard, and the retry a
supervisor spent are all the same number flowing through `world.record`.

The arbiter registry below is the pluggable heart of the blackboard. An arbiter
is a pure reducer over accumulated `(value, salience)` contenders that returns
the winning pair — or ``None`` to *abstain* (e.g. a quorum that wasn't met),
which leaves the previous winner standing. Adding a new coordination strategy
is one entry here; nothing else in the runtime changes.
"""
from __future__ import annotations

from typing import Any, Callable

# The neutral salience/priority. A plan or broadcast that names no number sits
# here, so unannotated programs keep their declaration-order behavior.
DEFAULT_SALIENCE = 0.5

# An arbiter reduces accumulated contenders to a single winner (or abstains).
Contenders = list[tuple[Any, float]]
Arbiter = Callable[[Contenders, int], "tuple[Any, float] | None"]


def split_arbiter(spec: str) -> tuple[str, int]:
    """Split an arbiter spec into ``(name, param)``.

    Accepts a bare name (``"weighted_vote"``) or a parameterized one
    (``"quorum(2)"`` / ``"quorum:2"``). The integer param defaults to 0 and is
    only meaningful to arbiters that read it (currently ``quorum``).
    """
    spec = (spec or "").strip()
    param = 0
    if "(" in spec and spec.endswith(")"):
        name, _, rest = spec.partition("(")
        try:
            param = int(rest[:-1].strip())
        except ValueError:
            param = 0
        return name.strip(), param
    if ":" in spec:
        name, _, rest = spec.partition(":")
        try:
            param = int(rest.strip())
        except ValueError:
            param = 0
        return name.strip(), param
    return spec, param


def _highest_salience(contenders: Contenders, param: int) -> tuple[Any, float] | None:
    return max(contenders, key=lambda c: c[1]) if contenders else None


def _last_write(contenders: Contenders, param: int) -> tuple[Any, float] | None:
    return contenders[-1] if contenders else None


def _tally(contenders: Contenders) -> dict[Any, float]:
    """Sum salience per distinct value. Unhashable values fall back to repr so
    structured drafts (lists/records) can still be voted on."""
    totals: dict[Any, float] = {}
    order: list[Any] = []
    for value, salience in contenders:
        try:
            key = value
            hash(key)
        except TypeError:
            key = repr(value)
        if key not in totals:
            totals[key] = 0.0
            order.append((key, value))
        totals[key] += salience
    # Re-key back to the original value objects, preserving first-seen order.
    return {orig: totals[key] for key, orig in order}


def _weighted_vote(contenders: Contenders, param: int) -> tuple[Any, float] | None:
    """Group identical contenders, sum their salience, highest total wins.

    Consensus where agreement compounds: ten quiet votes for the same answer
    beat one loud outlier. Ties break toward the first-broadcast value.
    """
    if not contenders:
        return None
    totals = _tally(contenders)
    winner = max(totals, key=lambda v: totals[v])
    return winner, totals[winner]


def _quorum(contenders: Contenders, param: int) -> tuple[Any, float] | None:
    """A value wins only once at least ``param`` contenders back it.

    The salience-weighted winner among values that clear the quorum; if none
    do, the arbiter abstains (returns ``None``) and the board keeps its prior
    winner. ``param <= 0`` degrades to ``weighted_vote`` (any agreement wins).
    """
    if not contenders:
        return None
    counts: dict[Any, int] = {}
    for value, _ in contenders:
        try:
            key = value
            hash(key)
        except TypeError:
            key = repr(value)
        counts[key] = counts.get(key, 0) + 1
    totals = _tally(contenders)
    eligible = {
        v: t for v, t in totals.items()
        if counts[v if _hashable(v) else repr(v)] >= max(param, 1)
    }
    if not eligible:
        return None
    winner = max(eligible, key=lambda v: eligible[v])
    return winner, eligible[winner]


def _hashable(v: Any) -> bool:
    try:
        hash(v)
        return True
    except TypeError:
        return False


# The registry. One entry per coordination strategy; the workspace reduces its
# contenders through whichever the program declared.
ARBITERS: dict[str, Arbiter] = {
    "highest_salience": _highest_salience,
    "last_write": _last_write,
    "weighted_vote": _weighted_vote,
    "quorum": _quorum,
}


def arbitrate(spec: str, contenders: Contenders) -> tuple[Any, float] | None:
    """Reduce contenders to a winner via the named arbiter.

    Unknown arbiter names fall back to ``last_write`` — the historical default —
    so a typo degrades gracefully instead of crashing a run.
    """
    name, param = split_arbiter(spec)
    fn = ARBITERS.get(name, _last_write)
    return fn(contenders, param)
