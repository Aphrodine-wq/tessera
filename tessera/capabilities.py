"""Canonical capability taxonomy.

Tessera capabilities are two-tier: a `Parent.Subtype` string identifies the
specific operation an agent declares it might perform. Declaring the parent
alone (`NetworkOut`) is shorthand for `NetworkOut.Any` — the broadest
subtype in that family.

The taxonomy is closed (per the closed-SIR-ops + opinionated-language
stance). Unknown capabilities produce a checker warning, not an error,
so legacy v0.1 files that used coarse capability labels keep working
while the migration to subtypes lands.

See:
- 2026-05-28 Tessera capability declarations are two-tier (parent.subtype)
- tessera/interp/eval.py — spawn capability gate (intersection rule)
"""
from __future__ import annotations

CAPABILITY_TAXONOMY: dict[str, list[str]] = {
    "NetworkOut": ["HTTPS", "HTTP", "DNS", "Any"],
    "NetworkIn":  ["HTTPS", "HTTP", "Any"],
    "FileSystem": ["ReadOnly", "ReadWrite", "TmpDir", "Any"],
    "Subprocess": ["Spawn", "Pipe", "Any"],
    "Environment": ["Read", "Write", "Any"],
    "VaultRead":  ["Any"],
    "VaultWrite": ["Any"],
    "Secrets":    ["Read", "Any"],
    "Time":       ["Any"],
    "Random":     ["Any"],
}

KNOWN_PARENTS: frozenset[str] = frozenset(CAPABILITY_TAXONOMY)


def _split(cap: str) -> tuple[str, str | None]:
    if "." in cap:
        parent, subtype = cap.split(".", 1)
        return parent, subtype
    return cap, None


def normalize(cap: str) -> str:
    """`NetworkOut` -> `NetworkOut.Any`; `NetworkOut.HTTPS` -> `NetworkOut.HTTPS`.

    Unknown parents pass through unchanged (treated as opaque legacy caps).
    """
    parent, subtype = _split(cap)
    if subtype is None and parent in KNOWN_PARENTS:
        return f"{parent}.Any"
    return cap


def is_subcap_of(child: str, parent: str) -> bool:
    """True when `child` is at-or-narrower-than `parent`.

    Same parent and same subtype → True.
    Same parent and parent is `.Any` → True (Any covers all subtypes).
    Different parents → False.
    Unknown caps fall back to string equality (no false positives).
    """
    c_parent, c_subtype = _split(normalize(child))
    p_parent, p_subtype = _split(normalize(parent))
    if c_parent != p_parent:
        return False
    if p_subtype == "Any":
        return True
    return c_subtype == p_subtype


def intersect(parent_caps: set[str], child_caps: set[str]) -> set[str]:
    """Auto-restrict rule: child gets the narrowest cap allowed for each family.

    For every cap the child declared, find the narrowest cap the PARENT holds
    in the same family. If the parent holds anything in the family the child
    gets that. If not, the cap is dropped.
    """
    granted: set[str] = set()
    p_norm = {normalize(c) for c in parent_caps}
    for raw_child in child_caps:
        child = normalize(raw_child)
        c_parent, c_subtype = _split(child)
        # Find best match among parent caps in the same family.
        candidates = [p for p in p_norm if _split(p)[0] == c_parent]
        if not candidates:
            continue
        # If parent has .Any, grant the more-specific child cap (parent allows it).
        if any(_split(p)[1] == "Any" for p in candidates):
            granted.add(child)
            continue
        # Otherwise grant only the EXACT subtypes the parent also holds.
        if child in candidates:
            granted.add(child)
    return granted


def validate(cap: str) -> str | None:
    """Return None when valid, or a one-line diagnostic message for the checker."""
    parent, subtype = _split(cap)
    if parent not in KNOWN_PARENTS:
        return None  # unknown parents are pass-through (legacy compat)
    if subtype is None or subtype in CAPABILITY_TAXONOMY[parent]:
        return None
    allowed = ", ".join(CAPABILITY_TAXONOMY[parent])
    return f"unknown capability subtype: {cap!r} (allowed for {parent}: {allowed})"
