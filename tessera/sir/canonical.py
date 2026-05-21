"""Canonicalize SIR — α-rename node ids in deterministic dependency order.

Two SIRs that differ only in random UUID-based node ids will hash to the same
verify cache key after canonicalization. This significantly improves cache hit
rate when re-compiling unchanged source.

Strategy:
  1. Walk regions in declaration order
  2. Walk nodes in declaration order within each region
  3. Assign sequential ids n0, n1, n2, ... matching that order
  4. Rewrite inputs to point at the new ids
  5. Re-emit the textual SIR

We do NOT rewrite the in-memory Module — canonicalization happens at the
textual SIR level after `emit_module()`, just before we hand to AEON.
"""
from __future__ import annotations

import re

_NODE_DEF_RE = re.compile(r"%(\w+)\s*=")
_NODE_REF_RE = re.compile(r"%(\w+)")


def canonicalize(sir_text: str) -> str:
    """Return a canonical form of the SIR text — same content modulo id renaming."""
    # Collect all defs in textual order, build remap table
    remap: dict[str, str] = {}
    counter = 0
    for m in _NODE_DEF_RE.finditer(sir_text):
        original = m.group(1)
        if original not in remap:
            remap[original] = f"n{counter:04d}"
            counter += 1

    # Also include nodes that only appear as references but not as defs
    # (shouldn't normally happen, but keeps the pass robust)
    for m in _NODE_REF_RE.finditer(sir_text):
        if m.group(1) not in remap:
            remap[m.group(1)] = f"n{counter:04d}"
            counter += 1

    def _replace(m: re.Match) -> str:
        ident = m.group(1)
        return "%" + remap.get(ident, ident)

    return _NODE_REF_RE.sub(_replace, sir_text)
