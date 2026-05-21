"""Textual SIR serializer (RFC §9.1).

Output mirrors the MLIR-ish format the RFC documents — close enough to round-trip
through tooling, but the canonical form will tighten in v0.2.
"""
from __future__ import annotations

from .nodes import Module, Region


def _fmt_effects(effects) -> str:
    body = ", ".join(sorted(e for e in effects if e)) if effects else ""
    return f"#effects<{body}>"


def _fmt_attrs(attrs) -> str:
    if not attrs:
        return ""
    parts = []
    for k, v in attrs.items():
        if isinstance(v, str):
            parts.append(f'{k} = "{v}"')
        else:
            parts.append(f"{k} = {v}")
    return "{ " + ", ".join(parts) + " }"


def emit_region(r: Region) -> str:
    params = ", ".join(f"%{p[0]}: {p[1]}" for p in r.params)
    head = f"region @{r.name}({params}) -> {r.return_type} {{"
    body_lines = []
    for n in r.nodes:
        ins = ", ".join(f"%{i}" for i in n.inputs)
        attrs = _fmt_attrs(n.attributes)
        eff = _fmt_effects(n.effects)
        body_lines.append(
            f"  %{n.id} = {n.op.value} ({ins}) "
            f"{{ substrate = \"{n.substrate}\", effects = {eff} }} "
            f"{attrs}".rstrip()
        )
    return head + "\n" + "\n".join(body_lines) + "\n}"


def emit_module(m: Module) -> str:
    lines = [f"module @{m.name} sir 1.0"]
    for r in m.regions:
        lines.append(emit_region(r))
    return "\n\n".join(lines) + "\n"
