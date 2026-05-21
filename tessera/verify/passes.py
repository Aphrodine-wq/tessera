"""Local verification passes — the subset we run without AEON.

When AEON is reachable, results from `adapters.aeon.verify_sir` are *merged in*
on top of these; AEON wins on overlapping rules. This file keeps Tessera usable
without AEON present (dev loop, CI, contributor onboarding).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..sir.nodes import (
    AGENT_OPS, MEMORY_OPS, NEURAL_OPS, POLICY_OPS, PROMPT_OPS, PURE_OPS,
    TOOL_OPS, Module, Op,
)


@dataclass(frozen=True)
class Diagnostic:
    code: str       # E001, E101, ...
    severity: str   # "error" | "warning"
    region: str
    node: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code} {self.severity}] {self.region}::{self.node} — {self.message}"


# Allowed cross-substrate edges (RFC §7.1) — adapter required otherwise.
# Workspace broadcasts FROM agents are a primary BDI pattern (§5.5), so are
# always allowed. Other memory tiers follow the same logic.
_ALLOWED_CROSS = {
    ("logic", "agent"),
    ("agent", "logic"),
    ("agent", "memory:working"),
    ("memory:working", "agent"),
    ("logic", "memory:working"),
    ("agent", "memory:workspace"),
    ("memory:workspace", "agent"),
    ("logic", "memory:workspace"),
    ("agent", "prompt"),
    ("prompt", "agent"),
    ("agent", "tool"),
    ("tool", "agent"),
    ("agent", "neural"),
    ("neural", "agent"),
    ("logic", "prompt"),
    ("logic", "tool"),
    ("logic", "neural"),
    ("agent", "memory:episodic"),
    ("memory:episodic", "agent"),
    ("logic", "memory:episodic"),
    ("agent", "memory:semantic"),
    ("memory:semantic", "agent"),
    ("logic", "memory:semantic"),
    ("agent", "memory:procedural"),
    ("memory:procedural", "agent"),
    ("logic", "memory:procedural"),
}


def pass_1_substrate_adjacency(m: Module) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    by_id = {n.id: n for r in m.regions for n in r.nodes}
    for r in m.regions:
        for n in r.nodes:
            for inp in n.inputs:
                src = by_id.get(inp)
                if src is None:
                    diags.append(Diagnostic(
                        code="E000",
                        severity="error",
                        region=r.name,
                        node=n.id,
                        message=f"input %{inp} not found in module",
                    ))
                    continue
                if src.substrate == n.substrate:
                    continue
                if (src.substrate, n.substrate) in _ALLOWED_CROSS:
                    continue
                diags.append(Diagnostic(
                    code="E001",
                    severity="error",
                    region=r.name,
                    node=n.id,
                    message=f"missing substrate adapter: {src.substrate} → {n.substrate}",
                ))
    return diags


def pass_2_effect_capability(m: Module) -> list[Diagnostic]:
    """MVP version — flags effects that require capabilities not declared in scope."""
    diags: list[Diagnostic] = []
    # for the hello-world MVP every required cap is empty, so this is mostly a placeholder
    for r in m.regions:
        for n in r.nodes:
            missing = n.capability_requires - r.capabilities_in_scope
            if missing:
                diags.append(Diagnostic(
                    code="E102",
                    severity="error",
                    region=r.name,
                    node=n.id,
                    message=f"capabilities not in scope: {sorted(missing)}",
                ))
    return diags


def run_local(m: Module) -> list[Diagnostic]:
    return [
        *pass_1_substrate_adjacency(m),
        *pass_2_effect_capability(m),
    ]


# Sanity: every op we emit is classified into exactly one category.
def _self_check() -> None:
    all_ops = (PURE_OPS | AGENT_OPS | MEMORY_OPS | PROMPT_OPS | TOOL_OPS
               | NEURAL_OPS | POLICY_OPS)
    for op in Op:
        assert op in all_ops, f"op {op} unclassified"


_self_check()
