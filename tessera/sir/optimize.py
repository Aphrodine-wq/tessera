"""SIR optimizer — small but ubiquitous compile-time wins.

Two passes:

  - **Constant folding**: ``BinOp(Const, Const)`` → ``Const`` if the binop
    is one of the side-effect-free arithmetic / comparison / logical ops.
    Cascades — a fold can create new fold opportunities, so we iterate
    until fixpoint or a small bound.

  - **Dead code elimination**: any node whose output is never consumed by
    another node's input AND that has no side effects (substrate in PURE_OPS)
    is removed. Conservative — we never elide ops that touch memory, spawn,
    send/recv, or any other observable effect.

Both passes are pure SIR-to-SIR transforms. They never change observable
behavior. Disable globally via ``TESSERA_NO_SIR_OPTIMIZE=1``.

Stats are stashed on the module under ``mod._optimize_stats`` for reporting.
"""
from __future__ import annotations

import os
from typing import Any

from .nodes import (
    AGENT_OPS, MEMORY_OPS, NEURAL_OPS, POLICY_OPS, PROMPT_OPS, PURE_OPS,
    TOOL_OPS, Module, Node, Op, Region,
)


SIDE_EFFECT_OPS = AGENT_OPS | MEMORY_OPS | PROMPT_OPS | TOOL_OPS | NEURAL_OPS | POLICY_OPS

# Control-flow ops whose attributes carry sub-regions — their bodies may have
# side effects we can't see from this region. Never eliminate them.
_CONTROL_FLOW_OPS = {Op.Until, Op.If, Op.Notice_Subscribe}

_CONST_FOLD_OPS = {"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">=",
                   "and", "or"}


def _try_eval_binop(op: str, a: Any, b: Any) -> tuple[bool, Any]:
    try:
        if op == "+":
            # string + non-string → string concat (matches interp semantics)
            if isinstance(a, str) and not isinstance(b, str):
                b = str(b)
            elif isinstance(b, str) and not isinstance(a, str):
                a = str(a)
            return True, a + b
        if op == "-": return True, a - b
        if op == "*": return True, a * b
        if op == "/": return True, a / b
        if op == "==": return True, a == b
        if op == "!=": return True, a != b
        if op == "<":  return True, a < b
        if op == "<=": return True, a <= b
        if op == ">":  return True, a > b
        if op == ">=": return True, a >= b
        if op == "and": return True, bool(a) and bool(b)
        if op == "or":  return True, bool(a) or bool(b)
    except Exception:
        pass
    return False, None


def _fold_constants_in(region: Region, stats: dict) -> bool:
    """One pass of constant folding. Returns True if anything changed."""
    by_id = {n.id: n for n in region.nodes}
    changed = False
    for n in region.nodes:
        if n.op is not Op.BinOp:
            continue
        op_sym = n.attributes.get("op", "")
        if op_sym not in _CONST_FOLD_OPS:
            continue
        if len(n.inputs) != 2:
            continue
        a = by_id.get(n.inputs[0])
        b = by_id.get(n.inputs[1])
        if a is None or b is None:
            continue
        if a.op is not Op.Const or b.op is not Op.Const:
            continue
        ok, value = _try_eval_binop(op_sym, a.attributes.get("value"),
                                    b.attributes.get("value"))
        if not ok:
            continue
        # Rewrite this BinOp node in-place into a Const
        n.op = Op.Const
        n.inputs = []
        n.attributes = {"value": value,
                        "type": "Bool" if isinstance(value, bool) else "any"}
        stats["folded"] = stats.get("folded", 0) + 1
        changed = True
    return changed


def _eliminate_dead_in(region: Region, stats: dict) -> None:
    """Remove nodes whose outputs are never consumed AND have no side effects.

    Conservative: keeps everything in SIDE_EFFECT_OPS, Return, and any node
    transitively reachable from a Return or side-effecting node.
    """
    if not region.nodes:
        return
    live: set[str] = set()
    by_id = {n.id: n for n in region.nodes}

    def _mark_live(node_id: str) -> None:
        if node_id in live or node_id not in by_id:
            return
        live.add(node_id)
        for inp in by_id[node_id].inputs:
            _mark_live(inp)

    for n in region.nodes:
        if (n.op in SIDE_EFFECT_OPS or n.op in _CONTROL_FLOW_OPS
                or n.op is Op.Return):
            _mark_live(n.id)

    before = len(region.nodes)
    region.nodes = [
        n for n in region.nodes
        if (n.id in live
            or n.op in SIDE_EFFECT_OPS
            or n.op in _CONTROL_FLOW_OPS)
    ]
    removed = before - len(region.nodes)
    if removed > 0:
        stats["eliminated"] = stats.get("eliminated", 0) + removed


def optimize(mod: Module, *, max_passes: int = 4) -> dict:
    """Run all optimization passes until fixpoint. Returns stats dict.

    Disabled when TESSERA_NO_SIR_OPTIMIZE=1 (returns empty stats).
    """
    if os.environ.get("TESSERA_NO_SIR_OPTIMIZE") == "1":
        return {}

    stats: dict = {"folded": 0, "eliminated": 0, "passes": 0}
    for _ in range(max_passes):
        any_change = False
        for region in mod.regions:
            if _fold_constants_in(region, stats):
                any_change = True
        stats["passes"] += 1
        if not any_change:
            break

    for region in mod.regions:
        _eliminate_dead_in(region, stats)

    mod._optimize_stats = stats  # type: ignore[attr-defined]
    return stats
