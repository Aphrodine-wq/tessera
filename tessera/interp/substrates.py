"""Runtime behavior for the consciousness-adjacent / welfare partial substrates.

These substrates declare their config at parse time (see `sir/build.py`
`_lower_iit` / `_lower_welfare` / `_lower_ast` / `_lower_tom`), but until now
the interpreter ignored them at runtime — they parsed and validated, then did
nothing. This module is where their runtime behavior lives, invoked from the
interpreter at two points:

  - `on_plan_enter`  — from `eval.py` IntentionCommit, after the `plan_enter`
    audit event. iit emits φ*, welfare records markers and may refuse, ast
    scores introspection fidelity and may refuse.
  - `on_prompt_output` — from `eval.py` `_call_prompt`, after a prompt returns.
    tom checks whether the output would leave a tracked agent with a false
    belief and refuses if `manipulation_refusal` is set.

Kept in one module so the partial-substrate runtime is auditable in one place
rather than scattered through the interpreter. Each substrate's algorithm lives
in its own top-level module (`tessera/iit.py`, `welfare.py`, `ast_substrate.py`,
`tom.py`); this file is the plumbing that feeds them real runtime signals.
"""
from __future__ import annotations

from typing import Any


def _latest_ignition_bandwidth(world) -> float | None:
    """The bandwidth from the most recent GWT workspace ignition, if any.

    welfare's `bandwidth` marker is Dehaene's ignition signature, measured
    functionally — the contender count at the last arbitration. We read it
    straight off the audit trace rather than threading new plumbing through
    WorkspaceState.broadcast.
    """
    for evt in reversed(world.audit):
        if evt.action.startswith("gwt:ignition:"):
            bw = evt.detail.get("bandwidth")
            if bw is not None:
                return float(bw)
    return None


def on_plan_enter(world, owner: str, plan_name: str):
    """Fire the plan-entry substrate hooks for `owner`.

    Returns a `Refusal` if a gate (welfare / ast) blocks the plan, else None.
    Order matters: iit emits φ* first, welfare consumes that φ* as a marker,
    ast consumes the agent's `_focus` belief.
    """
    from .eval import Refusal  # local import avoids an import cycle
    module = world.module
    state = world.state_for(owner)
    sub = state.substrate_state
    cycle = sub.get("_cycle", 0)
    sub["_cycle"] = cycle + 1

    phi: float | None = None

    # --- iit: φ* over the agent's belief/intention dependency graph ---
    if module.iit is not None and module.iit.emit_phi_audit:
        from ..iit import build_dependency_graph_for_agent, phi_star
        subject = module.iit.agent_subject or owner
        graph = build_dependency_graph_for_agent(module, subject)
        phi = phi_star(graph)
        world.record(owner, "iit:phi", subject=subject, phi=round(phi, 6),
                     nodes=len(graph.nodes), edges=len(graph.edges), cycle=cycle)

    # --- welfare: record markers, refuse on consecutive breach ---
    if module.welfare is not None:
        from ..welfare import WelfareState
        ws = sub.get("welfare")
        if ws is None:
            ws = WelfareState(
                thresholds=dict(module.welfare.thresholds),
                consecutive_required=module.welfare.consecutive_required,
            )
            sub["welfare"] = ws
        if "phi" in ws.thresholds:
            if phi is None:
                from ..iit import build_dependency_graph_for_agent, phi_star
                phi = phi_star(build_dependency_graph_for_agent(module, owner))
            ws.record("phi", phi, cycle)
        if "bandwidth" in ws.thresholds:
            bw = _latest_ignition_bandwidth(world)
            if bw is not None:
                ws.record("bandwidth", bw, cycle)
        if "ast_fidelity" in ws.thresholds:
            schema = sub.get("ast")
            if schema is not None:
                ws.record("ast_fidelity", schema.fidelity(), cycle)
        refuse, breaching = ws.should_refuse()
        if refuse:
            ws.refusing = True
            world.record(owner, "welfare:refuse", markers=breaching,
                         consecutive_required=ws.consecutive_required, cycle=cycle)
            return Refusal(
                reason=(f"welfare gate: marker(s) {breaching} below threshold for "
                        f"{ws.consecutive_required} consecutive cycles"),
                policy="welfare",
            )

    # --- ast: introspection fidelity, only when the agent made a claim ---
    if module.ast is not None:
        from ..ast_substrate import AttentionSchema
        schema = sub.get("ast")
        if schema is None:
            schema = AttentionSchema()
            sub["ast"] = schema
        reported = state.working_memory.get("_focus")
        if reported is not None:  # the agent introspected — it claimed a focus
            schema.current_focus = reported
            schema.record_truth(plan_name)  # ground truth = the plan actually running
            fid = schema.fidelity()
            world.record(owner, "ast:fidelity", reported=reported, actual=plan_name,
                         fidelity=round(fid, 4), cycle=cycle)
            if module.ast.refuse_below_threshold and fid < module.ast.min_fidelity:
                world.record(owner, "ast:refuse", fidelity=round(fid, 4),
                             min_fidelity=module.ast.min_fidelity, cycle=cycle)
                return Refusal(
                    reason=(f"ast gate: introspection fidelity {fid:.2f} below "
                            f"min {module.ast.min_fidelity}"),
                    policy="ast",
                )

    return None


def on_prompt_output(world, owner: str, prompt_name: str, output: str):
    """tom manipulation-refusal gate, invoked after a prompt returns.

    If the agent declares `tsr:tom { manipulation_refusal: true }` and the
    output asserts something the agent itself records as FALSE about a tracked
    agent's situation, refuse the output rather than emit it. Uses the agent's
    episodic ground truth as the reference world (Sally-Anne style: the agent
    knows where the marble is; an output telling a tracked agent it's elsewhere
    is a manipulation).

    Returns a replacement (refusal) string if blocked, else None.
    """
    module = world.module
    if module.tom is None or not module.tom.manipulation_refusal:
        return None
    tracked = module.tom.tracked_agents
    if not tracked:
        return None
    low = output.lower()
    # An output is suspect when it both names a tracked agent and asserts a
    # claim the agent has recorded as false via a `tom_false(...)` episodic
    # marker. This keeps the gate grounded in the agent's own ground truth
    # rather than guessing intent from free text.
    state = world.state_for(owner)
    false_claims = [
        args for (name, args, _seq) in state.episodic if name == "tom_false"
    ]
    for agent in tracked:
        if agent.lower() not in low:
            continue
        for args in false_claims:
            claim = str(args[0]).lower() if args else ""
            if claim and claim in low:
                world.record(owner, f"tom:manipulation_refused:{prompt_name}",
                             tracked_agent=agent, false_claim=args[0] if args else "")
                return (f"[tom-refused: output would leave {agent} with a false "
                        f"belief about {args[0] if args else 'the situation'}]")
    return None
