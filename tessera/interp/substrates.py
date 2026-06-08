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


def evaluate_contracts(
    world, owner: str | None, target_label: str, phase: str,
    *, value: Any, intent: str | None, caps, args=(),
):
    """Evaluate author-declared `tsr:contract` clauses for one effect boundary.

    `phase` is "before" or "after". Returns the first failing contract as
    `(ok=False, contract_name, failed_clause_src, on_violation)`; if every
    matching contract's `phase` clauses hold, returns `(True, None, None, None)`.

    Pure evaluation — no refusal/retry is performed here. The caller owns that
    policy (only the call site knows how to regenerate for `retry`), so this
    stays a side-effect-free predicate over the ActionContext.
    """
    contracts = getattr(world.module, "contracts", None)
    if not contracts:
        return (True, None, None, None)
    from ..policy_lang import ActionContext
    ctx = ActionContext(
        value=value, action=target_label, args=list(args),
        agent=owner, intent=intent,
        capabilities=frozenset(caps) if caps else frozenset(),
    )
    for c in contracts.values():
        if c.target_label != target_label:
            continue
        clauses = c.before if phase == "before" else c.after
        for expr, src in clauses:
            try:
                holds = bool(expr.eval(ctx))
            except Exception as e:  # a clause that errors is a failed guarantee
                world.record(owner, "contract:error", contract=c.name,
                             phase=phase, clause=src, error=str(e))
                return (False, c.name, src, c.on_violation)
            if not holds:
                return (False, c.name, src, c.on_violation)
    return (True, None, None, None)


def enforce_contract_refusal(world, owner, eval_result, target_label, phase):
    """Resolve a contract evaluation at a Refusal-style site (plan / tool).

    Returns a `Refusal` to block the action, or None to proceed. An `audit`
    violation records and proceeds; `refuse` (and `retry` at a site that can't
    regenerate) records and returns a Refusal.
    """
    ok, cname, clause, ov = eval_result
    if ok:
        return None
    from .eval import Refusal
    if ov and ov[0] == "audit":
        world.record(owner, "contract:audit", contract=cname, phase=phase,
                     clause=clause, target=target_label)
        return None
    world.record(owner, "contract:refuse", contract=cname, phase=phase,
                 clause=clause, target=target_label)
    return Refusal(reason=f"contract {cname}: {phase} clause failed — {clause}",
                   policy=f"contract:{cname}")


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

    # --- dual_process: route this plan fast vs slow (audit + store mode) ---
    if module.dual_process is not None:
        from ..dual_process import route
        dp = module.dual_process
        try:
            conf = float(state.working_memory.get("_confidence", dp.default_confidence))
        except (TypeError, ValueError):
            conf = dp.default_confidence
        hay = plan_name.lower()
        irreversible = any(t.lower() in hay for t in dp.irreversible_terms)
        decision = route(
            preferred=dp.preferred,
            confidence=conf,
            budget=1.0,  # budget tracking is not yet wired; assume ample
            confidence_threshold=dp.confidence_threshold,
            budget_threshold=dp.budget_threshold,
            irreversible=irreversible,
        )
        sub["dual_process_mode"] = decision.mode
        world.record(owner, "dual_process:route", mode=decision.mode,
                     rationale=decision.rationale, confidence=round(conf, 4),
                     forced_slow=decision.forced_slow, cycle=cycle)

    # --- contract: author-declared `before` clauses bound to this plan ---
    plan_intent = state.active_plan.intent if state.active_plan else None
    res = evaluate_contracts(world, owner, f"plan:{plan_name}", "before",
                             value=plan_name, intent=plan_intent,
                             caps=state.capabilities)
    refusal = enforce_contract_refusal(world, owner, res, f"plan:{plan_name}", "before")
    if refusal is not None:
        return refusal

    return None


def on_prompt_input(world, owner: str, prompt_name: str, rendered: str, caps):
    """Pre-generation action gates: precaution and moral_foundations.

    Both run BEFORE the LLM call (like tsr:autonomy), matching declared action
    classes / violation terms against the rendered prompt text + action label.
    Returns a refusal string to short-circuit the prompt, else None.
    """
    module = world.module
    label = f"prompt:{prompt_name}"
    hay = f"{rendered} {label}".lower()

    # --- precaution: refuse high-tail / irreversible action classes ---
    if module.precaution is not None:
        from ..precaution import HarmThreshold, precaution_gate
        for spec in module.precaution.thresholds:
            if spec.action_class.lower() not in hay:
                continue
            threshold = HarmThreshold(
                action_class=spec.action_class,
                harm_magnitude=spec.harm_magnitude,
                irreversible=spec.irreversible,
                max_tail_probability=spec.max_tail_probability,
            )
            # No evidence pins the tail down yet (a tsr:bayesian posterior will,
            # in Phase 4) — assume the declared default. Precaution shifts the
            # burden onto the action.
            decision = precaution_gate(threshold, module.precaution.default_tail)
            if decision.verdict == "refuse":
                world.record(owner, f"precaution:refuse:{prompt_name}",
                             action_class=spec.action_class,
                             tail=module.precaution.default_tail,
                             rationale=decision.rationale)
                return f"[precaution-refused: {spec.action_class} — {decision.rationale}]"

    # --- moral_foundations: refuse actions that violate a weighted axis ---
    if module.moral_foundations is not None:
        from ..moral_foundations import (ActionMoralScore, FoundationWeights,
                                          score_action)
        mf = module.moral_foundations
        matched: dict[str, float] = {}
        for axis, terms in mf.violations.items():
            if any(t.lower() in hay for t in terms):
                matched[axis] = -1.0
        if matched:
            action = ActionMoralScore(**matched)
            weights = FoundationWeights(**mf.weights) if mf.weights else FoundationWeights()
            decision = score_action(action, weights,
                                    accept_threshold=mf.accept_threshold)
            if not decision.accept:
                world.record(owner, f"moral_foundations:refuse:{prompt_name}",
                             refuse_axes=decision.refuse_axes,
                             weighted_total=round(decision.weighted_total, 4))
                return (f"[moral-refused: violates {decision.refuse_axes} "
                        f"(weighted_total={decision.weighted_total:.2f})]")

    return None


def on_prompt_output(world, owner: str, prompt_name: str, output: str):
    """Post-generation substrate hooks, invoked after a prompt returns.

    Runs tom (manipulation refusal), gricean (maxim scoring), and argumentative
    (adversarial downweight) in order. Returns a replacement (refusal) string
    if any gate blocks the output, else None.
    """
    module = world.module

    # --- tom: refuse outputs that would create a false belief in a tracked agent ---
    if (module.tom is not None and module.tom.manipulation_refusal
            and module.tom.tracked_agents):
        low = output.lower()
        # An output is suspect when it both names a tracked agent and asserts a
        # claim the agent has recorded as false via a `tom_false(...)` episodic
        # marker — grounded in the agent's own ground truth, not guessed intent.
        state = world.state_for(owner)
        false_claims = [
            args for (name, args, _seq) in state.episodic if name == "tom_false"
        ]
        for agent in module.tom.tracked_agents:
            if agent.lower() not in low:
                continue
            for args in false_claims:
                claim = str(args[0]).lower() if args else ""
                if claim and claim in low:
                    world.record(owner, f"tom:manipulation_refused:{prompt_name}",
                                 tracked_agent=agent, false_claim=args[0] if args else "")
                    return (f"[tom-refused: output would leave {agent} with a false "
                            f"belief about {args[0] if args else 'the situation'}]")

    # --- gricean: score the output against the cooperative maxims ---
    if module.gricean is not None:
        from ..gricean import check_all_maxims
        gd = module.gricean
        results = check_all_maxims(
            output,
            min_words=gd.min_words, max_words=gd.max_words,
            evidence_keywords=gd.evidence_keywords,
            topic_keywords=gd.topic_keywords,
        )
        violated = [r for r in results if r.violated]
        for r in violated:
            world.record(owner, f"gricean:violation:{r.maxim}", prompt=prompt_name,
                         score=round(r.score, 3), reason=r.reason,
                         gated=r.maxim in gd.gate_maxims)
        gating = [r for r in violated if r.maxim in gd.gate_maxims]
        if gating:
            maxims = [r.maxim for r in gating]
            return (f"[gricean-refused: output violates maxim(s) {maxims} — "
                    f"{'; '.join(r.reason for r in gating)}]")

    # --- argumentative: adversarial second pass downweights confidence ---
    if module.argumentative is not None and prompt_name != module.argumentative.critic:
        ad = module.argumentative
        critic_text = _run_critic(world, ad.critic, output)
        if critic_text is not None:
            from ..argumentative import (Claim, CounterArgument, decide_with_critic)
            strength = _critic_strength(critic_text, ad.refutation_markers)
            decision = decide_with_critic(
                Claim(content=output, confidence=ad.proposer_confidence),
                CounterArgument(content=critic_text, strength=strength),
                accept_threshold=ad.accept_threshold,
            )
            world.record(owner, f"argumentative:counter:{prompt_name}",
                         critic=ad.critic, strength=round(strength, 3))
            world.record(owner, f"argumentative:downweight:{prompt_name}",
                         final_confidence=round(decision.final_confidence, 3),
                         accept=decision.accept, rationale=decision.rationale)
            if not decision.accept:
                return (f"[argumentative-refused: critic downweighted confidence to "
                        f"{decision.final_confidence:.2f} < {ad.accept_threshold} — "
                        f"{decision.rationale}]")
    return None


def _critic_strength(critic_text: str, markers: list[str]) -> float:
    """Counter-argument strength from declared refutation markers in the critic
    output. Each distinct marker adds 0.25 over a 0.2 floor, capped at 0.95 —
    a critic raising several objections strongly downweights the proposer."""
    low = critic_text.lower()
    hits = sum(1 for mk in markers if mk.lower() in low)
    return min(0.95, 0.2 + 0.25 * hits)


def _run_critic(world, critic_name: str, claim_text: str) -> str | None:
    """Render + run the declared critic prompt against the proposer's claim.

    Calls the backend directly (not _call_prompt) so the critic pass doesn't
    recurse back through the substrate gates. Returns None if no critic prompt
    is declared/found — argumentative then has nothing to push against.
    """
    if not critic_name:
        return None
    pd = world.module.prompts.get(critic_name)
    if pd is None:
        return None
    tmpl = pd.template
    if pd.params:
        pname = pd.params[0][0]
        tmpl = tmpl.replace("{" + pname + "}", str(claim_text))
    from ..adapters.llm import get_backend
    return get_backend().complete(tmpl).text


def on_plan_exit(world, owner: str, plan_name: str, plan_region, result):
    """hindsight after-action review + contract `after` enforcement on plan exit.

    Compares declared ethics (from tsr:ethics) against the ethics actually
    applied on this plan's prompts (from the audit trail), records the outcome,
    and accumulates the review for tsr:evolve fitness via fitness_from_reviews.

    Returns a `Refusal` if a contract's `after` clause fails (the caller swaps
    it in for the plan's result); otherwise None. Plan results aren't re-driven,
    so `retry` here degrades to refuse.
    """
    module = world.module

    # --- contract: author-declared `after` clauses bound to this plan ---
    res = evaluate_contracts(world, owner, f"plan:{plan_name}", "after",
                             value=result, intent=plan_region.intent,
                             caps=world.state_for(owner).capabilities)
    refusal = enforce_contract_refusal(world, owner, res, f"plan:{plan_name}", "after")
    if refusal is not None:
        return refusal

    if module.hindsight is None or not module.hindsight.enabled:
        return None
    from ..hindsight import review
    declared = [p.name for p in module.ethics.principles] if module.ethics else []
    applied: list[str] = []
    for evt in world.audit:
        if evt.plan == plan_name and evt.agent == owner:
            for name in evt.detail.get("ethics_applied", []) or []:
                if name not in applied:
                    applied.append(name)
    r = review(
        plan_name,
        intended_outcome=plan_region.intent,
        actual_outcome=result,
        declared_ethics=declared,
        applied_ethics=applied,
    )
    world.state_for(owner).substrate_state.setdefault("hindsight_reviews", []).append(r)
    world.record(owner, f"hindsight:learning:{plan_name}",
                 outcome_matched=r.outcome_matched,
                 intended_ethics_missed=r.intended_ethics_missed,
                 unexpected_ethics_applied=r.unexpected_ethics_applied,
                 notes=r.notes)
