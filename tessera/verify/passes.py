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
from ..traits import KNOWN_TERMS, resolve_trait


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


def pass_3_trait_resolution(m: Module) -> list[Diagnostic]:
    """Cognitive-trait checks.

    E300 (error): an agent attaches a trait that resolves to neither a local
    `tsr:traits` definition nor a built-in.
    E301 (warning): a locally-defined trait names a trigger term outside the
    known vocabulary — it can never fire (a likely typo).
    """
    diags: list[Diagnostic] = []
    for r in m.regions:
        for name in r.trait_names:
            if resolve_trait(name, m) is None:
                diags.append(Diagnostic(
                    code="E300",
                    severity="error",
                    region=r.name,
                    node="-",
                    message=f"trait {name!r} not defined (no local or built-in trait)",
                ))
    for tname, decl in m.traits.items():
        for term in decl.trigger:
            if term not in KNOWN_TERMS:
                diags.append(Diagnostic(
                    code="E301",
                    severity="warning",
                    region=f"trait:{tname}",
                    node="-",
                    message=f"unknown trigger term {term!r} — it will never fire",
                ))
    return diags


def pass_4_intent(m: Module) -> list[Diagnostic]:
    """Intent checks — keep declared intent honest and auditable.

    E400 (error): an intent forbids an outcome that maps to no `tsr:policy` —
    purpose stated without the guardrail that backs it.
    E402 (error): an agent/plan serves an intent that was never declared.
    E403 (warning): a plan is bound to no intent (its actions can't be audited
    against a purpose).
    E404 (warning): a declared intent has no success criteria (nothing to audit
    the outcome against).
    """
    diags: list[Diagnostic] = []
    for iname, decl in m.intents.items():
        for forb in decl.forbidden:
            if forb not in m.policies:
                diags.append(Diagnostic(
                    code="E400",
                    severity="error",
                    region=f"intent:{iname}",
                    node="-",
                    message=f"forbids {forb!r} but no policy {forb!r} is declared",
                ))
        if not decl.success:
            diags.append(Diagnostic(
                code="E404",
                severity="warning",
                region=f"intent:{iname}",
                node="-",
                message="intent has no success criteria — its outcome cannot be audited",
            ))
    for r in m.regions:
        if r.intent is not None and r.intent not in m.intents:
            diags.append(Diagnostic(
                code="E402",
                severity="error",
                region=r.name,
                node="-",
                message=f"serves undeclared intent {r.intent!r}",
            ))
        elif r.name.startswith("plan:") and r.intent is None and m.intents:
            diags.append(Diagnostic(
                code="E403",
                severity="warning",
                region=r.name,
                node="-",
                message="plan is bound to no intent — its actions can't be audited against a purpose",
            ))
    return diags


_AUTONOMY_LEVELS = {"propose", "act_with_rollback", "act_freely"}
_ETHICS_CONFLICT = {"highest_weight", "first"}
_ETHICS_VIOLATION = {"refuse", "flag", "defer"}


def pass_5_governance(m: Module) -> list[Diagnostic]:
    """Ethics + autonomy consistency.

    E500 (error): ethics principle weight out of [0,1], or invalid on_conflict /
    on_violation. E501 (warning): a principle with no rule (nothing to inject).
    E502 (error): autonomy level outside the known set.
    """
    diags: list[Diagnostic] = []
    if m.ethics is not None:
        if m.ethics.on_conflict not in _ETHICS_CONFLICT:
            diags.append(Diagnostic("E500", "error", "ethics", "-",
                f"invalid on_conflict {m.ethics.on_conflict!r} (expected one of {sorted(_ETHICS_CONFLICT)})"))
        if m.ethics.on_violation not in _ETHICS_VIOLATION:
            diags.append(Diagnostic("E500", "error", "ethics", "-",
                f"invalid on_violation {m.ethics.on_violation!r} (expected one of {sorted(_ETHICS_VIOLATION)})"))
        for p in m.ethics.principles:
            if not (0.0 <= p.weight <= 1.0):
                diags.append(Diagnostic("E500", "error", f"ethics:{p.name}", "-",
                    f"weight {p.weight} out of range [0.0, 1.0]"))
            if not p.rule:
                diags.append(Diagnostic("E501", "warning", f"ethics:{p.name}", "-",
                    "principle has no rule — nothing to inject into prompts"))
    if m.autonomy is not None and m.autonomy.level not in _AUTONOMY_LEVELS:
        diags.append(Diagnostic("E502", "error", "autonomy", "-",
            f"invalid level {m.autonomy.level!r} (expected one of {sorted(_AUTONOMY_LEVELS)})"))
    return diags


def pass_7_spawn_cycle(m: Module) -> list[Diagnostic]:
    """Build a static spawn graph from Op.Spawn nodes in every agent region
    and refuse pure cycles (decision 11 static layer). Each detected cycle
    becomes one E700 DeadlockCertain error.

    Send/Recv-based data-flow deadlocks are NOT detected here — they require
    inter-region taint analysis. The dynamic per-recv timeout (decision 11
    runtime half) is the safety net for those.
    """
    id_to_region = {r.id: r for r in m.regions}

    def owner_agent_of(region: "Region") -> str | None:
        cur = region
        while cur is not None:
            n = cur.name
            if n.startswith("agent:") and ":notice_" not in n:
                return n[len("agent:"):]
            if cur.parent is None:
                return None
            cur = id_to_region.get(cur.parent)
        return None

    graph: dict[str, set[str]] = {a: set() for a in m.agents}
    for region in m.regions:
        owner = owner_agent_of(region)
        if owner is None:
            continue
        for node in region.nodes:
            if node.op is Op.Spawn:
                target = node.attributes.get("agent")
                if target:
                    graph.setdefault(owner, set()).add(target)

    cycles: list[list[str]] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    path: list[str] = []

    def visit(n: str) -> None:
        color[n] = GRAY
        path.append(n)
        for child in graph.get(n, ()):
            c = color.get(child, WHITE)
            if c == GRAY:
                idx = path.index(child)
                cycles.append(path[idx:] + [child])
            elif c == WHITE:
                color.setdefault(child, WHITE)
                visit(child)
        path.pop()
        color[n] = BLACK

    for n in list(graph):
        if color.get(n, WHITE) == WHITE:
            visit(n)

    diags: list[Diagnostic] = []
    seen: set[tuple[str, ...]] = set()
    for cyc in cycles:
        key = tuple(sorted(set(cyc)))
        if key in seen:
            continue
        seen.add(key)
        diags.append(Diagnostic(
            code="E700",
            severity="error",
            region=f"agent:{cyc[0]}",
            node="-",
            message=f"static spawn cycle detected: {' -> '.join(cyc)}",
        ))
    return diags


def pass_6_capability_taxonomy(m: Module) -> list[Diagnostic]:
    """Walk every capability declared on a region and validate it against the
    two-tier taxonomy. Unknown subtypes warn (don't error) so legacy v0.1
    files with coarse cap labels still compile.
    """
    from ..capabilities import validate
    diags: list[Diagnostic] = []
    seen: set[str] = set()
    for region in m.regions:
        for cap in sorted(region.capabilities_in_scope):
            if cap in seen:
                continue
            seen.add(cap)
            msg = validate(cap)
            if msg:
                diags.append(Diagnostic(
                    code="E600",
                    severity="warning",
                    region=region.name,
                    node="-",
                    message=msg,
                ))
    return diags


def pass_8_governance_consistency(m: Module) -> list[Diagnostic]:
    """Sampling-based satisfiability check on governance composition
    (decision 18). For each policy carrying `forbid_when` / `permit_when`
    constraint expressions, enumerate a small set of representative
    ActionContexts and verify at least one passes — if every sampled
    action triggers refusal, the policy is over-constrained and the
    composed governance set has no satisfying behavior. Emit E1000
    GovernanceContradiction.

    This is a fast approximation, not a SAT-complete proof. A real Z3
    backend lands later via AEON. Catches the obvious bugs (forbid when
    true, contradictory permit+forbid pairs) without taking compile time
    proportional to action-space size.
    """
    diags: list[Diagnostic] = []
    if not m.policies:
        return diags
    from ..policy_lang import ActionContext

    # Small sample action space — covers the common shapes
    samples: list[ActionContext] = [
        ActionContext(value="benign string", action="prompt:foo",
                      capabilities=frozenset({"NetworkOut.HTTPS"})),
        ActionContext(value="another value", action="tool:bar",
                      capabilities=frozenset({"FileSystem.ReadOnly"})),
        ActionContext(value="x", action="plan_enter:p",
                      capabilities=frozenset()),
        ActionContext(value="hello world", action="prompt:noop",
                      capabilities=frozenset({"NetworkOut.HTTPS",
                                              "FileSystem.ReadOnly"})),
    ]

    for pol_name, pol in m.policies.items():
        constraint_rules = [
            (kind, params) for (kind, params) in pol.rules
            if kind in ("forbid_when", "permit_when")
        ]
        if not constraint_rules:
            continue

        # A sample "passes" iff every constraint rule lets it through:
        # forbid_when must be False, permit_when must be True.
        def _passes(ctx) -> bool:
            for kind, params in constraint_rules:
                try:
                    val = bool(params["expr"].eval(ctx))
                except Exception:
                    return False
                if kind == "forbid_when" and val:
                    return False
                if kind == "permit_when" and not val:
                    return False
            return True

        if not any(_passes(s) for s in samples):
            diags.append(Diagnostic(
                code="E1000",
                severity="error",
                region=f"policy:{pol_name}",
                node="-",
                message=(
                    f"governance contradiction: policy {pol_name!r} refuses "
                    f"every sampled action — composed constraints are "
                    f"unsatisfiable on the default action space"
                ),
            ))
    return diags


def pass_9_consciousness_claim_check(m: Module) -> list[Diagnostic]:
    """Refuse modules whose consciousness-adjacent substrate declarations
    make bare metaphysical claims (per PHILOSOPHY.md).

    Cross-block: tsr:iit, tsr:welfare, and any future tsr:phenomenology
    block must operationalize their claims. Forbidden patterns are
    enumerated in tessera/iit.py::claim_violates_consciousness_discipline
    — "is conscious", "subjective experience", "phi > 0 means
    consciousness", etc.

    The individual substrate lowering functions already gate their own
    bodies (e.g., _lower_iit rejects forbidden phrases in the iit
    block). This pass adds a CROSS-FILE check: prose and frontmatter
    comments can ALSO not make bare claims when an iit or welfare
    substrate is declared in the same module.

    Emits E1100 ConsciousnessClaim (error).
    """
    diags: list[Diagnostic] = []
    if m.iit is None and m.welfare is None:
        return diags  # only gate modules that opt into consciousness-adjacent substrates
    from ..iit import claim_violates_consciousness_discipline
    # Check the module's name / region names — a crude but defensible
    # scan against the same lex. A real implementation would scan the
    # parsed module's prose; the prose is already gated when it's a
    # substrate body, so this pass catches escapes.
    for region in m.regions:
        reason = claim_violates_consciousness_discipline(region.name)
        if reason:
            diags.append(Diagnostic(
                code="E1100",
                severity="error",
                region=region.name,
                node="-",
                message=(
                    f"E1100 ConsciousnessClaim: {reason}. "
                    "See PHILOSOPHY.md — Tessera operationalizes "
                    "access-consciousness-adjacent properties only."
                ),
            ))
    return diags


def pass_10_precaution(m: Module) -> list[Diagnostic]:
    """Precaution gate lint. E800 (warning): a precaution block with no
    thresholds never fires — almost certainly an authoring mistake."""
    diags: list[Diagnostic] = []
    if m.precaution is not None and not m.precaution.thresholds:
        diags.append(Diagnostic(
            "E800", "warning", "precaution", "-",
            "precaution block declares no thresholds — the gate can never fire",
        ))
    return diags


def pass_11_dual_process(m: Module) -> list[Diagnostic]:
    """Dual-process router lint. E810 (warning): when `preferred: slow`,
    every plan already routes slow, so `irreversible_terms` are dead config."""
    diags: list[Diagnostic] = []
    dp = m.dual_process
    if dp is not None and dp.preferred == "slow" and dp.irreversible_terms:
        diags.append(Diagnostic(
            "E810", "warning", "dual_process", "-",
            "preferred=slow already routes every plan slow; irreversible_terms are redundant",
        ))
    return diags


def pass_12_moral_foundations(m: Module) -> list[Diagnostic]:
    """Moral-foundations lint. E820 (warning): declaring violation terms for a
    foundation whose weight is <= 0.1 is dead config — score_action only refuses
    on axes with weight > 0.1, so those violations can never trigger a refusal."""
    diags: list[Diagnostic] = []
    mf = m.moral_foundations
    if mf is not None:
        for axis in mf.violations:
            w = mf.weights.get(axis, 0.5)  # FoundationWeights default is 0.5
            if w <= 0.1:
                diags.append(Diagnostic(
                    "E820", "warning", f"moral_foundations:{axis}", "-",
                    f"violates {axis} declared but its weight ({w}) is <= 0.1 — "
                    "violations on this axis never refuse",
                ))
    return diags


def pass_13_gricean(m: Module) -> list[Diagnostic]:
    """Gricean lint. E900 (warning): gating the `quality` maxim without
    evidence keywords, or `relation` without topic keywords, makes that gate
    inert — those checks pass vacuously when the keyword list is empty."""
    diags: list[Diagnostic] = []
    g = m.gricean
    if g is not None:
        if "quality" in g.gate_maxims and not g.evidence_keywords:
            diags.append(Diagnostic(
                "E900", "warning", "gricean", "-",
                "gating 'quality' but no evidence keywords declared — the quality "
                "check passes vacuously, so the gate never fires",
            ))
        if "relation" in g.gate_maxims and not g.topic_keywords:
            diags.append(Diagnostic(
                "E900", "warning", "gricean", "-",
                "gating 'relation' but no topic keywords declared — the relation "
                "check passes vacuously, so the gate never fires",
            ))
    return diags


def pass_14_hindsight(m: Module) -> list[Diagnostic]:
    """Hindsight lint. E910 (warning): a review with no ethics and no intents
    to compare degrades to outcome-recording only — usually not what's wanted."""
    diags: list[Diagnostic] = []
    if (m.hindsight is not None and m.hindsight.enabled
            and m.ethics is None and not m.intents):
        diags.append(Diagnostic(
            "E910", "warning", "hindsight", "-",
            "hindsight enabled but no tsr:ethics and no tsr:intent declared — "
            "reviews record outcomes only, with nothing to compare against",
        ))
    return diags


def pass_15_argumentative(m: Module) -> list[Diagnostic]:
    """Argumentative lint. E920 (error): the named critic prompt must exist —
    without it the adversarial pass can never run and the substrate is inert."""
    diags: list[Diagnostic] = []
    a = m.argumentative
    if a is not None and a.critic and a.critic not in m.prompts:
        diags.append(Diagnostic(
            "E920", "error", "argumentative", "-",
            f"critic prompt {a.critic!r} is not defined — the adversarial pass "
            "can never run",
        ))
    return diags


def pass_16_rl(m: Module) -> list[Diagnostic]:
    """RL lint. E930: target agent must exist. E931: at least two actions —
    a one-armed bandit has nothing to learn."""
    diags: list[Diagnostic] = []
    decl = m.rl
    if decl is None:
        return diags
    if not decl.target_agent or decl.target_agent not in m.agents:
        diags.append(Diagnostic(
            "E930", "error", "rl", "-",
            f"tsr:rl target agent {decl.target_agent!r} is not defined",
        ))
    if len(decl.actions) < 2:
        diags.append(Diagnostic(
            "E931", "error", "rl", "-",
            "tsr:rl needs at least two actions — rl_choose has nothing to "
            "learn to prefer otherwise",
        ))
    return diags


def run_local(m: Module) -> list[Diagnostic]:
    return [
        *pass_1_substrate_adjacency(m),
        *pass_2_effect_capability(m),
        *pass_3_trait_resolution(m),
        *pass_4_intent(m),
        *pass_5_governance(m),
        *pass_6_capability_taxonomy(m),
        *pass_7_spawn_cycle(m),
        *pass_8_governance_consistency(m),
        *pass_9_consciousness_claim_check(m),
        *pass_10_precaution(m),
        *pass_11_dual_process(m),
        *pass_12_moral_foundations(m),
        *pass_13_gricean(m),
        *pass_14_hindsight(m),
        *pass_15_argumentative(m),
        *pass_16_rl(m),
    ]


# Sanity: every op we emit is classified into exactly one category.
def _self_check() -> None:
    all_ops = (PURE_OPS | AGENT_OPS | MEMORY_OPS | PROMPT_OPS | TOOL_OPS
               | NEURAL_OPS | POLICY_OPS)
    for op in Op:
        assert op in all_ops, f"op {op} unclassified"


_self_check()
