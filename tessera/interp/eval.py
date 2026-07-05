"""Tree-walking SIR interpreter (RFC §10).

MVP scope: enough to execute hello.t.md AND researcher.t.md end-to-end.

Concurrency model: a thread-per-message actor scheduler (default on). `send`
binds each message to its own future; `recv` / `recv all` await them, serialized
per child by a state lock so concurrent sends can't corrupt one agent's memory.
Children are supervised — a raise or Refusal is re-driven up to the spawn's
retry budget before it propagates.

Orchestration spine: every scheduling choice — which plan runs first, which
contender wins the blackboard, which child a supervisor re-drives — resolves
through one scalar, salience/priority in [0,1] (see interp/scheduling.py).

Capability gates: enforced at Spawn — a child can only receive caps the parent
holds AND has declared via its frontmatter `capabilities_requested`. Beyond
that, capabilities are still mostly decorative until we wire `Tool.Invoke` etc.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Lock
from typing import Any, Callable

import re as _re

from ..sir.nodes import Module, Node, Op, PolicyDecl, Region, TraitDecl, WorkspaceDecl
from ..traits import TriggerContext, fire_traits, resolve_trait, trait_preamble


@dataclass
class RuntimeError_(Exception):
    msg: str

    def __str__(self) -> str:
        return f"interp error: {self.msg}"


@dataclass
class Refusal:
    """First-class refusal value (RFC §5.6). Returned when a policy denies a step."""
    reason: str
    policy: str = ""

    def __repr__(self) -> str:
        return f"Refusal(reason={self.reason!r}, policy={self.policy!r})"

    def __bool__(self) -> bool:
        # Refusals are falsy so an agent can branch on `let x = ...; if not x ...`
        return False


class CalledVia(str, Enum):
    """Provenance of a tool dispatch (PRD §12 instruction/data boundary)."""
    DIRECT = "direct"  # a normal SIR Op.Apply tool call
    WIRE = "wire"      # parsed from a schema-bound prompt's emitted !call (Theme B)
    TEXT = "text"      # lifted from free-text / a =result body — no producer yet;
                       # refused by default so a smuggled call can't execute


def _check_tool_provenance(world, tool, called_via, agent_name) -> "Refusal | None":
    """Refuse a tool call sourced from untrusted text unless explicitly allowed.
    Composes with (does not replace) autonomy/policy, which gate intent/capability.
    `direct` and `wire` pass; only `text` is gated. Default behavior is unchanged
    because no current code path produces `text`."""
    if called_via != CalledVia.TEXT:
        return None
    allowed: list[str] = []
    if world.module.autonomy is not None:
        allowed = getattr(world.module.autonomy, "allow_text_calls", [])
    if tool.name in allowed:
        return None
    refusal = Refusal(
        reason=(f"tool {tool.name!r} call sourced from untrusted text "
                f"(called_via=text) refused; instruction/data boundary"),
        policy="provenance",
    )
    world.record(agent_name, "policy_violation", policy="provenance",
                 reason=refusal.reason, tool=tool.name, called_via="text")
    return refusal


def _check_policies(
    module: Module,
    value: object,
    *,
    action: str = "",
    agent: str | None = None,
    intent: str | None = None,
    capabilities: frozenset[str] | None = None,
) -> Refusal | None:
    """Apply all module-declared policies to a value. Returns Refusal on deny.

    The substring-form rules (forbid_contains / forbid_match / require_contains)
    walk the stringified value. The constraint-logic forms (forbid_when /
    permit_when) evaluate an AST against an ActionContext built from the
    caller-supplied agent / intent / capabilities + the value.
    """
    if not module.policies:
        return None
    if value is None:
        return None
    s = value if isinstance(value, str) else str(value)
    ctx = None  # built lazily; only constraint rules need it
    for pol_name, pol in module.policies.items():
        for kind, params in pol.rules:
            if kind == "forbid_contains":
                needle = params.get("needle", "")
                if needle and needle in s:
                    return Refusal(
                        reason=f"contains forbidden substring {needle!r}",
                        policy=pol_name,
                    )
            elif kind == "forbid_match":
                pattern = params.get("pattern", "")
                if pattern and _re.search(pattern, s):
                    return Refusal(
                        reason=f"matches forbidden pattern {pattern!r}",
                        policy=pol_name,
                    )
            elif kind == "require_contains":
                needle = params.get("needle", "")
                if needle and needle not in s:
                    return Refusal(
                        reason=f"missing required substring {needle!r}",
                        policy=pol_name,
                    )
            elif kind in ("forbid_when", "permit_when"):
                if ctx is None:
                    from ..policy_lang import ActionContext
                    ctx = ActionContext(
                        value=value,
                        action=action,
                        agent=agent,
                        intent=intent,
                        capabilities=capabilities or frozenset(),
                    )
                try:
                    result = bool(params["expr"].eval(ctx))
                except Exception as e:
                    return Refusal(
                        reason=f"policy expression error: {e}",
                        policy=pol_name,
                    )
                if kind == "forbid_when" and result:
                    return Refusal(
                        reason=f"forbid when {params.get('src', '<expr>')}",
                        policy=pol_name,
                    )
                if kind == "permit_when" and not result:
                    return Refusal(
                        reason=f"not permitted (predicate {params.get('src', '<expr>')} was false)",
                        policy=pol_name,
                    )
    return None


@dataclass
class PlanFrame:
    """One entry on an agent's runtime plan stack."""
    name: str
    traits: list[TraitDecl] = field(default_factory=list)
    intent: str | None = None


@dataclass
class AuditEvent:
    """One audited runtime action, stamped with the intent it served.

    The audit trace is the answer to 'what did this agent actually do, and
    why' — every action carries the plan and intent active when it fired, the
    capabilities it exercised, and the traits that shaped it.
    """
    seq: int
    agent: str | None
    plan: str | None
    intent: str | None
    action: str               # e.g. "prompt:assess", "tool:web_search", "spawn:Critic", "refusal"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq, "agent": self.agent, "plan": self.plan,
            "intent": self.intent, "action": self.action, **self.detail,
        }


@dataclass
class AgentState:
    """Per-agent working memory, mailbox, capability set, declared beliefs."""
    name: str
    working_memory: dict[str, Any] = field(default_factory=dict)
    mailbox: deque = field(default_factory=deque)
    capabilities: set[str] = field(default_factory=set)
    declared_beliefs: list[str] = field(default_factory=list)
    has_run: bool = False
    # Per-agent episodic event log: [(event_name, [arg_value, ...], timestamp_seq)]
    episodic: list[tuple[str, list[Any], int]] = field(default_factory=list)
    # Notice handlers registered on this agent: [(pred_region, handler_region)]
    notices: list[tuple[Any, Any]] = field(default_factory=list)
    # Track which notice fired last so we don't re-fire on the same true state
    notice_last_fired_seq: dict[int, int] = field(default_factory=dict)
    # Concurrent scheduler bookkeeping. One in-flight future per delivered
    # message (FIFO), so a parent can `send` several messages then `recv` (or
    # `recv all`) each reply — fan-out/gather composes instead of stranding
    # messages in a single slot.
    pending_futures: deque = field(default_factory=deque)
    last_result: Any = None                    # most recent plan return value
    mailbox_signal: Event = field(default_factory=Event)
    state_lock: Lock = field(default_factory=Lock)
    # Supervision budget granted at spawn: how many times a child that refuses
    # or raises is re-driven before its Refusal is allowed to propagate.
    supervise_retries: int = 0
    # Cognitive traits, resolved once per agent (see _ensure_traits_resolved).
    traits_resolved: bool = False
    per_call_traits: list[TraitDecl] = field(default_factory=list)
    per_plan_trait_defs: list[TraitDecl] = field(default_factory=list)
    global_traits: list[TraitDecl] = field(default_factory=list)   # fired global-scope
    # Runtime plan stack of PlanFrames (name + active per_plan traits + intent).
    plan_stack: list[PlanFrame] = field(default_factory=list)
    # Per-substrate runtime accumulators (welfare gate state, ast attention
    # schema, tom belief models, cycle counter). Keyed by substrate name; see
    # interp/substrates.py. Kept off the hot path until a partial substrate is
    # declared.
    substrate_state: dict[str, Any] = field(default_factory=dict)

    @property
    def active_plan_traits(self) -> list[TraitDecl]:
        return self.plan_stack[-1].traits if self.plan_stack else []

    @property
    def active_plan(self) -> PlanFrame | None:
        return self.plan_stack[-1] if self.plan_stack else None


@dataclass
class WorkspaceState:
    """Runtime view of a declared workspace.

    GWT extension (research B1): `gwt_bottleneck` enforces Baars's
    attention bottleneck — when more contenders pile up than the
    bottleneck allows, the lowest-salience ones are dropped before
    arbitration. `track_ignition` (set on the decl) gates whether
    arbitration broadcasts a `gwt:ignition` audit event recording the
    bandwidth (contender count) and the selected winner — Dehaene's
    ignition signature measured functionally.
    """
    decl: WorkspaceDecl
    contenders: list[tuple[Any, float]] = field(default_factory=list)  # (value, salience)
    draft_history: deque = field(default_factory=lambda: deque(maxlen=16))
    last_winner: Any = None
    # Provenance for ignition audit: set by World.ensure_workspace if needed.
    _world_ref: Any = None
    _ignition_seq: int = 0

    def broadcast(self, value: Any, salience: float) -> None:
        """Add a contender to the board.

        Contenders *accumulate* across a round — they are not consumed until the
        board is read. This is what lets multi-contender arbiters (``quorum``,
        ``weighted_vote``) see the whole field before deciding. Broadcasting
        still updates a live ``last_winner`` so peeking the board mid-round (as
        the GWT ignition path does) reflects the most salient draft so far.
        """
        self.contenders.append((value, salience))
        self.draft_history.append((value, salience))
        # Apply GWT bottleneck if declared (Baars's attention bottleneck): keep
        # only the most salient contenders so the board can't grow without bound.
        if self.decl.gwt_bottleneck > 0 and len(self.contenders) > self.decl.gwt_bottleneck:
            self.contenders.sort(key=lambda x: -x[1])
            self.contenders = self.contenders[: self.decl.gwt_bottleneck]
        # Peek: resolve a live winner but keep contenders for the rest of the round.
        self._arbitrate(commit=False)

    def read(self) -> Any:
        """Resolve the round and return the winner, then clear the board.

        Reading is the round boundary: the arbiter reduces every accumulated
        contender to one winner, the board is consumed, and the next broadcast
        starts a fresh round. Re-reading an empty board returns the retained
        last winner (an idempotent peek).
        """
        if self.contenders:
            self._arbitrate(commit=True)
        return self.last_winner

    def _arbitrate(self, commit: bool) -> None:
        if not self.contenders:
            return
        from . import scheduling
        bandwidth = len(self.contenders)
        result = scheduling.arbitrate(self.decl.arbiter, list(self.contenders))
        if commit:
            self.contenders.clear()
        if result is None:
            # The arbiter abstained (e.g. quorum not met): leave the prior
            # winner standing and record the abstention if ignition is tracked.
            if self.decl.track_ignition and self._world_ref is not None:
                self._ignition_seq += 1
                self._world_ref.record(
                    None,
                    f"gwt:ignition:{self.decl.name}",
                    workspace=self.decl.name,
                    bandwidth=bandwidth,
                    winner_salience=None,
                    arbiter=self.decl.arbiter,
                    abstained=True,
                    committed=commit,
                    bottleneck=self.decl.gwt_bottleneck,
                    cycle=self._ignition_seq + 1,
                )
            return
        winner_value, winner_salience = result
        self.last_winner = winner_value
        if self.decl.track_ignition and self._world_ref is not None:
            self._ignition_seq += 1
            self._world_ref.record(
                None,
                f"gwt:ignition:{self.decl.name}",
                workspace=self.decl.name,
                bandwidth=bandwidth,
                winner_salience=winner_salience,
                arbiter=self.decl.arbiter,
                committed=commit,
                bottleneck=self.decl.gwt_bottleneck,
                cycle=self._ignition_seq,
            )


@dataclass
class World:
    module: Module
    agents: dict[str, AgentState] = field(default_factory=dict)
    workspaces: dict[str, WorkspaceState] = field(default_factory=dict)
    region_results: dict[str, Any] = field(default_factory=dict)
    spawn_log: list[str] = field(default_factory=list)
    # Concurrent actor scheduler. None executor means synchronous within a
    # single agent's plan; spawned children still run concurrently when the
    # `concurrent` flag is True (the runtime default per decision 3).
    # Opt out via TESSERA_CONCURRENT_AGENTS=0 or concurrent=False.
    executor: ThreadPoolExecutor | None = None
    concurrent: bool = True
    # Append-only audit trace: every meaningful runtime action, stamped with the
    # active plan + intent. Exported via `tessera compile --run X --audit out`.
    audit: list[AuditEvent] = field(default_factory=list)
    _audit_seq: int = 0
    # Per-run shadow for non-persistent semantic schemas (those declared with
    # `persistent=false` on the memory:semantic fence). Keyed by schema name.
    ephemeral_semantic: dict[str, list[dict]] = field(default_factory=dict)
    ephemeral_relations: list[dict] = field(default_factory=list)
    # Constrained-decoding registry: prompt name -> tson.Compiled. When
    # populated, _call_prompt enforces that schema's grammar on the prompt's
    # output (see adapters/wire). Empty by default -> no behavior change.
    prompt_schemas: dict[str, Any] = field(default_factory=dict)

    def record(self, agent_name: str | None, action: str, **detail) -> None:
        st = self.agents.get(agent_name) if agent_name else None
        frame = st.active_plan if st else None
        # Outside a plan, fall back to the agent's declared top-level intent.
        intent = frame.intent if frame else None
        if intent is None and agent_name and agent_name in self.module.agents:
            intent = self.module.agents[agent_name].intent
        # KNOWN LIMITATION (concurrency): `_audit_seq` is not lock-guarded, so
        # with concurrent agents (run_agent concurrent=True) the seq numbers may
        # collide/skip and event *ordering* in `self.audit` is best-effort. The
        # `append` below is atomic under the GIL, so event *counts* are exact —
        # which is what tests/test_contracts_stress.py asserts on. A future fix
        # is a Lock here (and around semantic_cache_put). See CHANGELOG.
        self._audit_seq += 1
        evt = AuditEvent(
            seq=self._audit_seq,
            agent=agent_name,
            plan=frame.name if frame else None,
            intent=intent,
            action=action,
            detail=detail,
        )
        self.audit.append(evt)
        try:
            from ..adapters.audit import record_event
            record_event(evt.to_dict())
        except Exception:
            pass

    def ensure_workspace(self, name: str) -> WorkspaceState:
        if name not in self.workspaces:
            decl = self.module.workspaces.get(name) or WorkspaceDecl(name=name)
            ws = WorkspaceState(decl=decl)
            ws._world_ref = self  # lets ignition audit reach back into the world
            self.workspaces[name] = ws
        return self.workspaces[name]

    def state_for(self, agent_name: str) -> AgentState:
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentState(
                name=agent_name,
                declared_beliefs=self._declared_beliefs_of(agent_name),
            )
        return self.agents[agent_name]

    def _declared_beliefs_of(self, agent_name: str) -> list[str]:
        region = self.module.agents.get(agent_name)
        if not region:
            return []
        out = []
        for n in region.nodes:
            if n.op is Op.BeliefRead and n.attributes.get("declared"):
                out.append(n.attributes.get("name"))
        return out


_BUILTIN_BINOPS: dict[str, Callable[[Any, Any], Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: (a - b) if not isinstance(a, str) else a.replace(b, "", 1),
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "and": lambda a, b: bool(a) and bool(b),
    "or":  lambda a, b: bool(a) or bool(b),
}


def _as_str_safe(a: Any, b: Any) -> tuple[Any, Any]:
    if isinstance(a, str) and not isinstance(b, str):
        b = str(b)
    elif isinstance(b, str) and not isinstance(a, str):
        a = str(a)
    return a, b


def eval_region(region: Region, world: World, args: dict[str, Any] | None = None,
                agent_name: str | None = None) -> Any:
    args = args or {}
    values: dict[str, Any] = {}

    last: Any = None
    i = 0
    nodes = region.nodes
    while i < len(nodes):
        n = nodes[i]
        # Parallel group fast path: if this is the first node of a parallel
        # group AND we have an executor, dispatch the group concurrently.
        if (n.op is Op.WM_Write
                and world.executor is not None
                and "parallel_group" in n.attributes):
            group_id = n.attributes["parallel_group"]
            group_end = i
            # collect all consecutive nodes belonging to this group's
            # expression ranges, ending after the last WM_Write of the group
            while group_end < len(nodes):
                m = nodes[group_end]
                # break when we hit a WM_Write of a different group OR a
                # non-WM-related node that isn't part of group exprs
                if (m.op is Op.WM_Write
                        and m.attributes.get("parallel_group") != group_id):
                    break
                group_end += 1
            sub_nodes = nodes[i:group_end]
            _eval_parallel_group(sub_nodes, values, world, region, agent_name, args)
            for sn in sub_nodes:
                last = values.get(sn.id, last)
            i = group_end
            continue

        v = _eval_node(n, values, world, region, agent_name, args)
        values[n.id] = v
        last = v
        if n.op is Op.Return:
            world.region_results[region.id] = v
            return v
        i += 1

    world.region_results[region.id] = last
    return last


def _eval_parallel_group(sub_nodes, values, world, region, agent_name, params):
    """Evaluate a parallel group concurrently when possible.

    The strategy: each WM_Write's expression chain is a self-contained
    sub-DAG (parser-emitted in order). We split the sub-DAG into per-WM
    chunks and dispatch them to the executor. Pure ops within a chunk run
    sequentially in their own thread.
    """
    # Find WM boundaries within sub_nodes
    wm_indices = [idx for idx, n in enumerate(sub_nodes) if n.op is Op.WM_Write]
    if len(wm_indices) < 2:
        # Fallback: serial
        for n in sub_nodes:
            values[n.id] = _eval_node(n, values, world, region, agent_name, params)
        return

    # Split into chunks: each chunk includes everything from the previous WM
    # (exclusive) through and including the next WM.
    chunks = []
    prev = 0
    for wi in wm_indices:
        chunks.append(sub_nodes[prev:wi + 1])
        prev = wi + 1

    # Each chunk is evaluated by a worker function. Local dict per worker is
    # seeded from the shared values; results merged after.
    def _run_chunk(chunk):
        local_values = dict(values)
        for n in chunk:
            local_values[n.id] = _eval_node(n, local_values, world, region,
                                            agent_name, params)
        return {n.id: local_values[n.id] for n in chunk}

    futures = [world.executor.submit(_run_chunk, c) for c in chunks]
    for fut in futures:
        result = fut.result()
        values.update(result)


def _eval_node(n: Node, values: dict[str, Any], world: World, region: Region,
               agent_name: str | None, params: dict[str, Any]) -> Any:
    op = n.op

    if op is Op.Const:
        return n.attributes.get("value")

    if op is Op.BinOp:
        sym = n.attributes.get("op", "+")
        fn = _BUILTIN_BINOPS.get(sym)
        if fn is None:
            raise RuntimeError_(f"unknown binop {sym!r}")
        a, b = _as_str_safe(values[n.inputs[0]], values[n.inputs[1]])
        return fn(a, b)

    if op is Op.Apply:
        callee_name = n.attributes.get("callee") or values[n.inputs[0]]
        arg_vals = [values[i] for i in n.inputs[1:]]

        # 0. Built-in callables: value constructors (__list__/__record__ from
        #    the [..]/{..} literal syntax) + reasoning-tool callables.
        from .builtins import BUILTINS
        builtin = BUILTINS.get(callee_name)
        if builtin is not None:
            return builtin(n, arg_vals, world, agent_name)

        # 1. Plain logic function
        fn_region = world.module.functions.get(callee_name)
        if fn_region is not None:
            local = {pname: arg_vals[i] for i, (pname, _) in enumerate(fn_region.params)}
            return eval_region(fn_region, world, args=local, agent_name=agent_name)

        # 2. Prompt template → LLM call
        prompt = world.module.prompts.get(callee_name)
        if prompt is not None:
            return _call_prompt(prompt, arg_vals, world, agent_name)

        # 3. External tool (LangChain or bare python callable)
        tool = world.module.tools.get(callee_name)
        if tool is not None:
            from . import substrates
            label = f"tool:{callee_name}"
            intent_t = _active_intent(world, agent_name)
            caps_t = (frozenset(world.state_for(agent_name).capabilities)
                      if agent_name is not None else frozenset())
            # Contract `before` — gate inputs before the tool fires.
            if agent_name is not None:
                res_b = substrates.evaluate_contracts(
                    world, agent_name, label, "before",
                    value=arg_vals, intent=intent_t, caps=caps_t, args=arg_vals)
                ref = substrates.enforce_contract_refusal(world, agent_name, res_b, label, "before")
                if ref is not None:
                    return ref

            def _do_tool():
                r = _invoke_tool(tool, arg_vals, world,
                                 called_via=CalledVia.DIRECT, agent_name=agent_name)
                world.record(agent_name, label, called_via="direct")
                return r

            res = _do_tool()
            # Contract `after` — assert over the tool's result, retry re-invokes.
            if agent_name is not None:
                res = _enforce_after_contracts(
                    world, agent_name, label, res, intent_t, caps_t,
                    regen=_do_tool, as_refusal=True)
            return res

        # 4. Neural model — torch forward
        model = world.module.neural_models.get(callee_name)
        if model is not None:
            res = _forward_model(model, arg_vals, world)
            world.record(agent_name, f"model:{callee_name}")
            return res

        # 5. Procedural skill — named indirection over any of the above
        skill = world.module.skills.get(callee_name)
        if skill is not None:
            return _invoke_skill(skill, arg_vals, world, agent_name)

        raise RuntimeError_(f"callable {callee_name!r} not defined as function/prompt/tool/model/skill")

    if op is Op.BeliefRead:
        name = n.attributes.get("name")
        if n.attributes.get("declared"):
            return None
        if name in params:
            return params[name]
        if agent_name is not None:
            st = world.state_for(agent_name)
            if name in st.working_memory:
                return st.working_memory[name]
        return None

    if op is Op.WM_Write:
        name = n.attributes.get("name")
        val = values[n.inputs[0]]
        # Policy gate at plan-step boundary: if the value being written
        # violates a declared policy, replace it with a Refusal.
        caps_ = frozenset(world.state_for(agent_name).capabilities) if agent_name else frozenset()
        intent_ = None
        if agent_name and agent_name in world.module.agents:
            st_ = world.state_for(agent_name)
            frame_ = st_.active_plan if st_ else None
            intent_ = (frame_.intent if frame_ else None) or world.module.agents.get(agent_name).intent
        refusal = _check_policies(
            world.module, val,
            action=f"wm_write:{name}",
            agent=agent_name,
            intent=intent_,
            capabilities=caps_,
        )
        if refusal is not None:
            val = refusal
            world.record(agent_name, "refusal", policy=refusal.policy,
                         reason=refusal.reason, belief=name)
        if agent_name is not None:
            world.state_for(agent_name).working_memory[name] = val
            _check_notices(world, agent_name)
        return val

    if op is Op.WM_Read:
        name = n.attributes.get("name")
        if agent_name is not None:
            return world.state_for(agent_name).working_memory.get(name)
        return None

    if op is Op.Return:
        return values[n.inputs[0]] if n.inputs else None

    if op is Op.IntentionCommit:
        plan_region_id = n.attributes.get("region")
        plan_region = next((r for r in world.module.regions if r.id == plan_region_id), None)
        if plan_region is None:
            raise RuntimeError_(f"plan region {plan_region_id} missing")
        owner = region.name.removeprefix("agent:")
        plan_name = n.attributes.get("plan", "")
        _ensure_traits_resolved(world, owner)
        state = world.state_for(owner)
        # per_plan traits: evaluate the trigger ONCE at plan entry against the
        # plan's static prompt templates; if fired, active for every call below.
        active: list[TraitDecl] = []
        if state.per_plan_trait_defs:
            ctx = TriggerContext(
                text="\n".join(_static_prompt_templates(plan_region, world.module)),
                plan_name=plan_name,
                capabilities=frozenset(state.capabilities),
            )
            active = fire_traits(state.per_plan_trait_defs, ctx)
        state.plan_stack.append(PlanFrame(name=plan_name, traits=active,
                                          intent=plan_region.intent))
        world.record(owner, f"plan_enter:{plan_name}", intent_served=plan_region.intent,
                     priority=n.attributes.get("priority", 0.5))
        try:
            # Consciousness-adjacent / welfare partial substrates run their
            # plan-entry behavior here: iit emits φ*, welfare records markers
            # and may refuse, ast scores introspection fidelity and may refuse.
            from . import substrates
            refusal = substrates.on_plan_enter(world, owner, plan_name)
            if refusal is not None:
                return refusal
            result = eval_region(plan_region, world, agent_name=owner)
            # hindsight after-action review + contract `after` enforcement.
            replaced = substrates.on_plan_exit(world, owner, plan_name, plan_region, result)
            return replaced if replaced is not None else result
        finally:
            state.plan_stack.pop()

    if op is Op.Spawn:
        target_name = n.attributes.get("agent")
        requested_caps = set(n.attributes.get("capabilities") or [])
        # Auto-restrict (decision 10): child runs with the intersection of
        # (parent caps, child's declared caps). When the granted set is
        # smaller than what was requested, audit the narrowing so the
        # delta is queryable later via tessera audit query.
        from ..capabilities import intersect
        if agent_name is not None:
            parent = world.state_for(agent_name)
            granted = intersect(parent.capabilities, requested_caps)
        else:
            granted = set(requested_caps)
        dropped = requested_caps - granted
        # Create the child state with granted caps.
        child = world.state_for(target_name)
        child.capabilities |= granted
        # Supervision budget: how many times a refusing/raising child is
        # re-driven before its Refusal propagates (`spawn X supervise=retry(N)`).
        retries = int(n.attributes.get("retries") or 0)
        if retries > child.supervise_retries:
            child.supervise_retries = retries
        world.spawn_log.append(f"{agent_name or '<root>'} -> {target_name} with {sorted(granted)}")
        world.record(agent_name, f"spawn:{target_name}", caps_granted=sorted(granted),
                     supervise_retries=child.supervise_retries)
        if dropped:
            world.record(
                agent_name,
                f"caps_narrowed:{target_name}",
                requested=sorted(requested_caps),
                granted=sorted(granted),
                dropped=sorted(dropped),
            )
        return f"<agent:{target_name}>"

    if op is Op.Send:
        ref = values[n.inputs[0]]
        msg = values[n.inputs[1]]
        target_name = _agent_name_from_ref(ref)
        if target_name is None:
            raise RuntimeError_(f"send target is not an agent ref: {ref!r}")
        child = world.state_for(target_name)
        # Concurrent mode: bind THIS message to its own future so each send gets
        # exactly one reply on a later recv — fan-out composes. The future is
        # serialized per child by its state_lock (see _drive_child), so several
        # in-flight runs can't corrupt the child's working memory.
        if world.concurrent and world.executor is not None:
            retries = child.supervise_retries
            fut = world.executor.submit(_drive_child, world, target_name, msg, retries)
            child.pending_futures.append(fut)
        else:
            with child.state_lock:
                child.mailbox.append(msg)
                child.mailbox_signal.set()
        return None

    if op is Op.Recv:
        ref = values[n.inputs[0]]
        target_name = _agent_name_from_ref(ref)
        if target_name is None:
            raise RuntimeError_(f"recv target is not an agent ref: {ref!r}")
        child = world.state_for(target_name)
        gather = bool(n.attributes.get("gather"))
        # Resolve timeout: per-recv attribute > env var > default 30s.
        import os as _os
        from concurrent.futures import TimeoutError as _FutTimeout
        per_recv = n.attributes.get("timeout_s")
        env = _os.environ.get("TESSERA_RECV_TIMEOUT_S")
        try:
            timeout_s = float(per_recv) if per_recv is not None else (
                float(env) if env else 30.0
            )
        except (TypeError, ValueError):
            timeout_s = 30.0

        def _await(fut):
            try:
                return fut.result(timeout=timeout_s)
            except _FutTimeout:
                world.record(agent_name, "deadlock_suspected",
                             waiting_on=target_name, timeout_s=timeout_s)
                return Refusal(
                    reason=f"recv from {target_name} timed out after {timeout_s}s",
                    policy="<recv_timeout>",
                )

        # `recv all from X` — gather every reply still owed by X into a list.
        if gather:
            replies: list[Any] = []
            while child.pending_futures:
                replies.append(_await(child.pending_futures.popleft()))
            while child.mailbox:
                replies.append(_run_child_and_collect(world, target_name))
            return replies

        # `recv from X` — the next single reply. Prefer an in-flight future
        # (concurrent send); otherwise drive the child synchronously.
        if world.concurrent and child.pending_futures:
            return _await(child.pending_futures.popleft())
        return _run_child_and_collect(world, target_name)

    if op is Op.Workspace_Broadcast:
        ws_name = n.attributes.get("workspace")
        salience = float(n.attributes.get("salience", 0.5))
        ws = world.ensure_workspace(ws_name)
        ws.broadcast(values[n.inputs[0]], salience)
        return None

    if op is Op.Workspace_Read:
        ws_name = n.attributes.get("workspace")
        return world.ensure_workspace(ws_name).read()

    if op is Op.EM_Append:
        event_name = n.attributes.get("event")
        arg_values = [values[i] for i in n.inputs]
        if agent_name is not None:
            st = world.state_for(agent_name)
            st.episodic.append((event_name, arg_values, len(st.episodic)))
        return None

    if op is Op.EM_Query:
        event_name = n.attributes.get("event")
        if agent_name is None:
            return []
        st = world.state_for(agent_name)
        return [
            {"event": ev, "args": args, "seq": seq}
            for (ev, args, seq) in st.episodic
            if ev == event_name
        ]

    if op is Op.SM_Insert:
        import uuid as _uuid
        schema = n.attributes.get("schema", "")
        field_names = n.attributes.get("fields") or []
        arg_vals = [values[i] for i in n.inputs]
        schema_decl = world.module.knowledge_schemas.get(schema)
        field_types = schema_decl.fields if schema_decl else None
        if schema_decl and not schema_decl.persistent:
            # ephemeral — validate, mint an id so the fact is relatable too
            from ..adapters.semantic import validate_fields
            fields = validate_fields(schema, dict(zip(field_names, arg_vals)),
                                     field_types, ref_check=False)
            record = {"id": str(_uuid.uuid4()), "schema": schema, "fields": fields}
            world.ephemeral_semantic.setdefault(schema, []).append(record)
            return record
        from ..adapters.semantic import remember_fact
        fact_id = remember_fact(schema, dict(zip(field_names, arg_vals)), field_types=field_types)
        return {"id": fact_id, "schema": schema, "fields": dict(zip(field_names, arg_vals))}

    if op is Op.SM_Relate:
        predicate = n.attributes.get("predicate", "")
        subj, obj = values[n.inputs[0]], values[n.inputs[1]]
        subj_id = subj["id"] if isinstance(subj, dict) else subj
        obj_id = obj["id"] if isinstance(obj, dict) else obj
        # typed-endpoint check against the declared relation
        rel = world.module.relations.get(predicate)
        if rel is not None:
            def _schema_of(fid, val):
                if isinstance(val, dict):
                    return val.get("schema")
                from ..adapters.semantic import fact_schema
                return fact_schema(fid)
            ss, os_ = _schema_of(subj_id, subj), _schema_of(obj_id, obj)
            if ss is not None and ss != rel.subject_schema:
                raise RuntimeError(f"relate {predicate}: subject is {ss}, expected {rel.subject_schema}")
            if os_ is not None and os_ != rel.object_schema:
                raise RuntimeError(f"relate {predicate}: object is {os_}, expected {rel.object_schema}")
        # ephemeral if either endpoint is an ephemeral fact, else persistent
        eph = any(
            any(r.get("id") == fid for rs in world.ephemeral_semantic.values() for r in rs)
            for fid in (subj_id, obj_id)
        )
        if eph:
            world.ephemeral_relations.append(
                {"subject": subj_id, "predicate": predicate, "object": obj_id})
            return predicate
        from ..adapters.semantic import relate_facts
        return relate_facts(subj_id, predicate, obj_id)

    if op is Op.SM_Related:
        predicate = n.attributes.get("predicate")
        direction = n.attributes.get("direction", "out")
        val = values[n.inputs[0]]
        fid = val["id"] if isinstance(val, dict) else val
        if world.ephemeral_relations:
            by_id = {r["id"]: r for rs in world.ephemeral_semantic.values() for r in rs}
            out = []
            for e in world.ephemeral_relations:
                if predicate and e["predicate"] != predicate:
                    continue
                if direction in ("out", "both") and e["subject"] == fid and e["object"] in by_id:
                    out.append({**by_id[e["object"]], "_via": e["predicate"]})
                if direction in ("in", "both") and e["object"] == fid and e["subject"] in by_id:
                    out.append({**by_id[e["subject"]], "_via": e["predicate"]})
            if out:
                return out
        from ..adapters.semantic import related_facts
        return related_facts(fid, predicate=predicate, direction=direction)

    if op is Op.SM_Search:
        schema = n.attributes.get("schema", "")
        where_field = n.attributes.get("where_field")
        where_value = values[n.inputs[0]] if (where_field and n.inputs) else None
        schema_decl = world.module.knowledge_schemas.get(schema)
        if schema_decl and not schema_decl.persistent:
            return [
                r for r in world.ephemeral_semantic.get(schema, [])
                if where_field is None or r["fields"].get(where_field) == where_value
            ]
        from ..adapters.semantic import lookup_facts
        return lookup_facts(schema, where_field=where_field,
                            where_value=where_value)

    if op is Op.Until:
        pred_region = n.attributes.get("pred_region")
        body_region = n.attributes.get("body_region")
        max_iter = int(n.attributes.get("max_iter", 100))
        iters = 0
        last_val: Any = None
        while iters < max_iter:
            # Evaluate predicate against current world
            pred_val = eval_region(pred_region, world, args=params, agent_name=agent_name)
            if bool(pred_val):
                break
            last_val = eval_region(body_region, world, args=params, agent_name=agent_name)
            iters += 1
            _check_notices(world, agent_name)
        return last_val

    if op is Op.Notice_Subscribe:
        if agent_name is not None:
            world.state_for(agent_name).notices.append((
                n.attributes.get("pred_region"),
                n.attributes.get("handler_region"),
            ))
        return None

    raise RuntimeError_(f"op {op} not implemented in MVP interpreter")


def _check_notices(world: "World", agent_name: str | None) -> None:
    """Run any notice handlers whose predicate is now true (edge-triggered).

    Called after WM_Write and after each Until iteration. Skips re-firing
    on the same active-true state by tracking last-fired sequence.

    Predicates that crash (e.g., referenced beliefs not yet bound) are
    treated as 'not yet true' rather than runtime errors — notices are
    opportunistic observations, not assertions.
    """
    if agent_name is None:
        return
    state = world.state_for(agent_name)
    seq = len(state.episodic) + len(state.working_memory)
    for idx, (pred_region, handler_region) in enumerate(state.notices):
        last = state.notice_last_fired_seq.get(idx, -1)
        if last == seq:
            continue
        try:
            pred_val = eval_region(pred_region, world, agent_name=agent_name)
        except (TypeError, RuntimeError_):
            continue
        if bool(pred_val):
            eval_region(handler_region, world, agent_name=agent_name)
            state.notice_last_fired_seq[idx] = seq


def _agent_name_from_ref(ref: Any) -> str | None:
    if isinstance(ref, str) and ref.startswith("<agent:") and ref.endswith(">"):
        return ref[len("<agent:"):-1]
    return None


# Sentinel: "no message to bind" (distinct from a legitimate None message).
_NO_MESSAGE = object()


def _drive_child(world: World, target_name: str, msg: Any = _NO_MESSAGE,
                 retries: int = 0) -> Any:
    """Run a child agent once to produce a reply, with binding + supervision.

    Deterministic model:
      1. Bind the message (if any) to the child's first declared belief
         (the convention since v0.0.2).
      2. Execute the child agent region (IntentionCommit fires).
      3. Return whichever plan was the last to return — that's the recv value.

    Serialized per child by ``state_lock`` so concurrent sends to the same agent
    can't interleave on its working memory. **Supervision**: if the child raises
    or returns a Refusal, it is re-driven up to ``retries`` times before the
    Refusal is allowed to propagate — one bad child can't crash the orchestrator.
    """
    child = world.state_for(target_name)
    region = world.module.agents.get(target_name)
    if region is None:
        raise RuntimeError_(f"agent {target_name!r} not declared in module")
    with child.state_lock:
        _ensure_traits_resolved(world, target_name)
        if msg is not _NO_MESSAGE:
            if child.declared_beliefs:
                child.working_memory[child.declared_beliefs[0]] = msg
            else:
                child.working_memory["_msg"] = msg
        attempt = 0
        while True:
            try:
                eval_region(region, world, agent_name=target_name)
                result = _last_intention_value(world, target_name, region)
            except Exception as exc:  # supervision: a raising child becomes a Refusal
                result = Refusal(
                    reason=f"{target_name} raised {type(exc).__name__}: {exc}",
                    policy="<child_exception>",
                )
            child.has_run = True
            child.last_result = result
            if not isinstance(result, Refusal) or attempt >= retries:
                if isinstance(result, Refusal) and retries > 0 and attempt >= retries:
                    world.record(target_name, "supervised_exhausted",
                                 attempts=attempt + 1, reason=result.reason)
                return result
            attempt += 1
            world.record(target_name, "supervised_retry", attempt=attempt,
                         budget=retries, reason=result.reason)


def _last_intention_value(world: World, target_name: str, region: Region) -> Any:
    """The most recently executed plan's return value — the agent's reply."""
    last_value: Any = None
    for n in region.nodes:
        if n.op is Op.IntentionCommit:
            plan_id = n.attributes.get("region")
            if plan_id in world.region_results:
                last_value = world.region_results[plan_id]
    return last_value


def _run_child_and_collect(world: World, target_name: str) -> Any:
    """Synchronous recv path: drain one mailbox message, then drive the child."""
    child = world.state_for(target_name)
    msg = _NO_MESSAGE
    if child.mailbox:
        msg = child.mailbox.popleft()
    return _drive_child(world, target_name, msg, child.supervise_retries)


# ----- cognitive traits -----------------------------------------------------


def _static_prompt_templates(plan_region: Region, module: Module) -> list[str]:
    """Templates of prompts a plan calls — context for per_plan/global triggers.

    Catches direct prompt calls and prompt-bound skills. Prompts reached only
    through nested fn/until bodies are not enumerated here (a documented v1
    limitation); per_plan triggers still fire via plan_name / capabilities.
    """
    out: list[str] = []
    for n in plan_region.nodes:
        if n.op is not Op.Apply:
            continue
        callee = n.attributes.get("callee")
        if not callee:
            continue
        pd = module.prompts.get(callee)
        if pd is not None:
            out.append(pd.template)
            continue
        sk = module.skills.get(callee)
        if sk is not None and sk.binds_to_kind == "prompt":
            spd = module.prompts.get(sk.binds_to_name)
            if spd is not None:
                out.append(spd.template)
    return out


def _ensure_traits_resolved(world: World, agent_name: str) -> None:
    """Resolve an agent's attached traits once: partition by scope and fire the
    global-scope ones against the agent's whole-program context."""
    state = world.state_for(agent_name)
    if state.traits_resolved:
        return
    state.traits_resolved = True
    region = world.module.agents.get(agent_name)
    if region is None or not region.trait_names:
        return
    resolved = [resolve_trait(name, world.module) for name in region.trait_names]
    resolved = [t for t in resolved if t is not None]
    state.per_call_traits = [t for t in resolved if t.scope == "per_call"]
    state.per_plan_trait_defs = [t for t in resolved if t.scope == "per_plan"]
    global_defs = [t for t in resolved if t.scope == "global"]
    if global_defs:
        plan_names: list[str] = []
        templates: list[str] = []
        for n in region.nodes:
            if n.op is Op.IntentionCommit:
                plan_names.append(n.attributes.get("plan", ""))
                pr_id = n.attributes.get("region")
                pr = next((r for r in world.module.regions if r.id == pr_id), None)
                if pr is not None:
                    templates.extend(_static_prompt_templates(pr, world.module))
        ctx = TriggerContext(
            text="\n".join(templates),
            plan_name=" ".join(plan_names),
            capabilities=frozenset(state.capabilities),
        )
        state.global_traits = fire_traits(global_defs, ctx)


# ----- prompt / tool / neural dispatchers ----------------------------------


def _auto_recall_disabled() -> bool:
    import os
    return os.environ.get("TESSERA_NO_AUTO_RECALL") == "1"


def _build_recall(world, agent_name, state, query, *,
                  max_facts=4, max_events=4, char_budget=1200):
    """Assemble a <recalled-context> block from the agent's memory.

    Pulls relevant semantic facts (persistent store + in-World ephemeral
    shadow, ranked by keyword overlap with the prompt) and the most recent
    episodic events. Returns (block_str, n_semantic, n_episodic); block is ""
    when there's nothing to recall. This is what makes an agent use its memory
    without the plan author wiring an explicit lookup.
    """
    if _auto_recall_disabled():
        return "", 0, 0
    module = world.module
    lines: list[str] = []
    n_sem = 0

    if module.knowledge_schemas:
        from ..adapters.semantic import query_facts, rank_facts
        candidates: list[dict] = []
        for sname, sdecl in module.knowledge_schemas.items():
            if sdecl.persistent:
                candidates.extend(query_facts(schema=sname, limit=50))
            else:
                for r in world.ephemeral_semantic.get(sname, []):
                    candidates.append({"schema": sname, "fields": r["fields"],
                                       "created_at": ""})
        for f in rank_facts(candidates, query, limit=max_facts):
            fields = ", ".join(f"{k}={v}" for k, v in f["fields"].items())
            lines.append(f"- {f['schema']}: {fields}")
            n_sem += 1

    n_epi = 0
    if state and state.episodic:
        for (name, args, _seq) in state.episodic[-max_events:]:
            arglist = ", ".join(str(a) for a in args)
            lines.append(f"- event {name}({arglist})")
            n_epi += 1

    if not lines:
        return "", 0, 0
    block = "<recalled-context>\n" + "\n".join(lines) + "\n</recalled-context>\n"
    if len(block) > char_budget:
        block = block[:char_budget] + "\n</recalled-context>\n"
    return block, n_sem, n_epi


def _resolve_prompt_schema(world, prompt):
    """Compile + cache the call grammar for a prompt declared with `emits=<tool>`.

    Returns a tson ``Compiled`` (registered into ``world.prompt_schemas`` so it's
    compiled once per run), or None if the tool is unknown or tson is absent —
    in which case the prompt falls back to ordinary free-text completion.
    """
    tool = world.module.tools.get(prompt.emits)
    if tool is None:
        return None
    try:
        from ..adapters.wire import compile_tool
        compiled = compile_tool(tool)
    except Exception:
        return None
    world.prompt_schemas[prompt.name] = compiled
    return compiled


def _active_intent(world: World, agent_name: str | None) -> str | None:
    """The intent governing the agent right now — active plan's intent, falling
    back to the agent's declared top-level intent. Mirrors the audit stamping."""
    if agent_name is None:
        return None
    st = world.state_for(agent_name)
    frame = st.active_plan if st else None
    intent = frame.intent if frame else None
    if intent is None and agent_name in world.module.agents:
        intent = world.module.agents[agent_name].intent
    return intent


def _contract_refusal(cname: str, clause: str, phase: str, as_refusal: bool):
    """A blocked contract's return value — a `Refusal` for value-typed sites
    (tool), a refusal string for text-typed sites (prompt), matching each
    site's existing refusal convention."""
    if as_refusal:
        return Refusal(reason=f"contract {cname}: {phase} clause failed — {clause}",
                       policy=f"contract:{cname}")
    return f"[contract-refused: {cname} — {phase}: {clause}]"


def _enforce_after_contracts(world, agent_name, target_label, output, intent, caps,
                             regen, *, as_refusal: bool):
    """Run `after`-clause contracts for an effect, honoring on_violation:
    `audit` (record + keep), `refuse` (block), `retry(N) then F` (re-drive via
    the 0-arg `regen` up to N times, then fall back to F). Returns the final
    output, or a refusal value when blocked."""
    from . import substrates
    attempt = 0
    while True:
        ok, cname, clause, ov = substrates.evaluate_contracts(
            world, agent_name, target_label, "after",
            value=output, intent=intent, caps=caps)
        if ok:
            return output
        mode, n, fallback = ov
        if mode == "audit":
            world.record(agent_name, "contract:audit", contract=cname,
                         phase="after", clause=clause, target=target_label)
            return output
        if mode == "retry" and attempt < n:
            attempt += 1
            world.record(agent_name, "contract:retry", contract=cname,
                         attempt=attempt, max=n, target=target_label)
            output = regen()
            continue
        if mode == "retry" and fallback == "audit":
            world.record(agent_name, "contract:audit", contract=cname,
                         phase="after", clause=clause, target=target_label,
                         note="retry exhausted")
            return output
        world.record(agent_name, "contract:refuse", contract=cname,
                     phase="after", clause=clause, target=target_label,
                     attempts=attempt)
        return _contract_refusal(cname, clause, "after", as_refusal)


def _call_prompt(prompt, arg_vals, world, agent_name=None) -> Any:
    from ..adapters.llm import get_backend
    from ..cache import semantic_cache_lookup, semantic_cache_put
    from ..governance import approval_term, ethics_preamble
    from . import substrates

    bindings = {pname: arg_vals[i] for i, (pname, _) in enumerate(prompt.params)}
    rendered = prompt.template
    for k, v in bindings.items():
        rendered = rendered.replace("{" + k + "}", str(v))

    state = world.state_for(agent_name) if agent_name is not None else None
    plan_name = state.active_plan.name if (state and state.active_plan) else ""
    caps = frozenset(state.capabilities) if state else frozenset()

    # Autonomy gate — runs BEFORE any cost. A gated action is blocked at the
    # `propose` level, flagged at `act_with_rollback`, and silent at `act_freely`.
    auto = world.module.autonomy
    if auto is not None and agent_name is not None:
        actx = TriggerContext(text=rendered, plan_name=plan_name, capabilities=caps)
        term = approval_term(auto, actx, f"prompt:{prompt.name}")
        if term is not None:
            if auto.level == "propose":
                world.record(agent_name, f"approval_blocked:{prompt.name}",
                             needs=term, level=auto.level)
                return f"[approval-required: {term}]"
            if auto.level == "act_with_rollback":
                world.record(agent_name, f"approval_required:{prompt.name}",
                             needs=term, level=auto.level, acted=True)

    # Precaution + moral-foundations action gates — also BEFORE any cost.
    if agent_name is not None:
        blocked = substrates.on_prompt_input(world, agent_name, prompt.name, rendered, caps)
        if blocked is not None:
            return blocked

    # Author-declared contract `before` clauses — also BEFORE any cost. Inputs
    # can't be regenerated, so a violation refuses (or audits) rather than retries.
    intent = _active_intent(world, agent_name)
    if agent_name is not None:
        ok, cname, clause, ov = substrates.evaluate_contracts(
            world, agent_name, f"prompt:{prompt.name}", "before",
            value=rendered, intent=intent, caps=caps)
        if not ok:
            if ov[0] == "audit":
                world.record(agent_name, "contract:audit", contract=cname,
                             phase="before", clause=clause, target=f"prompt:{prompt.name}")
            else:
                world.record(agent_name, "contract:refuse", contract=cname,
                             phase="before", clause=clause, target=f"prompt:{prompt.name}")
                return _contract_refusal(cname, clause, "before", as_refusal=False)

    # Build the prompt preamble BEFORE the cache lookup (a framed prompt is a
    # genuinely different request, so it caches separately). Ethics is outermost
    # — values frame first, then cognitive posture (traits).
    preamble = ""
    ethics_applied: list[str] = []
    if world.module.ethics is not None:
        ep = ethics_preamble(world.module.ethics)
        if ep:
            preamble += ep
            ethics_applied = [p.name for p in world.module.ethics.principles]

    fired_names: list[str] = []
    if agent_name is not None:
        ctx = TriggerContext(text=rendered, plan_name=plan_name,
                             capabilities=caps, params=bindings)
        fired = list(state.global_traits) + list(state.active_plan_traits) \
            + fire_traits(state.per_call_traits, ctx)
        fired_names = [t.name for t in fired]
        preamble += trait_preamble(fired)

    # Dual-process: when this plan was routed to the slow path (low confidence
    # or irreversible), make it mean something — deliberate, and don't reuse a
    # fast cached answer for a decision the agent is unsure about.
    slow = bool(state and state.substrate_state.get("dual_process_mode") == "slow")
    if slow:
        preamble += ("<deliberate>Think step by step and enumerate the "
                     "alternatives before answering.</deliberate>\n")

    # Auto-recall: inject the agent's relevant memory (semantic facts +
    # recent episodic events) between the posture frame and the task body, so
    # it's part of the cache key. Values → posture → recalled context → task.
    recall_block = ""
    recalled = {"semantic": 0, "episodic": 0}
    if agent_name is not None:
        recall_block, n_sem, n_epi = _build_recall(world, agent_name, state, rendered)
        recalled = {"semantic": n_sem, "episodic": n_epi}

    if preamble or recall_block:
        rendered = preamble + recall_block + rendered

    # Constrained decoding: if this prompt is schema-bound, its expected output
    # is a wire record, not free text — so it caches separately (the grammar
    # hash is folded into the key) and goes through the wire adapter. A binding
    # comes from explicit registration (world.prompt_schemas) or, lazily, from a
    # prompt declared with `emits=<tool>` (resolved + cached on first use).
    compiled = world.prompt_schemas.get(prompt.name) if world.prompt_schemas else None
    if compiled is None and getattr(prompt, "emits", None):
        compiled = _resolve_prompt_schema(world, prompt)
    cache_key = rendered if compiled is None else f"{rendered}\n\x00grammar:{compiled.grammar_hash}"

    # Produce one output — cache short-circuit (slow/forced calls bypass it),
    # else a backend completion — then run the built-in output gates. Wrapped in
    # a closure so a contract `retry` can re-drive a fresh generation. Returns
    # (text, emitted_result_or_None, gated) where `gated` means a built-in gate
    # already replaced the text and no further enforcement should run.
    def _produce(force_fresh: bool):
        cached = None if (slow or force_fresh) else semantic_cache_lookup(cache_key)
        if cached is not None:
            world.region_results.setdefault("_semantic_cache_hits", 0)
            world.region_results["_semantic_cache_hits"] += 1
            world.record(agent_name, f"prompt:{prompt.name}", traits_fired=fired_names,
                         ethics_applied=ethics_applied, recalled=recalled,
                         routed=("slow" if slow else "fast"), cost=0.0, cached=True)
            text, emitted = cached["text"], None
        else:
            dp = world.module.dual_process
            slow_backend = dp.slow_backend if (slow and dp is not None) else None
            backend = get_backend(slow_backend)
            if compiled is not None:
                from ..adapters.wire import enforce_complete
                r = enforce_complete(backend, rendered, compiled)
            else:
                r = backend.complete(rendered)
            world.region_results.setdefault("_prompt_cost", 0.0)
            world.region_results["_prompt_cost"] += r.cost_dollars
            semantic_cache_put(cache_key, r.text, backend=r.backend, model=r.model)
            world.record(agent_name, f"prompt:{prompt.name}", traits_fired=fired_names,
                         ethics_applied=ethics_applied, recalled=recalled,
                         routed=("slow" if slow else "fast"), backend=r.backend,
                         cost=r.cost_dollars, cached=False)
            text, emitted = r.text, r
        if agent_name is not None:
            gated = substrates.on_prompt_output(world, agent_name, prompt.name, text)
            if gated is not None:
                return gated, None, True
        return text, emitted, False

    text, emitted, gated = _produce(force_fresh=False)
    if gated:
        return text

    # THEME B: close the loop. When the prompt opted in with execute=true AND is
    # schema-bound, dispatch the emitted !call and return the tool result. Runs
    # AFTER on_prompt_output so output governance still inspects the record first.
    # (The emitted-call path returns a tool value, not text, so contract `after`
    # clauses — which assert over text + intent — don't apply to it.)
    if emitted is not None and compiled is not None and getattr(prompt, "execute", False):
        return _dispatch_emitted_call(prompt, emitted.text, world, agent_name)

    # Author-declared contract `after` clauses, with retry re-driving _produce.
    if agent_name is not None:
        text = _enforce_after_contracts(
            world, agent_name, f"prompt:{prompt.name}", text, intent, caps,
            regen=lambda: _produce(force_fresh=True)[0], as_refusal=False)
    return text


def _dispatch_emitted_call(prompt, record_text, world, agent_name) -> Any:
    """Parse a validated wire record and dispatch its tool (called_via=wire)."""
    from ..adapters.wire import parse_emitted_record
    tool = world.module.tools.get(prompt.emits)
    if tool is None:                      # emits pointed at an unknown tool
        return record_text
    record = parse_emitted_record(record_text)
    # Positional args in ToolDecl.params declaration order (the contract the
    # grammar was built from) — NOT record field order. Values are already
    # type-coerced by the validator inside enforce_complete.
    try:
        arg_vals = [record.fields[pname] for pname, _ptype in tool.params]
    except KeyError as e:
        raise RuntimeError_(
            f"emitted record for tool {tool.name!r} missing field {e}")
    res = _invoke_tool(tool, arg_vals, world,
                       called_via=CalledVia.WIRE, agent_name=agent_name)
    world.record(agent_name, f"tool:{tool.name}",
                 called_via="wire", address=record.address)
    return res


def _invoke_tool(tool, arg_vals, world, *, called_via=CalledVia.DIRECT,
                 agent_name=None) -> Any:
    refusal = _check_tool_provenance(world, tool, called_via, agent_name)
    if refusal is not None:
        return refusal
    from ..adapters.langchain import invoke_tool, resolve_callable
    callable_obj = resolve_callable(tool.import_path)
    return invoke_tool(callable_obj, arg_vals, invoke_method=tool.invoke_method)


def _forward_model(model_decl, arg_vals, world) -> Any:
    from ..adapters.torch import forward
    return forward(model_decl, *arg_vals)


def _invoke_skill(skill, arg_vals, world, agent_name):
    """Dispatch a procedural skill to its underlying callable + track stats."""
    world.record(agent_name, f"skill:{skill.name}",
                 binds_to=f"{skill.binds_to_kind}:{skill.binds_to_name}")
    stats = world.region_results.setdefault("_skill_stats", {})
    counter = stats.setdefault(skill.name, {"calls": 0, "kind": skill.binds_to_kind,
                                            "target": skill.binds_to_name,
                                            "promotion_signaled": False})
    counter["calls"] += 1
    # Promotion plumbing (decision 16): when a skill declared `promote_to: neural`
    # crosses its threshold, emit a one-shot audit event so a future training
    # job can find it. Does NOT train; just lights the signal.
    if (skill.promote_to == "neural"
            and counter["calls"] >= skill.promote_threshold
            and not counter["promotion_signaled"]):
        counter["promotion_signaled"] = True
        world.record(
            agent_name,
            f"skill_promotion_pending:{skill.name}",
            skill_name=skill.name,
            binds_to_kind=skill.binds_to_kind,
            binds_to_name=skill.binds_to_name,
            promote_to=skill.promote_to,
            call_count=counter["calls"],
        )
    cache = world.region_results.setdefault("_skill_cache", {})
    # cheap input-keyed cache: repeated calls with same args short-circuit
    cache_key = (skill.name, tuple(repr(a) for a in arg_vals))
    if cache_key in cache:
        counter["cache_hits"] = counter.get("cache_hits", 0) + 1
        return cache[cache_key]

    kind = skill.binds_to_kind
    name = skill.binds_to_name
    if kind == "model":
        decl = world.module.neural_models.get(name)
        if decl is None:
            raise RuntimeError_(f"skill {skill.name!r} binds to model {name!r}, not defined")
        result = _forward_model(decl, arg_vals, world)
    elif kind == "prompt":
        decl = world.module.prompts.get(name)
        if decl is None:
            raise RuntimeError_(f"skill {skill.name!r} binds to prompt {name!r}, not defined")
        result = _call_prompt(decl, arg_vals, world, agent_name)
    elif kind == "tool":
        decl = world.module.tools.get(name)
        if decl is None:
            raise RuntimeError_(f"skill {skill.name!r} binds to tool {name!r}, not defined")
        result = _invoke_tool(decl, arg_vals, world,
                              called_via=CalledVia.DIRECT, agent_name=agent_name)
    elif kind == "fn":
        fn_region = world.module.functions.get(name)
        if fn_region is None:
            raise RuntimeError_(f"skill {skill.name!r} binds to fn {name!r}, not defined")
        local = {pname: arg_vals[i] for i, (pname, _) in enumerate(fn_region.params)}
        result = eval_region(fn_region, world, args=local, agent_name=agent_name)
    else:
        raise RuntimeError_(f"unknown skill binding kind {kind!r}")
    cache[cache_key] = result
    return result


def run_agent(module: Module, agent_name: str, initial_beliefs: dict[str, Any] | None = None,
              initial_capabilities: set[str] | None = None,
              concurrent: bool | None = None,
              max_workers: int = 8,
              world: "World | None" = None) -> Any:
    """Run an agent.

    concurrent=True spins up a ThreadPoolExecutor so spawned children run
    in parallel. **Default is now True (decision 3).** Multi-agent BDI is
    the language's core value prop, so the runtime should reflect that.
    Opt out via TESSERA_CONCURRENT_AGENTS=0 or concurrent=False explicitly.

    Pass an existing `world` to retain a reference after the run — e.g. to read
    `world.audit` for the audit trace. When None, a fresh World is created.
    """
    import os as _os
    region = module.agents.get(agent_name)
    if region is None:
        raise RuntimeError_(f"agent {agent_name!r} not found in module")

    if concurrent is None:
        # Default ON; opt out via TESSERA_CONCURRENT_AGENTS in {0, false, off}.
        env = _os.environ.get("TESSERA_CONCURRENT_AGENTS", "1").lower()
        concurrent = env not in {"0", "false", "off", "no"}

    if world is None:
        world = World(module=module, concurrent=concurrent)
    if concurrent and world.executor is None:
        world.executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="tessera-agent")
    try:
        state = world.state_for(agent_name)
        if initial_beliefs:
            state.working_memory.update(initial_beliefs)
        state.capabilities |= region.capabilities_in_scope
        if initial_capabilities:
            state.capabilities |= initial_capabilities
        _ensure_traits_resolved(world, agent_name)
        return eval_region(region, world, agent_name=agent_name)
    finally:
        if world.executor is not None:
            world.executor.shutdown(wait=True)
