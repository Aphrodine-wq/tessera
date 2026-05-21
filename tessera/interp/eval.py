"""Tree-walking SIR interpreter (RFC §10).

MVP scope: enough to execute hello.tsr.md AND researcher.tsr.md end-to-end.

Concurrency model: synchronous round-robin actor scheduler. When TeamLead does
`recv from researcher`, we deterministically run Researcher's next plan step
to produce a reply. Not real concurrency — that lands when we have a proper
async runtime — but the *semantics* of spawn/send/recv are correct.

Capability gates: enforced at Spawn — a child can only receive caps the parent
holds AND has declared via its frontmatter `capabilities_requested`. Beyond
that, capabilities are still mostly decorative until we wire `Tool.Invoke` etc.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any, Callable

import re as _re

from ..sir.nodes import Module, Node, Op, PolicyDecl, Region, WorkspaceDecl


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


def _check_policies(module: Module, value: object) -> Refusal | None:
    """Apply all module-declared policies to a value. Returns Refusal on deny."""
    if not module.policies:
        return None
    if value is None:
        return None
    s = value if isinstance(value, str) else str(value)
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
    return None


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
    # Concurrent scheduler bookkeeping
    pending_future: Future | None = None       # in-flight run of this agent
    last_result: Any = None                    # most recent plan return value
    mailbox_signal: Event = field(default_factory=Event)
    state_lock: Lock = field(default_factory=Lock)


@dataclass
class WorkspaceState:
    decl: WorkspaceDecl
    contenders: list[tuple[Any, float]] = field(default_factory=list)  # (value, salience)
    draft_history: deque = field(default_factory=lambda: deque(maxlen=16))
    last_winner: Any = None

    def broadcast(self, value: Any, salience: float) -> None:
        self.contenders.append((value, salience))
        self.draft_history.append((value, salience))
        self._arbitrate()

    def _arbitrate(self) -> None:
        if not self.contenders:
            return
        if self.decl.arbiter == "highest_salience":
            winner_value, _ = max(self.contenders, key=lambda x: x[1])
        else:
            # default: last-write-wins
            winner_value, _ = self.contenders[-1]
        self.last_winner = winner_value
        self.contenders.clear()


@dataclass
class World:
    module: Module
    agents: dict[str, AgentState] = field(default_factory=dict)
    workspaces: dict[str, WorkspaceState] = field(default_factory=dict)
    region_results: dict[str, Any] = field(default_factory=dict)
    spawn_log: list[str] = field(default_factory=list)
    # In-process semantic memory shadow — used when Synapse is unreachable or
    # in test mode. Real persistence still flows through the Synapse adapter
    # when TESSERA_ALLOW_REAL_VAULT=1.
    semantic_store: list[dict] = field(default_factory=list)
    # Concurrent actor scheduler. None means we run synchronously (sequential
    # legacy behavior). Created on first spawn when concurrent=True is asked
    # via run_agent(... concurrent=True) or TESSERA_CONCURRENT_AGENTS=1.
    executor: ThreadPoolExecutor | None = None
    concurrent: bool = False

    def ensure_workspace(self, name: str) -> WorkspaceState:
        if name not in self.workspaces:
            decl = self.module.workspaces.get(name) or WorkspaceDecl(name=name)
            self.workspaces[name] = WorkspaceState(decl=decl)
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

        # 1. Plain logic function
        fn_region = world.module.functions.get(callee_name)
        if fn_region is not None:
            local = {pname: arg_vals[i] for i, (pname, _) in enumerate(fn_region.params)}
            return eval_region(fn_region, world, args=local, agent_name=agent_name)

        # 2. Prompt template → LLM call
        prompt = world.module.prompts.get(callee_name)
        if prompt is not None:
            return _call_prompt(prompt, arg_vals, world)

        # 3. External tool (LangChain or bare python callable)
        tool = world.module.tools.get(callee_name)
        if tool is not None:
            return _invoke_tool(tool, arg_vals, world)

        # 4. Neural model — torch forward
        model = world.module.neural_models.get(callee_name)
        if model is not None:
            return _forward_model(model, arg_vals, world)

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
        refusal = _check_policies(world.module, val)
        if refusal is not None:
            val = refusal
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
        return eval_region(plan_region, world, agent_name=owner)

    if op is Op.Spawn:
        target_name = n.attributes.get("agent")
        requested_caps = set(n.attributes.get("capabilities") or [])
        if agent_name is not None:
            parent = world.state_for(agent_name)
            unauthorized = requested_caps - parent.capabilities
            if unauthorized:
                raise RuntimeError_(
                    f"agent {agent_name!r} cannot grant capabilities it does not hold: {sorted(unauthorized)}"
                )
        # Create the child state with granted caps.
        child = world.state_for(target_name)
        child.capabilities |= requested_caps
        world.spawn_log.append(f"{agent_name or '<root>'} -> {target_name} with {sorted(requested_caps)}")
        return f"<agent:{target_name}>"

    if op is Op.Send:
        ref = values[n.inputs[0]]
        msg = values[n.inputs[1]]
        target_name = _agent_name_from_ref(ref)
        if target_name is None:
            raise RuntimeError_(f"send target is not an agent ref: {ref!r}")
        child = world.state_for(target_name)
        with child.state_lock:
            child.mailbox.append(msg)
            child.mailbox_signal.set()
        # Concurrent mode: kick off the child's run eagerly so it overlaps
        # with any other work the caller does between this Send and the Recv.
        if world.concurrent and world.executor is not None and child.pending_future is None:
            child.pending_future = world.executor.submit(
                _run_child_and_collect, world, target_name
            )
        return None

    if op is Op.Recv:
        ref = values[n.inputs[0]]
        target_name = _agent_name_from_ref(ref)
        if target_name is None:
            raise RuntimeError_(f"recv target is not an agent ref: {ref!r}")
        child = world.state_for(target_name)
        # Concurrent path: if a future is in flight from the Send-eager submit,
        # block on it; otherwise run inline.
        if world.concurrent and child.pending_future is not None:
            fut = child.pending_future
            child.pending_future = None
            return fut.result()
        return _run_child_and_collect(world, target_name)

    if op is Op.Workspace_Broadcast:
        ws_name = n.attributes.get("workspace")
        salience = float(n.attributes.get("salience", 0.5))
        ws = world.ensure_workspace(ws_name)
        ws.broadcast(values[n.inputs[0]], salience)
        return None

    if op is Op.Workspace_Read:
        ws_name = n.attributes.get("workspace")
        return world.ensure_workspace(ws_name).last_winner

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
        schema = n.attributes.get("schema", "")
        field_names = n.attributes.get("fields") or []
        arg_vals = [values[i] for i in n.inputs]
        record = {"schema": schema, "fields": dict(zip(field_names, arg_vals))}
        world.semantic_store.append(record)
        # Best-effort persistence to Synapse (dry-run by default).
        try:
            from ..adapters.synapse import remember_fact
            remember_fact(schema, record["fields"], dry_run=True)
        except Exception:
            pass
        return record

    if op is Op.SM_Search:
        schema = n.attributes.get("schema", "")
        where_field = n.attributes.get("where_field")
        where_value = values[n.inputs[0]] if (where_field and n.inputs) else None
        # In-process matches first
        results = [
            r for r in world.semantic_store
            if r["schema"] == schema
            and (where_field is None or r["fields"].get(where_field) == where_value)
        ]
        # Best-effort: also query Synapse if reachable; merge.
        try:
            from ..adapters.synapse import lookup_facts
            extra = lookup_facts(schema, where_field=where_field,
                                 where_value=where_value)
            results = results + extra
        except Exception:
            pass
        return results

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


def _run_child_and_collect(world: World, target_name: str) -> Any:
    """Run the target agent's intention so it can produce a reply for recv.

    Synchronous, deterministic model:
      1. Drain one message from the child's mailbox.
      2. Bind it to the child's first declared belief (the convention for v0.0.2).
      3. Execute the child agent region (IntentionCommit fires).
      4. Return whichever plan was the last to return — that's the recv value.
    """
    child = world.state_for(target_name)
    if child.mailbox:
        msg = child.mailbox.popleft()
        if child.declared_beliefs:
            child.working_memory[child.declared_beliefs[0]] = msg
        else:
            child.working_memory["_msg"] = msg
    region = world.module.agents.get(target_name)
    if region is None:
        raise RuntimeError_(f"agent {target_name!r} not declared in module")
    eval_region(region, world, agent_name=target_name)
    child.has_run = True
    # The intention's plan stored its return value in region_results.
    # Find the most recently executed plan owned by this agent.
    last_value: Any = None
    for n in region.nodes:
        if n.op is Op.IntentionCommit:
            plan_id = n.attributes.get("region")
            if plan_id in world.region_results:
                last_value = world.region_results[plan_id]
    return last_value


# ----- prompt / tool / neural dispatchers ----------------------------------


def _call_prompt(prompt, arg_vals, world) -> Any:
    from ..adapters.llm import get_backend
    from ..cache import semantic_cache_lookup, semantic_cache_put

    bindings = {pname: arg_vals[i] for i, (pname, _) in enumerate(prompt.params)}
    rendered = prompt.template
    for k, v in bindings.items():
        rendered = rendered.replace("{" + k + "}", str(v))

    # Semantic cache — short-circuit if a near-identical prompt has been seen.
    cached = semantic_cache_lookup(rendered)
    if cached is not None:
        world.region_results.setdefault("_semantic_cache_hits", 0)
        world.region_results["_semantic_cache_hits"] += 1
        return cached["text"]

    backend = get_backend()
    result = backend.complete(rendered)
    world.region_results.setdefault("_prompt_cost", 0.0)
    world.region_results["_prompt_cost"] += result.cost_dollars
    semantic_cache_put(rendered, result.text, backend=result.backend, model=result.model)
    return result.text


def _invoke_tool(tool, arg_vals, world) -> Any:
    from ..adapters.langchain import invoke_tool, resolve_callable
    callable_obj = resolve_callable(tool.import_path)
    return invoke_tool(callable_obj, arg_vals, invoke_method=tool.invoke_method)


def _forward_model(model_decl, arg_vals, world) -> Any:
    from ..adapters.torch import forward
    return forward(model_decl, *arg_vals)


def _invoke_skill(skill, arg_vals, world, agent_name):
    """Dispatch a procedural skill to its underlying callable + track stats."""
    stats = world.region_results.setdefault("_skill_stats", {})
    counter = stats.setdefault(skill.name, {"calls": 0, "kind": skill.binds_to_kind,
                                            "target": skill.binds_to_name})
    counter["calls"] += 1
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
        result = _call_prompt(decl, arg_vals, world)
    elif kind == "tool":
        decl = world.module.tools.get(name)
        if decl is None:
            raise RuntimeError_(f"skill {skill.name!r} binds to tool {name!r}, not defined")
        result = _invoke_tool(decl, arg_vals, world)
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
              max_workers: int = 8) -> Any:
    """Run an agent.

    concurrent=True spins up a ThreadPoolExecutor so spawned children run
    in parallel. Defaults to env var TESSERA_CONCURRENT_AGENTS (1/true/on)
    or False if unset.
    """
    import os as _os
    region = module.agents.get(agent_name)
    if region is None:
        raise RuntimeError_(f"agent {agent_name!r} not found in module")

    if concurrent is None:
        concurrent = _os.environ.get("TESSERA_CONCURRENT_AGENTS", "").lower() in {"1", "true", "on"}

    world = World(module=module, concurrent=concurrent)
    if concurrent:
        world.executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="tessera-agent")
    try:
        state = world.state_for(agent_name)
        if initial_beliefs:
            state.working_memory.update(initial_beliefs)
        state.capabilities |= region.capabilities_in_scope
        if initial_capabilities:
            state.capabilities |= initial_capabilities
        return eval_region(region, world, agent_name=agent_name)
    finally:
        if world.executor is not None:
            world.executor.shutdown(wait=True)
