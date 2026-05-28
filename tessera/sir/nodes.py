"""SIR node definitions — subset of RFC §4 for the MVP.

Implemented operator categories: Pure (§4.1), Agent (§4.4), Memory:Working (§4.3).
Everything else (Tensor, Stream, Quantum, etc.) is deferred.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..parser.module import SourceSpan


class Op(str, Enum):
    # Pure (§4.1)
    Const = "tsr.const"
    Apply = "tsr.apply"
    Let = "tsr.let"
    BinOp = "tsr.binop"
    Lambda = "tsr.lambda"
    Return = "tsr.return"
    If = "tsr.if"

    # Memory: Working (§4.3)
    WM_Read = "tsr.wm.read"
    WM_Write = "tsr.wm.write"

    # Memory: Workspace (§4.3 — workspace tier)
    Workspace_Broadcast = "tsr.workspace.broadcast"
    Workspace_Read = "tsr.workspace.read"
    Workspace_Subscribe = "tsr.workspace.subscribe"

    # Memory: Episodic (§4.3 — episodic tier)
    EM_Append = "tsr.em.append"
    EM_Query = "tsr.em.query"

    # Memory: Semantic (§4.3 — semantic tier, local fact store)
    SM_Insert = "tsr.sm.insert"
    SM_Search = "tsr.sm.search"

    # Policy (§4.7)
    Policy_Check = "tsr.policy.check"
    Refuse = "tsr.refuse"

    # Memory: Procedural (§4.3 — procedural tier)
    PM_LoadSkill = "tsr.pm.load_skill"

    # Agent (§4.4)
    Spawn = "tsr.spawn"
    Send = "tsr.send"
    Recv = "tsr.recv"
    BeliefRead = "tsr.belief.read"
    BeliefRevise = "tsr.belief.revise"
    IntentionCommit = "tsr.intention.commit"

    # Notice (§4.15)
    Notice_Subscribe = "tsr.notice.subscribe"

    # Control flow (RFC §4.1 — Match-like)
    Until = "tsr.until"

    # Prompt (§4.5)
    Prompt_Render = "tsr.prompt.render"
    Prompt_Call = "tsr.prompt.call"

    # Tool (§4.6)
    Tool_Invoke = "tsr.tool.invoke"

    # Tensor / Neural (§4.2)
    Tensor_Const = "tsr.tensor.const"
    Tensor_Param = "tsr.tensor.param"
    Tensor_Linear = "tsr.tensor.linear"
    Tensor_Activation = "tsr.tensor.activation"
    Tensor_Apply = "tsr.tensor.apply"


PURE_OPS = {Op.Const, Op.Apply, Op.Let, Op.BinOp, Op.Lambda, Op.Return, Op.If, Op.Until}
AGENT_OPS = {
    Op.Spawn, Op.Send, Op.Recv,
    Op.BeliefRead, Op.BeliefRevise, Op.IntentionCommit,
    Op.Notice_Subscribe,
}
POLICY_OPS = {Op.Policy_Check, Op.Refuse}
MEMORY_OPS = {
    Op.WM_Read, Op.WM_Write,
    Op.Workspace_Broadcast, Op.Workspace_Read, Op.Workspace_Subscribe,
    Op.EM_Append, Op.EM_Query,
    Op.SM_Insert, Op.SM_Search,
    Op.PM_LoadSkill,
}
PROMPT_OPS = {Op.Prompt_Render, Op.Prompt_Call}
TOOL_OPS = {Op.Tool_Invoke}
NEURAL_OPS = {Op.Tensor_Const, Op.Tensor_Param, Op.Tensor_Linear,
              Op.Tensor_Activation, Op.Tensor_Apply}


# Effect names from RFC §6.1
class Effect(str, Enum):
    none = ""  # placeholder for empty set
    mem_working_r = "~mem.working.r"
    mem_working_w = "~mem.working.w"
    mem_workspace_w = "~mem.workspace.w"
    mem_workspace_r = "~mem.workspace.r"
    mem_workspace_subscribe = "~mem.workspace.subscribe"
    mem_episodic_w = "~mem.episodic.w"
    mem_episodic_r = "~mem.episodic.r"
    mem_semantic_w = "~mem.semantic.w"
    mem_semantic_r = "~mem.semantic.r"
    mem_procedural_r = "~mem.procedural.r"
    mem_procedural_w = "~mem.procedural.w"
    spawn = "~spawn"
    msg_send = "~msg.send"
    msg_recv = "~msg.recv"
    intention_commit = "~intention.commit"
    notice_subscribe = "~notice.subscribe"
    prompt = "~prompt"
    tool = "~tool"
    io = "~io"
    cost = "~cost"
    train = "~train"


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Node:
    id: str = field(default_factory=_new_id)
    op: Op = Op.Const
    inputs: list[str] = field(default_factory=list)         # node ids
    output_type: str = "unit"
    attributes: dict[str, Any] = field(default_factory=dict)
    substrate: str = "logic"
    effects: set[str] = field(default_factory=set)
    capability_requires: set[str] = field(default_factory=set)
    provenance: SourceSpan | None = None

    def __repr__(self) -> str:
        attrs = " ".join(f"{k}={v!r}" for k, v in self.attributes.items())
        eff = ",".join(sorted(self.effects)) or ""
        return f"<{self.id} {self.op.value} ({','.join(self.inputs)}) {attrs} #effects<{eff}>>"


@dataclass
class Region:
    """A single-entry, single-exit subgraph corresponding to a lexical scope (RFC §3.1, §3.3)."""

    id: str = field(default_factory=_new_id)
    name: str = ""
    params: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    return_type: str = "unit"
    nodes: list[Node] = field(default_factory=list)
    capabilities_in_scope: set[str] = field(default_factory=set)
    parent: str | None = None
    # Cognitive traits attached to this region (agent regions only). Names are
    # resolved against Module.traits (local) then BUILTIN_TRAITS at runtime.
    trait_names: list[str] = field(default_factory=list)
    # Intent this region serves (agent: `intends X`, plan: `serves X`). Plans
    # inherit their agent's intent when they don't declare their own.
    intent: str | None = None

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node


@dataclass
class WorkspaceDecl:
    """Declared workspace — instantiated at runtime by the actor scheduler."""
    name: str
    capacity: int = 1
    arbiter: str = "highest_salience"
    contenders: list[str] = field(default_factory=list)


@dataclass
class PromptDecl:
    """Declared prompt template — rendered with bindings, sent to an LLM."""
    name: str
    params: list[tuple[str, str]]  # (name, type)
    return_type: str
    template: str
    model_hint: str | None = None  # optional model override per prompt


@dataclass
class ToolDecl:
    """Declared external tool — resolved via importlib at runtime."""
    name: str
    params: list[tuple[str, str]]
    return_type: str
    import_path: str          # e.g., "langchain_community.tools.DuckDuckGoSearchRun"
    invoke_method: str = "invoke"   # LangChain convention; "__call__" for bare callables


@dataclass
class NeuralModelDecl:
    """Declared neural model — compiled lazily to a torch nn.Module."""
    name: str
    layers: list[dict]   # ordered: [{kind: "linear", in: 784, out: 128}, {kind: "relu"}, ...]


@dataclass
class EpisodicEventDecl:
    """A typed event declaration in an `episodic { event Foo(x: T, ...) }` block."""
    name: str
    fields: list[tuple[str, str]]  # (field_name, type_str)


@dataclass
class KnowledgeSchemaDecl:
    """A typed schema in a `knowledge { schema FactSheet(field: T, ...) }` block.

    `persistent=True` writes facts to the local SQLite store (`~/.tessera/semantic.db`)
    and queries them back across runs. `persistent=False` keeps facts in a per-World
    shadow that lives only for the duration of the run — useful for tests, dry runs,
    and any agent that needs semantic structure without disk persistence.
    """
    name: str
    fields: list[tuple[str, str]]
    persistent: bool = True


@dataclass
class PolicyDecl:
    """A runtime policy — checked at plan-step boundaries.

    MVP form: each rule is `(kind, params)` where kind is one of:
      - "forbid_contains" — refuse if any input contains the given substring
      - "forbid_match"    — refuse if any input matches the given regex
      - "require_contains" — refuse if input does NOT contain the substring
    """
    name: str
    rules: list[tuple[str, dict]]


@dataclass
class EvalCaseDecl:
    """One test case in an `eval { case "X" { ... } }` block."""
    name: str
    inputs: dict[str, str]                  # belief name → value
    expect_contains: str | None = None
    expect_equals: str | None = None
    expect_refusal: bool = False


@dataclass
class SkillDecl:
    """A learned skill declared in a `procedural { skill X from <kind> Y }` block.

    A skill is a named indirection over an underlying callable. The agent
    treats it like a function; the interpreter dispatches to whatever the
    skill is `from` (model, prompt, tool, or another skill).

    Why distinct from `tool` or `prompt`? Procedural skills carry semantic
    weight — they represent capabilities the agent has *internalized*. They
    accumulate call statistics and (in a later phase) get promoted from
    successful traces via `evolve`.
    """
    name: str
    params: list[tuple[str, str]]   # (name, type)
    return_type: str
    binds_to_kind: str              # "model" | "prompt" | "tool" | "fn"
    binds_to_name: str


@dataclass
class TraitDecl:
    """A cognitive trait — a channeled reasoning posture injected into prompts.

    Traits modify *how* an agent deliberates, not what it's allowed to do. A
    trait fires when any of its trigger terms match the current context; when it
    fires, its `behavior` text is injected as a preamble into the rendered
    prompt. See `tessera/traits.py` for the deterministic firing engine and the
    built-in trait registry.
    """
    name: str
    trigger: list[str]              # OR-set of trigger terms (split on "|")
    behavior: str                   # reasoning posture, injected into prompts
    priority: float = 0.5           # 0.0–1.0; resolves order + conflicts
    scope: str = "per_call"         # "per_call" | "per_plan" | "global"


@dataclass
class IntentDecl:
    """A declared intent — what an agent (or plan) is *for*.

    Intent is the thing you audit against: it states the goal, the checkable
    success criteria, and the outcomes that must never happen. `forbidden`
    entries name `tsr:policy` rules, binding purpose to enforcement so a stated
    intent can't be declared without the guardrails that back it.
    """
    name: str
    goal: str = ""
    success: list[str] = field(default_factory=list)   # checkable predicates (raw text MVP)
    forbidden: list[str] = field(default_factory=list)  # names of tsr:policy rules
    why: str = ""                                        # human rationale, for the audit narrative


@dataclass
class EthicsPrinciple:
    """One named value the agent reasons under."""
    name: str
    rule: str                 # the principle, injected into prompts as a values frame
    weight: float = 0.5       # 0.0–1.0; orders principles and resolves conflicts


@dataclass
class EthicsDecl:
    """The agent's ethical frame — values above hard policy.

    Principles are injected into every prompt (outermost, before cognitive
    posture) so the agent reasons under them, and every action records the frame
    it operated under. `on_violation` is the declared disposition surfaced in the
    audit trace, not a deterministic blocker (principles are natural-language).
    """
    principles: list[EthicsPrinciple] = field(default_factory=list)
    on_conflict: str = "highest_weight"   # "highest_weight" | "first"
    on_violation: str = "refuse"          # "refuse" | "flag" | "defer"


@dataclass
class AutonomyDecl:
    """How much the agent may do unsupervised.

    `level` sets the default disposition; `require_approval` names action classes
    (capabilities, trait-style terms like `payments`/`auth`, or plain keywords)
    that need a human. At `propose`, a gated action is blocked before it runs and
    logged; at `act_with_rollback` it proceeds but is flagged; at `act_freely` it
    proceeds silently. Autonomy with an audit trail = action you can still trace.
    """
    level: str = "propose"                          # "propose" | "act_with_rollback" | "act_freely"
    require_approval: list[str] = field(default_factory=list)
    escalate_when: str = ""
    boundary: str = ""


@dataclass
class Module:
    name: str
    regions: list[Region] = field(default_factory=list)
    agents: dict[str, Region] = field(default_factory=dict)
    functions: dict[str, Region] = field(default_factory=dict)
    workspaces: dict[str, WorkspaceDecl] = field(default_factory=dict)
    prompts: dict[str, PromptDecl] = field(default_factory=dict)
    tools: dict[str, ToolDecl] = field(default_factory=dict)
    neural_models: dict[str, NeuralModelDecl] = field(default_factory=dict)
    episodic_events: dict[str, EpisodicEventDecl] = field(default_factory=dict)
    knowledge_schemas: dict[str, KnowledgeSchemaDecl] = field(default_factory=dict)
    policies: dict[str, PolicyDecl] = field(default_factory=dict)
    eval_cases: list[EvalCaseDecl] = field(default_factory=list)
    skills: dict[str, SkillDecl] = field(default_factory=dict)
    traits: dict[str, TraitDecl] = field(default_factory=dict)
    intents: dict[str, IntentDecl] = field(default_factory=dict)
    ethics: "EthicsDecl | None" = None
    autonomy: "AutonomyDecl | None" = None

    def all_nodes(self):
        for r in self.regions:
            yield from r.nodes
