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

    # Memory: Semantic (§4.3 — semantic tier, Synapse-backed)
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
    """A typed schema in a `knowledge { schema FactSheet(field: T, ...) }` block."""
    name: str
    fields: list[tuple[str, str]]


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

    def all_nodes(self):
        for r in self.regions:
            yield from r.nodes
