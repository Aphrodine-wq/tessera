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
    SM_Relate = "tsr.sm.relate"      # typed edge: subject -predicate-> object
    SM_Related = "tsr.sm.related"    # neighbors of a fact via a predicate

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
    Op.SM_Insert, Op.SM_Search, Op.SM_Relate, Op.SM_Related,
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
    """Declared workspace — instantiated at runtime by the actor scheduler.

    Global Workspace Theory extension (research B1).
    Primary references:
      - Baars, B. (1988). A Cognitive Theory of Consciousness. Cambridge UP.
      - Dehaene, S. (2014). Consciousness and the Brain.

    `gwt_bottleneck` caps how many contenders the workspace can hold at
    once — Baars's attention bottleneck. When the limit is exceeded the
    lowest-salience contender drops out. `track_ignition` toggles whether
    arbitration emits a `gwt:ignition` audit event with the bandwidth
    (contender count) and selected winner — Dehaene's ignition signature
    measured functionally.
    """
    name: str
    capacity: int = 1
    arbiter: str = "highest_salience"
    contenders: list[str] = field(default_factory=list)
    gwt_bottleneck: int = 0          # 0 = unlimited; >0 = max contenders
    track_ignition: bool = False     # emit gwt:ignition audit on arbitrate


@dataclass
class PromptDecl:
    """Declared prompt template — rendered with bindings, sent to an LLM."""
    name: str
    params: list[tuple[str, str]]  # (name, type)
    return_type: str
    template: str
    model_hint: str | None = None  # optional model override per prompt
    emits: str | None = None       # tool name -> constrain output to that tool's
                                    # call grammar (block attr `emits=<tool>`).
                                    # See adapters/wire (tson constrained decoding).
    execute: bool = False          # if True AND emits is set: dispatch the emitted
                                    # !call and return the tool result (block attr
                                    # `execute=true`). Default: return the record text.


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
    """Declared neural model — compiled lazily to a torch nn.Module.

    `trainable` and the surrounding fields are populated when the author
    declares a `trainable { ... }` clause inside the `model` block. The
    runtime checks for a checkpoint at `~/.tessera/checkpoints/<name>.pt`
    and loads it before forward when training has previously run.
    """
    name: str
    layers: list[dict]   # ordered: [{kind: "linear", in: 784, out: 128}, {kind: "relu"}, ...]
    trainable: bool = False
    optimizer: str = "adam"        # "adam" | "sgd"
    learning_rate: float = 1e-3
    epochs: int = 50
    loss: str = "mse"              # "mse" | "cross_entropy"
    batch_size: int = 32


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
class RelationDecl:
    """A typed predicate in a `knowledge { relation cites(Claim -> Source) }` block.
    Edges created with `relate a -cites-> b` are checked: a must be a `subject_schema`
    fact and b an `object_schema` fact, so the graph stays typed."""
    name: str
    subject_schema: str
    object_schema: str


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
class ContractDecl:
    """A runtime contract — author-declared before/after assertions bound to a
    named effect (`prompt:X`, `tool:Y`, `plan:Z`).

    Unlike `tsr:policy` (a prohibition: `forbid when <expr>` refuses on True),
    a contract is a GUARANTEE: each `before`/`after` clause is an assertion that
    MUST hold, and a clause evaluating False is the violation. Clauses are
    parsed `policy_lang` expressions evaluated against an ActionContext —
    `before` clauses see the action's inputs, `after` clauses see its result via
    `value()` (and `intent_match()` for output-vs-intent drift).

    `on_violation` is `(mode, n, fallback)`:
      - ("refuse", 0, "")     — block the action, return a refusal
      - ("audit",  0, "")     — record the violation, let the action stand
      - ("retry",  N, F)      — re-drive the effect up to N times (after-clauses
                                only); on exhaustion fall back to F ("refuse" |
                                "audit"). before-clauses can't regenerate inputs,
                                so a retry there degrades to refuse (verify warns).
    """
    name: str
    target_kind: str                                   # "prompt" | "tool" | "plan"
    target_name: str                                   # the declared effect's name
    before: list[tuple[Any, str]] = field(default_factory=list)  # (parsed Expr, src)
    after: list[tuple[Any, str]] = field(default_factory=list)
    on_violation: tuple[str, int, str] = ("refuse", 0, "")

    @property
    def target_label(self) -> str:
        return f"{self.target_kind}:{self.target_name}"


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
    # Promotion plumbing (decision 16). `promote_to="neural"` means: after
    # `promote_threshold` calls accumulate in the operational audit store,
    # the runtime emits a `skill_promotion_pending` event. Actual training
    # (training corpus assembly, vast.ai job, checkpoint swap) is a
    # follow-up that consumes those events; this commit only lights the
    # signal so the surface is callable.
    promote_to: str | None = None
    promote_threshold: int = 100


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
    allow_text_calls: list[str] = field(default_factory=list)  # tool names permitted
                                                    # to run from a text-sourced (called_via=text)
                                                    # record. Default empty = all refused (PRD §12).


@dataclass
class WelfareDecl:
    """Welfare substrate config (research C4).

    Reference: Birch (2020). The search for invertebrate consciousness.
    Noûs 54(1):133-155.

    `thresholds` maps marker name to minimum acceptable value:
      - "phi"           — minimum phi* (from tsr:iit)
      - "bandwidth"     — minimum workspace ignition bandwidth
      - "ast_fidelity"  — minimum AST self-report fidelity
    `consecutive_required` is the number of consecutive cycles a marker
    must remain below threshold before refusal triggers.

    PHILOSOPHY.md: this is a BEHAVIORAL commitment. It is NOT a claim
    about phenomenal consciousness or moral status.
    """
    thresholds: dict[str, float] = field(default_factory=dict)
    consecutive_required: int = 3


@dataclass
class IITDecl:
    """IIT substrate config (research C1).

    Refs: Tononi (2004, 2016); Mediano et al. (2022).

    Tessera ships φ* — Mediano et al.'s geometric-loss approximation
    of integrated information. The canonical Tononi φ is intractable
    past ~6 nodes; φ* runs in polynomial time over the agent's
    belief/intention dependency graph.

    `emit_phi_audit` toggles whether plan_enter emits an iit:phi event
    with the computed score. `agent_subject` defaults to the host
    agent of the file.

    PHILOSOPHY.md: φ* is a STRUCTURAL measure. It is NOT consciousness.
    Any block making φ > 0 → conscious inferences is rejected by
    pass_9_consciousness_claim_check.
    """
    emit_phi_audit: bool = True
    agent_subject: str = ""  # "" = host agent of this module


@dataclass
class ToMDecl:
    """Theory of Mind substrate config (research C3).

    Refs: Premack & Woodruff (1978); Baker, Saxe, Tenenbaum (2009);
    Rabinowitz et al. (2018, Machine theory of mind).

    `tracked_agents` is the list of agent names this agent maintains a
    belief model for. `manipulation_refusal` (default true) makes the
    agent refuse to produce outputs likely to leave the listener with
    a false belief — pairs with the ethics substrate.
    """
    tracked_agents: list[str] = field(default_factory=list)
    manipulation_refusal: bool = True


@dataclass
class ASTDecl:
    """Attention Schema Theory substrate config (research C2).

    Reference: Graziano (2013, 2019); Graziano et al. (2020).

    `min_fidelity` is the threshold below which the agent refuses to
    introspect. Tracks fraction of past (reported_focus, actual_focus)
    pairs that matched. `refuse_below_threshold` toggles the refusal —
    when false, the agent still introspects but the fidelity score is
    audit-emitted so a downstream evaluator can decide.

    The substrate ships the MEASURE. It does NOT claim that maintaining
    an attention schema produces subjective experience (per Chalmers'
    hard problem, that question is left to philosophy).
    """
    min_fidelity: float = 0.7
    refuse_below_threshold: bool = True


@dataclass
class BayesianVarSpec:
    """One discrete random variable in a tsr:bayesian block.

    Reference: Blei, Kucukelbir, McAuliffe (2017). Variational inference:
    a review for statisticians. JASA. (MVP uses exact discrete inference,
    not VI; the citation marks the planned follow-up direction.)
    """
    name: str
    values: list[str]
    prior: list[float]


@dataclass
class BayesianLikelihoodSpec:
    """A conditional probability table: P(observed | latent)."""
    latent: str
    observed: str
    rows: dict[str, dict[str, float]]  # rows[latent_value][observed_value] = P


@dataclass
class BayesianDeclSIR:
    """The full tsr:bayesian declaration captured in SIR."""
    variables: list[BayesianVarSpec] = field(default_factory=list)
    likelihoods: list[BayesianLikelihoodSpec] = field(default_factory=list)


@dataclass
class CausalDAGDecl:
    """Declared causal DAG (research substrate D1).

    Reference: Pearl (2009). Causality: Models, Reasoning, and Inference.

    `variables` is the node set, `edges` is the directed edge list
    (parent, child). Compile-time checks: no cycles, no edges referencing
    undeclared variables.
    """
    name: str
    variables: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class MetacognitionDecl:
    """Calibration / metacognition substrate config (research substrate A1).

    Reference: Guo, Pleiss, Sun, Weinberger (2017). On Calibration of Modern
    Neural Networks. ICML.

    `temperature` is the scalar T applied to logits before softmax. T=1.0 means
    no scaling. `track_ece` toggles ECE-on-audit emission per plan_enter.
    `n_bins` controls binning granularity for the ECE estimator.
    """
    temperature: float = 1.0
    n_bins: int = 15
    track_ece: bool = True


@dataclass
class PrecautionThresholdSpec:
    """One author-declared harm threshold inside a tsr:precaution block."""
    action_class: str
    harm_magnitude: float = 1.0
    irreversible: bool = False
    max_tail_probability: float = 0.01


@dataclass
class PrecautionDecl:
    """Precautionary gate config (research 4.7, Hansson 2003).

    Each threshold names an action class (matched against the rendered prompt
    text + action label, like tsr:autonomy). `default_tail` is the assumed tail
    probability when no evidence (e.g. a tsr:bayesian posterior) pins it down —
    high by default, because precaution shifts the burden of proof onto the
    action. An irreversible action class with tail > 0.001 is refused outright.
    """
    thresholds: list[PrecautionThresholdSpec] = field(default_factory=list)
    default_tail: float = 0.5


@dataclass
class MoralFoundationsDecl:
    """Moral Foundations gate config (research 4.9, Haidt/Graham).

    `weights` is the agent's per-axis value vector (care/fairness/loyalty/
    authority/sanctity/liberty). `violations` maps a foundation to the terms
    that, when matched in an action, score that axis negative. An action that
    scores negative on a weighted axis (weight > 0.1) is refused — "even a small
    commitment to fairness rules out unfair actions."
    """
    weights: dict[str, float] = field(default_factory=dict)
    violations: dict[str, list[str]] = field(default_factory=dict)
    accept_threshold: float = 0.0


@dataclass
class DualProcessDecl:
    """Dual-process router config (research 4.1, Kahneman; Evans & Stanovich).

    At plan entry the router picks fast vs slow from the agent's confidence
    (the `_confidence` belief, else `default_confidence`), the remaining budget,
    and whether the plan touches an irreversible action term. The decision is
    audit-emitted as `dual_process:route` and stored on the agent so downstream
    steps can read the active mode.

    `slow_backend`, when set, is an LLM backend name (e.g. `anthropic`,
    `llamacpp`) that a slow-routed call escalates to instead of the module's
    default backend — "slow" then means an actually-smarter model, not just
    the default model told to deliberate more. Unset preserves the prior
    behavior (same backend, forced-fresh, a deliberation preamble).
    """
    preferred: str = "fast"                       # "fast" | "slow"
    confidence_threshold: float = 0.7
    budget_threshold: float = 0.2
    default_confidence: float = 1.0
    irreversible_terms: list[str] = field(default_factory=list)
    slow_backend: str | None = None


@dataclass
class GriceanDecl:
    """Gricean maxim checker config (research 4.5, Grice 1975).

    Runs after a prompt returns, scoring the output against the four maxims.
    `gate_maxims` names the maxims that REFUSE on violation; the rest only warn
    (audit). evidence/topic keywords drive the quality + relation checks.
    """
    min_words: int = 1
    max_words: int = 200
    evidence_keywords: list[str] = field(default_factory=list)
    topic_keywords: list[str] = field(default_factory=list)
    gate_maxims: list[str] = field(default_factory=list)  # subset of quantity/quality/relation/manner


@dataclass
class HindsightDecl:
    """After-action review config (research 4.10, Army AAR / Argyris & Schön).

    When enabled, every completed plan gets a hindsight:learning review
    comparing declared vs applied ethics (from the audit trail) and recording
    the actual outcome. Reviews accumulate per agent; `fitness_from_reviews`
    exposes them to tsr:evolve as a fitness signal.
    """
    enabled: bool = True


@dataclass
class ArgumentativeDecl:
    """Argumentative-reasoning config (research 4.12, Mercier & Sperber).

    After a (non-critic) prompt returns, fire the named `critic` prompt against
    the output, score the counter-argument's strength from declared
    `refutation_markers`, and log-odds-downweight the proposer's confidence.
    Below `accept_threshold` the answer is refused rather than shipped.
    """
    critic: str = ""                               # name of the critic prompt
    accept_threshold: float = 0.5
    proposer_confidence: float = 0.9
    refutation_markers: list[str] = field(default_factory=lambda: [
        "however", "but", "false", "incorrect", "wrong", "lacks", "no evidence",
    ])


@dataclass
class EvolveDecl:
    """A genetic evolution declaration (decision 17).

    Target agent gets N variants per generation; mutation operators
    perturb the targets (prompts/traits) and fitness is measured against
    the agent's eval cases. The best agent of each generation persists
    its prompts/traits + score to the governance audit store.
    """
    target_agent: str
    population: int = 4
    mutate_targets: list[str] = field(default_factory=lambda: ["prompts"])
    fitness: str = "eval_pass_rate"
    generations: int = 3


@dataclass
class RLDecl:
    """Reinforcement-learning substrate config (research B3; Sutton & Barto 2018).

    Plan-level tabular Q-learning. `rl_choose()` returns an ε-greedily chosen
    action label from `actions`; `rl_reward(action, reward)` updates the
    Q-table. `state_from` names the belief keys that form the Q-table state
    key. Q-tables persist per agent under ~/.tessera/rl/ (override
    TESSERA_RL_DIR).
    """
    target_agent: str = ""
    actions: list[str] = field(default_factory=list)
    state_from: list[str] = field(default_factory=list)
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    epsilon_decay_steps: int = 0  # 0 → fixed epsilon


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
    relations: dict[str, RelationDecl] = field(default_factory=dict)
    policies: dict[str, PolicyDecl] = field(default_factory=dict)
    contracts: dict[str, ContractDecl] = field(default_factory=dict)
    eval_cases: list[EvalCaseDecl] = field(default_factory=list)
    skills: dict[str, SkillDecl] = field(default_factory=dict)
    traits: dict[str, TraitDecl] = field(default_factory=dict)
    intents: dict[str, IntentDecl] = field(default_factory=dict)
    ethics: "EthicsDecl | None" = None
    autonomy: "AutonomyDecl | None" = None
    evolve: "EvolveDecl | None" = None
    metacognition: "MetacognitionDecl | None" = None
    causal_dags: dict[str, "CausalDAGDecl"] = field(default_factory=dict)
    bayesian: "BayesianDeclSIR | None" = None
    ast: "ASTDecl | None" = None
    tom: "ToMDecl | None" = None
    iit: "IITDecl | None" = None
    welfare: "WelfareDecl | None" = None
    precaution: "PrecautionDecl | None" = None
    moral_foundations: "MoralFoundationsDecl | None" = None
    dual_process: "DualProcessDecl | None" = None
    gricean: "GriceanDecl | None" = None
    hindsight: "HindsightDecl | None" = None
    argumentative: "ArgumentativeDecl | None" = None
    rl: "RLDecl | None" = None
    prose: str = ""  # document prose (outside substrate blocks) — scanned by pass_9

    def all_nodes(self):
        for r in self.regions:
            yield from r.nodes
