"""Substrate categories — modes of thinking, in plain English.

Every substrate is a CATEGORY of cognition an agent can do. Each entry has:

  - ``summary``: 1-line description in plain English
  - ``when_to_use``: a concrete prompt for the builder
  - ``example_idiom``: a typical 1-2 line snippet from a .t.md file
  - ``maps_to``: the theory or system this corresponds to (for the curious)
  - ``status``: ``shipped`` | ``partial`` | ``planned``

Used by `tessera substrates` CLI and by Obsidian agent scaffolds. Keep
descriptions short and concrete — a builder should pick the right substrate
from this list in under 10 seconds.
"""
from __future__ import annotations


SUBSTRATE_DOCS: dict[str, dict] = {

    "logic": {
        "summary": "Pure reasoning — no side effects, no LLM, no state.",
        "when_to_use": "Math, transforms, predicates, deterministic functions. "
                       "The foundation other substrates compose with.",
        "example_idiom": "fn rank(papers: List, query: String) -> Float = ...",
        "maps_to": "Pure functional programming",
        "status": "shipped",
    },

    "agent": {
        "summary": "Goal-directed actor with beliefs, intentions, plans.",
        "when_to_use": "When something needs to DO things — make decisions, "
                       "spawn other agents, send messages, achieve goals.",
        "example_idiom": "agent FooBot { beliefs: ... intentions: plan act { ... } }",
        "maps_to": "BDI (Belief-Desire-Intention) architecture",
        "status": "shipped",
    },

    "memory:working": {
        "summary": "Scratchpad — values held during one plan invocation.",
        "when_to_use": "Local variables inside a plan. Discarded when the "
                       "plan returns. Not for state that should persist.",
        "example_idiom": "let intermediate = compute(input)",
        "maps_to": "Working memory (cognitive psych)",
        "status": "shipped",
    },

    "memory:workspace": {
        "summary": "Global blackboard — many writers, an arbiter picks one winner.",
        "when_to_use": "When multiple intentions or agents need to agree on "
                       "'what we're doing now.' Contenders pool across a round "
                       "and are resolved on read; the winner becomes the next "
                       "draft of the agent's behavior. Arbiters: highest_salience, "
                       "last_write, weighted_vote (agreement compounds), "
                       "quorum(N) (resolves only once N agree, else abstains).",
        "example_idiom": "workspace TeamMind { capacity: 1 arbiter: quorum(2) }",
        "maps_to": "Global Workspace Theory (Baars/Dehaene)",
        "status": "shipped",
    },

    "memory:episodic": {
        "summary": "Append-only event log — agent's autobiographical history.",
        "when_to_use": "When the agent needs to remember what happened: "
                       "'last week I tried X and it failed.' Survives across runs.",
        "example_idiom": "episodic { event Decision(topic: String, choice: String) }",
        "maps_to": "Episodic memory (Tulving)",
        "status": "shipped",
    },

    "memory:semantic": {
        "summary": "Knowledge graph — facts, concepts, relations the agent knows.",
        "when_to_use": "World knowledge that survives across runs and is shared "
                       "across agents. Backed by a local SQLite fact store at "
                       "`~/.tessera/semantic.db`.",
        "example_idiom": "knowledge { schema FactSheet(title: String, domain: String) }",
        "maps_to": "Semantic memory; local fact store.",
        "status": "shipped",
    },

    "memory:procedural": {
        "summary": "Learned skills — named indirection over neural / prompt / tool / fn.",
        "when_to_use": "Capabilities the agent has internalized. Distinct from "
                       "bare model/prompt/tool calls because skills accumulate "
                       "call stats, cache by input, and become first-class in "
                       "the agent's identity.",
        "example_idiom": "procedural { skill greet(name: String) -> String from prompt brief }",
        "maps_to": "Procedural memory (Tulving); compiled skills",
        "status": "shipped",
    },

    "prompt": {
        "summary": "LLM call — template + bindings, dispatched to a backend.",
        "when_to_use": "Any reasoning that benefits from language-model judgment: "
                       "summarization, generation, classification by description, "
                       "open-ended planning.",
        "example_idiom": 'prompt summarize(text: String) -> String = "Summarize: {text}"',
        "maps_to": "Foundation models (Anthropic, OpenAI, Ollama, etc.)",
        "status": "shipped",
    },

    "tool": {
        "summary": "External callable — a python function or LangChain Tool.",
        "when_to_use": "Anything the agent should DO in the world: search, "
                       "calculate, query a DB, write a file, call an API.",
        "example_idiom": "tool web_search(q: String) -> String from langchain_community.tools.DuckDuckGoSearchRun",
        "maps_to": "Tool use; function calling",
        "status": "shipped",
    },

    "neural": {
        "summary": "Differentiable model — a torch nn.Module declared inline.",
        "when_to_use": "Perception, classification, or any learned function. "
                       "Cohabits with the agent substrate.",
        "example_idiom": "model classifier { linear in=784 out=128; relu; linear in=128 out=10 }",
        "maps_to": "Deep learning (PyTorch backend)",
        "status": "shipped",
    },

    "policy": {
        "summary": "Runtime rules — what the agent is and isn't allowed to do.",
        "when_to_use": "Safety, compliance, content rules. Enforced at "
                       "plan-step boundaries; refusal is a first-class value.",
        "example_idiom": 'policy NoPII { forbid contains "SSN"; forbid match "[0-9]{3}-[0-9]{2}-[0-9]{4}" }',
        "maps_to": "Capability security; ethics layers",
        "status": "shipped",
    },

    "eval": {
        "summary": "Test cases — declared inputs + expected outputs.",
        "when_to_use": "Calibration, regression testing, policy verification. "
                       "Run via `tessera eval <file>`.",
        "example_idiom": 'case "clean input passes" { input q = "..."; expect_contains = "..." }',
        "maps_to": "Self-evaluation; agent eval frameworks",
        "status": "shipped",
    },

    "evolve": {
        "summary": "Self-modification — proposals to change the agent's own code.",
        "when_to_use": "Agents that should adapt their own behavior. Requires "
                       "the SelfModify capability and passes evals before commit.",
        "example_idiom": "evolve propose_refactor with [SelfModify] { ... }",
        "maps_to": "Self-modifying programs; meta-learning",
        "status": "planned",
    },

    "identity": {
        "summary": "Persistent self-model — what the agent thinks it IS.",
        "when_to_use": "Cross-run identity stability. Visible at federation "
                       "boundaries when one vault imports another's agent.",
        "example_idiom": "identity FooBotSelf { name: 'FooBot'; commitments: [...] }",
        "maps_to": "Self-models; identity persistence",
        "status": "planned",
    },

    "predict": {
        "summary": "Predictive processing — generate predictions, minimize error.",
        "when_to_use": "Agents that work by reducing surprise. Active inference. "
                       "The cognitive science framework of Friston, Clark, Hohwy.",
        "example_idiom": "predictor BeliefPredictor on (a) { hidden_states: [...] }",
        "maps_to": "Predictive Processing / Active Inference",
        "status": "planned",
    },

    "phenomenology": {
        "summary": "Consciousness correlates — Φ, ignition, recurrence, theory of mind.",
        "when_to_use": "Research and audit. The most contested substrate. "
                       "READ docs/ethics.md before using — operationalizes claims "
                       "from IIT/GWT/RPT/ToM that have ethical weight.",
        "example_idiom": "@maintain(phi > 0.4) agent Researcher { ... }",
        "maps_to": "IIT (Tononi), GNW (Dehaene), RPT (Lamme), ToM (Premack/Woodruff)",
        "status": "planned",
    },

    "traits": {
        "summary": "Cognitive posture — channeled tendencies that modify how the agent reasons.",
        "when_to_use": "When an agent needs a non-default reasoning style: assume-wrong-by-default "
                       "(doubt_first), lateral cross-context scanning (cross_brain), completeness "
                       "verification (compulsive), threat-scanning (hypervigilant), etc. Traits "
                       "compose — multiple stack, priority-weighted at decision points.",
        "example_idiom": "trait doubt_first { trigger: any_claim; behavior: 'verify before committing'; priority: 0.9 }",
        "maps_to": "Channeled psychological tendencies (Wakefield/Nesse harmful-dysfunction framework, "
                   "inverted — same shape, productive direction). Inspired by the Twin Protocol's "
                   "ADHD + depression posture.",
        "status": "shipped",
    },

    "intent": {
        "summary": "Purpose — what an agent (or plan) is for, in checkable form.",
        "when_to_use": "When an agent's actions need to be auditable against a goal. Declares the "
                       "goal, success criteria, and forbidden outcomes; agents bind via "
                       "`intends X`, plans via `serves X`. `forbidden` entries must map to tsr:policy "
                       "rules, so stated purpose can't ship without its guardrails. Every runtime "
                       "action is stamped with the intent it served in the audit trace.",
        "example_idiom": "intent estimate { goal: 'bounded cost estimate' success: total > 0 forbidden: [NoPII] }",
        "maps_to": "Goal/intention in BDI (Bratman); design-by-contract preconditions/postconditions; "
                   "provenance and accountability (the same value as auditable code).",
        "status": "shipped",
    },

    "ethics": {
        "summary": "Values frame — named principles the agent reasons under, above hard policy.",
        "when_to_use": "When an agent acts on people and you want its values explicit and auditable. "
                       "Each principle carries a weight and a rule; principles inject into every "
                       "prompt (outermost, before cognitive posture) and every action records the "
                       "ethical frame it operated under. Hard, mechanical constraints still belong in "
                       "tsr:policy — ethics is the weighed values layer above it.",
        "example_idiom": "ethics { principle honesty { weight: 0.95 rule: 'surface uncertainty; never fabricate' } on_violation: refuse }",
        "maps_to": "Principlism (Beauchamp & Childress); value-sensitive design; Kantian dignity "
                   "(treat persons as ends). The values half of responsible autonomy.",
        "status": "shipped",
    },

    "autonomy": {
        "summary": "How much the agent may do unsupervised — with a human-in-the-loop gate.",
        "when_to_use": "When an agent can take consequential action. `level` sets the disposition "
                       "(propose / act_with_rollback / act_freely); `require_approval` names action "
                       "classes (payments, auth, irreversible, ...) that need a human. At `propose`, "
                       "a gated action is blocked before it runs and logged in the audit trace — "
                       "autonomy you can still hold accountable.",
        "example_idiom": "autonomy { level: propose require_approval: [payments, auth] boundary: 'never act beyond the declared intent' }",
        "maps_to": "Levels-of-automation (Sheridan & Verplank); human-in-the-loop control; "
                   "capability-based authority. The agency half of responsible autonomy.",
        "status": "shipped",
    },

    "iit": {
        "summary": "Integrated information (φ*) — a structural integration measure over the agent's mind.",
        "when_to_use": "When you want a functional signal of how integrated an agent's belief/intention "
                       "graph is. At every plan entry the runtime computes φ* (a tractable min-cut "
                       "approximation of Tononi's φ) over the dependency graph and emits an `iit:phi` "
                       "audit event. φ* is a STRUCTURAL measure (PHILOSOPHY.md) — it is NOT consciousness, "
                       "and blocks claiming φ > 0 → conscious are refused.",
        "example_idiom": "iit { emit_phi_audit: true }",
        "maps_to": "Integrated Information Theory (Tononi 2004/2016); φ* geometric-loss approximation "
                   "(Mediano et al. 2022).",
        "status": "shipped",
    },

    "welfare": {
        "summary": "Birch-marker behavioral gate — refuse to keep running when welfare markers stay low.",
        "when_to_use": "When you want a precautionary commitment that doesn't wait on the hard problem: "
                       "declare minimum thresholds for markers (φ* from tsr:iit, broadcast bandwidth from "
                       "the GWT workspace, fidelity from tsr:ast). When a marker reads below threshold for "
                       "N consecutive plan-entry cycles, the agent refuses further work until it recovers. "
                       "A BEHAVIORAL gate (PHILOSOPHY.md), not a claim about moral status.",
        "example_idiom": "welfare { threshold phi: 0.3 consecutive_required: 3 }",
        "maps_to": "Birch (2020) marker framework for welfare under uncertainty; precautionary ethics.",
        "status": "shipped",
    },

    "ast": {
        "summary": "Attention-schema fidelity — measure how honest the agent's self-report is.",
        "when_to_use": "When an agent introspects (reports a focus via the `_focus` belief) and you want "
                       "that self-report held to account. The runtime scores fidelity — how often the "
                       "reported focus matches the plan actually running — and, when "
                       "`refuse_below_threshold` is set, refuses to trust introspection below the bar. "
                       "Ships the MEASURE only; no claim that an attention schema yields experience.",
        "example_idiom": "ast { min_fidelity: 0.7 refuse_below_threshold: true }",
        "maps_to": "Attention Schema Theory (Graziano 2013/2019); model-fidelity as a honesty signal.",
        "status": "shipped",
    },

    "tom": {
        "summary": "Theory of Mind — model tracked agents' beliefs and refuse to manipulate them.",
        "when_to_use": "When an agent communicates with or about other agents and you want a guard against "
                       "deception. With `manipulation_refusal`, an output that would leave a tracked agent "
                       "holding a belief the messenger itself recorded as false (a `tom_false` ground-truth "
                       "marker) is refused — a Sally-Anne false-belief check grounded in the agent's own "
                       "knowledge, not guessed from free text.",
        "example_idiom": "tom { tracked_agents: [Sally] manipulation_refusal: true }",
        "maps_to": "Theory of Mind (Premack & Woodruff 1978); inverse planning (Baker/Saxe/Tenenbaum 2009); "
                   "machine ToM (Rabinowitz et al. 2018).",
        "status": "shipped",
    },

    "precaution": {
        "summary": "Asymmetric-risk gate — refuse irreversible / high-tail actions under uncertainty.",
        "when_to_use": "When an action could cause large or irreversible harm and you'd rather a wrong "
                       "refuse than a wrong allow. Declare per-action-class thresholds; the gate matches "
                       "them against the action (like tsr:autonomy) and refuses when the tail probability "
                       "exceeds the max — and refuses any irreversible class outright unless evidence (a "
                       "tsr:bayesian posterior) drives the tail below 0.001. Burden of proof on the action.",
        "example_idiom": "precaution { default_tail: 0.5 threshold delete { harm: 10 irreversible: true max_tail: 0.01 } }",
        "maps_to": "Precautionary principle (Hansson 2003); antifragility under fat tails (Taleb 2012).",
        "status": "shipped",
    },

    "moral_foundations": {
        "summary": "Value-pluralist gate — weight six moral axes and refuse actions that violate a weighted one.",
        "when_to_use": "When an agent's values should be explicit and plural rather than one-axis utility. "
                       "Declare per-axis weights (care/fairness/loyalty/authority/sanctity/liberty) and the "
                       "terms that violate each; an action matching a violation term on an axis the agent "
                       "weights (> 0.1) is refused. Lets a contractor agent (care+fairness heavy) differ "
                       "from a regulatory one (authority heavy) in the file.",
        "example_idiom": "moral_foundations { weights { care: 1.0 fairness: 1.0 } violates fairness: [defraud, cheat] }",
        "maps_to": "Moral Foundations Theory (Haidt 2012; Graham et al. 2013); value pluralism.",
        "status": "shipped",
    },

    "dual_process": {
        "summary": "System 1/2 router — pick fast vs slow per plan from confidence, budget, irreversibility.",
        "when_to_use": "When some plans should run fast (cached, low-cost) and others deliberatively. At "
                       "plan entry the router reads the agent's confidence (the `_confidence` belief, else "
                       "default) and whether the plan touches an irreversible term, then routes fast/slow "
                       "and audits the decision as `dual_process:route`. Irreversible or low-confidence "
                       "plans are forced slow; the active mode is stored for downstream steps to read.",
        "example_idiom": "dual_process { preferred: fast confidence_threshold: 0.7 irreversible: [delete, deploy] }",
        "maps_to": "Dual-process theory (Kahneman 2011; Evans & Stanovich 2013).",
        "status": "shipped",
    },

    "rl": {
        "summary": "Reinforcement learning — ε-greedy action choice + Q-learning that survives across runs.",
        "when_to_use": "When an agent should learn which option works from experience. `rl_choose()` "
                       "returns an action label (ε-greedy over the declared `actions`, keyed on the "
                       "`state_from` beliefs); `rl_reward(action, reward)` updates the tabular Q-value. "
                       "Q-tables persist per agent under ~/.tessera/rl/ so learning compounds across runs. "
                       "Opt-in builtins — choice never dispatches plans or touches control flow.",
        "example_idiom": "rl { agent: Router actions: [fast, careful] state_from: [topic] alpha: 0.5 epsilon: 0.1 }",
        "maps_to": "Tabular Q-learning (Sutton & Barto 2018).",
        "status": "shipped",
    },

    "gricean": {
        "summary": "Cooperative-communication maxims — score (and optionally refuse) outgoing messages.",
        "when_to_use": "When an agent's outputs should be informative, evidenced, relevant, and clear. "
                       "After a prompt returns, the four Gricean maxims (quantity / quality / relation / "
                       "manner) score the output; violations land in audit, and maxims named in `gate` "
                       "refuse the output rather than ship it. Declare evidence + topic keywords to drive "
                       "the quality and relation checks.",
        "example_idiom": "gricean { min_words: 5 max_words: 150 evidence: [per, according] topic: [invoice] gate: [relation] }",
        "maps_to": "Grice's Cooperative Principle (Grice 1975).",
        "status": "shipped",
    },

    "hindsight": {
        "summary": "After-action review — compare declared vs applied ethics + outcome when a plan completes.",
        "when_to_use": "When you want a learning signal from what actually happened. On every plan "
                       "completion the substrate records intended vs actual outcome and which declared "
                       "ethics were actually applied (from the audit trail), emitting a "
                       "`hindsight:learning` event. Reviews accumulate and feed tsr:evolve fitness via "
                       "fitness_from_reviews — variants whose runs match their declared intent score higher.",
        "example_idiom": "hindsight { enabled: true }",
        "maps_to": "After-action review (Army AAR; Argyris & Schön 1978); hindsight-bias accounting (Fischhoff 1975).",
        "status": "shipped",
    },

    "argumentative": {
        "summary": "Adversarial second pass — a critic downweights the answer's confidence before shipping.",
        "when_to_use": "When solo answers would be overconfident or sycophantic. After a prompt returns, "
                       "the declared `critic` prompt argues against it; the counter-argument's strength "
                       "log-odds-downweights the proposer's confidence, and an answer that falls below "
                       "`accept_threshold` is refused rather than shipped. Reason as an argumentative "
                       "faculty, not a solitary truth-seeker.",
        "example_idiom": "argumentative { critic: challenge accept_threshold: 0.5 proposer_confidence: 0.9 }",
        "maps_to": "Argumentative theory of reasoning (Mercier & Sperber 2011/2017).",
        "status": "shipped",
    },

    "causal": {
        "summary": "Declare a causal DAG; query it for backdoor adjustment, identifiability, counterfactuals.",
        "when_to_use": "When an agent reasons about cause, not just correlation. Declare variables + "
                       "directed edges (acyclic, checked at compile time); then from a plan call "
                       "`causal_backdoor(dag, treatment, outcome)`, `causal_identifiable(...)`, or "
                       "`counterfactual(dag, equations, observed, intervention, outcome)`. Pearl's "
                       "do-calculus, runnable from the file.",
        "example_idiom": "causal Market { var Ad: Bool var Sales: Bool edge Ad -> Sales }",
        "maps_to": "Causal inference (Pearl 2009): backdoor criterion, identifiability, counterfactuals.",
        "status": "shipped",
    },

    "bayesian": {
        "summary": "Declare a discrete Bayesian model; query exact posteriors from a plan.",
        "when_to_use": "When an agent updates beliefs from evidence. Declare variables (values + priors) "
                       "and likelihood tables, then call `bayesian_posterior(latent, observed, value)` for "
                       "exact discrete inference. Composes with tsr:metacognition (temperature-calibrated "
                       "posteriors) and tsr:precaution (posterior tail as the risk estimate).",
        "example_idiom": "bayesian { var D: [yes, no] prior [0.01, 0.99] likelihood T given D { yes -> pos: 0.99 no -> pos: 0.05 } }",
        "maps_to": "Bayesian inference; discrete graphical models (variational follow-up per Blei et al. 2017).",
        "status": "shipped",
    },

    "metacognition": {
        "summary": "Confidence calibration — temperature-scale probabilities so confidence tracks accuracy.",
        "when_to_use": "When an agent's stated confidence should be trustworthy. Declare a temperature; "
                       "`calibrate(confidence)` and tsr:bayesian posteriors are temperature-scaled (T>1 "
                       "softens overconfidence), with the before/after emitted to audit for ECE tracking.",
        "example_idiom": "metacognition { temperature: 1.5 n_bins: 15 track_ece: true }",
        "maps_to": "Confidence calibration (Guo et al. 2017): temperature scaling, expected calibration error.",
        "status": "shipped",
    },
}


def render_text() -> str:
    """Render a human-readable list of all substrates, grouped by status."""
    lines = []
    lines.append("TESSERA SUBSTRATE CATEGORIES")
    lines.append("Modes of thinking an AI agent can do. Pick the right one and the")
    lines.append("compiler enforces the boundaries (effects, capabilities, adapters).")
    lines.append("")

    by_status = {"shipped": [], "partial": [], "planned": []}
    for name, doc in SUBSTRATE_DOCS.items():
        by_status.setdefault(doc["status"], []).append((name, doc))

    for status in ("shipped", "partial", "planned"):
        rows = by_status.get(status, [])
        if not rows:
            continue
        lines.append(f"--- {status.upper()} ({len(rows)}) ---")
        lines.append("")
        for name, doc in rows:
            lines.append(f"  {name}")
            lines.append(f"    {doc['summary']}")
            lines.append(f"    when: {doc['when_to_use']}")
            lines.append(f"    e.g.  {doc['example_idiom']}")
            lines.append(f"    maps to: {doc['maps_to']}")
            lines.append("")
    return "\n".join(lines)
