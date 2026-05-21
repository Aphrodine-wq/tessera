"""Substrate categories — modes of thinking, in plain English.

Every substrate is a CATEGORY of cognition an agent can do. Each entry has:

  - ``summary``: 1-line description in plain English
  - ``when_to_use``: a concrete prompt for the builder
  - ``example_idiom``: a typical 1-2 line snippet from a .tsr.md file
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
                       "'what we're doing now.' The winner becomes the next "
                       "draft of the agent's behavior.",
        "example_idiom": "workspace TeamMind { capacity: 1 arbiter: highest_salience }",
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
                       "across agents. Backed by Synapse (Block/Edge graph).",
        "example_idiom": "knowledge { schema FactSheet(title: String, domain: String) }",
        "maps_to": "Semantic memory; knowledge bases (Synapse backend)",
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
