# Tessera

**Markdown-native programming language for AI agents.** Write agents in
`.t.md` files; the compiler verifies them via AEON, persists their knowledge
to a local SQLite fact store, discovers them in your Obsidian vault, and
runs them through real LLM / LangChain / PyTorch backends.

Read [`docs/architecture.md`](docs/architecture.md) for the cathedral view of
how it all fits together. This README is the quick start.

---

## Hello, agent

```bash
cd ~/Projects/walt/tessera
python3 -m venv .venv && .venv/bin/pip install -e .

# Run the simplest agent
.venv/bin/tessera compile examples/hello.t.md --run HelloAgent --set target=world
# → HelloAgent() = 'hello world'
```

The agent itself, in full:

```markdown
---
agent: HelloAgent
capabilities_requested: []
---

```tsr:logic
fn greet(name: String) -> String = "hello " + name
\```

```tsr:agent
agent HelloAgent {
  beliefs:
    @last_write target: String
  intentions:
    plan say_hello {
      let msg = greet(target)
      return msg
    }
}
\```
```

That's a Tessera program. Markdown all the way down.

---

## What's in the box

```
tessera/
├── parser/        .t.md → ParsedModule (frontmatter + substrate blocks)
├── sir/           SIR node taxonomy, region builder, textual serializer
├── verify/        local verification passes + diagnostics
├── interp/        tree-walking evaluator, actor scheduler, workspace runtime
├── substrate_docs.py    English descriptions of every substrate category
├── cli.py         tessera compile / vault / substrates commands
└── adapters/
    ├── aeon/      → AEON (formal verification, 73 engines)
    ├── semantic/  → local SQLite fact store for memory:semantic
    ├── obsidian/  → vault scan + scaffold
    ├── llm/       → Ollama (default) + Anthropic + noop
    ├── langchain/ → import any LangChain Tool by dotted path
    └── torch/     → tsr:neural blocks compile to nn.Sequential
```

---

## Substrates — modes of thinking

Each substrate is a typed code fence inside a `.t.md` file. The compiler
enforces boundaries between them (substrate adjacency, effect propagation,
capability gating).

```bash
tessera substrates    # prints English breakdown of all 16 categories
```

| Substrate | What it is | Example |
|---|---|---|
| `logic` | pure functions | `fn rank(papers, query) -> Float = ...` |
| `agent` | BDI actors with beliefs + plans | `agent FooBot { beliefs: ... }` |
| `memory:working` | per-invocation scratchpad | `let x = compute(input)` |
| `memory:workspace` | global blackboard, arbiter picks winner | `workspace TeamMind { arbiter: highest_salience }` |
| `memory:episodic` | append-only event log | `log Decision(topic, choice)` |
| `memory:semantic` | knowledge graph (local fact store) | `remember FactSheet(title=..., domain=...)` |
| `prompt` | LLM template + bindings | `prompt summarize(t) -> String = "..."` |
| `tool` | external callable (python or LangChain) | `tool web_search(q) from langchain_community.tools.DuckDuckGoSearchRun` |
| `neural` | torch nn.Module declared inline | `model classifier { linear in=4 out=8; relu; ... }` |

Eight more are planned (`policy`, `eval`, `evolve`, `identity`, `predict`,
`phenomenology`, `memory:procedural`). Run `tessera substrates` to see the
full menu.

---

## Examples

Every file in `examples/` is a runnable agent. Each demonstrates a different
slice of the language.

| Example | Demonstrates |
|---|---|
| `hello.t.md` | minimum: logic + agent, working memory |
| `researcher.t.md` | multi-agent (Researcher, Critic, TeamLead), workspace broadcast, spawn/send/recv |
| `researcher_full.t.md` | until-loops, notice handlers, comparison operators |
| `research_assistant.t.md` | prompt + tool + LangChain bridge |
| `perception.t.md` | PyTorch `neural` substrate |
| `vault_assistant.t.md` | `memory:episodic` event log |
| `knowledge_assistant.t.md` | `memory:semantic` round-trip into local SQLite |
| `migration_advisor.t.md` | built-in traits + ethics + audit (full governance stack) |

---

## Quick CLI tour

```bash
# Scan an Obsidian vault for agents
tessera vault scan ~/Desktop/TheVault

# Scaffold a new agent in the vault
tessera vault new ~/Desktop/TheVault/Agents/NewBot.t.md \
    --agent NewBot --template llm

# Compile + verify with AEON + run
tessera compile examples/researcher.t.md --aeon \
    --run TeamLead --set topic="fair pricing"

# Query the persistent audit store — provenance for any past run
tessera audit query --agent MigrationAdvisor --intent advise_safely --count
tessera audit query --action skill_promotion_pending
tessera audit purge --days 30   # operational only; governance untouched

# Pick an LLM backend
TESSERA_LLM_BACKEND=ollama  TESSERA_OLLAMA_MODEL=glm-4.6:cloud tessera compile ...
TESSERA_LLM_BACKEND=anthropic tessera compile ...   # uses ANTHROPIC_API_KEY
TESSERA_LLM_BACKEND=noop tessera compile ...        # deterministic stub
```

### v0.2 syntax

Files declare their language version in frontmatter:

```yaml
---
agent: MyAgent
tessera_version: 0.2
---
```

Files without a version are auto-migrated forward at parse time (default
`0.1`). Migrations live in `tessera/migrations/` and run before SIR
lowering — author files stay valid as the language evolves.

`memory:semantic` blocks opt in per-block to disk persistence:

```markdown
\`\`\`tsr:memory:semantic persistent=true
knowledge { schema FactSheet(title: String, domain: String) }
\`\`\`
```

`persistent=true` writes to `~/.tessera/semantic.db` (override via
`TESSERA_SEMANTIC_DB`). `persistent=false` keeps facts in a per-World
shadow that lives only for the run.

---

## Status

- **Tests:** 277 passing
- **Shipped substrates (29):** logic, agent, memory:working, memory:workspace,
  memory:episodic, memory:semantic, memory:procedural, prompt, tool, neural,
  traits, intent, ethics, autonomy, policy, eval, iit, welfare, ast, tom,
  precaution, moral_foundations, dual_process, gricean, hindsight,
  argumentative, causal, bayesian, metacognition. Run `tessera substrates`.
- **Reasoning-tool callables (from a plan, no new block):** `causal_backdoor`,
  `causal_identifiable`, `counterfactual` (over a declared `tsr:causal` DAG),
  `bayesian_posterior` (exact discrete inference), `calibrate`, `abductive`,
  `analogy` — enabled by list `[..]` and record `{..}` value-layer literals.
- **Built-in cognitive traits (10):** doubt_first, cross_brain, compulsive,
  hypervigilant, synesthetic, manic_burst, imposter_recursion,
  spectrum_directness, anxiety_simulation, insomniac_focus.
- **Persistent stores (Tessera-owned, no external service):**
  `~/.tessera/semantic.db` for `memory:semantic` facts;
  `~/.tessera/audit_governance.db` (forever) + `~/.tessera/audit_operational.db`
  (30-day rolling) for the audit / provenance graph.
- **External integrations:** AEON (verify), Obsidian (vault scan + scaffold),
  Ollama, Anthropic, LangChain, PyTorch.

Pre-alpha. Use at your own risk. PRs welcome — the design lives in
`docs/Tessera_PRD_v0.5.md` and the SIR spec in `docs/TESSERA-RFC-001-SIR.md`.

---

## Why does this exist?

Because writing an AI agent shouldn't require five frameworks, three vendor
SDKs, and a vector DB you have to babysit. It should require markdown and a
compiler that takes the substrate boundaries seriously.

The full thesis is in [`docs/architecture.md`](docs/architecture.md).
