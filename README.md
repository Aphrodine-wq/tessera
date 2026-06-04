# Tessera

**Markdown-native programming language for AI agents.** Write agents in
`.t.md` files; the compiler statically verifies them, persists their knowledge
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
├── cli.py         tessera compile / vault / substrates / audit / facts commands
└── adapters/
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
tessera substrates    # prints English breakdown of every substrate category
```

| Substrate | What it is | Example |
|---|---|---|
| `logic` | pure functions | `fn rank(papers, query) -> Float = ...` |
| `agent` | BDI actors with beliefs + plans | `agent FooBot { beliefs: ... }` |
| `memory:working` | per-invocation scratchpad | `let x = compute(input)` |
| `memory:workspace` | global blackboard; contenders pool, an arbiter picks the winner on read | `workspace TeamMind { arbiter: quorum(2) }` |
| `memory:episodic` | append-only event log | `log Decision(topic, choice)` |
| `memory:semantic` | knowledge graph (local fact store) | `remember FactSheet(title=..., domain=...)` |
| `prompt` | LLM template + bindings | `prompt summarize(t) -> String = "..."` |
| `tool` | external callable (python or LangChain) | `tool web_search(q) from langchain_community.tools.DuckDuckGoSearchRun` |
| `neural` | torch nn.Module declared inline | `model classifier { linear in=4 out=8; relu; ... }` |
| `rl` | ε-greedy choice + Q-learning that persists across runs | `rl { agent: Router actions: [fast, careful] state_from: [topic] }` |

A few more are planned (`evolve`, `identity`, `predict`, `phenomenology`). Run
`tessera substrates` to see the full menu.

### Runtime intelligence (v0.1.0)

Agents get smarter and faster without any extra wiring:

- **Auto-recall** — an agent with a `memory:semantic` / `memory:episodic` block
  has its relevant facts + recent events injected into every prompt
  automatically (ranked by keyword overlap). No manual `lookup` needed.
  Opt-out with `TESSERA_NO_AUTO_RECALL=1`.
- **Auto-confidence routing** — `bayesian_posterior` / `abductive` set the
  agent's `_confidence`, so `tsr:dual_process` routes low-confidence plans to
  the slow path on its own; slow-path prompts deliberate and skip the cache.
- **Faster prompts** — the semantic prompt cache has an exact-hash fast path
  and loads once per process instead of re-scanning the JSONL every call
  (≈280× faster warm lookups; see `tests/test_bench.py`).

### Orchestration — one currency, everywhere

Multi-agent coordination runs on a single scalar — **salience/priority in
`[0,1]`, higher wins** — so the whole layer reads as one idea
(`tessera/interp/scheduling.py`). The same number answers all three scheduling
questions, and every decision lands in the audit trail:

- **Which plan runs first?** `plan decide priority=0.9 { … }` — plans execute
  highest-priority first (stable, so unannotated agents are unchanged).
- **Which draft wins the blackboard?** Broadcasts *pool* across a round and an
  arbiter resolves them on read: `highest_salience`, `weighted_vote`
  (agreement compounds), or `quorum(N)` (resolves only once N agents agree,
  else abstains).
- **What happens when a child fails?** `spawn Critic supervise=retry(2)` —
  a child that raises or refuses is re-driven before its `Refusal` propagates,
  so one bad actor can't crash the run.

Plus **fan-out/gather**: `send` a child several messages, then `recv all from X`
to collect every reply at once. See [`examples/orchestration.t.md`](examples/orchestration.t.md)
for all of it in one agent.

---

## Examples

Every file in `examples/` is a runnable agent. Each demonstrates a different
slice of the language.

| Example | Demonstrates |
|---|---|
| `hello.t.md` | minimum: logic + agent, working memory |
| `researcher.t.md` | multi-agent (Researcher, Critic, TeamLead), workspace broadcast, spawn/send/recv |
| `orchestration.t.md` | the orchestration spine — plan `priority`, `quorum` arbiter, `recv all`, `supervise=retry` |
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

# Compile + verify + run
tessera compile examples/researcher.t.md \
    --run TeamLead --set topic="fair pricing"

# Query the persistent audit store — provenance for any past run
tessera audit query --agent MigrationAdvisor --intent advise_safely --count
tessera audit query --action skill_promotion_pending
tessera audit purge --days 30   # operational only; governance untouched

# Inspect / clean the memory:semantic fact store (~/.tessera/semantic.db)
tessera facts list                    # schema breakdown when unfiltered
tessera facts search retainage        # substring match across fields
tessera facts clear --schema note     # needs a filter (or --all) to delete

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

- **Tests:** 332 passing, 8 skipped (the skips need the optional `tson`
  package for constrained decoding — see `tessera/adapters/wire/`).
- **Shipped substrates (30):** logic, agent, memory:working, memory:workspace,
  memory:episodic, memory:semantic, memory:procedural, prompt, tool, neural,
  traits, intent, ethics, autonomy, policy, eval, iit, welfare, ast, tom,
  precaution, moral_foundations, dual_process, gricean, hindsight,
  argumentative, causal, bayesian, metacognition, rl. Run `tessera substrates`.
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
- **External integrations:** Obsidian (vault scan + scaffold),
  Ollama, Anthropic, LangChain, PyTorch.

Pre-alpha. Use at your own risk. PRs welcome — the design lives in
[`docs/architecture.md`](docs/architecture.md), the substrate philosophy in
[`PHILOSOPHY.md`](PHILOSOPHY.md), and the theory each substrate maps to in
[`CITATIONS.md`](CITATIONS.md).

---

## Why does this exist?

Because writing an AI agent shouldn't require five frameworks, three vendor
SDKs, and a vector DB you have to babysit. It should require markdown and a
compiler that takes the substrate boundaries seriously.

The full thesis is in [`docs/architecture.md`](docs/architecture.md).
