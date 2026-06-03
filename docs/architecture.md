# Tessera — the cathedral view

This document explains what Tessera is, what each of its neighbor systems is,
and what the **forest** of them together is useful for. Read it once before
writing your first agent.

## In one paragraph

**Tessera** is a markdown-native language for building AI agents. You write
agents in `.t.md` files using a small set of substrate-tagged code fences
(`tsr:agent`, `tsr:prompt`, `tsr:tool`, `tsr:neural`, `tsr:memory:*`). The
compiler verifies them via **AEON** (73-engine formal verifier), persists
their knowledge in a local SQLite fact store, discovers them in your
**Obsidian vault**, and runs them through a tree-walking interpreter that
dispatches to **Ollama / Anthropic / LangChain / PyTorch** for LLM calls,
tools, and learned models. Each system also stands alone — but the point is
that **they feed each other**, and the agent you write is sharper for it.

---

## The forest — four systems, one ecosystem

```
                          ┌────────────────────────┐
                          │   Obsidian / TheVault  │  ← the soil
                          │   markdown notes       │
                          └────────────┬───────────┘
                                       │ .t.md
                                       ▼
       ┌──────────────────────────────────────────────────────┐
       │                  Tessera                             │  ← the body
       │   parser → SIR → verify → interpreter                │
       │   substrates: logic, agent, memory:*, prompt,        │
       │               tool, neural, policy, eval, ...        │
       └─┬───────────────────────┬───────────────────────────┬┘
         │                       │                           │
   verify│                   call│                      embed│
         ▼                       ▼                           ▼
   ┌─────────┐             ┌──────────┐                ┌─────────┐
   │  AEON   │             │ Ollama / │                │ PyTorch │
   │ 73 eng. │             │ Anthropic│                │   nn.*  │
   │ verify  │             │ LangChain│                │  models │
   └─────────┘             └──────────┘                └─────────┘
   immune system           external cortex             skill memory
```

| System | Role | Repo |
|---|---|---|
| **Tessera** | Body — executable agents written in markdown | `~/Projects/walt/tessera` |
| **AEON** | Immune system — formal verification (73 engines, 22 cybersecurity) | `~/Projects/walt/aeon` |
| **Obsidian / TheVault** | Soil — markdown vault where agents are written, scanned, and read | `~/Desktop/TheVault` |
| **Ollama / Anthropic** | External cortex — language model completions | system-installed |
| **LangChain** | Hands — tool library (web search, calculators, vector DBs, ...) | optional pip dep |
| **PyTorch** | Skill memory — learned/differentiable models | optional pip dep |

Semantic memory (`memory:semantic`) is self-contained: facts persist to a
local SQLite file at `~/.tessera/semantic.db`. No external service required.

---

## What each does alone

### Tessera alone
Compiles `.t.md` files into runnable agents with a typed effect system, a
capability-gated actor scheduler, and substrate-aware compilation targets
(symbolic bytecode, neural graph, local fact store, quantum-semantic plan).
Without the other systems: agents run, semantic facts persist to local SQLite,
but verification is shallow, LLM calls go nowhere (stubbed), and they have no
soil to grow in.

### AEON alone
Formally verifies code in 21 languages against 73 engines (substrate adjacency,
effect inference, capability check, taint analysis, refinement types, Hoare
logic, 22 cybersecurity engines). Catches injection, auth bugs, crypto misuse,
PII leaks, race conditions. Has been doing this for general code; v0.0.4 of
Tessera added a `.sir` (Substrate IR) language adapter so AEON now verifies
Tessera agents the same way it verifies Python or Rust.

### Obsidian / TheVault alone
A directory of markdown notes. Daily notes, project notes, Zettelkasten. With
Tessera: any `.t.md` file in any folder of the vault is a runnable agent.
`tessera vault scan ~/Desktop/TheVault` lists every agent in the vault with
its substrates, capabilities, and dependencies.

### LLMs (Ollama / Anthropic), LangChain, PyTorch
Each is a useful tool in isolation. Tessera makes them callable from inside
substrate-typed agent code with effect tracking, cost accumulation, and
graceful degradation when one isn't available.

---

## What each system alone CANNOT do

| System | Limitation alone |
|---|---|
| **Tessera** | Verification is local-only without AEON; agents can't talk to LLMs / tools / models without adapters |
| **AEON** | Operates on code, not on agent runtime; doesn't know what an `agent.intention` IS until you give it the language adapter |
| **Obsidian** | Just markdown — no compiler, no runtime, no verification |
| **LLM / LangChain / PyTorch** | Provider-specific; no shared effect model; no capability gating; no audit trail |

---

## The forest — what only the COMBINATION unlocks

### 1. Agents you can verify before running
Write an agent in markdown → Tessera lowers it to SIR → AEON's 73 engines verify
the SIR text → diagnostics map back to RFC §Appendix-C error codes
(E001 substrate adjacency, E102 capability not in scope, E301 PII to non-sanitized
egress, ...). **No other agent framework does formal verification before run.**

### 2. Knowledge that survives across runs
`memory:semantic` writes hit a local SQLite fact store keyed by schema name
and agent author. Facts persist between invocations; agents that share a
schema see each other's writes. The store is owned by Tessera and lives at
`~/.tessera/semantic.db` (override via `TESSERA_SEMANTIC_DB`). **No external
service to run, no vendor lock-in.**

### 3. Agents written where you already think
Obsidian's vault is where you already write daily notes, Zettelkasten, and
project plans. Drop a `.t.md` file anywhere in the vault and `tessera vault
scan` finds it. **There's no "agent management tool" — the vault IS the tool.**

### 4. Capability discipline across the whole stack
A Tessera agent declares `capabilities_requested: [NetworkOut]` in frontmatter.
The compiler propagates that to every region. The actor scheduler refuses
to spawn child agents with caps the parent doesn't hold. The PII flow analysis
(AEON) catches `Tainted<T, pii>` reaching `Tool.Invoke` without a sanitizer.
**One declaration; enforced at compile, spawn, and runtime.**

### 5. Substrate-typed thinking
Each substrate is a *named mode of cognition*. You don't write "an agent that
uses LLMs and has memory and learns things." You write `tsr:agent` for goal-
directed action, `tsr:memory:workspace` for the global blackboard, `tsr:prompt`
for LLM calls, `tsr:neural` for learned functions. The compiler enforces the
boundaries. The substrate doc (`tessera substrates`) reads like a menu of
ways of thinking. **The architecture is legible.**

---

## Three cross-system flows

### Flow A — Build, verify, ship

```
1. you write    ~/Desktop/TheVault/Agents/Reviewer.t.md
                       │
2. tessera vault scan ~/Desktop/TheVault
                       │
                       │  Reviewer  ←  Agents/Reviewer.t.md
                       │      substrates: agent, prompt, memory:episodic
                       ▼
3. tessera compile ~/Desktop/TheVault/Agents/Reviewer.t.md --aeon
                       │
                       │  ✔ AEON: 4 functions, 0 errors, 0 warnings
                       ▼
4. tessera vault run ~/Desktop/TheVault/Agents/Reviewer.t.md \
        --agent Reviewer --set diff="..."
                       │
                       ▼
              real Ollama / Anthropic call
                       │
                       ▼
              return value + episodic log
```

Every step is one shell command. The vault is the source of truth.

### Flow B — Agent learns, facts persist

```
agent KnowledgeAssistant {
  intentions:
    plan teach {
      remember FactSheet(title="...", domain="construction")
      ...
    }
}
                       │
                       ▼
            Tessera SM_Insert node
                       │
                       ▼
        semantic adapter (writes by default)
                       │
                       ▼
       ~/.tessera/semantic.db (SQLite INSERT)
                       │
                       ▼
       facts row: schema=FactSheet
                  fields_json=<json>
                  agent_id, plan_id, created_at
                       │
                       ▼
       next plan invocation: lookup FactSheet where domain == "construction"
                       │
                       ▼
       reads back rows, returns matched facts
```

Knowledge survives across runs. No external service, no daemon — one SQLite
file Tessera owns end to end.

### Flow C — Vault audit at scale

```
$ time tessera vault scan ~/Desktop/TheVault
    VAULT: /Users/.../TheVault
    Found 17 agent(s)
    ...
    real    0m0.482s

$ aeon scan ~/Desktop/TheVault --profile portfolio
    [routes .t.md and .sir through TesseraTranslator]
    [73 engines run against each]
    [outputs per-agent diagnostics]
```

AEON's `portfolio` profile gets a free language target. Your whole vault of
agents is audited by the same security engines that run on your Python,
Rust, and Java code.

---

## What is this useful for?

Concrete use cases the forest unlocks:

### For solo AI developers
- **Write an agent in 30 seconds.** `tessera vault new ~/Desktop/TheVault/Agents/X.t.md --agent X --template llm`. Drop into Obsidian, edit. Run.
- **Verify before you ship.** AEON catches capability bugs, PII leaks, missing substrate adapters — *before* the agent does damage in prod.
- **No knowledge backend to babysit.** Semantic facts live in a Tessera-owned SQLite file. Inspect with any `sqlite3` client; back up with `cp`.

### For teams building agent products
- **Substrate boundaries as code review primitives.** "This agent has both `tool` and `policy` substrates — let's audit what the policy enforces before merging." The substrate is explicit; the review question writes itself.
- **Capability audit at PR time.** AEON portfolio scan of the vault catches new capability grants in PR diffs.
- **Welfare-ready architecture.** When the question of agent welfare becomes operationally real, the `eval:welfare` substrate slot is reserved (PRD §5.23). The substrate vocabulary forces honest framing — "this agent integrates information across X regions with Φ ≥ Y" is checkable; "this agent is conscious" is not.

### For research / cognitive science
- **Operationalize a theory of cognition in markdown.** Pick a substrate, write the operationalization, evaluate against running agents. The substrate doc cites the theory each one maps to (GWT, BDI, predictive processing, ToM, IIT, RPT). Tessera doesn't endorse any; it gives them all first-class APIs.
- **Empirical cross-theory comparison.** Build two agents — one `@active_inference`, one `@react` — and compare on the same benchmark. Today the language can express both; tomorrow it'll verify both.

### For James specifically (the cofounder context)
- **FTW agent layer.** When FTW needs an agent that explains a quote to a homeowner, you write it in `.t.md` in the FTW repo, AEON verifies it can't egress PII, the semantic store keeps conversation history, Ollama runs the LLM call. No agent framework. No vendor lock-in.
- **ConstructionAI inference + Tessera orchestration.** ConstructionAI is the model; Tessera is the agent that DECIDES when to call the model. `tsr:neural model classifier { ... }` cohabits with the agent. The model's output drives the next plan step.
- **MHP customer service agent.** Same shape. Different vault folder.

---

## Why markdown-native? (The soil thesis)

Programming languages live in two places: a syntax file, and a developer's
head. The friction between them is documentation. Markdown solves this by
making the syntax file ALSO the documentation, ALSO the spec, ALSO the
runbook.

A `.t.md` file is simultaneously:
- A **valid Obsidian note** — browsable, linkable, indexable
- A **compilable program** — `tessera compile` produces SIR + executable
- A **fact-producing runtime** — `remember` calls persist to the local store
- An **audited safety boundary** — AEON verifies it like any source file
- A **trainable artifact** — `@trainable_agent` regions get gradient flow
- An **introspectable cognitive architecture** — substrate boundaries are
  declared in fenced blocks

You don't need a separate IDE, a separate docs site, a separate agent
management tool. The vault IS the source of truth. The substrate fences are
both code and documentation. The frontmatter is both metadata and config.

This is the same insight that made Obsidian's plugin ecosystem work: in a
vault, structure emerges from links between notes, not from a top-down
hierarchy. Tessera adds a second dimension — substrate fences inside notes —
that lets the same emergence happen at the *agent* level. Your vault becomes
a workshop for cognition.

---

## Where this is going

| Substrate | Status |
|---|---|
| `logic`, `agent`, `memory:*`, `prompt`, `tool`, `neural`, `policy`, `eval` | shipped |
| `traits`, `intent`, `ethics`, `autonomy` | shipped (governance) |
| `iit`, `welfare`, `ast`, `tom` | shipped (consciousness-adjacent; PHILOSOPHY.md) |
| `precaution`, `moral_foundations`, `dual_process` | shipped (gates) |
| `gricean`, `hindsight`, `argumentative` | shipped (post-hooks) |
| `causal`, `bayesian`, `metacognition` | shipped (+ reasoning callables: counterfactual, abductive, analogy, …) |
| `identity`, `evolve`, `predict`, `phenomenology` | planned — `predict`/`phenomenology` heavy ethics; see PRD §12 first |

| System integration | Status |
|---|---|
| AEON verifies SIR via `.sir` language adapter | shipped |
| Local SQLite-backed `memory:semantic` | shipped |
| Obsidian vault scan + scaffold | shipped |
| Ollama backend (default) + Anthropic | shipped |
| LangChain tool resolution | shipped |
| PyTorch `neural` substrate | shipped (inference only; training planned) |
| AEON `aeon portfolio` routes `.t.md` directly | next (this turn) |
| Parse + verify caching for fast vault scans | next (this turn) |

---

## Quick reference

```bash
# Build the substrate vocabulary into your head
tessera substrates

# Scan a vault for agents
tessera vault scan ~/Desktop/TheVault

# Scaffold a new agent in the vault
tessera vault new ~/Desktop/TheVault/Agents/Bar.t.md --agent Bar --template llm

# Run an agent
tessera vault run ~/Desktop/TheVault/Agents/Bar.t.md --agent Bar --set q="..."

# Verify with AEON
tessera compile examples/researcher.t.md --aeon

# Inspect persisted semantic facts (the agent's accumulated knowledge)
sqlite3 ~/.tessera/semantic.db "SELECT schema, fields_json FROM facts LIMIT 10;"

# Pick an LLM backend
TESSERA_LLM_BACKEND=ollama TESSERA_OLLAMA_MODEL=llama3.2 tessera compile ...
TESSERA_LLM_BACKEND=anthropic tessera compile ...
TESSERA_LLM_BACKEND=noop tessera compile ...    # deterministic stub
```

This is the forest. Plant something in it.
