# Tessera — the cathedral view

This document explains what Tessera is, what each of its neighbor systems is,
and what the **forest** of them together is useful for. Read it once before
writing your first agent.

## In one paragraph

**Tessera** is a markdown-native language for building AI agents. You write
agents in `.tsr.md` files using a small set of substrate-tagged code fences
(`tsr:agent`, `tsr:prompt`, `tsr:tool`, `tsr:neural`, `tsr:memory:*`). The
compiler verifies them via **AEON** (73-engine formal verifier), persists
their knowledge via **Synapse** (block/edge graph with synaptic weighting),
discovers them in your **Obsidian vault**, and runs them through a tree-walking
interpreter that dispatches to **Ollama / Anthropic / LangChain / PyTorch** for
LLM calls, tools, and learned models. Each system also stands alone — but the
point is that **they feed each other**, and the agent you write is sharper
for it.

---

## The forest — five systems, one ecosystem

```
                          ┌────────────────────────┐
                          │   Obsidian / TheVault  │  ← the soil
                          │   markdown notes       │
                          └────────────┬───────────┘
                                       │ .tsr.md
                                       ▼
       ┌──────────────────────────────────────────────────────┐
       │                  Tessera                             │  ← the body
       │   parser → SIR → verify → interpreter                │
       │   substrates: logic, agent, memory:*, prompt,        │
       │               tool, neural, policy, eval, ...        │
       └─┬───────────────┬────────────────┬─────────────────┬─┘
         │               │                │                 │
   verify│       persist │            call│            embed│
         ▼               ▼                ▼                 ▼
   ┌─────────┐     ┌──────────┐     ┌──────────┐      ┌─────────┐
   │  AEON   │     │ Synapse  │     │ Ollama / │      │ PyTorch │
   │ 73 eng. │     │ Block/   │     │ Anthropic│      │   nn.*  │
   │ verify  │     │ Edge KG  │     │ LangChain│      │  models │
   └─────────┘     └──────────┘     └──────────┘      └─────────┘
   immune system    the brain       external cortex    skill memory
```

| System | Role | Repo |
|---|---|---|
| **Tessera** | Body — executable agents written in markdown | `~/Projects/walt/tessera` |
| **AEON** | Immune system — formal verification (73 engines, 22 cybersecurity) | `~/Projects/walt/aeon` |
| **Synapse** | Brain — knowledge graph with synaptic weighting + dream consolidation | `~/Projects/synapse` |
| **Obsidian / TheVault** | Soil — markdown vault where agents are written, scanned, and read | `~/Desktop/TheVault` |
| **Ollama / Anthropic** | External cortex — language model completions | system-installed |
| **LangChain** | Hands — tool library (web search, calculators, vector DBs, ...) | optional pip dep |
| **PyTorch** | Skill memory — learned/differentiable models | optional pip dep |

---

## What each does alone

### Tessera alone
Compiles `.tsr.md` files into runnable agents with a typed effect system, a
capability-gated actor scheduler, and substrate-aware compilation targets
(symbolic bytecode, neural graph, knowledge vault, quantum-semantic plan).
Without the other systems: agents run, but their verification is shallow,
their memory is in-process only, their LLM calls go nowhere (stubbed), and
they have no soil to grow in.

### AEON alone
Formally verifies code in 21 languages against 73 engines (substrate adjacency,
effect inference, capability check, taint analysis, refinement types, Hoare
logic, 22 cybersecurity engines). Catches injection, auth bugs, crypto misuse,
PII leaks, race conditions. Has been doing this for general code; v0.0.4 of
Tessera added a `.sir` (Substrate IR) language adapter so AEON now verifies
Tessera agents the same way it verifies Python or Rust.

### Synapse alone
Stores knowledge as Block/Edge graphs with synaptic weighting (BFS spreading
activation, Ebbinghaus forgetting curve, DBSCAN dream consolidation, HLC + CRDT
sync). Powers a SwiftUI canvas. Has an MCP server with 7 tools so agents can
read/write the graph. Without Tessera: it's a personal knowledge base. With
Tessera: the same graph becomes agent semantic memory.

### Obsidian / TheVault alone
A directory of markdown notes. Daily notes, project notes, Zettelkasten. With
Tessera: any `.tsr.md` file in any folder of the vault is a runnable agent.
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
| **Tessera** | Verification is local-only; memory is in-process; agents can't talk to LLMs / tools / models without adapters |
| **AEON** | Operates on code, not on agent runtime; doesn't know what an `agent.intention` IS until you give it the language adapter |
| **Synapse** | Knows nothing about agent declarations; can't run an agent or check its safety |
| **Obsidian** | Just markdown — no compiler, no runtime, no verification |
| **LLM / LangChain / PyTorch** | Provider-specific; no shared effect model; no capability gating; no audit trail |

---

## The forest — what only the COMBINATION unlocks

### 1. Agents you can verify before running
Write an agent in markdown → Tessera lowers it to SIR → AEON's 73 engines verify
the SIR text → diagnostics map back to RFC §Appendix-C error codes
(E001 substrate adjacency, E102 capability not in scope, E301 PII to non-sanitized
egress, ...). **No other agent framework does formal verification before run.**

### 2. Knowledge that survives and connects
`memory:semantic` writes hit Synapse blocks tagged with the schema name and
agent author. The same graph is browsable in Synapse's SwiftUI canvas with
activation spreading and dream consolidation. **Your agent's knowledge and
your personal knowledge live in the same brain.**

### 3. Agents written where you already think
Obsidian's vault is where you already write daily notes, Zettelkasten, and
project plans. Drop a `.tsr.md` file anywhere in the vault and `tessera vault
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
1. you write    ~/Desktop/TheVault/Agents/Reviewer.tsr.md
                       │
2. tessera vault scan ~/Desktop/TheVault
                       │
                       │  Reviewer  ←  Agents/Reviewer.tsr.md
                       │      substrates: agent, prompt, memory:episodic
                       ▼
3. tessera compile ~/Desktop/TheVault/Agents/Reviewer.tsr.md --aeon
                       │
                       │  ✔ AEON: 4 functions, 0 errors, 0 warnings
                       ▼
4. tessera vault run ~/Desktop/TheVault/Agents/Reviewer.tsr.md \
        --agent Reviewer --set diff="..."
                       │
                       ▼
              real Ollama / Anthropic call
                       │
                       ▼
              return value + episodic log
```

Every step is one shell command. The vault is the source of truth.

### Flow B — Agent learns, brain remembers

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
       Synapse adapter (dry-run by default)
                       │  TESSERA_ALLOW_REAL_VAULT=1
                       ▼
       Synapse vault.sqlite (real GRDB write)
                       │
                       ▼
       Block: author=tessera-compiler
              tags=[tessera, knowledge, FactSheet]
              content=<json>
                       │
                       ▼
        Synapse canvas visualizes it
        Activation spreading propagates
        Dream processor consolidates with related blocks
```

The agent's knowledge is YOUR knowledge — visible, browsable, weighted.

### Flow C — Vault audit at scale

```
$ time tessera vault scan ~/Desktop/TheVault
    VAULT: /Users/.../TheVault
    Found 17 agent(s)
    ...
    real    0m0.482s

$ aeon scan ~/Desktop/TheVault --profile portfolio
    [routes .tsr.md and .sir through TesseraTranslator]
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
- **Write an agent in 30 seconds.** `tessera vault new ~/Desktop/TheVault/Agents/X.tsr.md --agent X --template llm`. Drop into Obsidian, edit. Run.
- **Verify before you ship.** AEON catches capability bugs, PII leaks, missing substrate adapters — *before* the agent does damage in prod.
- **One brain for everything.** Your agents' knowledge and your personal Zettelkasten share Synapse. Cross-pollination is free.

### For teams building agent products
- **Substrate boundaries as code review primitives.** "This agent has both `tool` and `policy` substrates — let's audit what the policy enforces before merging." The substrate is explicit; the review question writes itself.
- **Capability audit at PR time.** AEON portfolio scan of the vault catches new capability grants in PR diffs.
- **Welfare-ready architecture.** When the question of agent welfare becomes operationally real, the `eval:welfare` substrate slot is reserved (PRD §5.23). The substrate vocabulary forces honest framing — "this agent integrates information across X regions with Φ ≥ Y" is checkable; "this agent is conscious" is not.

### For research / cognitive science
- **Operationalize a theory of cognition in markdown.** Pick a substrate, write the operationalization, evaluate against running agents. The substrate doc cites the theory each one maps to (GWT, BDI, predictive processing, ToM, IIT, RPT). Tessera doesn't endorse any; it gives them all first-class APIs.
- **Empirical cross-theory comparison.** Build two agents — one `@active_inference`, one `@react` — and compare on the same benchmark. Today the language can express both; tomorrow it'll verify both.

### For James specifically (the cofounder context)
- **FTW agent layer.** When FTW needs an agent that explains a quote to a homeowner, you write it in `.tsr.md` in the FTW repo, AEON verifies it can't egress PII, Synapse stores its conversation history, Ollama runs the LLM call. No agent framework. No vendor lock-in.
- **ConstructionAI inference + Tessera orchestration.** ConstructionAI is the model; Tessera is the agent that DECIDES when to call the model. `tsr:neural model classifier { ... }` cohabits with the agent. The model's output drives the next plan step.
- **MHP customer service agent.** Same shape. Different vault folder.

---

## Why markdown-native? (The soil thesis)

Programming languages live in two places: a syntax file, and a developer's
head. The friction between them is documentation. Markdown solves this by
making the syntax file ALSO the documentation, ALSO the spec, ALSO the
runbook.

A `.tsr.md` file is simultaneously:
- A **valid Obsidian note** — browsable, linkable, indexable
- A **compilable program** — `tessera compile` produces SIR + executable
- A **regenerated knowledge vault** — Κ target writes back to Synapse
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
| `logic`, `agent`, `memory:working`, `memory:workspace`, `memory:episodic`, `memory:semantic`, `prompt`, `tool`, `neural` | shipped |
| `policy`, `eval` | next (this turn) |
| `memory:procedural`, `identity`, `evolve` | planned |
| `predict`, `phenomenology` | planned — heavy ethics; see PRD §12 first |

| System integration | Status |
|---|---|
| AEON verifies SIR via `.sir` language adapter | shipped |
| Synapse-backed `memory:semantic` | shipped |
| Obsidian vault scan + scaffold | shipped |
| Ollama backend (default) + Anthropic | shipped |
| LangChain tool resolution | shipped |
| PyTorch `neural` substrate | shipped (inference only; training planned) |
| AEON `aeon portfolio` routes `.tsr.md` directly | next (this turn) |
| Parse + verify caching for fast vault scans | next (this turn) |

---

## Quick reference

```bash
# Build the substrate vocabulary into your head
tessera substrates

# Scan a vault for agents
tessera vault scan ~/Desktop/TheVault

# Scaffold a new agent in the vault
tessera vault new ~/Desktop/TheVault/Agents/Bar.tsr.md --agent Bar --template llm

# Run an agent
tessera vault run ~/Desktop/TheVault/Agents/Bar.tsr.md --agent Bar --set q="..."

# Verify with AEON
tessera compile examples/researcher.tsr.md --aeon

# Persist to Synapse (dry-run by default; needs TESSERA_ALLOW_REAL_VAULT=1)
tessera compile examples/researcher.tsr.md --synapse-write

# Pick an LLM backend
TESSERA_LLM_BACKEND=ollama TESSERA_OLLAMA_MODEL=llama3.2 tessera compile ...
TESSERA_LLM_BACKEND=anthropic tessera compile ...
TESSERA_LLM_BACKEND=noop tessera compile ...    # deterministic stub
```

This is the forest. Plant something in it.
