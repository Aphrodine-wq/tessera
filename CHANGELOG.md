# Changelog

## 2026-06-03 — Wire the orphan substrates: 13 cognitive substrates made real

The research wave (commits tagged `research 4.x / A / B / C / D`) had built 16
cognitive-science modules as tested standalone Python, but they were never
wired into the language pipeline — declaring them did nothing at runtime, and
the `tessera substrates` catalog, CITATIONS, and this changelog all disagreed
about what shipped. This batch closes that gap. The catalog goes from **16 → 29
shipped substrates**; tests from 242 → **277 green**.

### Phase 1 — partial substrates made real at runtime (commit `0cb95a7`, `dd41f83`)

Seven substrates already parsed + lowered but the interpreter ignored them. New
`interp/substrates.py` is the one place their runtime lives:

- **iit** — compute φ* over the belief/intention dependency graph at plan entry,
  emit `iit:phi`. (Fixed a latent `Op.BeliefWrite` bug — real op is `BeliefRevise`.)
- **welfare** — record markers (φ* from iit, bandwidth from `gwt:ignition`,
  ast_fidelity) and refuse after N consecutive sub-threshold cycles.
- **ast** — score introspection fidelity (the `_focus` belief vs the running plan);
  refuse below `min_fidelity`.
- **tom** — manipulation-refusal gate on prompt output (Sally-Anne false-belief
  check grounded in `tom_false` markers).
- **causal / bayesian / metacognition** — declaration blocks whose runtime is the
  Phase-4 callable/calibration path.

### Phase 2 — gate substrates (commit `c379e9e`)

**precaution**, **moral_foundations**, **dual_process** — all driven by the proven
`tsr:autonomy` term-matching mechanism. precaution + moral_foundations gate
before the model call; dual_process routes fast/slow at plan entry. Verify lints
E800 / E810 / E820.

### Phase 3 — post-hook substrates (commit `232f61b`)

**gricean** (maxim scoring), **argumentative** (critic-pass confidence downweight),
**hindsight** (after-action review feeding tsr:evolve fitness) — via a shared
`on_prompt_output` + `on_plan_exit` seam. Verify lints E900 / E910 / E920.

### Phase 4 — value-layer literals + reasoning-tool callables (commit `dd41f83`)

Added list `[..]` and record `{..}` literals to the expression grammar — the
first value-layer extension since v0.2. On top of them, five reasoning tools
become callable from a plan (no new block): `causal_backdoor`,
`causal_identifiable`, `counterfactual` (over a declared tsr:causal DAG),
`bayesian_posterior` (exact discrete inference, metacognition-calibrated),
`calibrate`, `abductive`, `analogy`.

### Error-code bands added

E200–E250 reserved for partial-substrate validation; E800 precaution, E810
dual_process, E820 moral_foundations, E900 gricean, E910 hindsight, E920
argumentative (E920 errors when the critic prompt is undefined).

### Still planned (genuinely not built)

`evolve`, `identity`, `predict`, `phenomenology` remain `planned`. `rl_loop`,
`active_inference`, `concept_formation`, `semantic_embedding` remain research
notes in CITATIONS, not substrates.

## 2026-05-28 — Synapse → local SQLite + v0.2 + governance/audit batch

Twelve commits across one architecture session + two execution batches.
Twenty high-stakes decisions logged in `~/Desktop/TheVault/700 Decisions/`;
the commits below ship the implementations that fit inside one focused
sitting.

### Architecture session output (decisions logged in TheVault)

Twenty decisions logged: per-block memory persistence, AEON-owns-semantic-
verification, concurrent-by-default runtime, runtime capability sandbox,
audit-as-queryable-SQLite, versioned syntax with auto-migration, strict
BDI only, Obsidian load-bearing, open-source-from-day-one (later parked
as INTENT — see below), two-tier capability declarations, both-layer
deadlock detection, constraint-logic policies, ship training for
`tsr:neural`, Rust runtime port (`tessera-rs`) deferred, scientific-
scaffolded predict/phenomenology substrates, audit-derived skill
promotion, governance composition with consistency proof, tiered audit
retention, first-party-only LLM providers.

### Decision deferred to post-FTW-MVP (2026-07-01 review)

- Open-source rollout (PyPI release, cross-platform CI, Diataxis docs,
  Rust port scaffolding) — INTENT, not commitment. See decision
  `2026-05-28 Tessera open-source release — intent, deferred to
  post-FTW-MVP.md`. Eight other decisions cascade-depend on this one;
  their urgency drops until the deferral lifts.

### Shipped commits

- `45f1122` **drop Synapse dependency, replace with local SQLite
  semantic store.** Rip `tessera/adapters/synapse/` (311 LOC), replace
  with `tessera/adapters/semantic/` — single-table fact store, no
  external service, env override via `TESSERA_SEMANTIC_DB`.
- `3dd1b12` **tessera v0.2 — versioned syntax + per-block persistence
  for memory:semantic.** Frontmatter `tessera_version` + migration
  framework (`tessera/migrations/`). Memory:semantic fences accept
  `persistent=true|false`.
- `5f018f9` **mirror full Twin trait set.** `BUILTIN_TRAITS` grows from
  6 to 10: adds imposter_recursion, spectrum_directness,
  anxiety_simulation, insomniac_focus with deterministic
  keyword/capability matchers.
- `e6d8a36` **audit-as-queryable-SQLite + tessera audit query CLI.**
  Every audit event persists to `~/.tessera/audit.db` (single-table v1);
  new `tessera audit query` subcommand filters by agent/intent/action/
  time.
- `20583d0` **migration_advisor example + built-in trait fallback proven
  end-to-end.** Tests prove built-in traits resolve by name without a
  local `tsr:traits` block.
- `909e512` **two-tier capability taxonomy + E600 unknown-subtype
  warning.** `tessera/capabilities.py` canonical taxonomy + verify pass
  warns on unknown subtypes (legacy-compat soft migration).
- `819bb47` **spawn auto-restrict — finishes decision 10 with
  caps_narrowed audit.** `Op.Spawn` no longer raises on unauthorized
  caps; intersects requested with parent's set, audits the delta.
- `d7c795e` **pass_7_spawn_cycle — static deadlock detection.** Verify
  pass builds a spawn graph and emits E700 DeadlockCertain for pure
  cycles.
- `d480e4b` **tiered audit retention — governance forever, operational
  rolling.** Single `audit.db` splits into `audit_governance.db` (proof
  trail, no purge) and `audit_operational.db` (rolling window, purgeable
  via `tessera audit purge`).
- `03ab1f7` **skill promote_to: neural plumbing.** Procedural skill
  declarations accept `promote_to: neural { threshold: N }`. Runtime
  emits one `skill_promotion_pending` audit event when the threshold
  trips. Actual training is a follow-up.

### Decisions still deferred (not in this batch)

- Concurrent-default flip (`TESSERA_CONCURRENT_AGENTS` defaults to 1)
- Dynamic per-recv timeout (decision 11 runtime half)
- Constraint-logic policies (decision 12) — needs constraint solver
- Runtime capability sandbox (decision 4) — lands in `tessera-rs`
- Actual neural training for promoted skills (rest of decision 16)
- Genetic evolve substrate (decision 17)
- Governance composition with consistency proof (decision 18)

### Test count

- Session start: 33 passing
- After Synapse rip-out: 67
- After v0.2: 69
- After full Twin trait set: 70
- After audit CLI: 72
- After migration_advisor: 73
- After two-tier taxonomy: 76
- After spawn auto-restrict + deadlock detection + tiered audit + skill
  promotion: **81 passing**.
