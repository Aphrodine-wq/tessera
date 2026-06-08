# Changelog

## 2026-06-04 — Contract hardening: inspection tooling, stress + property tests

Followed the runtime-contracts ship (below) with coverage, stress, and ways to
*see* contracts working. Tests **353 → 393 green**.

- **`tessera contracts <file>`.** Lists every contract — target, before/after
  clauses, on_violation — and cross-references the audit graph for live
  fire/refuse/retry/error counts. The contract-specific inspection surface.
- **`tessera doctor <file>`.** One-stop health for any `.t.md`: verify error/
  warning summary, eval results, substrate + contract inventory, recent-audit
  breakdown. Exits non-zero on verify errors or eval failures.
- **`tessera eval` now sees contract refusals.** `expect_refusal` matched only
  first-class `Refusal` values, so prompt-contract refusals (a `[contract-
  refused: …]` string) slipped through. A shared `_is_refusal` helper now
  recognizes both — which also fixes the same blind spot for precaution/tom/
  approval string gates.
- **E833: unknown-predicate lint.** A bareword like `holds(NetworkOut)` parses
  as a zero-arg predicate `NetworkOut` and fails-closed at runtime (silently
  refusing everything). The contract pass now flags it at compile time and
  suggests `holds("NetworkOut")`.
- **Contract refusals are governance-tier.** `contract:refuse` / `contract:error`
  now route to the permanent governance audit store like every other refusal
  (they were landing in the 30-day operational store). `contract:retry` /
  `contract:audit` stay operational (routine churn).
- **Stress + property tests.** A 12-agent concurrent storm through one contract,
  supervision re-driving a contract `Refusal`, a 50-deep retry budget, 100-event
  volume, plus hypothesis property tests (on_violation round-trips, random
  contracts always compile/verify, `intent_match()` bounded). Declared
  `hypothesis` in dev deps.

### Known limitation (concurrency)

Surfaced by the stress tests, **not introduced by contracts** — these are
pre-existing and affect every substrate:

- `World.record` increments `_audit_seq` without a lock, so under concurrent
  agents seq numbers may collide/skip and audit *ordering* is best-effort.
  `list.append` is GIL-atomic, so event *counts* stay exact.
- `semantic_cache_put` updates `_SEM_MEM` + appends JSONL without a lock; under
  concurrency a put can be lost or interleaved.

Neither affects whether a contract *fires* (enforcement runs on whatever text is
produced). The fix is a `Lock` in both spots — deferred as a focused follow-up.

## 2026-06-04 — Runtime contracts: author-declarable guarantees, enforced as it runs

Tessera's verification was static-only — the `verify/` passes check substrate
and capability boundaries before a run, but couldn't say anything about an LLM's
*actual* behavior. The hardcoded runtime gates (precaution, moral_foundations,
welfare, ast, tom) proved the hooks existed; they just weren't author-reachable.
The new `tsr:contract` substrate generalizes them into a first-class guarantee.
Tests **332 → 353 green**.

- **`tsr:contract` substrate.** `contract C on prompt:X { before: … after: … on_violation: … }`
  binds before/after assertions to a named effect — `prompt:X`, `tool:Y`, or
  `plan:Z`. A clause is the **inverse of `tsr:policy`**: where a policy `forbid
  when <expr>` refuses on *true*, a contract clause is a guarantee that must
  *hold* — false is the violation. Clauses are `policy_lang` expressions, so the
  whole predicate set (`contains_pii`, `holds`, `extracts`, `action_class`, …)
  is reused as-is.
- **Output-vs-intent drift.** New `intent_match()` predicate returns lexical
  overlap in `[0,1]` between an action's result and the active intent, so an
  `after: intent_match() >= 0.3` clause catches an LLM that wandered off the
  plan it was serving. Deterministic and dependency-free (upgrades to embedding
  cosine when sentence-transformers is present).
- **`on_violation` policy.** `refuse` (block + audit), `audit` (record + let it
  stand), or `retry(N) then refuse|audit` — an `after` violation re-drives the
  effect up to N times before falling back. `before` clauses can't regenerate
  inputs, so a retry there degrades to refuse (verify warns, E831).
- **Enforced at the real boundaries.** `before`/`after` plug into the existing
  hook sites (`on_prompt_input`/`on_prompt_output`, the tool-call chokepoint,
  `on_plan_enter`/`on_plan_exit`). Every check is audited — `contract:refuse`,
  `contract:retry`, `contract:audit`.
- **Verify pass `pass_17_contracts`.** E830 (target effect doesn't exist —
  the contract can never fire), E831 (retry on a `before`-only contract is
  inert), E832 (no clauses). New example `examples/contracts.t.md` and
  `tests/test_contracts.py` (13 tests).

## 2026-06-04 — Orchestration spine: one currency, five connected moves

Multi-agent orchestration now reads as a single idea — **salience/priority is
one scalar in `[0,1]`, and higher wins everywhere**. Five previously-missing or
inert capabilities were added and routed through one place
(`tessera/interp/scheduling.py`). Tests **320 → 332 green**.

- **Plan priority is live.** `plan Name priority=0.8 { ... }` now orders plan
  execution highest-first (a stable sort, so unannotated agents are unchanged).
  The priority that was declared on plans but never read at runtime now is, and
  it's recorded on the `plan_enter` audit event.
- **Blackboard accumulates.** `memory:workspace` contenders pool across a round
  and are consumed on read, instead of every broadcast immediately arbitrating
  and clearing. This is what lets multi-contender arbiters see the whole field.
  Broadcasting still keeps a live winner so the GWT ignition peek is unchanged.
- **New arbiters.** Beyond `highest_salience` / `last_write`: `weighted_vote`
  (agreement compounds — many quiet votes beat one loud outlier) and
  `quorum(N)` (resolves only once N contenders agree, else *abstains* and the
  board keeps its prior winner). One registry entry per strategy in `scheduling.py`.
- **Multi-message recv.** `send` binds each message to its own future, so a
  parent can `send` several times then `recv` each reply — or `recv all from X`
  to gather every owed reply into a list. Fan-out/gather composes instead of
  stranding messages in a single slot.
- **Supervision.** `spawn X supervise=retry(N)` re-drives a child that raises or
  returns a `Refusal` up to N times before the refusal propagates; a raising
  child becomes a `Refusal` instead of crashing the orchestrator. Retries and
  exhaustion are audited (`supervised_retry`, `supervised_exhausted`).
- **New example + tests.** `examples/orchestration.t.md` exercises all five
  together (a review panel reaching quorum); `tests/test_orchestration.py` adds
  12 tests.

## 2026-06-04 — Verify goes local-only; docs reconciled to reality

- **AEON integration removed.** Verification is now a first-party, local-only
  pass system (`tessera/verify/`); there is no external verifier and no
  `--aeon` flag. The earlier `AEON-owns-semantic-verification` decision (logged
  2026-05-28, below) is superseded. `docs/architecture.md` rewritten to match.
- **`tson` tests skip cleanly when the package is absent.** The `wire`
  (constrained-decoding) adapter degrades gracefully without the standalone
  `tson` package, but `tests/test_wire_*` raised instead of skipping. They now
  guard on `wire.AVAILABLE` (`pytest.mark.skipif`), so a fresh clone is green:
  **320 passing, 8 skipped** (the skips need `tson`).
- **Doc drift fixed.** `pyproject` version `0.0.1` → `0.1.0` (matches README /
  CHANGELOG); README test count and shipped-substrate count (now 30, incl.
  `rl`) corrected; dangling `docs/Tessera_PRD_v0.5.md` / `TESSERA-RFC-001-SIR.md`
  links repointed to `docs/architecture.md`, `PHILOSOPHY.md`, `CITATIONS.md`.

## 2026-06-03 — v0.1.0: Smarter + faster across the .t.md

A hot-path overhaul of `_call_prompt` (the spine every agent run flows through)
on two axes, plus the first reinforcement-learning substrate. Version 0.0.1 →
**0.1.0**; tests 285 → **316 green**.

### Faster

- **Indexed, in-memory caches.** `semantic_cache_lookup` re-read the JSONL and
  recomputed cosine against every row on every prompt call; `verify_cache_get`
  linear-scanned its file each compile. Both now load once per file version
  into a single-slot structure keyed by (path, mtime). The prompt cache gains
  an exact-hash fast path — an identical rendered prompt returns O(1) with no
  embedding (≈280× faster warm lookups, `tests/test_bench.py`). Also fixed a
  latent bug: the cache dir was frozen at import, so the test suite wrote to
  the dev's real `~/.cache/tessera`; it now resolves dynamically and is
  isolated per test.
- **Trait hot path.** `TriggerContext._haystack` is memoized per context — a
  single `fire_traits()` no longer re-normalizes the prompt ~14× (once per
  built-in trigger).

### Smarter

- **Auto-recall (RAG).** Agents with a `memory:semantic` / `memory:episodic`
  block get relevant facts (ranked by keyword overlap, from the persistent
  store and the in-World shadow) + recent episodic events injected into every
  prompt automatically. Opt-out `TESSERA_NO_AUTO_RECALL=1`; counts stamped as
  `recalled` on each prompt audit event.
- **Auto-confidence routing.** `bayesian_posterior` / `abductive` write their
  max posterior to `_confidence`, which `tsr:dual_process` reads at the next
  plan entry — so reasoning steers fast/slow routing on its own. Slow-routed
  prompts inject a deliberation frame and bypass the cache; `routed=fast|slow`
  audited.
- **`tsr:rl` substrate.** The orphaned `rl.py` is now first-class: parser/SIR
  (`RLDecl`), checker (`pass_16_rl`: E930 unknown agent, E931 needs ≥2
  actions), and two opt-in builtins — `rl_choose()` returns an ε-greedy action
  label keyed on the `state_from` beliefs; `rl_reward(action, reward)` updates
  the tabular Q-table, persisted per agent under `~/.tessera/rl/`. Actions are
  labels (not plans) so choice never dispatches or intercepts control flow.
  Catalog **29 → 30 shipped**; `examples/rl_router.t.md`.

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
