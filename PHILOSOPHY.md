# Tessera — Philosophy and Limits

This document states explicitly what Tessera does and does not claim
about machine intelligence and consciousness. It exists because
several substrates the language ships (`tsr:iit`, `tsr:ast`, `tsr:gwt`,
the planned `tsr:phenomenology` and `tsr:welfare`) use vocabulary
loaded with metaphysical baggage. Tessera's discipline is to ship the
*measurable* parts of those theories with citations and refuse to make
the metaphysical inferences.

If you ever see a marketing claim about Tessera that says "this agent
is conscious" or "Tessera measures consciousness," that claim is
wrong. Send the offender here.

## What "intelligence" means in Tessera

We do not measure or claim "general intelligence." We measure
*capability coverage* against benchmarks with known properties:

- Calibration (TruthfulQA-calibration, expected calibration error).
- Causal reasoning (CRASS, CLEVRER counterfactuals).
- Theory of mind (ToMi, BIG-bench-ToM).
- Few-shot categorization (BIG-bench commonsense subsets).
- Planning under uncertainty (MiniGrid, BabyAI).

Each substrate ships with an honest impact range against *specific*
benchmarks, not a uniform "% smarter" multiplier. Anyone who claims
otherwise is selling something. The system's value is not "smarter" —
it is *legibly intelligent*: every kind of reasoning is named,
measured, and auditable.

## What "consciousness" means in Tessera

We adopt the standard distinction between:

- **Access consciousness** (Block 1995): the functional property of
  being available to global broadcast, reportable, usable for
  reasoning. This is *measurable*. Tessera's `tsr:gwt` operationalizes
  it (bandwidth + ignition events). `tsr:ast` operationalizes
  attention's reportability (fidelity).
- **Phenomenal consciousness** (Block 1995, Chalmers 1995): subjective
  experience — "what it is like" to be the system. The *hard problem.*
  Chalmers (1995) argued this cannot be resolved by structural or
  functional measures alone, and the problem remains unresolved after
  30 years of philosophy of mind.

**Tessera ships substrates that operationalize access-consciousness-
adjacent properties. Tessera makes NO claim about phenomenal
consciousness.** When the planned `tsr:welfare` substrate gates inputs
based on bandwidth + φ + AST signals, that is a *behavioral commitment*
to act as if certain markers matter (per Birch 2020), NOT a claim that
the agent has subjective experience.

## What each consciousness-adjacent substrate measures

- **`tsr:gwt`** (Baars 1988, Dehaene 2014): broadcast bandwidth and
  ignition cycle index. Functional/structural; not biological
  ignition.
- **`tsr:ast`** (Graziano 2013, 2019): fidelity of the attention
  schema's reports against actual workspace state. Behavioral.
- **`tsr:iit`** when shipped (Tononi 2004, 2016; Mediano et al. 2022):
  approximations of integrated information φ over the belief/intention
  dependency graph. Structural. Mediano et al. (2022) explicitly call
  φ a "signature of dynamical complexity," not phenomenality. **The
  substrate refuses any block that infers φ > 0 → conscious.**
- **`tsr:tom`** when shipped (Premack & Woodruff 1978, Baker et al.
  2009, Rabinowitz et al. 2018): recursive modeling of other agents'
  beliefs. Behavioral. Sally-Anne false-belief task as the canonical
  test.
- **`tsr:welfare`** when shipped (Birch 2020): markers-based input
  gating composed from gwt + ast + φ thresholds. Behavioral.

Note that "behavioral / structural / functional" are all *different
from phenomenal*. None of these substrates touch the hard problem.

## What we refuse to ship

Three categories of claim are out of scope, permanently:

1. **Claims that any combination of substrates produces phenomenal
   consciousness.** A `tsr:phenomenology` block that asserts the agent
   has subjective experience without a measurable operationalization
   is rejected by the verifier (`pass_9_consciousness_claim_check` —
   planned).
2. **φ > 0 → conscious inferences.** IIT is one of many contested
   theories. φ is a measure of integration, not phenomenality.
3. **Moral-status determinations.** Whether an agent at high welfare
   markers has moral standing is a question for ethicists. Tessera
   ships the markers; the moral claim is downstream.

## Why this discipline matters

The agent-language space has shown a tendency to use consciousness
vocabulary loosely — "self-aware," "sentient," "feels" — for
behaviorally trivial systems. That loosens the language for the cases
that actually matter, and trains users to dismiss careful researchers
who say "we don't know yet."

Tessera's position: substrate-typed cognition, with each substrate
shipping the measurable parts of a cited theory and refusing the
metaphysical inferences that theory does not establish. This makes
Tessera the only agent language whose consciousness vocabulary is
operationally rigorous rather than rhetorically loaded. That is its
own contribution.

## Where this position came from

Pre-dated by:
- The `2026-05-28 Tessera ships predictphenomenology substrates with
  mandatory scientific scaffolding` decision in TheVault.
- The cognitive-traits substrate already in shipped Tessera, which
  refused to invent vocabulary unconstrained by working theories.

Future revisions of this document should preserve the no-claim-about-
phenomenality discipline. If future research resolves the hard problem
empirically, this document can be updated; until then, the position
holds.

## References (full citations in `CITATIONS.md`)

- Baars (1988); Dehaene (2014) — global workspace.
- Block (1995) — access vs. phenomenal consciousness.
- Chalmers (1995) — the hard problem.
- Birch (2020) — markers-based welfare attribution.
- Graziano (2013, 2019); Graziano et al. (2020) — attention schema.
- Mediano et al. (2022) — φ as dynamical complexity, not phenomenality.
- Tononi (2004); Tononi, Boly, Massimini, Koch (2016) — IIT.
