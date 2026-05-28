"""Genetic evolution for Tessera agents (decision 17).

A `tsr:evolve AgentName { ... }` block declares the population, mutation
targets, fitness metric, and generation count. `tessera evolve <file>`
runs the loop:

  generation 0 = the agent as written
  for each generation:
    1. spawn N variants by mutating prompts/traits of the survivors
    2. score each variant against the agent's declared eval cases
    3. select top-K = max(1, N // 2)
    4. record best score + lineage to the governance audit store

MVP mutation operators:
  - prompts: append a deterministic suffix (varied per index) to the
    prompt template. Real semantic mutation comes later.
  - traits: toggle one built-in trait on/off from the agent's attached
    set.

MVP fitness:
  - eval_pass_rate: fraction of declared eval cases the variant passes.
    Pass = result is not a Refusal AND any expect_contains/expect_equals
    constraint matches.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .adapters.audit import record_event


@dataclass
class GenerationResult:
    generation: int
    best_score: float
    best_variant_id: int
    scores: list[float]


_PROMPT_SUFFIXES = [
    "",
    " Think step by step.",
    " Be concise.",
    " Provide reasoning before the answer.",
    " Consider edge cases first.",
    " Cite sources when relevant.",
    " Answer in one sentence.",
    " Hedge appropriately.",
]


_OPTIONAL_TRAITS = [
    "doubt_first",
    "cross_brain",
    "compulsive",
    "synesthetic",
    "imposter_recursion",
]


def _mutate_prompts(module, variant_id: int):
    """Return a copy of the module with mutated prompt templates."""
    m = copy.deepcopy(module)
    suffix = _PROMPT_SUFFIXES[variant_id % len(_PROMPT_SUFFIXES)]
    for p in m.prompts.values():
        if suffix and suffix.strip() not in p.template:
            p.template = (p.template.rstrip() + suffix).strip()
    return m


def _mutate_traits(module, variant_id: int, agent_name: str):
    """Return a copy of the module with one trait toggled on the target agent."""
    m = copy.deepcopy(module)
    region = m.agents.get(agent_name)
    if region is None:
        return m
    trait_to_add = _OPTIONAL_TRAITS[variant_id % len(_OPTIONAL_TRAITS)]
    if trait_to_add not in region.trait_names:
        region.trait_names.append(trait_to_add)
    return m


def _score_variant(module, agent_name: str) -> float:
    """eval_pass_rate fitness: pass = case result is not a Refusal AND any
    declared expect_contains/expect_equals constraint matches."""
    if not module.eval_cases:
        return 0.0
    from .interp.eval import run_agent, Refusal
    passes = 0
    total = 0
    for case in module.eval_cases:
        total += 1
        try:
            result = run_agent(module, agent_name, initial_beliefs=case.inputs,
                               concurrent=False)
        except Exception:
            continue
        if isinstance(result, Refusal):
            if case.expect_refusal:
                passes += 1
            continue
        ok = True
        if case.expect_contains is not None:
            ok = ok and (isinstance(result, str) and case.expect_contains in result)
        if case.expect_equals is not None:
            ok = ok and (result == case.expect_equals)
        if ok:
            passes += 1
    return passes / total if total else 0.0


def evolve(module) -> list[GenerationResult]:
    """Run the evolve loop declared on `module.evolve`. Returns per-generation
    results and emits one audit event per generation."""
    decl = module.evolve
    if decl is None:
        raise RuntimeError("module has no tsr:evolve block")
    target = decl.target_agent
    if target not in module.agents:
        raise RuntimeError(f"evolve target agent {target!r} not found in module")

    survivors: list[Any] = [module]
    history: list[GenerationResult] = []

    for gen in range(decl.generations):
        variants: list[Any] = []
        for i in range(decl.population):
            base = survivors[i % len(survivors)]
            v = base
            if "prompts" in decl.mutate_targets:
                v = _mutate_prompts(v, gen * decl.population + i)
            if "traits" in decl.mutate_targets:
                v = _mutate_traits(v, gen * decl.population + i + 1, target)
            variants.append(v)

        scores = [_score_variant(v, target) for v in variants]
        ranked = sorted(range(len(variants)), key=lambda i: -scores[i])
        keep = max(1, decl.population // 2)
        survivors = [variants[i] for i in ranked[:keep]]
        best_id = ranked[0]
        best_score = scores[best_id]

        history.append(GenerationResult(
            generation=gen,
            best_score=best_score,
            best_variant_id=best_id,
            scores=scores,
        ))

        record_event({
            "seq": gen,
            "agent": target,
            "plan": None,
            "intent": None,
            "action": f"evolve:generation_{gen}",
            "population": decl.population,
            "best_score": best_score,
            "scores": scores,
            "generations_target": decl.generations,
        })

    return history
