"""Analogical reasoning via structure-mapping (research 4.4).

Primary reference: Gentner, D. (1983). Structure-mapping: a theoretical
framework for analogy. Cognitive Science 7(2):155-170.

Also: Falkenhainer, B., Forbus, K. D., Gentner, D. (1989). The
structure-mapping engine: algorithm and examples. Artificial
Intelligence 41(1):1-63 — the SME implementation Gentner's theory
inspired.

Structure-mapping theory says analogies are mappings between two
relational structures (the SOURCE and the TARGET) that preserve
relational structure even when individual objects differ. The
canonical example: "the atom is like the solar system" — the source
domain has [sun, planets, gravity-attracts] and the target has
[nucleus, electrons, electrical-attraction]; the analogy maps
sun↔nucleus and planets↔electrons because the relation
attracts(sun, planet) maps to attracts(nucleus, electron) even
though "sun" and "nucleus" share no surface features.

This MVP ships:
- Typed relation tuples: (predicate, [arg1, arg2, ...]).
- A greedy structure-mapping search that finds an object-binding
  M: source_objects → target_objects maximizing structural overlap
  (count of source relations whose argument mapping yields a target
  relation with the same predicate).
- A systematicity bonus (Gentner's key principle: higher-order
  relations beat shallow attribute matches).

Honest scope: symbolic SME, not LLM-augmented. Real-world analogy in
LLMs uses dense embeddings; the substrate's symbolic engine is the
verifiable spine. Author-declared relations are required; the system
cannot infer them from prose. A follow-up can layer LLM-extracted
relations on top while keeping this engine's contract stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Iterable


@dataclass(frozen=True)
class Relation:
    """One typed relation in a domain: predicate(arg1, arg2, ...)."""
    predicate: str
    args: tuple[str, ...]


@dataclass
class Domain:
    """A source or target domain — typed relations over named objects."""
    name: str
    objects: list[str] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass
class Mapping:
    """A candidate object-mapping with its structural-overlap score."""
    bindings: dict[str, str]   # source_object → target_object
    matched_relations: list[tuple[Relation, Relation]]
    score: float


def _relation_under_mapping(
    source_rel: Relation, bindings: dict[str, str]
) -> Relation:
    """Re-express a source relation under the candidate object-mapping."""
    mapped_args = tuple(bindings.get(a, a) for a in source_rel.args)
    return Relation(predicate=source_rel.predicate, args=mapped_args)


def _score_mapping(
    source: Domain,
    target: Domain,
    bindings: dict[str, str],
) -> tuple[float, list[tuple[Relation, Relation]]]:
    """Count source relations whose mapped form appears in target.

    Systematicity bonus: higher-arity relations contribute more
    (Gentner — higher-order structure preferred).
    """
    target_set = set(target.relations)
    matched: list[tuple[Relation, Relation]] = []
    score = 0.0
    for sr in source.relations:
        mapped = _relation_under_mapping(sr, bindings)
        if mapped in target_set:
            matched.append((sr, mapped))
            # Systematicity: arity > 1 gets a small bonus
            score += 1.0 + 0.25 * max(0, len(sr.args) - 1)
    return score, matched


def find_best_mapping(
    source: Domain,
    target: Domain,
    *,
    max_search: int = 5040,  # 7! cap
) -> Mapping | None:
    """Greedy + bounded-exhaustive mapping search.

    For small domains (≤7 source objects), enumerate every injective
    mapping into the target and pick the highest scorer. Above that,
    use a greedy heuristic: bind objects in order of relational
    degree (most-mentioned first), at each step picking the target
    that maximizes incremental score.

    Returns None when no relations match under any mapping (the
    domains have no shared structure).
    """
    if not source.objects or not target.objects:
        return None

    # Bounded exhaustive
    from math import factorial
    n_s = len(source.objects)
    n_t = len(target.objects)
    if factorial(min(n_s, n_t)) <= max_search and n_s <= 7:
        best: Mapping | None = None
        for perm in permutations(target.objects, n_s):
            bindings = dict(zip(source.objects, perm))
            score, matched = _score_mapping(source, target, bindings)
            if score > 0 and (best is None or score > best.score):
                best = Mapping(bindings=bindings, matched_relations=matched,
                               score=score)
        return best

    # Greedy fallback for larger domains
    bindings: dict[str, str] = {}
    used: set[str] = set()
    # Order source objects by relational degree, descending
    degree = {o: 0 for o in source.objects}
    for r in source.relations:
        for a in r.args:
            if a in degree:
                degree[a] += 1
    ordered = sorted(source.objects, key=lambda o: -degree[o])
    for so in ordered:
        best_to = None
        best_inc = -1.0
        for to in target.objects:
            if to in used:
                continue
            trial = {**bindings, so: to}
            inc, _ = _score_mapping(source, target, trial)
            if inc > best_inc:
                best_inc = inc
                best_to = to
        if best_to is None:
            break
        bindings[so] = best_to
        used.add(best_to)

    score, matched = _score_mapping(source, target, bindings)
    if score == 0:
        return None
    return Mapping(bindings=bindings, matched_relations=matched, score=score)
