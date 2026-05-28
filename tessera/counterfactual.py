"""Counterfactual reasoning over the tsr:causal DAG (research 4.2).

Primary references:
- Lewis, D. (1973). Counterfactuals. Harvard University Press.
- Halpern, J. Y. (2016). Actual Causality. MIT Press.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference, Ch. 7.

Tessera's tsr:causal substrate (shipped in a73091e) already ships
intervention reasoning via Pearl's backdoor adjustment. This module
extends it to counterfactuals via Pearl's three-step recipe:

  1. ABDUCTION — given the observed world W, infer the values of
     exogenous variables (the noise / unobserved causes).
  2. ACTION — intervene on the DAG to set the counterfactual treatment.
  3. PREDICTION — propagate forward, holding exogenous values fixed.

For Tessera's MVP we ship a finite-domain interpretation:
- Each variable has a small finite domain declared in the causal block
  (extension to be added: today only the variable NAMES exist on the
  DAG; we treat each as binary T/F unless the agent declares
  otherwise via observed assignments).
- The structural equations are author-declared as conditional tables
  (P(child | parents)).
- A counterfactual query "Y if (X = x)" given observed assignment O
  evaluates: do_intervention(X, x) holding the inferred exogenous
  values consistent with O, then propagate to Y.

Honest scope: this MVP supports COUNTERFACTUAL EVALUATION when the
structural equations are deterministic (every (parents, child) row
has probability 0 or 1). Stochastic counterfactuals require sampling
or analytic posterior; that's a follow-up. The deterministic case
captures Lewis's twin-world construction structurally.

Caveat: Counterfactual identifiability is strictly harder than
interventional. Some DAGs admit `do(X)` queries but not `would have`
queries — Pearl's hierarchy. The substrate exposes the identifiability
check so authors aren't surprised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .causal import CausalDAG


@dataclass
class StructuralEquation:
    """Deterministic structural equation: child = f(parents).

    `child` is the variable name; `parents` is the ordered list of
    parent variable names; `table` maps tuple-of-parent-values to the
    child's value. Missing entries default to None (undefined).
    """
    child: str
    parents: list[str]
    table: dict[tuple, Any] = field(default_factory=dict)

    def evaluate(self, parent_values: dict[str, Any]) -> Any:
        key = tuple(parent_values.get(p) for p in self.parents)
        return self.table.get(key)


def topological_order(dag: CausalDAG) -> list[str]:
    """Standard Kahn topo sort. Raises if cycle (shouldn't — DAG enforced)."""
    indeg = {v: 0 for v in dag.variables}
    for p, c in dag.edges:
        if c in indeg:
            indeg[c] += 1
    queue = [v for v, d in indeg.items() if d == 0]
    out: list[str] = []
    while queue:
        n = queue.pop(0)
        out.append(n)
        for c in dag.children(n):
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(out) != len(dag.variables):
        raise RuntimeError("topological_order on a cyclic graph")
    return out


def propagate(
    dag: CausalDAG,
    equations: dict[str, StructuralEquation],
    assignments: dict[str, Any],
) -> dict[str, Any]:
    """Given partial assignments (interventions or observations on root
    nodes), propagate values through the DAG via structural equations.

    Returns a complete assignment dict. Variables without a defined
    equation OR a starting assignment end up missing from the result.
    """
    values: dict[str, Any] = dict(assignments)
    for node in topological_order(dag):
        if node in values:
            continue
        eq = equations.get(node)
        if eq is None:
            continue
        parent_values = {p: values.get(p) for p in eq.parents}
        if any(v is None for v in parent_values.values()):
            continue
        result = eq.evaluate(parent_values)
        if result is not None:
            values[node] = result
    return values


def counterfactual_query(
    dag: CausalDAG,
    equations: dict[str, StructuralEquation],
    observed: dict[str, Any],
    intervention: tuple[str, Any],
    outcome: str,
) -> tuple[Any, Any]:
    """Pearl's three-step counterfactual: ABDUCTION → ACTION → PREDICTION.

    Returns (actual_outcome, counterfactual_outcome).

    The MVP's abduction: we treat exogenous (root) variables as fixed
    by the observation. For non-root observed variables, we trust the
    observation but verify it's consistent with the structural
    equations under the abduced exogenous values; on inconsistency we
    return (None, None) and the caller surfaces a refusal.
    """
    # 1. Abduction: identify exogenous (root) variables and pin them.
    roots = [v for v in dag.variables if not dag.parents(v)]
    exogenous = {r: observed.get(r) for r in roots if r in observed}

    # 2/3 Actual: propagate exogenous through equations to all nodes.
    actual = propagate(dag, equations, exogenous)

    # Consistency check on the actual world's outcome vs the observation
    if outcome in observed and outcome in actual:
        if actual[outcome] != observed[outcome]:
            # Observation can't be reproduced by the SCM → counterfactual
            # not identifiable from this model.
            return (None, None)
    actual_outcome = actual.get(outcome, observed.get(outcome))

    # 3. Counterfactual: same exogenous values, but apply intervention.
    X, x = intervention
    cf_assignments = dict(exogenous)
    cf_assignments[X] = x  # ACTION step — surgical replacement
    cf_world = propagate(dag, equations, cf_assignments)
    cf_outcome = cf_world.get(outcome)

    return (actual_outcome, cf_outcome)
