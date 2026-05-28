"""Tests for the causal substrate + Pearl do-calculus backdoor adjustment."""
import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail
from tessera.causal import (
    CausalDAG,
    find_backdoor_adjustment_set,
    query_effect_identifiable,
)


# --------- DAG core ---------


def test_dag_parents_children_descendants():
    dag = CausalDAG(
        name="d",
        variables=["A", "B", "C", "D"],
        edges=[("A", "B"), ("B", "C"), ("A", "D")],
    )
    assert dag.parents("B") == {"A"}
    assert dag.children("A") == {"B", "D"}
    assert dag.descendants("A") == {"B", "C", "D"}
    assert dag.descendants("C") == set()


def test_dag_cycle_detected():
    dag = CausalDAG(
        name="loop",
        variables=["A", "B"],
        edges=[("A", "B"), ("B", "A")],
    )
    assert dag.has_cycle()


def test_dag_no_cycle_on_tree():
    dag = CausalDAG(
        name="tree",
        variables=["root", "left", "right"],
        edges=[("root", "left"), ("root", "right")],
    )
    assert not dag.has_cycle()


# --------- Backdoor adjustment ---------


def test_no_confounder_empty_adjustment():
    # T -> Y, nothing else. No backdoor path → empty Z is admissible.
    dag = CausalDAG(name="d", variables=["T", "Y"], edges=[("T", "Y")])
    Z = find_backdoor_adjustment_set(dag, "T", "Y")
    assert Z == set()


def test_classical_confounder_requires_conditioning():
    # Confounder C: C -> T, C -> Y. Backdoor T <- C -> Y.
    # Admissible set must contain C.
    dag = CausalDAG(
        name="d",
        variables=["T", "Y", "C"],
        edges=[("C", "T"), ("C", "Y"), ("T", "Y")],
    )
    Z = find_backdoor_adjustment_set(dag, "T", "Y")
    assert Z == {"C"}


def test_collider_blocks_without_conditioning():
    # Collider M: T -> M <- Y is NOT a backdoor (path starts T -> M, outgoing).
    # The only path T -> M <- Y starts with an outgoing arrow from T — not a
    # backdoor — so no adjustment needed.
    dag = CausalDAG(
        name="d",
        variables=["T", "Y", "M"],
        edges=[("T", "M"), ("Y", "M"), ("T", "Y")],
    )
    Z = find_backdoor_adjustment_set(dag, "T", "Y")
    assert Z == set()


def test_descendant_of_treatment_excluded_from_adjustment():
    # D is descendant of T; cannot include in adjustment set.
    # C is real confounder. Best Z = {C}, not {C, D}.
    dag = CausalDAG(
        name="d",
        variables=["T", "Y", "C", "D"],
        edges=[("C", "T"), ("C", "Y"), ("T", "Y"), ("T", "D")],
    )
    Z = find_backdoor_adjustment_set(dag, "T", "Y")
    assert "D" not in Z
    assert "C" in Z


def test_query_effect_identifiable_on_simple_chain():
    # Chain T -> M -> Y. Effect identifiable with empty Z.
    dag = CausalDAG(
        name="d",
        variables=["T", "M", "Y"],
        edges=[("T", "M"), ("M", "Y")],
    )
    ok, Z = query_effect_identifiable(dag, "T", "Y")
    assert ok
    assert Z == set()


# --------- Substrate parsing ---------


def test_causal_substrate_parses_into_module():
    src = """---
agent: Reasoner
tessera_version: 0.2
---

```tsr:causal
causal MarketDAG {
  var price: Float
  var demand: Float
  var season: String
  edge season -> price
  edge season -> demand
  edge price -> demand
}
```

```tsr:agent
agent Reasoner {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert "MarketDAG" in module.causal_dags
    dag = module.causal_dags["MarketDAG"]
    assert set(dag.variables) == {"price", "demand", "season"}
    assert ("season", "price") in dag.edges
    assert ("price", "demand") in dag.edges


def test_causal_substrate_rejects_cycles():
    src = """---
agent: BadDAG
tessera_version: 0.2
---

```tsr:causal
causal Loop {
  var A: Float
  var B: Float
  edge A -> B
  edge B -> A
}
```

```tsr:agent
agent BadDAG {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    with pytest.raises(SyntaxFail, match="cycle"):
        lower(pm)
