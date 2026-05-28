"""Tests for counterfactual reasoning (research 4.2)."""
import pytest

from tessera.causal import CausalDAG
from tessera.counterfactual import (
    StructuralEquation,
    topological_order,
    propagate,
    counterfactual_query,
)


# ----- Core propagation -----


def test_topological_order_on_chain():
    dag = CausalDAG(name="d", variables=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    order = topological_order(dag)
    assert order.index("A") < order.index("B") < order.index("C")


def test_propagate_simple_chain():
    """A → B → C. B = NOT(A). C = NOT(B). So A=True propagates to C=True."""
    dag = CausalDAG(name="d", variables=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    eqs = {
        "B": StructuralEquation(
            child="B", parents=["A"],
            table={(True,): False, (False,): True},
        ),
        "C": StructuralEquation(
            child="C", parents=["B"],
            table={(True,): False, (False,): True},
        ),
    }
    out = propagate(dag, eqs, {"A": True})
    assert out["A"] is True
    assert out["B"] is False
    assert out["C"] is True


# ----- Counterfactual query (canonical example) -----


def test_counterfactual_query_canonical_match_dependence():
    """Classic Halpern example. Two switches; light = OR(S1, S2).

    Observed: S1=True, S2=False, Light=True.
    Query: would the Light have been on if S1 had been False?
    Answer: Light would be False (S2 was also False).
    """
    dag = CausalDAG(
        name="lights",
        variables=["S1", "S2", "Light"],
        edges=[("S1", "Light"), ("S2", "Light")],
    )
    eqs = {
        "Light": StructuralEquation(
            child="Light", parents=["S1", "S2"],
            table={
                (False, False): False,
                (False, True): True,
                (True, False): True,
                (True, True): True,
            },
        ),
    }
    actual, cf = counterfactual_query(
        dag, eqs,
        observed={"S1": True, "S2": False, "Light": True},
        intervention=("S1", False),
        outcome="Light",
    )
    assert actual is True   # Light was on
    assert cf is False      # Would have been off had S1 been off


def test_counterfactual_query_or_with_backup():
    """Same lights DAG, but observed S1=True, S2=True, Light=True.

    Query: would Light have been on if S1 had been False?
    Answer: Yes — S2 still on, OR semantics keeps Light on.
    """
    dag = CausalDAG(
        name="lights",
        variables=["S1", "S2", "Light"],
        edges=[("S1", "Light"), ("S2", "Light")],
    )
    eqs = {
        "Light": StructuralEquation(
            child="Light", parents=["S1", "S2"],
            table={
                (False, False): False,
                (False, True): True,
                (True, False): True,
                (True, True): True,
            },
        ),
    }
    actual, cf = counterfactual_query(
        dag, eqs,
        observed={"S1": True, "S2": True, "Light": True},
        intervention=("S1", False),
        outcome="Light",
    )
    assert actual is True
    assert cf is True


def test_counterfactual_query_inconsistent_observation_returns_none():
    """If the observation can't be reproduced by the SCM, return (None, None)."""
    dag = CausalDAG(name="d", variables=["A", "B"], edges=[("A", "B")])
    eqs = {
        "B": StructuralEquation(
            child="B", parents=["A"],
            table={(True,): False, (False,): True},
        ),
    }
    # Observation says A=True, B=True — but SCM says A=True → B=False.
    actual, cf = counterfactual_query(
        dag, eqs,
        observed={"A": True, "B": True},
        intervention=("A", False),
        outcome="B",
    )
    assert actual is None
    assert cf is None
