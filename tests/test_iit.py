"""Tests for the IIT substrate + φ* approximation (research C1)."""
import math
import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail
from tessera.iit import (
    DependencyGraph,
    all_bipartitions,
    phi_star,
    claim_violates_consciousness_discipline,
)


# ----- Math core -----


def test_phi_star_zero_on_trivial_graph():
    """≤1 node or no edges → φ* = 0 by convention."""
    assert phi_star(DependencyGraph(nodes=[], edges={})) == 0.0
    assert phi_star(DependencyGraph(nodes=["a"], edges={})) == 0.0


def test_phi_star_high_on_fully_connected():
    """A fully connected dense graph: every cut crosses many edges →
    high φ* (close to (n-1)/n for n nodes since most edges cross)."""
    g = DependencyGraph(
        nodes=["a", "b", "c"],
        edges={("a", "b"): 1.0, ("b", "c"): 1.0, ("a", "c"): 1.0,
               ("b", "a"): 1.0, ("c", "b"): 1.0, ("c", "a"): 1.0},
    )
    score = phi_star(g)
    # Best partition for 3 fully-connected nodes leaves 1 internal edge
    # (one node alone, two together → 2 edges intra, 4 cross). cross = 4/6.
    assert score > 0.5


def test_phi_star_low_on_two_disconnected_clusters():
    """Two clusters with one weak link between → low φ* (we can almost
    bisect with minimal damage)."""
    g = DependencyGraph(
        nodes=["a1", "a2", "b1", "b2"],
        edges={
            ("a1", "a2"): 10.0, ("a2", "a1"): 10.0,   # cluster A
            ("b1", "b2"): 10.0, ("b2", "b1"): 10.0,   # cluster B
            ("a1", "b1"): 0.1,                         # weak bridge
        },
    )
    score = phi_star(g)
    # The optimal cut puts {a1, a2} vs {b1, b2}. Only the 0.1 bridge crosses.
    # Total weight = 40.1; cross = 0.1; φ* = 0.1/40.1 ≈ 0.0025.
    assert score < 0.05


def test_all_bipartitions_count():
    """For n nodes, all_bipartitions enumerates 2^(n-1) - 1 cuts."""
    nodes = ["a", "b", "c"]  # 2^2 - 1 = 3 distinct bipartitions
    parts = all_bipartitions(nodes)
    assert len(parts) == 3


# ----- PHILOSOPHY.md claim guard -----


def test_claim_violates_catches_forbidden_patterns():
    forbidden_examples = [
        "this agent is conscious",
        "demonstrates subjective experience",
        "phi > 0 means consciousness",
        "with high phi the agent has consciousness",
    ]
    for ex in forbidden_examples:
        assert claim_violates_consciousness_discipline(ex) is not None, ex


def test_claim_violates_allows_measured_language():
    safe_examples = [
        "computes phi* as a structural measure",
        "tracks information integration",
        "audit emits iit:phi events",
        "",
    ]
    for ex in safe_examples:
        assert claim_violates_consciousness_discipline(ex) is None, ex


# ----- Substrate parsing -----


def test_iit_substrate_parses_defaults():
    src = """---
agent: Mind
tessera_version: 0.2
---

```tsr:iit
iit {
  emit_phi_audit: true
}
```

```tsr:agent
agent Mind {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert module.iit is not None
    assert module.iit.emit_phi_audit is True


def test_iit_substrate_rejects_forbidden_claim_at_compile_time():
    src = """---
agent: Overclaim
tessera_version: 0.2
---

```tsr:iit
iit {
  emit_phi_audit: true
  // this agent is conscious when phi > 0
}
```

```tsr:agent
agent Overclaim {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    with pytest.raises(SyntaxFail, match="forbidden consciousness claim"):
        lower(pm)
