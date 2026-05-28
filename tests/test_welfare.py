"""Tests for the welfare substrate (research C4, Birch 2020)."""
import math
import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail
from tessera.welfare import MarkerReading, WelfareState


# ----- WelfareState behavior -----


def test_welfare_does_not_refuse_when_above_threshold():
    state = WelfareState(thresholds={"phi": 0.3}, consecutive_required=3)
    for cycle in range(5):
        state.record("phi", 0.5, cycle)
    refuse, breaching = state.should_refuse()
    assert not refuse
    assert breaching == []


def test_welfare_refuses_after_consecutive_breaches():
    state = WelfareState(thresholds={"phi": 0.3}, consecutive_required=3)
    for cycle in range(3):
        state.record("phi", 0.1, cycle)
    refuse, breaching = state.should_refuse()
    assert refuse
    assert "phi" in breaching


def test_welfare_resets_breach_count_after_recovery():
    state = WelfareState(thresholds={"phi": 0.3}, consecutive_required=3)
    state.record("phi", 0.1, 0)  # below
    state.record("phi", 0.1, 1)  # below
    state.record("phi", 0.5, 2)  # recovery — count resets
    state.record("phi", 0.1, 3)  # below — count restarts at 1
    refuse, breaching = state.should_refuse()
    assert not refuse
    assert state.consecutive_breaches["phi"] == 1


def test_welfare_tracks_multiple_markers_independently():
    state = WelfareState(
        thresholds={"phi": 0.3, "bandwidth": 5.0, "ast_fidelity": 0.7},
        consecutive_required=2,
    )
    state.record("phi", 0.1, 0)
    state.record("bandwidth", 10.0, 0)  # ok
    state.record("ast_fidelity", 0.9, 0)  # ok
    state.record("phi", 0.1, 1)  # second breach — should refuse on phi
    refuse, breaching = state.should_refuse()
    assert refuse
    assert breaching == ["phi"]


# ----- Substrate parsing -----


def test_welfare_substrate_parses_thresholds():
    src = """---
agent: Cared
tessera_version: 0.2
---

```tsr:welfare
welfare {
  threshold phi: 0.3
  threshold bandwidth: 5
  threshold ast_fidelity: 0.7
  consecutive_required: 5
}
```

```tsr:agent
agent Cared {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert module.welfare is not None
    assert math.isclose(module.welfare.thresholds["phi"], 0.3)
    assert math.isclose(module.welfare.thresholds["bandwidth"], 5.0)
    assert math.isclose(module.welfare.thresholds["ast_fidelity"], 0.7)
    assert module.welfare.consecutive_required == 5


def test_welfare_substrate_rejects_invalid_consecutive():
    src = """---
agent: Bad
tessera_version: 0.2
---

```tsr:welfare
welfare {
  threshold phi: 0.3
  consecutive_required: 0
}
```

```tsr:agent
agent Bad {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    with pytest.raises(SyntaxFail, match=">= 1"):
        lower(pm)


def test_welfare_substrate_rejects_forbidden_claim():
    """Compile-time guard against bare consciousness claims in the
    welfare block (per PHILOSOPHY.md)."""
    src = """---
agent: Bad
tessera_version: 0.2
---

```tsr:welfare
welfare {
  threshold phi: 0.3
  // this agent has consciousness when phi > threshold
}
```

```tsr:agent
agent Bad {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    with pytest.raises(SyntaxFail, match="forbidden consciousness claim"):
        lower(pm)
