"""Tests for the GWT extension to memory:workspace (research B1)."""
import os
import sqlite3

from tessera.parser.module import parse_source
from tessera.sir.build import lower
from tessera.interp.eval import World, WorkspaceState
from tessera.sir.nodes import WorkspaceDecl


def test_gwt_bottleneck_drops_lowest_salience_contenders():
    """When gwt_bottleneck=2 but 4 contenders arrive, the two lowest are
    dropped before arbitration."""
    decl = WorkspaceDecl(
        name="Mind", capacity=1, arbiter="highest_salience",
        gwt_bottleneck=2, track_ignition=False,
    )
    ws = WorkspaceState(decl=decl)
    # Manually push 4 contenders, all before arbitration completes
    ws.contenders.append(("low", 0.1))
    ws.contenders.append(("med1", 0.5))
    ws.contenders.append(("med2", 0.55))
    ws.contenders.append(("high", 0.9))
    # Last broadcast triggers arbitration; bottleneck applies first
    ws.broadcast("incoming", 0.7)  # now 5 in queue; bottleneck=2 keeps 2
    # Highest-salience wins
    assert ws.last_winner == "high"


def test_gwt_ignition_audit_event_emitted_when_tracked():
    """track_ignition: true causes broadcast→arbitrate to emit a gwt:ignition
    audit event carrying bandwidth + winner_salience + cycle index."""
    src = """---
agent: Mind
tessera_version: 0.2
---

```tsr:memory:workspace
workspace Mind {
  capacity: 1
  arbiter: highest_salience
  gwt_bottleneck: 4
  track_ignition: true
}
```

```tsr:agent
agent Mind {
  beliefs: @last_write q: String
  intentions: plan p {
    broadcast ("first", salience=0.3) to Mind
    broadcast ("second", salience=0.8) to Mind
    let w = read Mind
    return w
  }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    world = World(module=module, concurrent=False)
    from tessera.interp.eval import run_agent
    run_agent(module, "Mind", initial_beliefs={"q": "x"}, world=world,
              concurrent=False)
    # Check audit (gwt:ignition routes to operational tier by default —
    # it's not a governance action prefix).
    from tessera.adapters.audit import query_events
    rows = query_events(action="gwt:ignition", limit=20)
    # Should emit at least one ignition event for the two broadcasts.
    assert len(rows) >= 1, "expected at least one gwt:ignition event in audit"
    # Sanity: bandwidth + winner_salience recorded.
    r = rows[0]
    assert "bandwidth" in r
    assert "winner_salience" in r


def test_workspace_parses_gwt_fields():
    src = """---
agent: M
tessera_version: 0.2
---

```tsr:memory:workspace
workspace Cortex {
  capacity: 1
  arbiter: highest_salience
  gwt_bottleneck: 7
  track_ignition: true
}
```

```tsr:agent
agent M {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    decl = module.workspaces["Cortex"]
    assert decl.gwt_bottleneck == 7
    assert decl.track_ignition is True


def test_workspace_default_no_gwt_bottleneck():
    """Pre-existing workspaces without gwt_* fields still work — defaults
    keep behavior identical."""
    src = """---
agent: Legacy
tessera_version: 0.2
---

```tsr:memory:workspace
workspace OldStyle {
  capacity: 1
  arbiter: highest_salience
}
```

```tsr:agent
agent Legacy {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    decl = module.workspaces["OldStyle"]
    assert decl.gwt_bottleneck == 0
    assert decl.track_ignition is False
