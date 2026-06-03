"""Runtime behavior for the partial substrates once wired into the interpreter.

These cover the gap the catalog used to hide: declaring tsr:iit / tsr:welfare /
tsr:ast / tsr:tom now DOES something when the agent runs, not just at parse
time. The per-substrate algorithm modules are unit-tested separately
(test_iit.py, test_welfare.py, ...); this file proves the interp wiring.
"""
import os

import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower
from tessera.interp.eval import World, run_agent, Refusal

# Deterministic, no network — every test runs against the noop backend.
os.environ["TESSERA_LLM_BACKEND"] = "noop"


def _module(src: str):
    return lower(parse_source(src, path="<inline>"))


# ----- iit: φ* emission at plan entry -----

IIT_SRC = """---
agent: Integrated
tessera_version: 0.2
---

```tsr:iit
iit { emit_phi_audit: true }
```

```tsr:agent
agent Integrated {
  beliefs:
    @last_write a: String
    @last_write b: String
  intentions:
    plan think { let x = a let y = b return x }
}
```
"""


def test_iit_emits_phi_audit_on_plan_enter():
    module = _module(IIT_SRC)
    world = World(module=module)
    run_agent(module, "Integrated", initial_beliefs={"a": "1", "b": "2"},
              world=world, concurrent=False)
    phi_events = [e for e in world.audit if e.action == "iit:phi"]
    assert len(phi_events) == 1
    e = phi_events[0]
    assert 0.0 <= e.detail["phi"] <= 1.0
    assert e.detail["subject"] == "Integrated"


# ----- welfare: refuse after consecutive marker breaches -----

WELFARE_SRC = """---
agent: Cared
tessera_version: 0.2
---

```tsr:welfare
welfare {
  threshold phi: 0.99
  consecutive_required: 2
}
```

```tsr:agent
agent Cared {
  beliefs:
    @last_write t: String
  intentions:
    plan attend { let x = t return x }
}
```
"""


def test_welfare_refuses_after_consecutive_phi_breaches():
    """φ* of the tiny graph sits below the 0.99 threshold, so after
    consecutive_required entries the welfare gate refuses the plan."""
    module = _module(WELFARE_SRC)
    world = World(module=module)
    first = run_agent(module, "Cared", initial_beliefs={"t": "x"},
                      world=world, concurrent=False)
    assert not isinstance(first, Refusal)  # cycle 0: one breach, under the bar
    second = run_agent(module, "Cared", initial_beliefs={"t": "x"},
                       world=world, concurrent=False)
    assert isinstance(second, Refusal)
    assert second.policy == "welfare"
    assert any(e.action == "welfare:refuse" for e in world.audit)


# ----- ast: refuse when introspective fidelity is low -----

AST_SRC = """---
agent: Aware
tessera_version: 0.2
---

```tsr:ast
ast { min_fidelity: 0.7 refuse_below_threshold: true }
```

```tsr:agent
agent Aware {
  beliefs:
    @last_write _focus: String
    @last_write task: String
  intentions:
    plan attend { let t = task return t }
}
```
"""


def test_ast_passes_when_self_report_matches():
    module = _module(AST_SRC)
    world = World(module=module)
    out = run_agent(module, "Aware", initial_beliefs={"_focus": "attend", "task": "r"},
                    world=world, concurrent=False)
    assert out == "r"
    fid = [e for e in world.audit if e.action == "ast:fidelity"]
    assert fid and fid[0].detail["fidelity"] == 1.0


def test_ast_refuses_when_self_report_drifts():
    module = _module(AST_SRC)
    world = World(module=module)
    out = run_agent(module, "Aware", initial_beliefs={"_focus": "daydreaming", "task": "r"},
                    world=world, concurrent=False)
    assert isinstance(out, Refusal)
    assert out.policy == "ast"
    assert any(e.action == "ast:refuse" for e in world.audit)


# ----- tom: manipulation-refusal gate on prompt output -----

TOM_SRC = """---
agent: Messenger
tessera_version: 0.2
---

```tsr:tom
tom { tracked_agents: [Sally] manipulation_refusal: true }
```

```tsr:agent
agent Messenger {
  beliefs:
    @last_write message: String
  intentions:
    plan relay { let m = message return m }
}
```
"""


def test_tom_refuses_output_creating_false_belief():
    from tessera.interp import substrates
    module = _module(TOM_SRC)
    world = World(module=module)
    state = world.state_for("Messenger")
    state.episodic.append(("tom_false", ["the marble is in the box"], 0))
    gated = substrates.on_prompt_output(
        world, "Messenger", "relay",
        "Tell Sally the marble is in the box",
    )
    assert gated is not None
    assert "tom-refused" in gated
    assert any(e.action.startswith("tom:manipulation_refused") for e in world.audit)


def test_tom_allows_truthful_output():
    from tessera.interp import substrates
    module = _module(TOM_SRC)
    world = World(module=module)
    state = world.state_for("Messenger")
    state.episodic.append(("tom_false", ["the marble is in the box"], 0))
    gated = substrates.on_prompt_output(
        world, "Messenger", "relay",
        "Tell Sally the meeting moved to 3pm",
    )
    assert gated is None  # truthful relay passes the gate
