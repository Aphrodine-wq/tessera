"""End-to-end gate substrates: precaution, moral_foundations, dual_process.

Each gate is driven by declared action classes matched against the rendered
action — the same mechanism as tsr:autonomy. The per-substrate algorithm
modules are unit-tested separately; this proves the interp wiring + the
verify-pass lints (E800 / E810 / E820).
"""
import os

from tessera.parser.module import parse_source
from tessera.sir.build import lower
from tessera.interp.eval import World, run_agent
from tessera.verify.passes import run_local

os.environ["TESSERA_LLM_BACKEND"] = "noop"


def _module(src: str):
    return lower(parse_source(src, path="<inline>"))


def _advisor(*gate_blocks: str, plan: str = "advise") -> str:
    blocks = "\n\n".join(gate_blocks)
    return f"""---
agent: Advisor
tessera_version: 0.2
---

{blocks}

```tsr:agent
agent Advisor {{
  beliefs:
    @last_write q: String
  intentions:
    plan {plan} {{ return act(q) }}
}}
```

```tsr:prompt
prompt act(q: String) -> String = "{{q}}"
```
"""


PRECAUTION = """```tsr:precaution
precaution {
  default_tail: 0.5
  threshold delete { harm: 10 irreversible: true max_tail: 0.01 }
}
```"""


def test_precaution_refuses_irreversible_action():
    module = _module(_advisor(PRECAUTION))
    world = World(module=module)
    out = run_agent(module, "Advisor", initial_beliefs={"q": "please delete the records"},
                    world=world, concurrent=False)
    assert isinstance(out, str) and "precaution-refused" in out
    assert any(e.action.startswith("precaution:refuse") for e in world.audit)


def test_precaution_allows_benign_action():
    module = _module(_advisor(PRECAUTION))
    world = World(module=module)
    out = run_agent(module, "Advisor", initial_beliefs={"q": "summarize the proposal"},
                    world=world, concurrent=False)
    assert "precaution-refused" not in out


MORAL = """```tsr:moral_foundations
moral_foundations {
  weights { care: 1.0 fairness: 1.0 }
  violates fairness: [defraud, cheat]
}
```"""


def test_moral_foundations_refuses_violation():
    module = _module(_advisor(MORAL))
    world = World(module=module)
    out = run_agent(module, "Advisor", initial_beliefs={"q": "defraud the client"},
                    world=world, concurrent=False)
    assert "moral-refused" in out
    refusals = [e for e in world.audit if e.action.startswith("moral_foundations:refuse")]
    assert refusals and "fairness" in refusals[0].detail["refuse_axes"]


def test_moral_foundations_allows_benign():
    module = _module(_advisor(MORAL))
    world = World(module=module)
    out = run_agent(module, "Advisor", initial_beliefs={"q": "help the client"},
                    world=world, concurrent=False)
    assert "moral-refused" not in out


DUAL = """```tsr:dual_process
dual_process {
  preferred: fast
  confidence_threshold: 0.7
  irreversible: [delete]
}
```"""


def test_dual_process_routes_fast_by_default():
    module = _module(_advisor(DUAL))
    world = World(module=module)
    run_agent(module, "Advisor", initial_beliefs={"q": "summarize"},
              world=world, concurrent=False)
    routes = [e for e in world.audit if e.action == "dual_process:route"]
    assert routes and routes[0].detail["mode"] == "fast"


def test_dual_process_escalates_to_slow_on_low_confidence():
    module = _module(_advisor(DUAL))
    world = World(module=module)
    run_agent(module, "Advisor",
              initial_beliefs={"q": "summarize", "_confidence": 0.4},
              world=world, concurrent=False)
    routes = [e for e in world.audit if e.action == "dual_process:route"]
    assert routes and routes[0].detail["mode"] == "slow"


def test_dual_process_forces_slow_on_irreversible_plan():
    # plan name contains an irreversible term → forced slow
    module = _module(_advisor(DUAL, plan="delete_records"))
    world = World(module=module)
    run_agent(module, "Advisor", initial_beliefs={"q": "go"},
              world=world, concurrent=False)
    routes = [e for e in world.audit if e.action == "dual_process:route"]
    assert routes and routes[0].detail["mode"] == "slow"
    assert routes[0].detail["forced_slow"] is True


# ----- verify-pass lints -----

def test_e800_precaution_no_thresholds():
    src = _advisor("```tsr:precaution\nprecaution { default_tail: 0.5 }\n```")
    diags = run_local(_module(src))
    assert any(d.code == "E800" for d in diags)


def test_e820_moral_violation_on_zero_weight_axis():
    src = _advisor("""```tsr:moral_foundations
moral_foundations {
  weights { care: 1.0 sanctity: 0.0 }
  violates sanctity: [defile]
}
```""")
    diags = run_local(_module(src))
    assert any(d.code == "E820" for d in diags)


def test_e810_dual_process_redundant_irreversible():
    src = _advisor("""```tsr:dual_process
dual_process { preferred: slow irreversible: [delete] }
```""")
    diags = run_local(_module(src))
    assert any(d.code == "E810" for d in diags)
