"""Auto-confidence routing.

Two halves:
  - reasoning posteriors (bayesian / abductive) write `_confidence` so
    dual_process routing picks it up automatically;
  - a plan routed to the slow path makes prompts deliberate and bypass the
    cache, stamped `routed=slow` in the audit.
"""
from __future__ import annotations

import os

from tessera.interp.eval import World, run_agent
from tessera.parser.module import parse_source
from tessera.sir.build import lower

os.environ["TESSERA_LLM_BACKEND"] = "noop"

BAYES = """```tsr:bayesian
bayesian {
  var Disease: [yes, no] prior [0.01, 0.99]
  likelihood Test given Disease {
    yes -> pos: 0.99
    no -> pos: 0.05
  }
}
```"""


def test_bayesian_posterior_sets_confidence():
    src = f"""---
agent: A
tessera_version: 0.2
---

{BAYES}

```tsr:agent
agent A {{
  beliefs:
    @last_write q: String
  intentions:
    plan go {{ return bayesian_posterior("Disease", "Test", "pos") }}
}}
```
"""
    m = lower(parse_source(src, path="<inline>"))
    w = World(module=m)
    run_agent(m, "A", initial_beliefs={"q": "x"}, world=w, concurrent=False)
    conf = w.state_for("A").working_memory.get("_confidence")
    assert conf is not None
    assert 0.0 < conf <= 1.0


DUAL = """```tsr:dual_process
dual_process {
  preferred: fast
  confidence_threshold: 0.7
  irreversible: [delete]
}
```"""


def _advisor() -> str:
    return f"""---
agent: Advisor
tessera_version: 0.2
---

{DUAL}

```tsr:agent
agent Advisor {{
  beliefs:
    @last_write q: String
  intentions:
    plan advise {{ return act(q) }}
}}
```

```tsr:prompt
prompt act(q: String) -> String = "{{q}}"
```
"""


def _routed(confidence):
    m = lower(parse_source(_advisor(), path="<inline>"))
    w = World(module=m)
    run_agent(m, "Advisor",
              initial_beliefs={"q": "summarize", "_confidence": confidence},
              world=w, concurrent=False)
    prompts = [e for e in w.audit if e.action == "prompt:act"]
    assert prompts, "expected a prompt:act audit event"
    return prompts[0].detail.get("routed")


def test_low_confidence_prompt_routed_slow():
    assert _routed(0.4) == "slow"


def test_high_confidence_prompt_routed_fast():
    assert _routed(0.9) == "fast"


DUAL_WITH_ESCALATION = """```tsr:dual_process
dual_process {
  preferred: fast
  confidence_threshold: 0.7
  slow_backend: escalated
}
```"""


def _advisor_with_escalation() -> str:
    return f"""---
agent: Advisor
tessera_version: 0.2
---

{DUAL_WITH_ESCALATION}

```tsr:agent
agent Advisor {{
  beliefs:
    @last_write q: String
  intentions:
    plan advise {{ return act(q) }}
}}
```

```tsr:prompt
prompt act(q: String) -> String = "{{q}}"
```
"""


def _backend_used(confidence, monkeypatch):
    from tessera.adapters.llm import CompletionResult

    class _Stub:
        def __init__(self, name):
            self.name = name
            self.cost_dollars = 0.0

        def complete(self, prompt, **opts):
            return CompletionResult(text="ok", backend=self.name, model="stub")

    def fake_get_backend(name=None):
        return _Stub(name or "noop")

    monkeypatch.setattr("tessera.adapters.llm.get_backend", fake_get_backend)

    m = lower(parse_source(_advisor_with_escalation(), path="<inline>"))
    w = World(module=m)
    run_agent(m, "Advisor",
              initial_beliefs={"q": "summarize", "_confidence": confidence},
              world=w, concurrent=False)
    prompts = [e for e in w.audit if e.action == "prompt:act"]
    assert prompts, "expected a prompt:act audit event"
    return prompts[0].detail.get("backend")


def test_low_confidence_escalates_to_configured_slow_backend(monkeypatch):
    assert _backend_used(0.4, monkeypatch) == "escalated"


def test_high_confidence_uses_default_backend(monkeypatch):
    assert _backend_used(0.9, monkeypatch) == "noop"
