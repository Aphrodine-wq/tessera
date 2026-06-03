"""tsr:rl substrate — choose (ε-greedy) + learn (Q-update), end to end.

Stores are isolated per-test by conftest (incl. TESSERA_RL_DIR), so Q-tables
written here never touch the dev's ~/.tessera/rl.
"""
from __future__ import annotations

import os

from tessera.interp.eval import World, run_agent
from tessera.parser.module import parse_source
from tessera.rl import load_qtable, state_key
from tessera.sir.build import lower
from tessera.verify.passes import run_local

os.environ["TESSERA_LLM_BACKEND"] = "noop"

ROUTER = """---
agent: Router
tessera_version: 0.2
---

```tsr:rl
rl {
  agent: Router
  actions: [fast, careful]
  state_from: [topic]
  alpha: 0.5
  gamma: 0.9
  epsilon: 0.0
}
```

```tsr:agent
agent Router {
  beliefs:
    @last_write topic: String
  intentions:
    plan triage {
      let choice = rl_choose()
      let q = rl_reward(choice, 1.0)
      return q
    }
}
```
"""


def _module(src=ROUTER):
    return lower(parse_source(src, path="<inline>"))


def test_rl_module_verifies_clean():
    errors = [d for d in run_local(_module()) if d.severity == "error"]
    assert not errors, errors


def test_rl_choose_returns_declared_action_and_learns():
    m = _module()
    w = World(module=m)
    out = run_agent(m, "Router", initial_beliefs={"topic": "billing"},
                    world=w, concurrent=False)
    # alpha=0.5, reward=1.0, fresh Q=0 → new Q = 0.5
    assert out == 0.5
    chosen = [e for e in w.audit if e.action == "rl:choose"]
    assert chosen and chosen[0].detail["chosen"] in ("fast", "careful")
    # Q-table persisted with the chosen action's value
    table = load_qtable("Router")
    skey = state_key({"topic": "billing"})
    assert table.get(skey, chosen[0].detail["chosen"]) == 0.5


def test_rl_q_accumulates_across_runs():
    m = _module()
    for _ in range(3):
        run_agent(m, "Router", initial_beliefs={"topic": "billing"},
                  world=World(module=m), concurrent=False)
    table = load_qtable("Router")
    skey = state_key({"topic": "billing"})
    # Greedy (epsilon 0) re-picks the same action; Q climbs toward the reward.
    best = max(table.q.get(skey, {}).values(), default=0.0)
    assert best > 0.5  # more than a single update


def test_e930_unknown_target_agent():
    src = ROUTER.replace("agent: Router\n  actions", "agent: Ghost\n  actions")
    diags = run_local(_module(src))
    assert any(d.code == "E930" for d in diags)


def test_e931_too_few_actions():
    src = ROUTER.replace("actions: [fast, careful]", "actions: [only]")
    diags = run_local(_module(src))
    assert any(d.code == "E931" for d in diags)
