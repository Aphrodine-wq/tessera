"""Auto-memory recall: an agent uses its memory without an explicit lookup.

The autouse conftest fixture isolates the semantic store + caches to tmp and
forces the noop backend, so `remember_fact` here lands in an isolated db.
"""
from __future__ import annotations

from tessera.adapters.semantic import rank_facts, remember_fact
from tessera.interp.eval import World, run_agent
from tessera.parser.module import parse_source
from tessera.sir.build import lower

AGENT = '''---
agent: Recaller
tessera_version: 0.2
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

```tsr:memory:semantic persistent=true
knowledge {
  schema FactSheet(title: String, domain: String)
}
```

```tsr:prompt
prompt advise(topic: String) -> String = "Advise on {topic}"
```

```tsr:agent
agent Recaller {
  beliefs:
    topic: String
  intentions:
    plan run {
      let a = advise(topic)
      return a
    }
}
```
'''


def test_rank_facts_prefers_keyword_overlap():
    facts = [
        {"schema": "F", "fields": {"title": "retainage in construction"}, "created_at": "1"},
        {"schema": "F", "fields": {"title": "the weather is nice today"}, "created_at": "2"},
    ]
    top = rank_facts(facts, "tell me about retainage", limit=1)
    assert top[0]["fields"]["title"] == "retainage in construction"


def test_rank_facts_falls_back_to_recency_when_no_overlap():
    facts = [
        {"schema": "F", "fields": {"title": "alpha"}, "created_at": "1"},
        {"schema": "F", "fields": {"title": "beta"}, "created_at": "2"},
    ]
    top = rank_facts(facts, "totally unrelated query", limit=1)
    assert top[0]["fields"]["title"] == "beta"  # most recent


def _run():
    module = lower(parse_source(AGENT))
    world = World(module=module)
    run_agent(module, "Recaller", initial_beliefs={"topic": "retainage"}, world=world)
    return [e for e in world.audit if e.action == "prompt:advise"]


def test_auto_recall_injects_semantic_fact():
    remember_fact("FactSheet",
                  {"title": "contractor retainage is 5-10%", "domain": "construction"})
    events = _run()
    assert events, "expected a prompt:advise audit event"
    assert events[0].detail.get("recalled", {}).get("semantic", 0) >= 1


def test_auto_recall_opt_out_env(monkeypatch):
    monkeypatch.setenv("TESSERA_NO_AUTO_RECALL", "1")
    remember_fact("FactSheet", {"title": "x", "domain": "y"})
    events = _run()
    assert events
    assert events[0].detail.get("recalled", {}).get("semantic", 0) == 0
