"""Integration tests — exercise multiple substrates together end-to-end.

Each test composes several recently-shipped pieces into a single flow
that catches regressions when one piece changes in a way that breaks
its consumers.
"""
from pathlib import Path

from tessera.parser.module import parse_file
from tessera.sir.build import lower
from tessera.interp.eval import run_agent, Refusal


EXAMPLES = Path(__file__).parent.parent / "examples"


def test_complaint_router_full_pipeline_with_audit_tiering():
    """The complaint_router exercises intent + ethics + autonomy + traits +
    capabilities + audit. After running it, the audit_governance store
    has rows (ethics_applied non-empty); the operational store has rows
    (routine plan_enter and prompt actions without intent_served binding)."""
    from tessera.adapters.audit import query_events
    import os
    import sqlite3
    pm = parse_file(EXAMPLES / "complaint_router.t.md")
    module = lower(pm)
    run_agent(module, "ComplaintRouter",
              initial_beliefs={"complaint": "rerouting test"})
    gov_db = os.environ["TESSERA_AUDIT_GOV_DB"]
    with sqlite3.connect(gov_db) as c:
        gov_count = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert gov_count > 0, "expected governance events (ethics_applied)"
    # And we can query both tiers in one call
    rows = query_events(agent="ComplaintRouter")
    assert rows, "expected merged events across both tiers"


def test_evolve_persists_to_governance_and_query_returns_lineage():
    """Running evolve produces evolve:generation_N events; tessera audit query
    can reconstruct the lineage of best scores across generations."""
    from tessera.evolve import evolve
    from tessera.adapters.audit import query_events
    pm = parse_file(EXAMPLES / "evolve_researcher.t.md")
    module = lower(pm)
    history = evolve(module)
    rows = query_events(action="evolve")
    # One audit row per generation, plus history matches.
    gens = sorted([r for r in rows if "best_score" in r],
                  key=lambda r: r.get("seq") or 0)
    assert len(gens) >= len(history)
    # Best score per generation is captured.
    for h, row in zip(history, gens):
        assert abs(h.best_score - row["best_score"]) < 1e-6


def test_spawn_auto_restrict_audit_links_to_governance():
    """Spawn auto-restrict emits caps_narrowed → governance tier.
    Verifying via tessera/adapters/audit::query_events that the caps_narrowed
    event is reachable with a tier=governance filter."""
    from tessera.adapters.audit import query_events
    pm = parse_file(EXAMPLES / "researcher.t.md")
    module = lower(pm)
    from tessera.interp.eval import World, eval_region
    world = World(module=module)
    # Don't grant TeamLead the NetworkOut cap — its child spawns should auto-narrow.
    state = world.state_for("TeamLead")
    state.working_memory["topic"] = "x"
    eval_region(module.agents["TeamLead"], world, agent_name="TeamLead")
    rows = query_events(action="caps_narrowed", tier="governance")
    assert rows, "caps_narrowed must land in the governance tier"


def test_policy_pii_guard_refusal_lands_in_governance_audit():
    """A PII refusal emits a `refusal` action that routes to governance."""
    from tessera.adapters.audit import query_events
    pm = parse_file(EXAMPLES / "policy_pii_guard.t.md")
    module = lower(pm)
    result = run_agent(module, "PIIGuard",
                       initial_beliefs={"message": "SSN 123-45-6789"})
    assert isinstance(result, Refusal)
    rows = query_events(action="refusal", tier="governance")
    assert rows, "refusal events must land in governance tier"


def test_skill_promotion_to_corpus_full_flow():
    """summarizer_with_promotion runs → skill_promotion_pending fires →
    training_corpus.assemble_for_skill writes a JSONL file with pairs."""
    from tessera.training_corpus import assemble_for_skill
    from tessera.adapters.audit import query_events
    pm = parse_file(EXAMPLES / "summarizer_with_promotion.t.md")
    module = lower(pm)
    run_agent(module, "Summarizer", initial_beliefs={"article": "x"})
    # Promotion event was emitted
    pending = query_events(action="skill_promotion_pending")
    assert pending, "expected skill_promotion_pending after threshold"
    # And the corpus assembler reads + writes pairs
    path, n = assemble_for_skill("summarize")
    assert path.exists()
    assert n > 0, "expected pairs in the corpus"


def test_governance_consistency_catches_always_refuse():
    """An always-refusing policy lands E1000 from pass_8."""
    from tessera.parser.module import parse_source
    from tessera.verify.passes import run_local
    src = """---
agent: BadGov
tessera_version: 0.2
---

```tsr:policy
policy AlwaysRefuse {
  forbid when true
}
```

```tsr:agent
agent BadGov {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    module = lower(parse_source(src, path="<inline>"))
    diags = run_local(module)
    assert any(d.code == "E1000" for d in diags), \
        f"expected E1000, got codes: {[d.code for d in diags]}"
