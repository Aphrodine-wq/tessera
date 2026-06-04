"""Tests for the orchestration spine — the single salience/priority currency
that ties plan ordering, the blackboard arbiters, multi-message recv, and
supervision together (see tessera/interp/scheduling.py)."""
from pathlib import Path

from tessera.parser.module import parse_source, parse_file
from tessera.sir.build import lower
from tessera.interp.eval import (
    World, WorkspaceState, run_agent, Refusal, eval_region,
)
from tessera.sir.nodes import WorkspaceDecl, Op
from tessera.interp import scheduling


def _run(src, agent, **beliefs):
    module = lower(parse_source(src, path="<inline>"))
    world = World(module=module)
    result = run_agent(module, agent, initial_beliefs=beliefs or None, world=world)
    return result, world


# ------------------------------------------------------------------ #1 priority

def test_plans_run_highest_priority_first():
    """Declaration order is low-then-high, but priority reorders execution so
    the high-priority plan's plan_enter is audited first."""
    src = """---
agent: Orderly
tessera_version: 0.2
---

```tsr:agent
agent Orderly {
  beliefs: @last_write q: String
  intentions:
    plan low priority=0.1 { return "low" }
    plan high priority=0.9 { return "high" }
}
```
"""
    _, world = _run(src, "Orderly", q="x")
    entered = [e.action.split(":", 1)[1] for e in world.audit
               if e.action.startswith("plan_enter:")]
    assert entered == ["high", "low"], entered
    # The priority is on the audit event so the currency is queryable.
    highs = [e for e in world.audit if e.action == "plan_enter:high"]
    assert highs and highs[0].detail.get("priority") == 0.9


def test_unannotated_plans_keep_declaration_order():
    """Plans without a priority default to 0.5 and the stable sort leaves their
    declaration order untouched — backward compatible."""
    src = """---
agent: Plain
tessera_version: 0.2
---

```tsr:agent
agent Plain {
  beliefs: @last_write q: String
  intentions:
    plan first { return "1" }
    plan second { return "2" }
}
```
"""
    _, world = _run(src, "Plain", q="x")
    entered = [e.action.split(":", 1)[1] for e in world.audit
               if e.action.startswith("plan_enter:")]
    assert entered == ["first", "second"], entered


def test_plan_priority_parses_with_serves_in_either_order():
    src = """---
agent: A
tessera_version: 0.2
---

```tsr:intent
intent Goal { goal: "do" }
```

```tsr:agent
agent A intends Goal {
  beliefs: @last_write q: String
  intentions:
    plan p priority=0.7 serves Goal { return q }
}
```
"""
    module = lower(parse_source(src, path="<inline>"))
    region = module.agents["A"]
    commit = next(n for n in region.nodes if n.op is Op.IntentionCommit)
    assert commit.attributes["priority"] == 0.7


# -------------------------------------------------------- #2/#3 blackboard

def test_broadcast_accumulates_until_read():
    """Contenders pool across broadcasts and are consumed only on read — the
    fix that lets multi-contender arbiters see the whole round."""
    ws = WorkspaceState(decl=WorkspaceDecl(name="W", arbiter="weighted_vote"))
    ws.broadcast("X", 0.3)
    ws.broadcast("X", 0.3)
    assert len(ws.contenders) == 2          # accumulated, not cleared on broadcast
    assert ws.read() == "X"
    assert len(ws.contenders) == 0          # the read closed the round


def test_weighted_vote_lets_agreement_beat_a_loud_outlier():
    """Three quiet votes for A (0.4*3=1.2) beat one loud B (0.9). highest_salience
    would pick B; weighted_vote picks the consensus."""
    ws = WorkspaceState(decl=WorkspaceDecl(name="W", arbiter="weighted_vote"))
    ws.broadcast("A", 0.4)
    ws.broadcast("B", 0.9)
    ws.broadcast("A", 0.4)
    ws.broadcast("A", 0.4)
    assert ws.read() == "A"
    # Contrast: highest_salience over the same field picks B.
    ws2 = WorkspaceState(decl=WorkspaceDecl(name="W", arbiter="highest_salience"))
    for v, s in [("A", 0.4), ("B", 0.9), ("A", 0.4), ("A", 0.4)]:
        ws2.broadcast(v, s)
    assert ws2.read() == "B"


def test_quorum_abstains_until_threshold_met():
    ws = WorkspaceState(decl=WorkspaceDecl(name="W", arbiter="quorum(2)"))
    ws.broadcast("A", 0.5)
    assert ws.read() is None                # one vote, quorum=2 → abstain
    ws.broadcast("A", 0.5)
    ws.broadcast("A", 0.6)
    assert ws.read() == "A"                 # quorum reached


def test_arbitrate_registry_split_and_fallback():
    assert scheduling.split_arbiter("quorum(3)") == ("quorum", 3)
    assert scheduling.split_arbiter("quorum:3") == ("quorum", 3)
    assert scheduling.split_arbiter("highest_salience") == ("highest_salience", 0)
    # Unknown arbiter degrades to last_write rather than crashing.
    assert scheduling.arbitrate("bogus", [("a", 0.1), ("b", 0.2)]) == ("b", 0.2)


# ----------------------------------------------------------- #4 multi-message

def test_recv_all_gathers_every_reply():
    """Two sends, one `recv all` → a list with both replies, in send order."""
    src = """---
agent: Parent
tessera_version: 0.2
---

```tsr:agent
agent Parent {
  beliefs: @last_write go: String
  intentions:
    plan run {
      let c = spawn Child with []
      send c "a"
      send c "b"
      let replies = recv all from c
      return replies
    }
}

agent Child {
  beliefs:
    @last_write m: String
  intentions:
    plan echo { return m }
}
```
"""
    result, _ = _run(src, "Parent", go="x")
    assert result == ["a", "b"], result


def test_recv_all_parses_gather_flag():
    src = """---
agent: P
tessera_version: 0.2
---

```tsr:agent
agent P {
  beliefs: @last_write go: String
  intentions:
    plan run {
      let c = spawn C with []
      let rs = recv all from c
      return rs
    }
}
agent C { beliefs: @last_write m: String  intentions: plan e { return m } }
```
"""
    module = lower(parse_source(src, path="<inline>"))
    recvs = [n for r in module.regions for n in r.nodes if n.op is Op.Recv]
    assert recvs and recvs[0].attributes.get("gather") is True


# -------------------------------------------------------------- #5 supervision

def test_supervised_child_retries_then_refuses():
    """A child whose plan raises is re-driven up to the budget, then its Refusal
    propagates instead of crashing the run. The retries are audited."""
    src = """---
agent: Boss
tessera_version: 0.2
---

```tsr:agent
agent Boss {
  beliefs: @last_write go: String
  intentions:
    plan run {
      let c = spawn Flaky with [] supervise=retry(2)
      send c "task"
      let r = recv from c
      return r
    }
}

agent Flaky {
  beliefs: @last_write m: String
  intentions:
    plan work { return boom(m) }
}
```
"""
    result, world = _run(src, "Boss", go="x")
    assert isinstance(result, Refusal), result
    retries = [e for e in world.audit if e.action == "supervised_retry"]
    exhausted = [e for e in world.audit if e.action == "supervised_exhausted"]
    assert len(retries) == 2, retries
    assert len(exhausted) == 1


def test_orchestration_example_reaches_quorum_consensus():
    """The shipped examples/orchestration.t.md runs all five moves together:
    two reviewers pool verdicts, the quorum arbiter resolves them, and the
    high-priority plan runs first."""
    example = Path(__file__).parent.parent / "examples" / "orchestration.t.md"
    module = lower(parse_file(example))
    world = World(module=module)
    run_agent(module, "Lead", initial_beliefs={"proposal": "ship it"}, world=world)
    # Both reviewers agreed → quorum(2) resolved to the shared verdict.
    assert world.workspaces["Verdict"].last_winner == "approve: ship it"
    # Plan priority ordered decide (0.9) ahead of housekeeping (0.1).
    entered = [e.action.split(":", 1)[1] for e in world.audit
               if e.action.startswith("plan_enter:")]
    assert entered.index("decide") < entered.index("housekeeping")
    # The quorum arbiter emitted an ignition event naming itself.
    ignitions = [e for e in world.audit if e.action.startswith("gwt:ignition")]
    assert any(e.detail.get("arbiter") == "quorum(2)" for e in ignitions)


def test_supervise_parses_retry_budget():
    src = """---
agent: P
tessera_version: 0.2
---

```tsr:agent
agent P {
  beliefs: @last_write go: String
  intentions:
    plan run {
      let c = spawn C with [] supervise=retry(3)
      return c
    }
}
agent C { beliefs: @last_write m: String  intentions: plan e { return m } }
```
"""
    module = lower(parse_source(src, path="<inline>"))
    spawn = next(n for r in module.regions for n in r.nodes if n.op is Op.Spawn)
    assert spawn.attributes.get("retries") == 3
