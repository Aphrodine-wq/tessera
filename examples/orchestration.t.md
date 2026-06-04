---
agent: Lead
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Orchestration spine — one currency, five moves

This example shows the whole orchestration layer working as a single idea:
**salience/priority is one scalar, and higher wins everywhere.**

- **Plan priority** — `Lead` runs `decide` (0.9) before `housekeeping` (0.1).
- **Blackboard accumulation** — reviewers' verdicts *pool* on the `Verdict`
  workspace across the round instead of overwriting each other.
- **Quorum arbiter** — `Verdict` only resolves once two reviewers agree.
- **Multi-message recv** — `recv all from panel` gathers every reply at once.
- **Supervision** — `spawn ... supervise=retry(1)` re-drives a reviewer that
  fails before its refusal would propagate.

```tsr:memory:workspace
workspace Verdict {
  capacity: 1
  arbiter: quorum(2)
  track_ignition: true
}
```

```tsr:agent
agent Reviewer {
  beliefs:
    @last_write item: String
  intentions:
    plan assess {
      let verdict = "approve: " + item
      broadcast (verdict, salience=0.8) to Verdict
      return verdict
    }
}

agent Lead {
  beliefs:
    @last_write proposal: String
  intentions:
    plan decide priority=0.9 {
      let panel = spawn Reviewer with [] supervise=retry(1)
      send panel proposal
      send panel proposal
      let votes = recv all from panel
      let consensus = read Verdict
      return consensus
    }
    plan housekeeping priority=0.1 {
      return "audit logged"
    }
}
```
