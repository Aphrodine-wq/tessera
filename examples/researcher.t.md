---
agent: TeamLead
capabilities_requested: [NetworkOut]
max_cost: { dollars: 0.00, tokens: 0 }
---

# Researcher Team — researcher_lite

Three agents coordinating through a workspace.

- **Researcher** finds papers (stub) and broadcasts a finding to `TeamMind`.
- **Critic** reviews the finding and broadcasts a critique.
- **TeamLead** spawns both, ferries messages, reads the winning workspace draft,
  and emits the final report.

```tsr:logic
fn join(a: String, b: String) -> String = a + " | " + b
```

```tsr:memory:workspace
workspace TeamMind {
  capacity: 1
  arbiter: highest_salience
  contenders: [findings, critiques]
}
```

```tsr:agent
agent Researcher {
  beliefs:
    @last_write topic: String
  intentions:
    plan find_papers {
      let finding = "5 papers on " + topic
      broadcast (finding, salience=0.7) to TeamMind
      return finding
    }
}

agent Critic {
  beliefs:
    @last_write target: String
  intentions:
    plan critique {
      let crit = "needs more rigor on " + target
      broadcast (crit, salience=0.6) to TeamMind
      return crit
    }
}

agent TeamLead {
  beliefs:
    @last_write topic: String
  intentions:
    plan run_team {
      let researcher = spawn Researcher with [NetworkOut]
      let critic = spawn Critic with [NetworkOut]
      send researcher topic
      let finding = recv from researcher
      send critic finding
      let critique = recv from critic
      let report = join(finding, critique)
      return report
    }
}
```
