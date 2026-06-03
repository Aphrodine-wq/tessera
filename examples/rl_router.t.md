---
agent: Router
tessera_version: 0.2
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# RL Router — choose / learn (research B3, Sutton & Barto 2018)

The `tsr:rl` substrate gives an agent plan-level tabular Q-learning. The agent
picks a triage strategy with `rl_choose()` (ε-greedy over the declared
`actions`, keyed on the `state_from` beliefs) and learns from the outcome with
`rl_reward(choice, reward)`. Q-tables persist per agent under `~/.tessera/rl/`
(override `TESSERA_RL_DIR`) so learning survives across runs.

`rl_choose` returns a label for the plan to act on — it never dispatches plans
or touches control flow. Learning is the explicit `rl_reward` call.

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
