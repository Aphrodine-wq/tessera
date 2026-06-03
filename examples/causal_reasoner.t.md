# Causal reasoner — reasoning tools as callables over a declared DAG

This agent declares a causal DAG and a Bayesian model, then *calls* the
reasoning tools from its plan — no new substrate block, just functions usable
inside a plan body, made possible by the value layer's list `[..]` and record
`{..}` literals. `causal_backdoor` / `counterfactual` operate on the declared
`tsr:causal` DAG; `bayesian_posterior` queries the declared `tsr:bayesian`
model; `abductive` and `analogy` operate on inline data.

```tsr:causal
causal Market {
  var Season: Bool
  var Ad: Bool
  var Sales: Bool
  edge Season -> Ad
  edge Season -> Sales
  edge Ad -> Sales
}
```

```tsr:bayesian
bayesian {
  var Demand: [high, low] prior [0.3, 0.7]
  likelihood Signal given Demand {
    high -> strong: 0.8
    low -> strong: 0.2
  }
}
```

```tsr:agent
agent Analyst {
  beliefs:
    @last_write q: String
  intentions:
    plan analyze {
      let adjust = causal_backdoor("Market", "Ad", "Sales")
      let posterior = bayesian_posterior("Demand", "Signal", "strong")
      let cause = abductive([{name: "ad_worked", prior: 0.4, complexity: 1.0, likelihood: {sales_up: 0.9}}], ["sales_up"])
      return {confounders: adjust, demand: posterior, best_cause: cause}
    }
}
```

```bash
tessera compile examples/causal_reasoner.t.md --run Analyst --set q=go --audit /tmp/causal.jsonl
# → returns {confounders: ["Season"], demand: {high: ~0.63, low: ~0.37}, best_cause: "ad_worked"}
#   with bayesian:posterior, causal-adjustment, and abductive:rank events in the trace.
```
