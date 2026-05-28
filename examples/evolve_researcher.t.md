---
agent: Reflector
tessera_version: 0.2
capabilities_requested: []
max_cost: { dollars: 0.05, tokens: 5000 }
---

# Evolve Researcher — genetic prompt evolution (decision 17)

A minimal evolve example: the agent has one prompt, one eval case, and
a `tsr:evolve` block that runs four generations of prompt mutation.
Each generation spawns three variants by appending a different suffix
to the prompt template, scores each against the eval case, and keeps
the top survivor.

Per-generation results land in the governance audit store as
`evolve:generation_<N>` events with `best_score` and the full
`scores` list.

```
tessera evolve examples/evolve_researcher.t.md
# target agent: Reflector
#   gen 0: best_score=1.000 variant=0
#   gen 1: best_score=1.000 variant=2
#   ...

tessera audit query --action evolve --count
```

```tsr:logic
fn echo(s: String) -> String = s
```

```tsr:prompt
prompt think(q: String) -> String = "Reflect on the question: {q}"
```

```tsr:agent
agent Reflector {
  beliefs:
    @last_write q: String
  intentions:
    plan run {
      let r = think(q)
      return r
    }
}
```

```tsr:eval
case "contains-reflect" {
  input q = "what is fairness"
  expect_contains = "Reflect"
}
```

```tsr:evolve
evolve Reflector {
  population: 3
  generations: 4
  mutate: [prompts]
  fitness: eval_pass_rate
}
```
