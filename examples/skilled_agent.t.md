---
agent: SkilledAgent
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Skilled Agent — memory:procedural demo

Three skills bound to three different underlying callables. Same call site
in the plan; different machinery underneath. Skills are cached per-input
(repeated calls short-circuit) and tracked for stats.

```tsr:logic
fn upper(s: String) -> String = s + "!"
```

```tsr:prompt
prompt brief(question: String) -> String = "Briefly: {question}"
```

```tsr:memory:procedural
procedural {
  skill summarize(question: String) -> String from prompt brief
  skill emphasize(text: String) -> String from fn upper
}
```

```tsr:agent
agent SkilledAgent {
  beliefs:
    @last_write topic: String

  intentions:
    plan answer {
      let s1 = summarize(topic)
      let s2 = emphasize(s1)
      return s2
    }
}
```
