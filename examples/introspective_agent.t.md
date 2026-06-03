# Introspective agent — AST fidelity gate

Attention Schema Theory (Graziano): the agent keeps a model of its own
attention and reports from that model. The `tsr:ast` block measures **fidelity**
— how well the agent's reported focus (the `_focus` introspection belief)
matches the plan it is actually running. If the agent's self-report drifts below
the threshold, the substrate refuses to keep trusting its introspection.

The substrate ships the MEASURE (PHILOSOPHY.md) — it makes no claim that an
attention schema produces subjective experience.

```tsr:ast
ast {
  min_fidelity: 0.7
  refuse_below_threshold: true
}
```

```tsr:agent
agent Aware {
  beliefs:
    @last_write _focus: String
    @last_write task: String
  intentions:
    plan attend {
      let t = task
      return t
    }
}
```

Run it with an honest self-report (`_focus` matches the running plan `attend`):

```bash
tessera compile examples/introspective_agent.t.md --run Aware \
  --set _focus=attend --set task=review --audit /tmp/ast.jsonl
# → reported focus matches actual plan, fidelity = 1.0, the gate stays open.
#   Set _focus to something else and the fidelity gate refuses.
```
