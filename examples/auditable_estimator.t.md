---
agent: Estimator
capabilities_requested: [NetworkOut]
max_cost: { dollars: 0.05, tokens: 5000 }
---

# Auditable estimator — intent in the syntax

This agent declares *what it is for* with a `tsr:intent` block, not just how it
works. The intent states a goal, a checkable success criterion, and a forbidden
outcome — and that forbidden outcome (`NoPII`) must map to a real `tsr:policy`,
so the purpose can't be declared without the guardrail that backs it.

The agent binds to the intent with `intends`, and its plan with `serves`. At
runtime every action is stamped with the intent it served. Export the trace:

```
tessera compile examples/auditable_estimator.t.md --run Estimator \
  --set scope="kitchen remodel" --audit trace.jsonl
```

```tsr:intent
intent produce_estimate {
  goal: "Return a defensible, bounded construction cost estimate"
  success: estimate_has_line_items
  forbidden: [NoPII]
  why: "Homeowners act on this number directly — it has to be auditable and must never leak PII"
}
```

```tsr:policy
policy NoPII {
  forbid match "[0-9]{3}-[0-9]{2}-[0-9]{4}"
}
```

```tsr:prompt
prompt price(scope: String) -> String = "Give a line-item cost estimate for: {scope}."
```

```tsr:agent
agent Estimator intends produce_estimate {
  beliefs:
    @last_write scope: String
  intentions:
    plan build_estimate serves produce_estimate {
      let est = price(scope)
      return est
    }
}
```
