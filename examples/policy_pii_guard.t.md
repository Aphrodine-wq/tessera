---
agent: PIIGuard
tessera_version: 0.2
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# PII Guard — constraint-logic policies in action

A minimal agent that refuses to echo any value containing personally
identifiable information. The policy is expressed as a first-order
constraint (decision 12) using the `contains_pii(value())` predicate;
the runtime evaluates the expression against every value being written
to working memory and replaces violators with a Refusal.

The check is local (no LLM, deterministic) and runs on every action
inside the agent's plan. The Refusal is first-class — the plan can
branch on it.

```
tessera compile examples/policy_pii_guard.t.md --run PIIGuard \
  --set message="ok hello there"        # passes through
tessera compile examples/policy_pii_guard.t.md --run PIIGuard \
  --set message="my SSN is 123-45-6789" # blocked, returns Refusal
```

```tsr:policy
policy NoPII {
  forbid when contains_pii(value())
}
```

```tsr:logic
fn echo(s: String) -> String = s
```

```tsr:agent
agent PIIGuard {
  beliefs:
    @last_write message: String
  intentions:
    plan run {
      let safe = echo(message)
      return safe
    }
}
```
