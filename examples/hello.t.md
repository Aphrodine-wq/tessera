---
agent: HelloAgent
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Hello Agent

The simplest possible Tessera program. Exists to exercise the full
parse → SIR → verify → emit → interpret loop end-to-end.

```tsr:logic
fn greet(name: String) -> String = "hello " + name
```

```tsr:agent
agent HelloAgent {
  beliefs:
    @last_write target: String

  intentions:
    plan say_hello {
      let msg = greet(target)
      return msg
    }
}
```
