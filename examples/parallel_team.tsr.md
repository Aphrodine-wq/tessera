---
agent: Dispatcher
capabilities_requested: [NetworkOut]
max_cost: { dollars: 0.00, tokens: 0 }
---

# Parallel Team — demonstrates concurrent actor scheduler

Dispatcher spawns three independent specialists, sends each a question,
then recvs all three. With the synchronous scheduler this runs in 3× the
single-call latency (serial). With `TESSERA_CONCURRENT_AGENTS=1` it runs
in ~1× — each specialist runs in its own thread.

```tsr:logic
fn concat3(a: String, b: String, c: String) -> String = a + " // " + b + " // " + c
```

```tsr:agent
agent SpecialistA {
  beliefs:
    @last_write question: String
  intentions:
    plan answer { return "A: " + question }
}

agent SpecialistB {
  beliefs:
    @last_write question: String
  intentions:
    plan answer { return "B: " + question }
}

agent SpecialistC {
  beliefs:
    @last_write question: String
  intentions:
    plan answer { return "C: " + question }
}

agent Dispatcher {
  beliefs:
    @last_write topic: String

  intentions:
    plan poll_all {
      let a = spawn SpecialistA with [NetworkOut]
      let b = spawn SpecialistB with [NetworkOut]
      let c = spawn SpecialistC with [NetworkOut]
      send a topic
      send b topic
      send c topic
      let ra = recv from a
      let rb = recv from b
      let rc = recv from c
      let report = concat3(ra, rb, rc)
      return report
    }
}
```
