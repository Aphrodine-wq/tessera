---
agent: VaultAssistant
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Vault Assistant

An agent that logs every question it receives to an **episodic memory**, then
answers using whatever LLM backend is configured. The episodic log is the
agent's **autobiographical memory** — it persists for the lifetime of the
run and can be queried with `recall`.

This is the simplest demonstration of the `memory:episodic` substrate: an
agent that knows what it has done.

```tsr:memory:episodic
episodic {
  event Question(asked: String)
  event Answer(reply: String)
}
```

```tsr:prompt
prompt brief(question: String) -> String = "Answer briefly: {question}"
```

```tsr:agent
agent VaultAssistant {
  beliefs:
    @last_write question: String

  intentions:
    plan answer {
      log Question(question)
      let reply = brief(question)
      log Answer(reply)
      return reply
    }
}
```
