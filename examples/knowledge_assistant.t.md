---
agent: KnowledgeAssistant
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Knowledge Assistant — memory:semantic

Agent that **remembers** typed facts and **looks them up** by domain in
subsequent plan invocations. The `memory:semantic` substrate writes through
a local SQLite fact store at `~/.tessera/semantic.db` (override with the
`TESSERA_SEMANTIC_DB` env var).

```tsr:memory:semantic
knowledge {
  schema FactSheet(title: String, domain: String)
}
```

```tsr:agent
agent KnowledgeAssistant {
  beliefs:
    @last_write topic: String

  intentions:
    plan teach {
      remember FactSheet(title="contractor retainage is typically 5-10%", domain="construction")
      remember FactSheet(title="net-30 to net-90 terms are normal for subs", domain="construction")
      remember FactSheet(title="QuickBooks ACH transfer fee is 1%", domain="payments")
      let facts = lookup FactSheet where domain == "construction"
      return facts
    }
}
```
