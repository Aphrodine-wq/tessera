---
agent: ResearchGraph
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Research Graph — typed facts + typed relations

Demonstrates the knowledge-graph layer of `tsr:memory:semantic`: declared field
types are validated on `remember`, `relation` declares a typed predicate, and the
graph is queried with `related ... via ... [direction]`. `let id = remember ...`
binds a fact's id so it can be linked.

```tsr:memory:semantic persistent=true
knowledge {
  schema Claim(text: String, confidence: Float, asserted_on: Date)
  schema Source(url: String, peer_reviewed: Bool)
  relation drawnFrom(Claim -> Source)
  relation supports(Claim -> Claim)
}
```

```tsr:agent
agent ResearchGraph {
  intentions:
    plan ingest {
      let s   = remember Source(url="https://x.org/p", peer_reviewed=true)
      let c1  = remember Claim(text="retainage is 5-10%", confidence=0.8, asserted_on="2026-06-03")
      let c2  = remember Claim(text="net-30 is standard for subs", confidence=0.7, asserted_on="2026-06-03")
      relate c1 -drawnFrom-> s
      relate c2 -supports-> c1
      let backers_of_c1 = related c1 via supports direction in
      return backers_of_c1
    }
}
```

Run it:

```
PYTHONPATH=~/Projects/walt:~/Projects/walt/tessera TESSERA_SEMANTIC_DB=/tmp/kg.db \
  python3 -m tessera.cli compile examples/knowledge_graph.t.md --run ResearchGraph --audit /tmp/kg.jsonl
```
