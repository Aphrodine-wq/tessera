---
agent: TeamLead
capabilities_requested: [NetworkOut]
max_cost: { dollars: 0.00, tokens: 0 }
---

# Researcher (full) — notice + until + comparison ops

Same shape as RFC §12.3 but with our actually-shipped grammar:

- **`until rank_quality > threshold or steps > max_steps`** — bounded loop
- **`notice when quality < 0.5`** — predicate-triggered handler

```tsr:memory:episodic
episodic {
  event LowQuality(score: Int)
  event RaisedConcern(reason: String)
}
```

```tsr:agent
agent TeamLead {
  beliefs:
    @last_write topic: String

  intentions:
    plan iterate {
      let steps = 0
      let rank_quality = 0
      until rank_quality > 85 or steps > 5 {
        let steps = steps + 1
        let rank_quality = rank_quality + 30
      }
      return rank_quality
    }

  notice when rank_quality < 50 {
    log LowQuality(rank_quality)
    log RaisedConcern("quality below threshold")
  }
}
```
