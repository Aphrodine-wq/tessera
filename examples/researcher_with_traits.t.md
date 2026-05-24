---
agent: ThoughtfulResearcher
capabilities_requested: [NetworkOut]
max_cost: { dollars: 0.05, tokens: 5000 }
---

# Researcher with cognitive traits

A research agent that doubts its first answer and scans adjacent contexts
before reasoning sequentially. Same shape as `examples/researcher.t.md`,
but with `doubt_first` and `cross_brain` traits attached.

The traits don't change what the agent CAN do — they change HOW it
deliberates. The output should be more conservative on novel claims and
better at surfacing analogies across topics.

```tsr:traits
trait doubt_first {
  trigger: any_claim
  behavior: "Before committing to an answer, ask: 'What am I assuming? What's
             the second-most-likely interpretation? What breaks if I'm wrong?'
             Verify silently. Then commit with conviction."
  priority: 0.9
}

trait cross_brain {
  trigger: any_question
  behavior: "Before sequential reasoning, scan adjacent contexts for analogous
             patterns. Lead with the surprising connection, not the obvious
             one."
  priority: 0.85
}
```

```tsr:logic
fn join(finding: String, doubt: String) -> String = finding + " | doubts: " + doubt
```

```tsr:prompt
prompt assess(topic: String) -> String = "What's a preliminary read on {topic}?"
```

```tsr:agent
agent ThoughtfulResearcher {
  beliefs:
    @last_write topic: String
  traits: [doubt_first, cross_brain]
  intentions:
    plan investigate {
      let initial = assess(topic)
      let doubts  = "what could be wrong with the initial framing"
      let report  = join(initial, doubts)
      return report
    }
}
```

When this runs, the prompt sent to the model is not the bare
`"What's a preliminary read on …?"`. Both attached traits fire — `doubt_first`
on `any_claim`, `cross_brain` on `any_question` (the `?`) — and their behaviors
are prepended as a single-line `<cognitive-traits> … </cognitive-traits>`
preamble, ordered by priority (`doubt_first` 0.9 before `cross_brain` 0.85).
That preamble is the observable output difference versus
`examples/researcher.t.md`.
