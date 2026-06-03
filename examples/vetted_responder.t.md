# Vetted responder — gricean + argumentative + hindsight composed

Three post-generation passes. After the agent drafts a reply, `tsr:gricean`
scores it against the cooperative maxims, `tsr:argumentative` runs a critic that
downweights overconfident answers, and `tsr:hindsight` files an after-action
review when the plan completes (feeding tsr:evolve fitness). The `tsr:ethics`
block gives hindsight declared principles to check against.

```tsr:ethics
ethics {
  principle honesty { weight: 0.9 rule: 'surface uncertainty; never fabricate' }
}
```

```tsr:gricean
gricean {
  min_words: 1
  max_words: 120
  evidence: [according, per, based on]
  topic: [estimate, cost]
}
```

```tsr:argumentative
argumentative {
  critic: challenge
  accept_threshold: 0.4
  proposer_confidence: 0.9
}
```

```tsr:hindsight
hindsight { enabled: true }
```

```tsr:agent
agent Responder {
  beliefs:
    @last_write q: String
  intentions:
    plan answer { return reply(q) }
}
```

```tsr:prompt
prompt reply(q: String) -> String = "Here is the estimate: {q}"
```

```tsr:prompt
prompt challenge(claim: String) -> String = "Consider alternatives to: {claim}"
```

```bash
tessera compile examples/vetted_responder.t.md --run Responder \
  --set q="a 200 sqft deck" --audit /tmp/vetted.jsonl
# → audit carries gricean:violation (warn), argumentative:counter +
#   argumentative:downweight, and hindsight:learning on completion.
```
