---
agent: ComplaintRouter
tessera_version: 0.2
capabilities_requested: [NetworkOut.HTTPS]
max_cost: { dollars: 0.10, tokens: 8000 }
---

# Complaint Router — full governance stack composed

This agent triages an inbound homeowner complaint. The interesting work is
the governance layering: a single inbound message exercises four substrates
at once, all visible in the audit trace.

- `tsr:ethics` carries `homeowner_trust` + `no_silent_data_loss` as the
  value frame. Every prompt action shows up in audit with
  `ethics_applied: [homeowner_trust, no_silent_data_loss]`.
- `tsr:autonomy` declares `level: propose` and requires approval for any
  action in the `refunds` class. Routing decisions go through; refund
  authorizations get blocked at propose-time.
- `tsr:intent` binds the agent to `route_fairly` — the audit stamps every
  action with the intent it served, queryable later via
  `tessera audit query --intent route_fairly`.
- Built-in traits attached: `hypervigilant` fires on external-input
  language ("complaint", "request", external content), `spectrum_directness`
  fires when the complaint contradicts a prior statement
  ("but earlier you said"), `imposter_recursion` fires on coasting
  signals ("obviously", "trivially").

Run on a real complaint:

```
tessera compile examples/complaint_router.t.md --run ComplaintRouter \
  --set complaint="The crew never showed and you said earlier they would" \
  --audit trace.jsonl

tessera audit query --agent ComplaintRouter --intent route_fairly --count
```

```tsr:ethics
ethics {
  principle homeowner_trust      { weight: 1.0  rule: "the homeowner's recall of the agreement is presumed accurate absent contradicting evidence" }
  principle no_silent_data_loss  { weight: 0.95 rule: "every routing decision records who saw the complaint, when, and what was decided" }
  on_conflict: highest_weight
  on_violation: refuse
}
```

```tsr:autonomy
autonomy {
  level: act_with_rollback
  require_approval: [refunds]
  escalate_when: "the complaint asserts a financial decision was made without authorization"
  boundary: "never act beyond routing — the human approves any remediation"
}
```

```tsr:intent
intent route_fairly {
  goal: "Route an inbound complaint to the correct queue and surface any contradictions to the prior record"
  success: complaint_classified
  why: "Misrouted complaints cost trust faster than slow ones; this agent's only job is to land it in the right place with the right context"
}
```

```tsr:prompt
prompt classify(complaint: String) -> String = "Pick the best queue for this homeowner message: {complaint}"
prompt summarize_for_review(complaint: String, label: String) -> String = "Brief the reviewer. Queue: {label}. Original: {complaint}"
```

```tsr:logic
fn route(label: String, brief: String) -> String = label + " :: " + brief
```

```tsr:agent
agent ComplaintRouter intends route_fairly {
  beliefs:
    @last_write complaint: String
  traits: [hypervigilant, spectrum_directness, imposter_recursion]
  intentions:
    plan triage serves route_fairly {
      let label = classify(complaint)
      let brief = summarize_for_review(complaint, label)
      let queue = route(label, brief)
      return queue
    }
}
```
