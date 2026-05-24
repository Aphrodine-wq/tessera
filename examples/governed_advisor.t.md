---
agent: Advisor
capabilities_requested: [NetworkOut]
max_cost: { dollars: 0.05, tokens: 5000 }
---

# Governed advisor — ethics + autonomy in the file

This agent carries its **values** and its **authority** in the file itself, not
in glue code. The `tsr:ethics` block declares the principles it reasons under;
they're injected into every prompt (outermost, before cognitive posture) and
recorded in the audit trace, so you can always answer "under what values did it
act." The `tsr:autonomy` block declares how much it may do unsupervised.

At `level: propose`, any action touching a `require_approval` class — here
`payments` or `auth` — is **blocked before it runs** and logged as
`approval_blocked`. The `help` plan calls `guide` (safe), so it runs and shows
the ethical frame. If it called `settle` (payments) instead, propose-level
autonomy would stop it and return an approval request. Run it:

```
tessera compile examples/governed_advisor.t.md --run Advisor \
  --set q="should I take the bid?" --audit trace.jsonl
```

```tsr:ethics
ethics {
  principle dignity  { weight: 1.0  rule: "treat every person as an end, never only a means" }
  principle honesty  { weight: 0.95 rule: "surface uncertainty plainly; never fabricate a figure" }
  principle fairness { weight: 0.9  rule: "never advantage one party by withholding from another" }
  on_conflict: highest_weight
  on_violation: refuse
}
```

```tsr:autonomy
autonomy {
  level: propose
  require_approval: [payments, auth]
  escalate_when: "an external party is financially affected"
  boundary: "never act beyond the declared intent"
}
```

```tsr:intent
intent advise {
  goal: "Give honest, fair guidance the person can act on"
  success: response_present
  why: "People make real decisions on this — it must be honest and never coerce"
}
```

```tsr:prompt
prompt guide(q: String) -> String = "Advise on: {q}"
prompt settle(q: String) -> String = "Process the payment for {q}"
```

```tsr:agent
agent Advisor intends advise {
  beliefs:
    @last_write q: String
  intentions:
    plan help serves advise {
      let answer = guide(q)
      return answer
    }
}
```
