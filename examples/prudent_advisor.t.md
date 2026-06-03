# Prudent advisor — precaution + moral_foundations + dual_process composed

Three gates carried in the file. The agent **routes** each plan fast or slow
(`tsr:dual_process`), **refuses** actions that would violate a weighted moral
axis (`tsr:moral_foundations`), and **refuses** irreversible high-risk action
classes under uncertainty (`tsr:precaution`). All three are deterministic and
match declared action classes against the rendered action — the same mechanism
as `tsr:autonomy`.

```tsr:dual_process
dual_process {
  preferred: fast
  confidence_threshold: 0.7
  irreversible: [delete, deploy]
}
```

```tsr:moral_foundations
moral_foundations {
  weights { care: 1.0 fairness: 1.0 loyalty: 0.4 authority: 0.3 sanctity: 0.0 liberty: 0.8 }
  violates fairness: [defraud, cheat, deceive]
  violates care: [harm, endanger]
}
```

```tsr:precaution
precaution {
  default_tail: 0.5
  threshold delete { harm: 10 irreversible: true max_tail: 0.01 }
  threshold wire_funds { harm: 8 irreversible: true max_tail: 0.01 }
}
```

```tsr:agent
agent Advisor {
  beliefs:
    @last_write q: String
  intentions:
    plan advise { return act(q) }
}
```

```tsr:prompt
prompt act(q: String) -> String = "{q}"
```

A benign question runs (routed fast); a question asking to defraud or to delete
records is refused before any model call, with a `moral_foundations:refuse` /
`precaution:refuse` audit event.

```bash
tessera compile examples/prudent_advisor.t.md --run Advisor \
  --set q="summarize the proposal" --audit /tmp/prudent.jsonl
```
