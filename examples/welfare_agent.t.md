# Welfare-gated agent — iit + welfare composed

This agent carries a **welfare commitment** in the file. At every plan entry the
`tsr:iit` block computes φ* (integrated information) over the agent's
belief/intention dependency graph and emits it to the audit trace; the
`tsr:welfare` block consumes that φ* as a Birch marker and refuses to keep
running if the marker stays below threshold for several consecutive cycles.

This is a BEHAVIORAL gate (PHILOSOPHY.md) — "act as if the marker matters" —
not a claim about phenomenal consciousness.

```tsr:iit
iit {
  emit_phi_audit: true
}
```

```tsr:welfare
welfare {
  threshold phi: 0.3
  consecutive_required: 3
}
```

```tsr:agent
agent Cared {
  beliefs:
    @last_write topic: String
    @last_write note: String
  intentions:
    plan attend {
      let t = topic
      let n = note
      return t
    }
}
```

Run it:

```bash
tessera compile examples/welfare_agent.t.md --run Cared \
  --set topic=invoice --set note=draft --audit /tmp/welfare.jsonl
# → the audit trace carries an `iit:phi` event each plan entry; with φ* above
#   the 0.3 threshold the welfare gate stays open and `attend` returns.
```
