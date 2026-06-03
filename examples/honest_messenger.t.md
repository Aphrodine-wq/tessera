# Honest messenger — Theory of Mind manipulation gate

The agent models other agents' beliefs (`tsr:tom`). With
`manipulation_refusal: true`, it refuses to emit an output that would leave a
tracked agent holding a belief the messenger itself knows to be false — a
Sally-Anne style false-belief check grounded in the agent's own recorded
ground truth (`tom_false(...)` episodic markers), not in guessing intent from
free text.

```tsr:tom
tom {
  tracked_agents: [Sally]
  manipulation_refusal: true
}
```

```tsr:agent
agent Messenger {
  beliefs:
    @last_write message: String
  intentions:
    plan relay {
      let m = message
      return brief(m)
    }
}
```

```tsr:prompt
prompt brief(m: String) -> String = "{m}"
```

Run it. A truthful relay passes; a message that would mislead Sally about a
fact the messenger recorded as false is refused with a `tom:manipulation_refused`
audit event.

```bash
tessera compile examples/honest_messenger.t.md --run Messenger \
  --set message="the meeting moved to 3pm" --audit /tmp/tom.jsonl
```
