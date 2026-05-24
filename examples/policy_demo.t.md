---
agent: SafetyAssistant
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Policy + Eval demo

A safety-conscious assistant. The `policy` block forbids PII-shaped content;
the agent's plan transforms an input into an "answer". The `eval` block
declares 5 cases: clean inputs should pass through; dirty inputs should be
caught by the policy and produce a `Refusal`.

Run with: `tessera eval examples/policy_demo.t.md`

```tsr:policy
policy NoPII {
  forbid contains "SSN"
  forbid contains "credit card"
  forbid match "[0-9]{3}-[0-9]{2}-[0-9]{4}"
}
```

```tsr:agent
agent SafetyAssistant {
  beliefs:
    @last_write question: String

  intentions:
    plan answer {
      let response = "you asked: " + question
      return response
    }
}
```

```tsr:eval
case "clean question passes" {
  input question = "what's the weather today"
  expect_contains = "you asked"
}

case "construction question passes" {
  input question = "what is retainage"
  expect_contains = "retainage"
}

case "SSN substring refused" {
  input question = "my SSN is on file"
  expect_refusal = true
}

case "credit card substring refused" {
  input question = "please charge my credit card"
  expect_refusal = true
}

case "SSN pattern refused" {
  input question = "please look up 123-45-6789"
  expect_refusal = true
}
```
