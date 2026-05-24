---
agent: Perception
capabilities_requested: []
max_cost: { dollars: 0.00, tokens: 0 }
---

# Perception — neural substrate demo

A tiny classifier declared in `tsr:neural`, instantiated and called from an
agent. Demonstrates the `neural` substrate compiling to a real PyTorch
`nn.Sequential`. Run with:

```
pip install torch
tessera compile examples/perception.t.md --run Perception --set features="[0.1, 0.2, 0.3, 0.4]"
```

If torch isn't installed the verifier still passes; only the runtime call
into the model raises (with a clean message).

```tsr:neural
model classifier {
  linear in=4 out=8
  relu
  linear in=8 out=3
  softmax dim=0
}
```

```tsr:agent
agent Perception {
  beliefs:
    @last_write features: String

  intentions:
    plan predict {
      let logits = classifier(features)
      return logits
    }
}
```
