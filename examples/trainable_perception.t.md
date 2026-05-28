---
agent: Perception
tessera_version: 0.2
capabilities_requested: []
max_cost: { dollars: 0.05, tokens: 0 }
---

# Trainable Perception — `model X { ... } trainable { ... }` (decision 13)

A neural model declared with an attached `trainable` clause. Running
`tessera compile examples/trainable_perception.t.md --train` invokes
the optimizer (Adam, lr=1e-3, 20 epochs over synthetic data matching
the model's in=4 out=2 shape) and writes a checkpoint to
`~/.tessera/checkpoints/classifier.pt`.

At inference time, `adapters/torch.forward` checks for that checkpoint
and loads it — subsequent runs use the trained weights, not the random
init.

```
tessera compile examples/trainable_perception.t.md --train
# trained checkpoint → ~/.tessera/checkpoints/classifier.pt
```

The training corpus today is synthetic (random vectors with a known
linear target) — proves the optimizer + checkpoint cycle. Real eval-
driven training lands when the eval substrate is extended to carry
numeric (input, output) pairs.

```tsr:neural
model classifier {
  linear in=4 out=8
  relu
  linear in=8 out=2
} trainable {
  optimizer: adam(lr=0.005)
  epochs: 20
  loss: mse
  batch_size: 16
}
```

```tsr:agent
agent Perception {
  beliefs:
    @last_write x: String
  intentions:
    plan p {
      return "perception agent — see --train mode"
    }
}
```
