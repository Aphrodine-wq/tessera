"""Local CPU training for `trainable { ... }` neural models (decision 13).

Author declares an MLP with an attached trainable clause:

    model perception {
      linear in=4 out=8
      relu
      linear in=8 out=2
    } trainable {
      optimizer: adam(lr=1e-3)
      epochs: 50
      loss: mse
      batch_size: 16
    }

`tessera compile --train file.t.md` runs a PyTorch optimizer loop on the
declared model and writes a checkpoint to `~/.tessera/checkpoints/<name>.pt`.
At inference time, `tessera/adapters/torch::forward` (the existing path)
checks for that checkpoint and loads it before the forward pass.

MVP training corpus = synthetic random vectors matching the model's
inferred input/output shape. Real eval-driven training (eval cases as
the (x, y) pairs) is a follow-up — that wants the eval substrate
extended to carry numeric vectors.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT_DIR = Path.home() / ".tessera" / "checkpoints"
ENV_CHECKPOINT_DIR = "TESSERA_CHECKPOINTS_DIR"


def _resolve_checkpoint_dir() -> Path:
    env = os.environ.get(ENV_CHECKPOINT_DIR)
    return Path(env) if env else DEFAULT_CHECKPOINT_DIR


def checkpoint_path(model_name: str) -> Path:
    return _resolve_checkpoint_dir() / f"{model_name}.pt"


def _infer_io_dims(decl) -> tuple[int, int]:
    """Walk the model layers to infer first-linear input and last-linear output."""
    first_in = None
    last_out = None
    for layer in decl.layers:
        if layer.get("kind") == "linear":
            if first_in is None and "in" in layer:
                first_in = int(layer["in"])
            if "out" in layer:
                last_out = int(layer["out"])
    if first_in is None or last_out is None:
        raise RuntimeError(
            f"cannot infer I/O dimensions for model {decl.name!r} — declare at "
            "least one `linear in=N out=M` layer"
        )
    return first_in, last_out


def train_model(decl, *, n_samples: int = 128, seed: int = 0) -> Path:
    """Train the declared model on synthetic data, write a checkpoint.

    Returns the checkpoint path. Raises RuntimeError if torch isn't
    installed.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for trainable models. "
            "pip install tessera-lang[torch] to enable."
        ) from e

    from .adapters.torch import compile_model
    model = compile_model(decl)
    in_dim, out_dim = _infer_io_dims(decl)

    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_dim)
    # Synthetic linear target — a real (random) linear map from in_dim to out_dim
    # so the model has something learnable.
    W = torch.randn(in_dim, out_dim) * 0.5
    Y = X @ W

    if decl.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=decl.learning_rate)
    elif decl.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=decl.learning_rate)
    else:
        raise RuntimeError(f"unknown optimizer {decl.optimizer!r}")

    if decl.loss == "mse":
        loss_fn = nn.MSELoss()
    elif decl.loss == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise RuntimeError(f"unknown loss {decl.loss!r}")

    model.train()
    batch_size = max(1, min(decl.batch_size, n_samples))
    history: list[float] = []
    for epoch in range(decl.epochs):
        # Shuffle each epoch
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], Y[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append(epoch_loss / max(1, n_batches))

    ck_dir = _resolve_checkpoint_dir()
    ck_dir.mkdir(parents=True, exist_ok=True)
    path = ck_dir / f"{decl.name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "loss_history": history,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "optimizer": decl.optimizer,
        "learning_rate": decl.learning_rate,
        "epochs": decl.epochs,
        "loss": decl.loss,
    }, path)
    return path


def train_all_trainable(module) -> list[Path]:
    """Train every model with `trainable: True` in a module. Returns paths."""
    paths: list[Path] = []
    for decl in module.neural_models.values():
        if decl.trainable:
            paths.append(train_model(decl))
    return paths


def load_checkpoint_if_present(model, name: str) -> bool:
    """If a checkpoint exists for `name`, load it into `model` and return True."""
    path = checkpoint_path(name)
    if not path.exists():
        return False
    try:
        import torch
    except ImportError:
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    return True
