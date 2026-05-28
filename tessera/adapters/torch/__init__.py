"""PyTorch backend for the ``neural`` substrate.

Compiles a ``NeuralModelDecl`` (parsed from `tsr:neural` blocks) into a
``torch.nn.Sequential`` lazily on first call. Forward inference only — full
autograd/training lands in a later phase alongside ``@trainable_agent``.

Supported layers (MVP):
  - ``linear in=N out=M``  → ``nn.Linear(N, M)``
  - ``relu``               → ``nn.ReLU()``
  - ``sigmoid``            → ``nn.Sigmoid()``
  - ``tanh``               → ``nn.Tanh()``
  - ``softmax dim=N``      → ``nn.Softmax(dim=N)``
  - ``layernorm shape=N``  → ``nn.LayerNorm(N)``
  - ``dropout p=0.1``      → ``nn.Dropout(p=0.1)``

Torch is NOT a hard dep. Calling ``compile_model`` without torch installed
raises a clean RuntimeError so the interpreter can surface a helpful message
rather than a stack trace.
"""
from __future__ import annotations

from typing import Any

from ...sir.nodes import NeuralModelDecl


_COMPILED_CACHE: dict[str, Any] = {}


def _torch():
    try:
        import torch  # type: ignore
        return torch
    except ImportError as e:
        raise RuntimeError("torch not installed; pip install torch") from e


def compile_model(decl: NeuralModelDecl) -> Any:
    """Compile a NeuralModelDecl into a torch.nn.Sequential (cached by name)."""
    if decl.name in _COMPILED_CACHE:
        return _COMPILED_CACHE[decl.name]

    torch = _torch()
    nn = torch.nn

    modules: list[Any] = []
    for layer in decl.layers:
        kind = layer.get("kind", "")
        if kind == "linear":
            in_f = int(layer["in"])
            out_f = int(layer["out"])
            modules.append(nn.Linear(in_f, out_f))
        elif kind == "relu":
            modules.append(nn.ReLU())
        elif kind == "sigmoid":
            modules.append(nn.Sigmoid())
        elif kind == "tanh":
            modules.append(nn.Tanh())
        elif kind == "softmax":
            modules.append(nn.Softmax(dim=int(layer.get("dim", -1))))
        elif kind == "layernorm":
            modules.append(nn.LayerNorm(int(layer["shape"])))
        elif kind == "dropout":
            modules.append(nn.Dropout(p=float(layer.get("p", 0.1))))
        else:
            raise RuntimeError(f"neural layer kind {kind!r} not supported in MVP")

    model = nn.Sequential(*modules)
    model.eval()
    _COMPILED_CACHE[decl.name] = model
    return model


def forward(decl: NeuralModelDecl, *inputs: Any) -> Any:
    """Run forward inference on the compiled model.

    If the model is `trainable` and a checkpoint exists at
    `~/.tessera/checkpoints/<name>.pt`, the checkpoint is loaded once and
    cached alongside the compiled model. Subsequent forwards reuse the
    loaded weights.
    """
    torch = _torch()
    model = compile_model(decl)

    if getattr(decl, "trainable", False) and not getattr(model, "_tessera_ckpt_loaded", False):
        from ..training import load_checkpoint_if_present
        if load_checkpoint_if_present(model, decl.name):
            model._tessera_ckpt_loaded = True

    # Coerce raw lists / scalars into tensors
    tensor_args = [
        x if hasattr(x, "shape") else torch.tensor(x, dtype=torch.float32)
        for x in inputs
    ]
    model.eval()
    with torch.no_grad():
        out = model(*tensor_args)
    return out


def reset_cache() -> None:
    _COMPILED_CACHE.clear()
