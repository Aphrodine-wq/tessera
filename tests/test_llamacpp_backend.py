"""Regression guard for the raw-completion / chat-template bug.

LlamaCppBackend and LlamaServerBackend used to hand the prompt straight to
llama.cpp's raw completion path, which never applies the GGUF's chat template.
An instruct model then treats the prompt as plain continuation and tends to
write out its own turn markers (``<|assistant|>``, ``<|im_start|>assistant``,
``[/INST]``, ...) as literal text instead of answering — silently tanking
unconstrained completion quality independent of any grammar constraint. This
is exactly the bug that deflated tson's grammar_yield bench control arm to
1/200 before it was traced and fixed (both here and in tson's bench scripts).

Gated on a real GGUF so it only runs when one is available; skips cleanly
otherwise so CI stays green.
"""
import os
import re

import pytest

llama_cpp = pytest.importorskip("llama_cpp")
GGUF = os.environ.get("TESSERA_WIRE_GGUF")
if not GGUF:
    pytest.skip("set TESSERA_WIRE_GGUF to a local .gguf to run", allow_module_level=True)

# Generic across chat-template families: ChatML/Phi-3-style <|...|> markers,
# Llama-2-style [INST]/[/INST]. A correctly templated completion should never
# surface any of these as literal output text.
_LEAKED_TURN_MARKER = re.compile(r"<\|\w+\|>|\[/?INST\]")


def test_llamacpp_backend_applies_chat_template():
    from tessera.adapters.llm import LlamaCppBackend

    backend = LlamaCppBackend(GGUF)
    result = backend.complete("Say hello in exactly three words.", max_tokens=32, temperature=0.2)

    assert result.text.strip(), "completion was empty"
    assert not _LEAKED_TURN_MARKER.search(result.text), (
        f"chat template not applied — turn marker leaked into completion: {result.text!r}"
    )
