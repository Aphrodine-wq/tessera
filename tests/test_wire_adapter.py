"""Tessera <-> tson integration (constrained decoding seam).

Proves the adapter bridges a ToolDecl to wire artifacts, selects the right
enforcement tier per backend, and that a schema-bound prompt routes through the
constrained path in _call_prompt (with a distinct cache key) while unbound
prompts are byte-identical to before.
"""
from dataclasses import dataclass

import pytest

from tessera.adapters import wire
from tessera.adapters.wire import (
    BACKEND_CAPS,
    compile_tool,
    constrain_opts,
    enforce_complete,
    schema_from_tool,
    tier_for,
)
from tessera.adapters.llm import CompletionResult
from tessera.sir.nodes import Module, PromptDecl, ToolDecl
from tessera.interp.eval import World, _call_prompt

# The constrained-decoding path needs the standalone `tson` package. The wire
# adapter degrades gracefully without it, so these tests skip rather than fail.
requires_tson = pytest.mark.skipif(
    not wire.AVAILABLE, reason="tson not installed; constrained decoding unavailable"
)


def _tool():
    return ToolDecl(
        name="get_weather",
        params=[("location", "String"), ("units", "String")],
        return_type="String",
        import_path="x",
    )


@dataclass
class _StubBackend:
    name: str
    text: str = ""
    cost_dollars: float = 0.0

    def complete(self, prompt, **opts):
        self.last_opts = opts
        return CompletionResult(text=self.text, backend=self.name, model="stub")


@requires_tson
def test_schema_from_tool_maps_params():
    s = schema_from_tool(_tool())
    assert s.name == "get_weather"
    assert [(f.name, f.type, f.required) for f in s.fields] == [
        ("location", "str", True),
        ("units", "str", True),
    ]


def test_tier_selection():
    assert tier_for(_StubBackend("llamacpp")) == "gbnf"
    assert tier_for(_StubBackend("llama_server")) == "gbnf"
    assert tier_for(_StubBackend("ollama")) == "jsonschema"
    assert tier_for(_StubBackend("anthropic")) == "none"
    assert tier_for(_StubBackend("totally_unknown")) == "none"


@requires_tson
def test_constrain_opts_per_tier():
    c = compile_tool(_tool())
    assert constrain_opts(c, _StubBackend("llamacpp")) == {"grammar": c.gbnf}
    assert constrain_opts(c, _StubBackend("ollama")) == {"format": c.json_schema}
    assert constrain_opts(c, _StubBackend("anthropic")) == {}


def test_backend_caps_known_backends_covered():
    for name in ("llamacpp", "llama_server", "ollama", "openai_compat",
                 "noop", "anthropic", "gemini", "cohere", "bedrock"):
        assert name in BACKEND_CAPS


@requires_tson
def test_enforce_complete_constrained_tier_validates():
    c = compile_tool(_tool())
    backend = _StubBackend("llamacpp", text='!get_weather #c1 location:Paris units:f')
    out = enforce_complete(backend, "prompt", c)
    assert out.text == "!get_weather #c1 location:Paris units:f"
    assert backend.last_opts.get("grammar") == c.gbnf  # grammar was passed


@requires_tson
def test_enforce_complete_none_tier_repairs_once():
    c = compile_tool(_tool())
    # First reply is garbage; the repair re-ask returns a valid record.
    class _TwoShot:
        name = "anthropic"
        cost_dollars = 0.0
        def __init__(self):
            self.calls = 0
        def complete(self, prompt, **opts):
            self.calls += 1
            text = ("not a record" if self.calls == 1
                    else "!get_weather #c1 location:Paris units:c")
            return CompletionResult(text=text, backend="anthropic", model="stub")
    b = _TwoShot()
    out = enforce_complete(b, "prompt", c)
    assert b.calls == 2  # one repair round happened
    assert out.text == "!get_weather #c1 location:Paris units:c"


@requires_tson
def test_bound_prompt_uses_distinct_cache_key(monkeypatch):
    """A schema-bound prompt folds the grammar hash into its cache key, so it
    never collides with the same prompt text unconstrained."""
    import tessera.cache as cache

    keys: list[str] = []
    monkeypatch.setattr(cache, "semantic_cache_lookup", lambda k, **kw: keys.append(k) or None)
    monkeypatch.setattr(cache, "semantic_cache_put", lambda k, *a, **kw: None)
    monkeypatch.setattr("tessera.adapters.llm.get_backend",
                        lambda *a, **kw: _StubBackend("noop", text="[noop]"))

    w = World(module=Module(name="t"))
    p = PromptDecl(name="greet", params=[], return_type="str", template="hello")

    _call_prompt(p, [], w)                       # unbound
    w.prompt_schemas["greet"] = compile_tool(_tool())
    _call_prompt(p, [], w)                        # bound

    assert len(keys) == 2
    assert "\x00grammar:" not in keys[0]
    assert "\x00grammar:" in keys[1]
