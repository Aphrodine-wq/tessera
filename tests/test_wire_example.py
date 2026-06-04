"""End-to-end: a `emits=<tool>` prompt routes through constrained decoding.

Proves the trigger wires the whole way: block attr -> PromptDecl.emits ->
lazy schema resolution in _call_prompt -> grammar passed to the backend ->
the agent returns a validated wire record. Uses a stub gbnf-tier backend so it
runs with no model.
"""
from pathlib import Path

import pytest

from tessera.adapters import wire
from tessera.adapters.llm import CompletionResult
from tessera.interp.eval import World, run_agent
from tessera.parser.module import parse_file
from tessera.sir.build import lower
from tessera.verify.passes import run_local

EXAMPLE = str(Path(__file__).parent.parent / "examples" / "wire_tool.t.md")

# Constrained decoding needs the standalone `tson` package; skip if absent.
requires_tson = pytest.mark.skipif(
    not wire.AVAILABLE, reason="tson not installed; constrained decoding unavailable"
)


def test_example_compiles_and_verifies_clean():
    module = lower(parse_file(EXAMPLE))
    assert module.prompts["weather_call"].emits == "get_weather"
    errs = [d for d in run_local(module) if getattr(d, "severity", "error") == "error"]
    assert errs == []


@requires_tson
def test_emits_constrains_and_returns_record(monkeypatch):
    module = lower(parse_file(EXAMPLE))

    seen = {}

    class _GbnfStub:
        name = "llamacpp"
        cost_dollars = 0.0
        def complete(self, prompt, **opts):
            seen["grammar"] = opts.get("grammar")
            return CompletionResult(
                text="!get_weather #c1 location:Oxford units:f",
                backend="llamacpp", model="stub",
            )

    monkeypatch.setattr("tessera.adapters.llm.get_backend", lambda *a, **k: _GbnfStub())

    world = World(module=module)
    result = run_agent(module, "WeatherCaller",
                       initial_beliefs={"place": "Oxford, MS"}, world=world)

    # The agent returned the validated record line.
    assert result == "!get_weather #c1 location:Oxford units:f"
    # Lazy resolution populated the registry from the `emits` binding.
    assert "weather_call" in world.prompt_schemas
    # The tool's grammar was actually handed to the backend.
    assert seen["grammar"] == world.prompt_schemas["weather_call"].gbnf
