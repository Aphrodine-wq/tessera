"""Theme B (close the loop) + Theme C (provenance boundary) tests.

Uses a stub gbnf-tier backend so it runs with no model. Asserts: execute=true
dispatches the tool and returns its result with audit called_via=wire; the
field→positional-args mapping follows ToolDecl.params order; a text-sourced call
is refused by default and allowed when opted in; the direct path is unchanged.
"""
from pathlib import Path

import pytest

from tessera.adapters import wire
from tessera.adapters.llm import CompletionResult
from tessera.interp.eval import (
    CalledVia,
    Refusal,
    World,
    _invoke_tool,
    run_agent,
)
from tessera.parser.module import parse_file
from tessera.sir.build import lower
from tessera.sir.nodes import AutonomyDecl, ToolDecl

EXAMPLE = str(Path(__file__).parent.parent / "examples" / "wire_tool_execute.t.md")

# The wire-sourced (constrained-decoding) path needs the standalone `tson`
# package; skip those tests when it is not installed.
requires_tson = pytest.mark.skipif(
    not wire.AVAILABLE, reason="tson not installed; constrained decoding unavailable"
)


class _GbnfStub:
    name = "llamacpp"
    cost_dollars = 0.0

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, **opts):
        return CompletionResult(text=self.text, backend="llamacpp", model="stub")


def test_example_compiles_with_execute_flag():
    module = lower(parse_file(EXAMPLE))
    p = module.prompts["weather_call"]
    assert p.emits == "get_weather" and p.execute is True


@requires_tson
def test_execute_dispatches_and_returns_tool_result(monkeypatch):
    module = lower(parse_file(EXAMPLE))
    stub = _GbnfStub("!get_weather #c1 location:Oxford units:f")
    monkeypatch.setattr("tessera.adapters.llm.get_backend", lambda *a, **k: stub)

    world = World(module=module)
    result = run_agent(module, "WeatherActor",
                       initial_beliefs={"place": "Oxford, MS"}, world=world)

    # The tool RAN and the plan bound its result (not the wire text).
    assert result == "Oxford: 72 (f)"
    assert any(e.action == "tool:get_weather" and e.detail.get("called_via") == "wire"
               for e in world.audit)


@requires_tson
def test_field_mapping_follows_param_order(monkeypatch):
    """Args are pulled in ToolDecl.params order, not record field order."""
    module = lower(parse_file(EXAMPLE))
    # record lists units BEFORE location; params are (location, units).
    stub = _GbnfStub("!get_weather #c1 units:c location:Tupelo")
    monkeypatch.setattr("tessera.adapters.llm.get_backend", lambda *a, **k: stub)
    result = run_agent(module, "WeatherActor", initial_beliefs={"place": "x"})
    assert result == "Tupelo: 72 (c)"  # location=Tupelo, units=c — order correct


def test_text_sourced_call_refused_by_default():
    module = lower(parse_file(EXAMPLE))
    tool = module.tools["get_weather"]
    world = World(module=module)
    out = _invoke_tool(tool, ["X", "f"], world,
                       called_via=CalledVia.TEXT, agent_name="A")
    assert isinstance(out, Refusal) and out.policy == "provenance"
    assert any(e.action == "policy_violation" and e.detail.get("called_via") == "text"
               for e in world.audit)


def test_text_call_allowed_when_opted_in():
    module = lower(parse_file(EXAMPLE))
    module.autonomy = AutonomyDecl(allow_text_calls=["get_weather"])
    tool = module.tools["get_weather"]
    world = World(module=module)
    out = _invoke_tool(tool, ["Oxford", "f"], world,
                       called_via=CalledVia.TEXT, agent_name="A")
    assert out == "Oxford: 72 (f)"  # ran, no refusal


def test_direct_tool_call_unchanged_and_tagged():
    module = lower(parse_file(EXAMPLE))
    tool = module.tools["get_weather"]
    world = World(module=module)
    out = _invoke_tool(tool, ["Oxford", "f"], world,
                       called_via=CalledVia.DIRECT, agent_name="A")
    assert out == "Oxford: 72 (f)"  # direct path passes the gate, runs normally


def test_autonomy_propose_blocks_direct_tool_call():
    """Closes the gap where `_invoke_tool` only ran provenance — a `propose`-
    level autonomy gate now blocks tool calls too, not just prompt calls."""
    module = lower(parse_file(EXAMPLE))
    module.autonomy = AutonomyDecl(level="propose", require_approval=["get_weather"])
    tool = module.tools["get_weather"]
    world = World(module=module)
    out = _invoke_tool(tool, ["Oxford", "f"], world,
                       called_via=CalledVia.DIRECT, agent_name="A")
    assert isinstance(out, Refusal) and out.policy == "autonomy"
    assert any(e.action == "approval_blocked:get_weather" for e in world.audit)


def test_autonomy_act_freely_allows_direct_tool_call():
    module = lower(parse_file(EXAMPLE))
    module.autonomy = AutonomyDecl(level="act_freely", require_approval=["get_weather"])
    tool = module.tools["get_weather"]
    world = World(module=module)
    out = _invoke_tool(tool, ["Oxford", "f"], world,
                       called_via=CalledVia.DIRECT, agent_name="A")
    assert out == "Oxford: 72 (f)"  # ran despite matching a require_approval term
    assert not any(e.action.startswith("approval_blocked") for e in world.audit)


def test_wire_dispatch_not_double_gated_by_autonomy():
    """A `wire` dispatch is a continuation of a prompt call already gated under
    `prompt:{name}` in `_call_prompt` — `_invoke_tool` must skip the tool-level
    autonomy check for it, or the same decision gets gated twice (and risks a
    false negative if the tool-name term doesn't match what the prompt text did)."""
    module = lower(parse_file(EXAMPLE))
    module.autonomy = AutonomyDecl(level="propose", require_approval=["get_weather"])
    tool = module.tools["get_weather"]
    world = World(module=module)
    out = _invoke_tool(tool, ["Oxford", "f"], world,
                       called_via=CalledVia.WIRE, agent_name="A")
    assert out == "Oxford: 72 (f)"  # wire path skips the tool-level gate, runs normally
    assert not any(e.action.startswith("approval_blocked") for e in world.audit)
