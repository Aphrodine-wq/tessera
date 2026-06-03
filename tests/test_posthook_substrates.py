"""End-to-end post-hook substrates: gricean, hindsight, argumentative.

These run AFTER generation (gricean scores the output, argumentative runs a
critic, hindsight reviews on plan completion). The algorithm modules are
unit-tested separately; this proves the interp wiring + the E900/E910/E920
verify lints.
"""
import os

from tessera.parser.module import parse_source
from tessera.sir.build import lower
from tessera.interp.eval import World, run_agent
from tessera.verify.passes import run_local

os.environ["TESSERA_LLM_BACKEND"] = "noop"


def _module(src: str):
    return lower(parse_source(src, path="<inline>"))


def _responder(*blocks: str, extra_prompts: str = "") -> str:
    body = "\n\n".join(blocks)
    return f"""---
agent: Responder
tessera_version: 0.2
---

{body}

```tsr:agent
agent Responder {{
  beliefs:
    @last_write q: String
  intentions:
    plan answer {{ return reply(q) }}
}}
```

```tsr:prompt
prompt reply(q: String) -> String = "{{q}}"
```
{extra_prompts}
"""


# ----- gricean -----

def test_gricean_gate_refuses_short_output():
    # noop echo is short; min_words 50 with quantity gated → refuse.
    src = _responder("```tsr:gricean\ngricean { min_words: 50 gate: [quantity] }\n```")
    world = World(module=_module(src))
    out = run_agent(world.module, "Responder", initial_beliefs={"q": "hi"},
                    world=world, concurrent=False)
    assert "gricean-refused" in out
    assert any(e.action.startswith("gricean:violation:quantity") for e in world.audit)


def test_gricean_warn_only_does_not_block():
    src = _responder("```tsr:gricean\ngricean { min_words: 50 }\n```")  # no gate
    world = World(module=_module(src))
    out = run_agent(world.module, "Responder", initial_beliefs={"q": "hi"},
                    world=world, concurrent=False)
    assert "gricean-refused" not in out
    assert any(e.action.startswith("gricean:violation") for e in world.audit)


# ----- argumentative -----

CRITIC_STRONG = '\n```tsr:prompt\nprompt challenge(c: String) -> String = "However this is false and incorrect: {c}"\n```'
CRITIC_WEAK = '\n```tsr:prompt\nprompt challenge(c: String) -> String = "Consider: {c}"\n```'


def test_argumentative_refuses_when_critic_strong():
    src = _responder(
        "```tsr:argumentative\nargumentative { critic: challenge accept_threshold: 0.5 proposer_confidence: 0.9 }\n```",
        extra_prompts=CRITIC_STRONG,
    )
    world = World(module=_module(src))
    out = run_agent(world.module, "Responder", initial_beliefs={"q": "claim"},
                    world=world, concurrent=False)
    assert "argumentative-refused" in out
    assert any(e.action.startswith("argumentative:downweight") for e in world.audit)


def test_argumentative_accepts_when_critic_weak():
    src = _responder(
        "```tsr:argumentative\nargumentative { critic: challenge accept_threshold: 0.5 proposer_confidence: 0.9 }\n```",
        extra_prompts=CRITIC_WEAK,
    )
    world = World(module=_module(src))
    out = run_agent(world.module, "Responder", initial_beliefs={"q": "claim"},
                    world=world, concurrent=False)
    assert "argumentative-refused" not in out
    counters = [e for e in world.audit if e.action.startswith("argumentative:counter")]
    assert counters  # the critic ran


# ----- hindsight -----

def test_hindsight_records_review_on_completion():
    src = _responder(
        "```tsr:ethics\nethics { principle honesty { weight: 0.9 rule: 'no fabrication' } }\n```",
        "```tsr:hindsight\nhindsight { enabled: true }\n```",
    )
    world = World(module=_module(src))
    run_agent(world.module, "Responder", initial_beliefs={"q": "x"},
              world=world, concurrent=False)
    reviews = [e for e in world.audit if e.action.startswith("hindsight:learning")]
    assert reviews
    # honesty was declared and (via the ethics preamble) applied on the prompt,
    # so it should not show up as missed.
    assert reviews[0].detail["intended_ethics_missed"] == []


# ----- verify-pass lints -----

def test_e900_gricean_quality_gate_without_evidence():
    src = _responder("```tsr:gricean\ngricean { gate: [quality] }\n```")
    diags = run_local(_module(src))
    assert any(d.code == "E900" for d in diags)


def test_e910_hindsight_without_ethics_or_intent():
    src = _responder("```tsr:hindsight\nhindsight { enabled: true }\n```")
    diags = run_local(_module(src))
    assert any(d.code == "E910" for d in diags)


def test_e920_argumentative_critic_undefined():
    src = _responder("```tsr:argumentative\nargumentative { critic: nope }\n```")
    diags = run_local(_module(src))
    assert any(d.code == "E920" and d.severity == "error" for d in diags)
