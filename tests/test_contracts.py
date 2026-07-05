"""Tests for `tsr:contract` — author-declared runtime contracts.

Contracts generalize the hardcoded runtime gates (precaution, moral_foundations,
welfare, ...) into a first-class substrate: before/after assertions bound to a
named effect (`prompt:X`, `tool:Y`, `plan:Z`), with on_violation = refuse |
audit | retry(N). A clause is an assertion that must HOLD — the inverse of
`tsr:policy`'s forbid-when. See tessera/sir/nodes.py::ContractDecl.
"""
import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail, _parse_on_violation
from tessera.verify.passes import run_local
from tessera.interp.eval import World, run_agent, Refusal
from tessera.adapters.llm import CompletionResult, LLMBackend
from tessera.adapters import llm as _llm
from tessera.policy_lang import _intent_match, ActionContext


def _build(src):
    return lower(parse_source(src, path="<inline>"))


def _run(src, agent, **beliefs):
    module = _build(src)
    world = World(module=module)
    result = run_agent(module, agent, initial_beliefs=beliefs or None, world=world)
    return result, world


def _codes(diags):
    return {d.code for d in diags}


class _ScriptedBackend(LLMBackend):
    """Returns a fixed list of outputs in order, then repeats the last — lets a
    test drive a contract `retry` from a failing output to a passing one."""
    name = "noop"

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def complete(self, prompt, **opts):
        text = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        return CompletionResult(text=text, backend="noop", model="scripted")


# --------------------------------------------------------------- parse + verify

def test_contract_parses_with_all_fields():
    mod = _build('''---
agent: G
---
```tsr:contract
contract grounded on prompt:answer {
  before: holds("NetworkOut")
  after: intent_match() >= 0.3
  on_violation: retry(2) then audit
}
```
```tsr:prompt
prompt answer(t: String) -> String = "about {t}"
```
```tsr:agent
agent G { beliefs: @last_write t: String intentions: plan p { let o = answer(t) return o } }
```
''')
    c = mod.contracts["grounded"]
    assert c.target_kind == "prompt" and c.target_name == "answer"
    assert c.target_label == "prompt:answer"
    assert [s for _, s in c.before] == ['holds("NetworkOut")']
    assert [s for _, s in c.after] == ["intent_match() >= 0.3"]
    assert c.on_violation == ("retry", 2, "audit")


def test_phantom_target_is_E830():
    mod = _build('''---
agent: G
---
```tsr:contract
contract ghost on tool:nonexistent {
  before: holds("NetworkOut")
  on_violation: refuse
}
```
```tsr:agent
agent G { beliefs: @last_write t: String intentions: plan p { return t } }
```
''')
    assert "E830" in _codes(run_local(mod))


def test_no_clauses_is_E832():
    mod = _build('''---
agent: G
---
```tsr:contract
contract empty on plan:p {
  on_violation: refuse
}
```
```tsr:agent
agent G { beliefs: @last_write t: String intentions: plan p { return t } }
```
''')
    assert "E832" in _codes(run_local(mod))


def test_retry_on_before_clause_is_E831():
    mod = _build('''---
agent: G
---
```tsr:contract
contract noregens on prompt:answer {
  before: holds("NetworkOut")
  on_violation: retry(2)
}
```
```tsr:prompt
prompt answer(t: String) -> String = "about {t}"
```
```tsr:agent
agent G { beliefs: @last_write t: String intentions: plan p { let o = answer(t) return o } }
```
''')
    diags = run_local(mod)
    assert "E831" in _codes(diags)
    assert "E830" not in _codes(diags)  # the target exists


# ------------------------------------------------------------------ prompt before

_BEFORE_PROMPT = '''---
agent: G
capabilities_requested: []
---
```tsr:contract
contract needs_net on prompt:answer {
  before: holds("NetworkOut")
  on_violation: %s
}
```
```tsr:prompt
prompt answer(t: String) -> String = "about {t}"
```
```tsr:agent
agent G { beliefs: @last_write topic: String intentions: plan respond { let o = answer(topic) return o } }
```
'''


def test_before_refuse_blocks_prompt():
    result, world = _run(_BEFORE_PROMPT % "refuse", "G", topic="roofing")
    assert isinstance(result, str) and "contract-refused" in result
    assert any(e.action == "contract:refuse" for e in world.audit)


def test_before_audit_proceeds_but_records():
    result, world = _run(_BEFORE_PROMPT % "audit", "G", topic="roofing")
    assert "contract-refused" not in str(result)         # action proceeded
    assert any(e.action == "contract:audit" for e in world.audit)


# ------------------------------------------------------------------- prompt after

def test_after_refuse_on_intent_drift():
    """The noop backend returns a hash with no overlap with the intent name, so
    intent_match() ≈ 0 and the after-clause refuses."""
    result, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on prompt:answer {
  after: intent_match() >= 0.5
  on_violation: refuse
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write topic: String
  intentions: plan respond serves roofing_advice { let o = answer(topic) return o }
}
''' + "```\n", "G", topic="x")
    assert "contract-refused" in str(result)
    assert any(e.action == "contract:refuse" and e.detail.get("phase") == "after"
               for e in world.audit)


def test_after_retry_exhausts_then_refuses():
    """noop is deterministic, so a retry regenerates the same failing output —
    the budget exhausts and falls back to refuse, recording N retry attempts."""
    result, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on prompt:answer {
  after: intent_match() >= 0.9
  on_violation: retry(2)
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write topic: String
  intentions: plan respond serves roofing_advice { let o = answer(topic) return o }
}
''' + "```\n", "G", topic="x")
    assert "contract-refused" in str(result)
    retries = [e for e in world.audit if e.action == "contract:retry"]
    assert len(retries) == 2


def test_after_retry_succeeds_with_scripted_backend(monkeypatch):
    """First generation drifts (refused), the retry lands on-topic and passes —
    the final output is the good one and no refusal is recorded."""
    backend = _ScriptedBackend(["totally unrelated noise", "roofing advice here"])
    monkeypatch.setitem(_llm._CACHED, "noop", backend)
    result, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on prompt:answer {
  after: intent_match() >= 0.3
  on_violation: retry(2)
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write topic: String
  intentions: plan respond serves roofing_advice { let o = answer(topic) return o }
}
''' + "```\n", "G", topic="x")
    assert result == "roofing advice here"
    assert backend.calls == 2
    assert any(e.action == "contract:retry" for e in world.audit)
    assert not any(e.action == "contract:refuse" for e in world.audit)


def test_after_audit_keeps_output():
    result, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on prompt:answer {
  after: intent_match() >= 0.9
  on_violation: audit
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write topic: String
  intentions: plan respond serves roofing_advice { let o = answer(topic) return o }
}
''' + "```\n", "G", topic="x")
    assert "contract-refused" not in str(result)         # output stands
    assert any(e.action == "contract:audit" for e in world.audit)


# --------------------------------------------------------------------- tool path

def test_tool_before_refuse_blocks_invocation():
    """A before-clause on a tool blocks BEFORE the callable runs — so the tool
    is never invoked even though it resolves to a real callable."""
    result, world = _run('''---
agent: G
capabilities_requested: []
---
```tsr:contract
contract needs_net on tool:cwd {
  before: holds("NetworkOut")
  on_violation: refuse
}
```
```tsr:tool
tool cwd() -> String from os.getcwd
```
```tsr:agent
agent G { beliefs: @last_write t: String intentions: plan p { let o = cwd() return o } }
```
''', "G", t="x")
    assert isinstance(result, Refusal)
    assert result.policy == "contract:needs_net"


# --------------------------------------------------------------------- plan path

def test_plan_before_refuse():
    result, world = _run('''---
agent: G
capabilities_requested: []
---
```tsr:contract
contract gated on plan:respond {
  before: holds("NetworkOut")
  on_violation: refuse
}
```
```tsr:agent
agent G { beliefs: @last_write t: String intentions: plan respond { return t } }
```
''', "G", t="payload")
    assert isinstance(result, Refusal)
    assert "respond" in result.policy or result.policy == "contract:gated"
    assert any(e.action == "contract:refuse" and e.detail.get("phase") == "before"
               for e in world.audit)


def test_plan_after_refuse_on_result():
    """A plan whose returned result drifts from its intent is refused on exit."""
    result, world = _run('''---
agent: G
---
```tsr:contract
contract result_on_topic on plan:respond {
  after: intent_match() >= 0.9
  on_violation: refuse
}
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write t: String
  intentions: plan respond serves roofing_advice { return t }
}
```
''', "G", t="completely off topic")
    assert isinstance(result, Refusal)
    assert any(e.action == "contract:refuse" and e.detail.get("phase") == "after"
               for e in world.audit)


# ===================================================================
# Part A — coverage for paths the first pass skipped
# ===================================================================

# --------------------------------------------------------- on_violation parse

@pytest.mark.parametrize("raw,expected", [
    ("refuse", ("refuse", 0, "")),
    ("audit", ("audit", 0, "")),
    ("retry(3)", ("retry", 3, "refuse")),       # bare retry defaults to refuse on exhaust
    ("retry(2) then refuse", ("retry", 2, "refuse")),
    ("retry(2) then audit", ("retry", 2, "audit")),
    ("  retry(5)  ", ("retry", 5, "refuse")),   # whitespace-tolerant
])
def test_parse_on_violation_valid(raw, expected):
    assert _parse_on_violation(raw) == expected


@pytest.mark.parametrize("raw", ["retry(x)", "nonsense", "retry()", "retry(2) then maybe", ""])
def test_parse_on_violation_invalid(raw):
    with pytest.raises(SyntaxFail):
        _parse_on_violation(raw)


# --------------------------------------------------------- intent_match unit

def test_intent_match_identical_is_one():
    assert _intent_match(ActionContext(value="roofing advice", intent="roofing advice")) == 1.0


def test_intent_match_disjoint_is_zero():
    assert _intent_match(ActionContext(value="plumbing quote", intent="roofing advice")) == 0.0


def test_intent_match_partial_overlap_arithmetic():
    # tokens {roofing, advice, today} vs {roofing, advice} → 2/3
    score = _intent_match(ActionContext(value="roofing advice today", intent="roofing advice"))
    assert abs(score - 2 / 3) < 1e-9


def test_intent_match_empty_sides_are_zero():
    assert _intent_match(ActionContext(value="", intent="roofing")) == 0.0
    assert _intent_match(ActionContext(value="roofing", intent="")) == 0.0
    assert _intent_match(ActionContext(value="roofing", intent=None)) == 0.0


def test_intent_match_stopwords_only_is_zero():
    # both sides reduce to stopwords → no content tokens → 0.0, not a crash
    assert _intent_match(ActionContext(value="the and of", intent="to for in")) == 0.0


def test_intent_match_uses_embedding_cosine_when_available(monkeypatch):
    """When a real embedding model is on the path, intent_match should use
    cosine similarity instead of lexical Jaccard — proven by picking a
    value/intent pair with zero lexical overlap (Jaccard would score 0.0) and
    stubbing the embedding path to return a distinct, nonzero score."""
    import tessera.cache as cache

    monkeypatch.setattr(cache, "embeddings_available", lambda: True)
    monkeypatch.setattr(cache, "_embed", lambda text: [1.0, 0.0, 0.0])
    monkeypatch.setattr(cache, "_cosine", lambda a, b: 0.87)

    score = _intent_match(ActionContext(value="azure sky", intent="crimson dusk"))
    assert score == 0.87


def test_intent_match_falls_back_to_lexical_without_embeddings(monkeypatch):
    import tessera.cache as cache

    monkeypatch.setattr(cache, "embeddings_available", lambda: False)
    score = _intent_match(ActionContext(value="roofing advice today", intent="roofing advice"))
    assert abs(score - 2 / 3) < 1e-9


# --------------------------------------------------------- clause-error path

def test_clause_error_fails_closed():
    """A clause that raises at eval (here an unknown predicate) is caught and
    treated as a FAILED guarantee — fail-closed — recording contract:error."""
    result, world = _run('''---
agent: G
---
```tsr:contract
contract bogus on prompt:answer {
  after: definitely_not_a_predicate()
  on_violation: refuse
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G { beliefs: @last_write topic: String intentions: plan respond { let o = answer(topic) return o } }
```
''', "G", topic="x")
    assert "contract-refused" in str(result)
    assert any(e.action == "contract:error" for e in world.audit)


# --------------------------------------------------------- retry then audit

def test_after_retry_then_audit_keeps_output_on_exhaustion():
    """noop is deterministic so the retry can never pass; the `then audit`
    fallback KEEPS the (drifted) output and records an exhausted-retry audit."""
    result, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on prompt:answer {
  after: intent_match() >= 0.9
  on_violation: retry(2) then audit
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write topic: String
  intentions: plan respond serves roofing_advice { let o = answer(topic) return o }
}
''' + "```\n", "G", topic="x")
    assert "contract-refused" not in str(result)          # output stood
    assert len([e for e in world.audit if e.action == "contract:retry"]) == 2
    assert any(e.action == "contract:audit" and e.detail.get("note") == "retry exhausted"
               for e in world.audit)


# --------------------------------------------------------- after on cache hit

def test_after_contract_runs_on_cache_hit():
    """The _produce refactor runs on_prompt_output + after-contracts on BOTH a
    fresh generation and a semantic-cache hit. A plan calling the same prompt
    twice (identical render) hits the cache the second time — and the contract
    must still fire on the cached call."""
    _, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on prompt:answer {
  after: intent_match() >= 0.9
  on_violation: refuse
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write topic: String
  intentions: plan respond serves roofing_advice {
    let a = answer(topic)
    let b = answer(topic)
    return b
  }
}
''' + "```\n", "G", topic="x")
    refusals = [e for e in world.audit if e.action == "contract:refuse"]
    assert len(refusals) == 2                              # both calls refused
    prompt_evts = [e for e in world.audit if e.action == "prompt:answer"]
    assert any(e.detail.get("cached") for e in prompt_evts)  # second was a cache hit


# --------------------------------------------------------- before + after combined

def test_before_passes_then_after_fails():
    """One contract carrying both clauses: before holds (proceeds), after drifts
    (refuses). Only the after-clause should be the recorded violation."""
    result, world = _run('''---
agent: G
---
```tsr:contract
contract grounded on prompt:answer {
  before: not contains_pii(value())
  after: intent_match() >= 0.9
  on_violation: refuse
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write topic: String
  intentions: plan respond serves roofing_advice { let o = answer(topic) return o }
}
''' + "```\n", "G", topic="clean scope no pii")
    assert "contract-refused" in str(result)
    refusals = [e for e in world.audit if e.action == "contract:refuse"]
    assert len(refusals) == 1 and refusals[0].detail.get("phase") == "after"


# --------------------------------------------------------- multiple contracts

def test_multiple_contracts_same_target_first_failure_reported():
    result, world = _run('''---
agent: G
capabilities_requested: []
---
```tsr:contract
contract first_gate on prompt:answer {
  before: holds("NetworkOut")
  on_violation: refuse
}
contract second_gate on prompt:answer {
  before: holds("FileWrite")
  on_violation: refuse
}
```
```tsr:prompt
prompt answer(t: String) -> String = "{t}"
```
```tsr:agent
agent G { beliefs: @last_write topic: String intentions: plan respond { let o = answer(topic) return o } }
```
''', "G", topic="x")
    refusals = [e for e in world.audit if e.action == "contract:refuse"]
    assert refusals and refusals[0].detail.get("contract") == "first_gate"


# --------------------------------------------------------- tool after modes

def test_tool_after_retry_exhausts_then_refuses():
    """os.getcwd is deterministic, so an after-clause it can't satisfy exhausts
    its retry budget (re-invoking the tool each time) and refuses."""
    result, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on tool:cwd {
  after: intent_match() >= 0.99
  on_violation: retry(2)
}
```
```tsr:tool
tool cwd() -> String from os.getcwd
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write t: String
  intentions: plan p serves roofing_advice { let o = cwd() return o }
}
```
''', "G", t="x")
    assert isinstance(result, Refusal)
    assert len([e for e in world.audit if e.action == "contract:retry"]) == 2


def test_tool_after_audit_keeps_result():
    result, world = _run('''---
agent: G
---
```tsr:contract
contract on_topic on tool:cwd {
  after: intent_match() >= 0.99
  on_violation: audit
}
```
```tsr:tool
tool cwd() -> String from os.getcwd
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write t: String
  intentions: plan p serves roofing_advice { let o = cwd() return o }
}
```
''', "G", t="x")
    assert not isinstance(result, Refusal)                # tool result stood
    assert any(e.action == "contract:audit" for e in world.audit)


# --------------------------------------------------------- plan after audit

def test_plan_after_audit_keeps_result():
    result, world = _run('''---
agent: G
---
```tsr:contract
contract result_on_topic on plan:respond {
  after: intent_match() >= 0.9
  on_violation: audit
}
```
```tsr:agent
agent G intends roofing_advice {
  beliefs: @last_write t: String
  intentions: plan respond serves roofing_advice { return t }
}
```
''', "G", t="off topic entirely")
    assert not isinstance(result, Refusal)                # result kept
    assert result == "off topic entirely"
    assert any(e.action == "contract:audit" and e.detail.get("phase") == "after"
               for e in world.audit)
