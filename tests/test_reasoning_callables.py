"""Value-layer literals + reasoning-tool callables (Phase 4).

list/record literals make the value layer rich enough to call the reasoning
modules from a plan. Each callable bridges scalar/list/record args into the
research module and (for causal/bayesian) operates on a declared block.
"""
import os

from tessera.parser.module import parse_source
from tessera.sir.build import lower
from tessera.interp.eval import World, run_agent

os.environ["TESSERA_LLM_BACKEND"] = "noop"


def _run(src: str, beliefs=None):
    m = lower(parse_source(src, path="<inline>"))
    w = World(module=m)
    out = run_agent(m, _agent_name(m), initial_beliefs=beliefs or {"q": "x"},
                    world=w, concurrent=False)
    return out, w


def _agent_name(m):
    return next(iter(m.agents))


def _plan_agent(plan_body: str, *blocks: str) -> str:
    extra = "\n\n".join(blocks)
    return f"""---
agent: A
tessera_version: 0.2
---

{extra}

```tsr:agent
agent A {{
  beliefs:
    @last_write q: String
  intentions:
    plan go {{ {plan_body} }}
}}
```
"""


# ----- value-layer literals -----

def test_list_literal():
    out, _ = _run(_plan_agent("let xs = [1, 2, 3] return xs"))
    assert out == [1, 2, 3]


def test_record_literal():
    out, _ = _run(_plan_agent('let r = {a: 1, b: "two"} return r'))
    assert out == {"a": 1, "b": "two"}


def test_nested_literal():
    out, _ = _run(_plan_agent('let r = {items: [1, 2], meta: {n: 2}} return r'))
    assert out == {"items": [1, 2], "meta": {"n": 2}}


# ----- causal-DAG queries -----

CAUSAL = """```tsr:causal
causal Conf {
  var Z: Bool
  var X: Bool
  var Y: Bool
  edge Z -> X
  edge Z -> Y
  edge X -> Y
}
```"""


def test_causal_backdoor_finds_confounder():
    out, _ = _run(_plan_agent('return causal_backdoor("Conf", "X", "Y")', CAUSAL))
    assert out == ["Z"]


def test_causal_identifiable():
    out, _ = _run(_plan_agent('return causal_identifiable("Conf", "X", "Y")', CAUSAL))
    assert out is True


# ----- counterfactual over a declared DAG -----

CHAIN = """```tsr:causal
causal Chain {
  var X: Str
  var Y: Str
  edge X -> Y
}
```"""


def test_counterfactual_query():
    body = (
        'let eqs = {Y: {parents: ["X"], table: {T: "T", F: "F"}}}\n'
        '    return counterfactual("Chain", eqs, {X: "F", Y: "F"}, {X: "T"}, "Y")'
    )
    out, _ = _run(_plan_agent(body, CHAIN))
    assert out == "T"  # had X been T, Y would have been T


# ----- bayesian posterior (+ metacognition calibration) -----

BAYES = """```tsr:bayesian
bayesian {
  var Disease: [yes, no] prior [0.01, 0.99]
  likelihood Test given Disease {
    yes -> pos: 0.99
    no -> pos: 0.05
  }
}
```"""


def test_bayesian_posterior():
    out, _ = _run(_plan_agent('return bayesian_posterior("Disease", "Test", "pos")', BAYES))
    # base-rate fact: P(disease | positive) ≈ 0.166 despite a 99% test
    assert abs(out["yes"] - 0.166) < 0.01


def test_metacognition_calibrates_posterior():
    src = _plan_agent('return bayesian_posterior("Disease", "Test", "pos")',
                      BAYES, "```tsr:metacognition\nmetacognition { temperature: 2.0 track_ece: true }\n```")
    out, w = _run(src)
    assert any(e.action == "metacog:calibrated" for e in w.audit)
    # T=2 softens the distribution toward uniform → yes-prob rises from 0.166
    assert out["yes"] > 0.166


def test_calibrate_scalar():
    out, w = _run(_plan_agent("return calibrate(0.9)",
                              "```tsr:metacognition\nmetacognition { temperature: 2.0 }\n```"))
    assert 0.5 < out < 0.9  # T>1 pulls an overconfident 0.9 toward 0.5


# ----- abductive + analogy -----

def test_abductive_best_explanation():
    body = (
        'let hyps = [{name: "rain", prior: 0.3, complexity: 1.0, likelihood: {wet: 0.9}}, '
        '{name: "sprinkler", prior: 0.1, complexity: 2.0, likelihood: {wet: 0.8}}]\n'
        '    return abductive(hyps, ["wet"])'
    )
    out, _ = _run(_plan_agent(body))
    assert out == "rain"


def test_analogy_structure_mapping():
    body = (
        'let src = {name: "solar", objects: ["sun", "planet"], '
        'relations: [{pred: "attracts", args: ["sun", "planet"]}]}\n'
        '    let tgt = {name: "atom", objects: ["nucleus", "electron"], '
        'relations: [{pred: "attracts", args: ["nucleus", "electron"]}]}\n'
        '    return analogy(src, tgt)'
    )
    out, _ = _run(_plan_agent(body))
    assert out.get("sun") == "nucleus" and out.get("planet") == "electron"
