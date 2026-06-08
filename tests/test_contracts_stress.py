"""Stress + property-based tests for `tsr:contract`.

Three pressures the unit tests don't apply: concurrency (many agents through one
contract at once), volume/depth (deep retry budgets, hundreds of events), and
randomized input (hypothesis over on_violation strings, clause combinations, and
intent_match).

CONCURRENCY NOTE: `World._audit_seq += 1` is not lock-guarded, so under
concurrent agents the per-event sequence numbers may collide or skip and event
*ordering* is best-effort (documented in eval.py::World.record). `list.append`
is atomic under the GIL, so event *counts* are still exact — which is what these
tests assert on. The semantic cache (`semantic_cache_put`) can also race under
concurrency; again it affects which path a call takes (fresh vs cached), not
whether the contract fires. See the CHANGELOG "Known limitation" note.
"""
from hypothesis import given, settings, strategies as st

from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail, _parse_on_violation
from tessera.verify.passes import run_local
from tessera.interp.eval import World, run_agent, Refusal
from tessera.policy_lang import _intent_match, ActionContext


def _build(src):
    return lower(parse_source(src, path="<inline>"))


# ===================================================================
# Stress
# ===================================================================

def _storm_src(n: int) -> str:
    """A TeamLead that spawns `n` Workers, each calling a prompt whose after-
    contract always fails (intent drift) → every worker refuses."""
    spawns = "\n".join(f"      let w{i} = spawn Worker with []" for i in range(n))
    sends = "\n".join(f"      send w{i} topic" for i in range(n))
    recvs = "\n".join(f"      let r{i} = recv from w{i}" for i in range(n))
    return (
        "---\nagent: TeamLead\n---\n"
        "```tsr:contract\n"
        "contract on_topic on prompt:work {\n"
        "  after: intent_match() >= 0.99\n"
        "  on_violation: refuse\n"
        "}\n```\n"
        "```tsr:prompt\n"
        "prompt work(t: String) -> String = \"{t}\"\n```\n"
        "```tsr:agent\n"
        "agent Worker intends do_work {\n"
        "  beliefs: @last_write task: String\n"
        "  intentions: plan run serves do_work { let o = work(task) return o }\n"
        "}\n"
        "agent TeamLead {\n"
        "  beliefs: @last_write topic: String\n"
        "  intentions: plan lead {\n"
        f"{spawns}\n{sends}\n{recvs}\n"
        "      return \"done\"\n"
        "    }\n"
        "}\n```\n"
    )


def test_concurrent_agent_storm_all_refuse():
    """12 workers (> the 8-worker pool, so they queue) all run the contracted
    prompt concurrently. Every one must refuse, and the count is exact despite
    the documented audit-seq / cache races."""
    n = 12
    module = _build(_storm_src(n))
    world = World(module=module)
    run_agent(module, "TeamLead", initial_beliefs={"topic": "x"},
              concurrent=True, world=world)
    refusals = [e for e in world.audit
                if e.action == "contract:refuse" and e.detail.get("target") == "prompt:work"]
    assert len(refusals) == n


def test_supervision_redrives_contract_refusal():
    """A child whose plan returns a contract `Refusal` value (here from a
    tool-contract `before` that the child lacks the capability for) is re-driven
    by the parent's `supervise=retry(2)` twice before the Refusal propagates.

    This is exactly why tool/plan contracts return a `Refusal` (not a string):
    only a Refusal flowing back as the plan's value is re-drivable by
    supervision. NOTE: a plan-`before` refusal via on_plan_enter short-circuits
    *before* the plan result is recorded, so it is NOT re-driven — only a Refusal
    value returned through the plan body is. Hence the tool contract here."""
    src = (
        "---\nagent: Boss\ncapabilities_requested: []\n---\n"
        "```tsr:contract\n"
        "contract needs_net on tool:risky {\n"
        "  before: holds(\"NetworkOut\")\n"
        "  on_violation: refuse\n"
        "}\n```\n"
        "```tsr:tool\ntool risky() -> String from os.getcwd\n```\n"
        "```tsr:agent\n"
        "agent Risky { beliefs: @last_write task: String intentions: plan work { let o = risky() return o } }\n"
        "agent Boss {\n"
        "  beliefs: @last_write topic: String\n"
        "  intentions: plan lead {\n"
        "      let r = spawn Risky with [] supervise=retry(2)\n"
        "      send r topic\n"
        "      let out = recv from r\n"
        "      return out\n"
        "    }\n"
        "}\n```\n"
    )
    module = _build(src)
    world = World(module=module)
    out = run_agent(module, "Boss", initial_beliefs={"topic": "x"}, world=world)
    redrives = [e for e in world.audit if e.action == "supervised_retry"]
    assert len(redrives) == 2
    assert isinstance(out, Refusal)                       # propagated after exhaustion
    assert any(e.action == "contract:refuse" for e in world.audit)


def test_deep_retry_budget_exhausts():
    """retry(50) with the deterministic noop backend can never pass — exactly 50
    re-drives are recorded before the refusal. Stresses the retry loop bound."""
    src = (
        "---\nagent: G\n---\n"
        "```tsr:contract\n"
        "contract on_topic on prompt:answer {\n"
        "  after: intent_match() >= 0.99\n"
        "  on_violation: retry(50)\n"
        "}\n```\n"
        "```tsr:prompt\nprompt answer(t: String) -> String = \"{t}\"\n```\n"
        "```tsr:agent\n"
        "agent G intends roofing_advice {\n"
        "  beliefs: @last_write topic: String\n"
        "  intentions: plan respond serves roofing_advice { let o = answer(topic) return o }\n"
        "}\n```\n"
    )
    module = _build(src)
    world = World(module=module)
    result = run_agent(module, "G", initial_beliefs={"topic": "x"}, world=world)
    assert "contract-refused" in str(result)
    assert len([e for e in world.audit if e.action == "contract:retry"]) == 50


def test_high_audit_volume():
    """One plan calls a contracted prompt 100 times (audit mode keeps output) →
    exactly 100 contract:audit events and the run completes."""
    n = 100
    calls = "\n".join(f"      let x{i} = answer(topic)" for i in range(n))
    src = (
        "---\nagent: G\n---\n"
        "```tsr:contract\n"
        "contract on_topic on prompt:answer {\n"
        "  after: intent_match() >= 0.99\n"
        "  on_violation: audit\n"
        "}\n```\n"
        "```tsr:prompt\nprompt answer(t: String) -> String = \"{t}\"\n```\n"
        "```tsr:agent\n"
        "agent G intends roofing_advice {\n"
        "  beliefs: @last_write topic: String\n"
        "  intentions: plan respond serves roofing_advice {\n"
        f"{calls}\n"
        "      return topic\n"
        "    }\n"
        "}\n```\n"
    )
    module = _build(src)
    world = World(module=module)
    result = run_agent(module, "G", initial_beliefs={"topic": "x"}, world=world)
    assert result == "x"
    assert len([e for e in world.audit if e.action == "contract:audit"]) == n


# ===================================================================
# Property-based (hypothesis)
# ===================================================================

@settings(max_examples=100)
@given(n=st.integers(min_value=0, max_value=9999),
       fallback=st.sampled_from(["refuse", "audit"]))
def test_property_retry_on_violation_roundtrips(n, fallback):
    assert _parse_on_violation(f"retry({n}) then {fallback}") == ("retry", n, fallback)
    assert _parse_on_violation(f"retry({n})") == ("retry", n, "refuse")


@settings(max_examples=50)
@given(text=st.text(min_size=0, max_size=200))
def test_property_intent_match_is_bounded(text):
    score = _intent_match(ActionContext(value=text, intent="roofing advice estimate"))
    assert 0.0 <= score <= 1.0


_PREDS = [
    'holds("NetworkOut")',
    "not contains_pii(value())",
    "intent_match() >= 0.5",
    "not extracts(value())",
    "cost_remaining() > 0",
]
_OVS = ["refuse", "audit", "retry(2)", "retry(3) then audit"]


@settings(max_examples=60, deadline=None)
@given(
    before=st.lists(st.sampled_from(_PREDS), max_size=3),
    after=st.lists(st.sampled_from(_PREDS), max_size=3),
    ov=st.sampled_from(_OVS),
)
def test_property_random_contracts_compile_and_verify(before, after, ov):
    """Any combination of known predicates as before/after clauses with any
    on_violation must lower + verify without raising — run_local only ever
    returns a diagnostics list, never throws."""
    clauses = "".join(f"  before: {c}\n" for c in before)
    clauses += "".join(f"  after: {c}\n" for c in after)
    src = (
        "---\nagent: G\ncapabilities_requested: [NetworkOut]\n---\n"
        "```tsr:contract\n"
        "contract fuzz on prompt:answer {\n"
        f"{clauses}"
        f"  on_violation: {ov}\n"
        "}\n```\n"
        "```tsr:prompt\nprompt answer(t: String) -> String = \"{t}\"\n```\n"
        "```tsr:agent\n"
        "agent G { beliefs: @last_write topic: String intentions: plan respond { let o = answer(topic) return o } }\n"
        "```\n"
    )
    module = _build(src)
    diags = run_local(module)
    assert isinstance(diags, list)
    # The target exists, so no phantom-target error should ever fire.
    assert not any(d.code == "E830" for d in diags)
