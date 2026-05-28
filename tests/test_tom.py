"""Tests for the ToM substrate (research C3)."""
import math
import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower
from tessera.tom import (
    BeliefProposition,
    BeliefAboutAgent,
    sally_anne_update,
    has_false_belief,
    GoalBeliefHypothesis,
    score_hypothesis,
    rank_hypotheses,
)


# ----- Sally-Anne canonical -----


def test_sally_anne_witness_updates_belief():
    """Sally watches the marble move from basket to box → her belief updates."""
    world = [BeliefProposition("marble", "in_basket")]
    sally = BeliefAboutAgent(other_agent="Sally",
                             propositions=[BeliefProposition("marble", "in_basket")])
    agents = {"Sally": sally}
    sally_anne_update(
        world_view=world,
        agent_views=agents,
        event_id="ev1",
        witnesses=["Sally"],
        belief_change=(
            BeliefProposition("marble", "in_basket"),
            BeliefProposition("marble", "in_box"),
        ),
    )
    assert sally.believes(BeliefProposition("marble", "in_box"))
    assert not sally.believes(BeliefProposition("marble", "in_basket"))


def test_sally_anne_absent_keeps_false_belief():
    """The canonical signature: Sally LEAVES; Anne moves the marble. Sally's
    model retains the false belief that the marble is still in the basket."""
    world = [BeliefProposition("marble", "in_basket")]
    sally = BeliefAboutAgent(other_agent="Sally",
                             propositions=[BeliefProposition("marble", "in_basket")])
    agents = {"Sally": sally}
    # Sally is NOT in witnesses (she's left the room).
    sally_anne_update(
        world_view=world,
        agent_views=agents,
        event_id="anne_moves",
        witnesses=["Anne"],
        belief_change=(
            BeliefProposition("marble", "in_basket"),
            BeliefProposition("marble", "in_box"),
        ),
    )
    # World moved on
    assert world == [BeliefProposition("marble", "in_box")]
    # Sally's model didn't
    assert sally.believes(BeliefProposition("marble", "in_basket"))
    # Diagnostic
    fbs = has_false_belief(world, sally)
    assert any(p.subject == "marble" and p.predicate == "in_basket" for p in fbs)


def test_has_false_belief_returns_empty_when_aligned():
    world = [BeliefProposition("ball", "in_box")]
    view = BeliefAboutAgent(other_agent="X",
                            propositions=[BeliefProposition("ball", "in_box")])
    assert has_false_belief(world, view) == []


# ----- Inverse planning -----


def test_inverse_planning_picks_consistent_goal():
    """Given two observed actions, the hypothesis whose goal makes
    those actions most likely should rank highest."""
    actions = ["open_box", "take_marble"]
    h_marble = GoalBeliefHypothesis(goal="get_marble", belief_state=[])
    h_doll = GoalBeliefHypothesis(goal="get_doll", belief_state=[])
    likelihood = {
        ("open_box", "get_marble"): 0.9,
        ("take_marble", "get_marble"): 0.95,
        ("open_box", "get_doll"): 0.4,
        ("take_marble", "get_doll"): 0.05,
    }
    ranked = rank_hypotheses(actions, [h_marble, h_doll], likelihood)
    assert ranked[0].goal == "get_marble"
    assert ranked[0].score > ranked[1].score


def test_score_hypothesis_handles_unknown_actions():
    """An action not in the likelihood table gets a small smoothing prior;
    the score is finite (not -inf)."""
    h = GoalBeliefHypothesis(goal="g", belief_state=[])
    s = score_hypothesis(["unknown"], h, {})
    assert s != -1e9 * 1  # smoothing applied → not floor
    assert math.isfinite(s)


# ----- Substrate parsing -----


def test_tom_substrate_parses():
    src = """---
agent: Listener
tessera_version: 0.2
---

```tsr:tom
tom {
  tracked_agents: [Homeowner, Subcontractor]
  manipulation_refusal: true
}
```

```tsr:agent
agent Listener {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert module.tom is not None
    assert module.tom.tracked_agents == ["Homeowner", "Subcontractor"]
    assert module.tom.manipulation_refusal is True


def test_tom_substrate_default_refusal_true():
    src = """---
agent: L
tessera_version: 0.2
---

```tsr:tom
tom {
  tracked_agents: [X]
}
```

```tsr:agent
agent L {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert module.tom.manipulation_refusal is True
