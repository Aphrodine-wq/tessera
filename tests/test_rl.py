"""Tests for the RL substrate (research B3)."""
import math
import os
import random
from pathlib import Path

import pytest

from tessera.rl import (
    QTable,
    state_key,
    epsilon_at_step,
    save_qtable,
    load_qtable,
)


def test_state_key_is_stable_under_dict_order():
    """Same beliefs → same key regardless of insertion order."""
    a = state_key({"x": 1, "y": "hello"})
    b = state_key({"y": "hello", "x": 1})
    assert a == b


def test_qtable_get_returns_zero_for_unknown_state():
    t = QTable(agent="A")
    assert t.get("unseen", "plan_a") == 0.0


def test_qtable_update_moves_toward_reward():
    """Reward = 1.0, learning rate 0.5, no next-state → Q should rise."""
    t = QTable(agent="A")
    t.update("s1", "plan_a", reward=1.0, next_state="s1",
             next_plans=["plan_a"], alpha=0.5, gamma=0.0)
    # First update with alpha=0.5, no discount: Q = 0 + 0.5*(1 + 0 - 0) = 0.5
    assert math.isclose(t.get("s1", "plan_a"), 0.5)


def test_qtable_update_with_discount_propagates_value():
    """Two-step update: reward 0 then 1, with γ=0.9 — first state's Q
    should rise because of the discounted future value."""
    t = QTable(agent="A")
    # First teach: from s2, plan_b → reward 1
    t.update("s2", "plan_b", reward=1.0, next_state="terminal",
             next_plans=["plan_b"], alpha=0.5, gamma=0.9)
    # Now from s1, plan_a → reward 0, next state s2
    t.update("s1", "plan_a", reward=0.0, next_state="s2",
             next_plans=["plan_b"], alpha=0.5, gamma=0.9)
    # s1.plan_a should be > 0 because of bootstrapped future value
    assert t.get("s1", "plan_a") > 0


def test_qtable_visit_count_increments():
    t = QTable(agent="A")
    for _ in range(3):
        t.update("s", "p", reward=0.5, next_state="s",
                 next_plans=["p"], alpha=0.1)
    assert t.visit_count("s", "p") == 3


def test_choose_explores_with_high_epsilon():
    t = QTable(agent="A")
    t.q["s"] = {"p1": 10.0, "p2": 0.0}
    rng = random.Random(0)
    # With ε=1.0, every choice is uniform random
    counts = {"p1": 0, "p2": 0}
    for _ in range(200):
        choice = t.choose("s", ["p1", "p2"], epsilon=1.0, rng=rng)
        counts[choice] += 1
    # Both should be picked roughly equally
    assert 0.3 < counts["p1"] / 200 < 0.7


def test_choose_exploits_with_zero_epsilon():
    t = QTable(agent="A")
    t.q["s"] = {"p1": 10.0, "p2": 0.0}
    rng = random.Random(0)
    for _ in range(20):
        choice = t.choose("s", ["p1", "p2"], epsilon=0.0, rng=rng)
        assert choice == "p1"  # always picks the higher-Q plan


def test_choose_tie_breaks_by_visit_count():
    """When two plans have equal Q, the less-visited one wins."""
    t = QTable(agent="A")
    t.q["s"] = {"p1": 1.0, "p2": 1.0}
    t.visits["s"] = {"p1": 10, "p2": 1}
    rng = random.Random(0)
    choice = t.choose("s", ["p1", "p2"], epsilon=0.0, rng=rng)
    assert choice == "p2"


def test_epsilon_decay_schedule():
    assert math.isclose(epsilon_at_step(0, eps_start=1.0, eps_end=0.05,
                                        decay_steps=100), 1.0)
    assert math.isclose(epsilon_at_step(100, eps_start=1.0, eps_end=0.05,
                                        decay_steps=100), 0.05)
    mid = epsilon_at_step(50, eps_start=1.0, eps_end=0.05, decay_steps=100)
    assert 0.4 < mid < 0.6


def test_save_and_load_round_trip(tmp_path):
    t = QTable(agent="A")
    t.update("s1", "plan_a", reward=1.0, next_state="s1",
             next_plans=["plan_a"])
    t.update("s2", "plan_b", reward=0.5, next_state="s2",
             next_plans=["plan_b"])
    path = save_qtable(t, rl_dir=tmp_path)
    assert path.exists()
    loaded = load_qtable("A", rl_dir=tmp_path)
    assert loaded.get("s1", "plan_a") == t.get("s1", "plan_a")
    assert loaded.get("s2", "plan_b") == t.get("s2", "plan_b")


def test_load_missing_returns_empty_table(tmp_path):
    loaded = load_qtable("nonexistent", rl_dir=tmp_path)
    assert loaded.agent == "nonexistent"
    assert loaded.q == {}
