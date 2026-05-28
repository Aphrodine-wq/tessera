"""Reinforcement-learning substrate — per-plan Q-learning (research B3).

Primary reference: Sutton, R. S., Barto, A. G. (2018). Reinforcement
Learning: An Introduction (2nd ed.). MIT Press.

Tessera's RL substrate operates at the PLAN level, not the action
level. Each (state, plan_name) pair carries a Q-value. State is the
canonicalized tuple of the agent's relevant beliefs. Plan choice uses
ε-greedy with declared ε decay schedule.

After each plan completes, the runtime delivers a reward (declared
externally — author plugs in eval_pass_rate, user-thumbs, or a
custom signal). The Q-update follows the standard rule:

    Q(s, p) ← Q(s, p) + α [r + γ · max_p' Q(s', p') − Q(s, p)]

where α is learning rate, γ is discount, s' is the resulting state.

Q-tables persist to ~/.tessera/rl/<agent>.qtable.json (override via
TESSERA_RL_DIR). Pure-Python — no numpy dependency.

Honest scope: this is tabular RL. State space is the agent's declared
belief tuple, hashed lexicographically. Non-trivial state spaces
(continuous beliefs, large tuples) hit the curse of dimensionality.
A follow-up can add function-approximation via the tsr:neural
substrate.

Caveat: specification gaming (Krakovna et al. 2020). Reward must be
ground-truth-aligned; the substrate doesn't enforce that. Composes
with shipped tsr:precaution to gate against reward-hacking-induced
catastrophe.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RL_DIR = Path.home() / ".tessera" / "rl"
ENV_RL_DIR = "TESSERA_RL_DIR"


def _resolve_rl_dir() -> Path:
    env = os.environ.get(ENV_RL_DIR)
    return Path(env) if env else DEFAULT_RL_DIR


def state_key(beliefs: dict[str, Any]) -> str:
    """Canonicalize a beliefs dict into a stable, hashable state key.

    Sorts keys, converts values to repr, joins. A future commit can
    discretize continuous values via author-declared bins.
    """
    parts = [f"{k}={v!r}" for k, v in sorted(beliefs.items())]
    return "|".join(parts)


@dataclass
class QTable:
    """Per-agent Q-table mapping (state_key, plan_name) → Q-value."""
    agent: str
    q: dict[str, dict[str, float]] = field(default_factory=dict)
    visits: dict[str, dict[str, int]] = field(default_factory=dict)

    def get(self, state: str, plan: str) -> float:
        return self.q.get(state, {}).get(plan, 0.0)

    def visit_count(self, state: str, plan: str) -> int:
        return self.visits.get(state, {}).get(plan, 0)

    def update(
        self,
        state: str,
        plan: str,
        reward: float,
        next_state: str,
        next_plans: list[str],
        *,
        alpha: float = 0.1,
        gamma: float = 0.9,
    ) -> float:
        """Standard tabular Q-update. Returns the new Q-value."""
        current = self.get(state, plan)
        next_max = max(
            (self.get(next_state, p) for p in next_plans),
            default=0.0,
        )
        new_q = current + alpha * (reward + gamma * next_max - current)
        self.q.setdefault(state, {})[plan] = new_q
        self.visits.setdefault(state, {})[plan] = (
            self.visits.get(state, {}).get(plan, 0) + 1
        )
        return new_q

    def choose(
        self,
        state: str,
        plans: list[str],
        *,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ) -> str:
        """ε-greedy plan choice.

        With probability ε, pick uniformly at random. Otherwise pick
        the plan with highest Q-value (ties broken by visit count low
        first — favors exploration on tied plans).
        """
        if not plans:
            raise ValueError("choose() needs at least one plan")
        rng = rng or random.Random()
        if rng.random() < epsilon:
            return rng.choice(plans)
        # Greedy with tie-breaking by visit count (less-visited wins)
        scored = [
            (self.get(state, p), -self.visit_count(state, p), p)
            for p in plans
        ]
        scored.sort(reverse=True)
        return scored[0][2]


def epsilon_at_step(
    step: int, *, eps_start: float = 1.0, eps_end: float = 0.05,
    decay_steps: int = 1000,
) -> float:
    """Linear ε decay from eps_start to eps_end over decay_steps."""
    if step >= decay_steps:
        return eps_end
    frac = step / max(1, decay_steps)
    return eps_start + (eps_end - eps_start) * frac


def save_qtable(table: QTable, *, rl_dir: Path | None = None) -> Path:
    rl_dir = rl_dir or _resolve_rl_dir()
    rl_dir.mkdir(parents=True, exist_ok=True)
    path = rl_dir / f"{table.agent}.qtable.json"
    with open(path, "w") as fh:
        json.dump({"q": table.q, "visits": table.visits}, fh)
    return path


def load_qtable(agent: str, *, rl_dir: Path | None = None) -> QTable:
    rl_dir = rl_dir or _resolve_rl_dir()
    path = rl_dir / f"{agent}.qtable.json"
    if not path.exists():
        return QTable(agent=agent)
    with open(path) as fh:
        data = json.load(fh)
    return QTable(agent=agent, q=data.get("q", {}), visits=data.get("visits", {}))
