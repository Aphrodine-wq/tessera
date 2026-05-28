"""Abductive reasoning — inference to the best explanation (research 4.3).

Primary references:
- Peirce, C. S. (1903). Pragmatism as a Principle and Method of Right
  Thinking. Lectures at Harvard.
- Lipton, P. (2004). Inference to the Best Explanation, 2nd ed.
  Routledge.
- Douven, I. (2017). Abduction. Stanford Encyclopedia of Philosophy.

Given observations O and a hypothesis space H, score each h by:

    score(h) = log P(O | h) + log prior(h) + log parsimony(h)

where parsimony is a Solomonoff-flavored simplicity penalty (we use
2^-complexity, with author-declared complexity per hypothesis).

The substrate ships the ranker. Authors declare hypotheses with
likelihood functions for each observation; the engine returns ranked
hypotheses with confidence-normalized scores.

Honest scope (Lipton's loveliness vs likeliness distinction):
- "Best" is the hypothesis MOST LIKELY given the data + prior. This is
  "likeliness."
- LOVELINESS (what would make a great explanation if true) is also
  important, and this MVP does NOT capture it. Document; treat as
  follow-up.

Caveat: hypothesis space is enumerated by the author. Out-of-space
explanations cannot be inferred. Document this explicitly when an
agent's best score is below an author-declared threshold — that's the
signal to expand the hypothesis space, not to commit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Hypothesis:
    """One candidate explanation in an abductive query.

    `name` is a short label. `prior` is P(h) before any data.
    `likelihood` is a callable: `likelihood(observation_id) -> P(O | h)`.
    `complexity` is a parsimony cost in arbitrary units (higher = more
    complex = penalized).
    """
    name: str
    prior: float = 0.5
    likelihood: Callable[[str], float] = field(default=lambda obs: 0.5)
    complexity: float = 1.0


@dataclass
class RankedHypothesis:
    name: str
    log_score: float
    posterior: float


def _log(x: float) -> float:
    return math.log(x) if x > 0 else -1e9


def rank_hypotheses(
    hypotheses: list[Hypothesis],
    observations: list[str],
) -> list[RankedHypothesis]:
    """Score each hypothesis and return a ranked list with normalized
    posteriors.

    log_score(h) = sum over obs of log P(obs | h) + log prior(h) +
                   log(2^-complexity)

    Posterior is exp(log_score) normalized across hypotheses.
    """
    log_scores: list[float] = []
    for h in hypotheses:
        s = _log(h.prior) + (-h.complexity * math.log(2))
        for obs in observations:
            s += _log(h.likelihood(obs))
        log_scores.append(s)

    # Normalize to posteriors via softmax
    m = max(log_scores) if log_scores else 0.0
    exps = [math.exp(s - m) for s in log_scores]
    total = sum(exps) or 1.0
    posteriors = [e / total for e in exps]

    ranked = sorted(
        [
            RankedHypothesis(name=h.name, log_score=log_scores[i], posterior=posteriors[i])
            for i, h in enumerate(hypotheses)
        ],
        key=lambda r: -r.log_score,
    )
    return ranked


def best_explanation(
    hypotheses: list[Hypothesis],
    observations: list[str],
    *,
    confidence_threshold: float = 0.5,
) -> tuple[RankedHypothesis | None, list[RankedHypothesis]]:
    """Return (best, all_ranked).

    `best` is None when the top hypothesis's posterior is below
    `confidence_threshold` — the signal to expand the hypothesis space
    rather than commit to a weak winner.
    """
    if not hypotheses:
        return (None, [])
    ranked = rank_hypotheses(hypotheses, observations)
    if ranked[0].posterior < confidence_threshold:
        return (None, ranked)
    return (ranked[0], ranked)
