"""Bayesian belief substrate (research A2, discrete MVP).

Primary references:
- Bayes, T. (1763). An essay towards solving a problem in the doctrine
  of chances. Philosophical Transactions of the Royal Society.
- Blei, Kucukelbir, McAuliffe (2017). Variational inference: a review
  for statisticians. JASA 112:518. https://arxiv.org/abs/1601.00670

A `tsr:bayesian` block declares discrete random variables with priors:

    bayesian {
      var weather: [sunny, cloudy, rainy] prior [0.5, 0.3, 0.2]
      var traffic: [light, heavy] prior [0.7, 0.3]
    }

The runtime supports two operations against these:
  - `update X with evidence (Y=y)` — exact Bayes posterior over X
    given a likelihood P(Y=y | X) declared as a tsr:bayesian table.
  - `posterior X` — return the current belief distribution.

Posteriors persist to the semantic store under schema "BayesianBelief"
so cross-run inference accumulates evidence. Each update emits a
`bayesian:update` audit event with prior, likelihood vector, and
posterior, providing full provenance.

Honest scope: discrete vars + exact inference only. Continuous-valued
beliefs and variational inference (Blei et al.) are explicit follow-
ups; they want a real numerical backend (scipy or PyTorch). The
discrete MVP exercises the math + audit + persistence cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class DiscreteVarDecl:
    """A discrete random variable with a finite outcome space.

    `values` is the ordered list of outcome names; `prior` is the
    list of P(value_i) with sum 1. Likelihood tables are stored
    separately on the BayesianDecl.
    """
    name: str
    values: list[str]
    prior: list[float]

    def __post_init__(self):
        if len(self.values) != len(self.prior):
            raise ValueError(
                f"var {self.name}: values ({len(self.values)}) and "
                f"prior ({len(self.prior)}) length mismatch"
            )
        s = sum(self.prior)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"var {self.name}: prior must sum to 1.0 (got {s:.4f})"
            )
        if any(p < 0 for p in self.prior):
            raise ValueError(f"var {self.name}: prior must be non-negative")


@dataclass
class LikelihoodTable:
    """P(observed=o | latent=l) for each (l, o) pair.

    Stored as `entries[l_value][o_value] = float`. Each row (fixed l)
    does NOT need to sum to 1 — likelihoods are not distributions; they
    are conditional probabilities indexed by the observation.
    """
    latent: str   # name of the latent variable
    observed: str  # name of the observation variable
    entries: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BayesianDecl:
    """The complete tsr:bayesian declaration."""
    variables: dict[str, DiscreteVarDecl] = field(default_factory=dict)
    likelihoods: dict[tuple[str, str], LikelihoodTable] = field(default_factory=dict)


def bayes_update(
    var: DiscreteVarDecl,
    likelihood_row: Mapping[str, float],
) -> list[float]:
    """Exact discrete Bayes update.

    Args:
      var: DiscreteVarDecl with current prior (or previous posterior).
      likelihood_row: {value_i: P(observed | latent=value_i)} for each i.

    Returns the normalized posterior as a list aligned with var.values.

    P(latent=v_i | obs) ∝ P(obs | latent=v_i) * P(latent=v_i)
    """
    unnorm = [
        likelihood_row.get(v, 0.0) * p
        for v, p in zip(var.values, var.prior)
    ]
    s = sum(unnorm)
    if s == 0:
        # Evidence rules out every outcome — return uniform (impossible state)
        # to avoid divide-by-zero. Caller can detect this via the audit.
        n = len(var.values)
        return [1.0 / n] * n
    return [u / s for u in unnorm]


def update_var(
    decl: BayesianDecl,
    latent_name: str,
    observed_name: str,
    observed_value: str,
) -> tuple[list[float], list[float]]:
    """Look up the likelihood table, apply Bayes, mutate the variable's
    prior in place (so chained updates accumulate evidence).

    Returns (prior_before, posterior). The audit caller logs both.
    """
    if latent_name not in decl.variables:
        raise KeyError(f"unknown latent variable {latent_name!r}")
    if (latent_name, observed_name) not in decl.likelihoods:
        raise KeyError(
            f"no likelihood table declared for "
            f"observed={observed_name} given latent={latent_name}"
        )
    var = decl.variables[latent_name]
    table = decl.likelihoods[(latent_name, observed_name)]
    likelihood_row = {
        latent_v: table.entries.get(latent_v, {}).get(observed_value, 0.0)
        for latent_v in var.values
    }
    prior_before = list(var.prior)
    posterior = bayes_update(var, likelihood_row)
    var.prior = posterior
    return prior_before, posterior


def posterior(decl: BayesianDecl, name: str) -> list[float]:
    """Return the current posterior distribution over `name`."""
    if name not in decl.variables:
        raise KeyError(f"unknown variable {name!r}")
    return list(decl.variables[name].prior)
