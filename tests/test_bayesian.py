"""Tests for the Bayesian substrate + discrete inference (research A2)."""
import math
import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail
from tessera.bayesian import (
    DiscreteVarDecl,
    LikelihoodTable,
    BayesianDecl,
    bayes_update,
    update_var,
    posterior,
)


# --------- Core math ---------


def test_var_decl_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        DiscreteVarDecl(name="x", values=["a", "b"], prior=[1.0])


def test_var_decl_rejects_non_normalized_prior():
    with pytest.raises(ValueError, match="sum to 1"):
        DiscreteVarDecl(name="x", values=["a", "b"], prior=[0.5, 0.7])


def test_var_decl_rejects_negative_prior():
    with pytest.raises(ValueError, match="non-negative"):
        DiscreteVarDecl(name="x", values=["a", "b"], prior=[1.5, -0.5])


def test_bayes_update_uniform_prior_proportional_to_likelihood():
    var = DiscreteVarDecl(name="x", values=["a", "b"], prior=[0.5, 0.5])
    # Likelihood 0.8 on a, 0.2 on b → posterior 0.8, 0.2
    post = bayes_update(var, {"a": 0.8, "b": 0.2})
    assert math.isclose(post[0], 0.8, abs_tol=1e-9)
    assert math.isclose(post[1], 0.2, abs_tol=1e-9)


def test_bayes_update_strong_prior_dominates_weak_likelihood():
    var = DiscreteVarDecl(name="x", values=["a", "b"], prior=[0.99, 0.01])
    # Slight evidence against a — posterior should still favor a
    post = bayes_update(var, {"a": 0.4, "b": 0.6})
    assert post[0] > post[1]


def test_update_var_chains_through_likelihood_table():
    decl = BayesianDecl()
    decl.variables["weather"] = DiscreteVarDecl(
        name="weather",
        values=["sunny", "rainy"],
        prior=[0.7, 0.3],
    )
    # P(umbrella | weather): rain → 0.9 umbrella, sun → 0.1 umbrella
    decl.likelihoods[("weather", "umbrella")] = LikelihoodTable(
        latent="weather",
        observed="umbrella",
        entries={
            "sunny": {"yes": 0.1, "no": 0.9},
            "rainy": {"yes": 0.9, "no": 0.1},
        },
    )
    prior_before, post = update_var(decl, "weather", "umbrella", "yes")
    assert prior_before == [0.7, 0.3]
    # Bayes: P(rain | umbrella) = 0.9*0.3 / (0.1*0.7 + 0.9*0.3) = 0.79...
    assert math.isclose(post[1], 0.7941, abs_tol=1e-3)
    # Mutation: subsequent posterior() reads the updated state
    assert posterior(decl, "weather")[1] > 0.7


def test_update_var_with_impossible_evidence_returns_uniform():
    decl = BayesianDecl()
    decl.variables["x"] = DiscreteVarDecl(name="x", values=["a", "b"], prior=[0.5, 0.5])
    decl.likelihoods[("x", "obs")] = LikelihoodTable(
        latent="x",
        observed="obs",
        entries={
            "a": {"impossible": 0.0},
            "b": {"impossible": 0.0},
        },
    )
    _, post = update_var(decl, "x", "obs", "impossible")
    assert math.isclose(post[0], 0.5)
    assert math.isclose(post[1], 0.5)


# --------- Substrate parsing ---------


def test_bayesian_substrate_parses_into_module():
    src = """---
agent: Bayes
tessera_version: 0.2
---

```tsr:bayesian
bayesian {
  var weather: [sunny, rainy] prior [0.7, 0.3]
  var traffic: [light, heavy] prior [0.6, 0.4]

  likelihood umbrella given weather {
    sunny -> yes: 0.1
    sunny -> no: 0.9
    rainy -> yes: 0.9
    rainy -> no: 0.1
  }
}
```

```tsr:agent
agent Bayes {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert module.bayesian is not None
    by = module.bayesian
    assert len(by.variables) == 2
    weather = next(v for v in by.variables if v.name == "weather")
    assert weather.values == ["sunny", "rainy"]
    assert math.isclose(sum(weather.prior), 1.0)
    assert len(by.likelihoods) == 1
    lik = by.likelihoods[0]
    assert lik.latent == "weather"
    assert lik.observed == "umbrella"
    assert math.isclose(lik.rows["rainy"]["yes"], 0.9)


def test_bayesian_substrate_rejects_unnormalized_prior():
    src = """---
agent: Bad
tessera_version: 0.2
---

```tsr:bayesian
bayesian {
  var x: [a, b] prior [0.5, 0.7]
}
```

```tsr:agent
agent Bad {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    with pytest.raises(SyntaxFail, match="sum to 1"):
        lower(pm)
