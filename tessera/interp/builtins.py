"""Built-in callables for tsr:logic and plan bodies.

Two groups:
  - value constructors `__list__` / `__record__`, emitted by the `[..]` / `{..}`
    literal syntax (see sir/build.py `_emit_builtin_call`).
  - reasoning-tool callables that bridge scalar / list / record arguments into
    the research modules, operating on declared tsr:causal / tsr:bayesian blocks
    and inline data. These are the "reasoning-tools as callables" surface — no
    new top-level substrate block, just functions you call from a plan.

Handler signature: `(node, args, world, agent_name) -> value`. Registered in
BUILTINS; eval.py's Apply handler dispatches here before the module
function/prompt/tool lookup.
"""
from __future__ import annotations

import math
from typing import Any


def _err(msg: str):
    from .eval import RuntimeError_
    return RuntimeError_(msg)


# ----- value constructors (literal syntax) -----

def _set_auto_confidence(world, owner, conf) -> None:
    """Record reasoning-derived confidence so dual_process routing picks it up
    at the next plan entry. A plan can still set `_confidence` explicitly; the
    freshest write wins. No-op if confidence isn't a finite number."""
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return
    world.state_for(owner).working_memory["_confidence"] = c


def _b_list(node, args, world, owner) -> list:
    return list(args)


def _b_record(node, args, world, owner) -> dict:
    keys = node.attributes.get("record_keys", [])
    return dict(zip(keys, args))


# ----- causal-DAG queries (operate on a declared tsr:causal block) -----

def _runtime_dag(world, name: str, who: str):
    decl = world.module.causal_dags.get(name)
    if decl is None:
        raise _err(f"{who}: no declared causal DAG named {name!r}")
    from ..causal import CausalDAG
    return CausalDAG(name=decl.name, variables=list(decl.variables),
                     edges=list(decl.edges))


def _b_causal_backdoor(node, args, world, owner) -> list:
    dag = _runtime_dag(world, args[0], "causal_backdoor")
    from ..causal import find_backdoor_adjustment_set
    s = find_backdoor_adjustment_set(dag, args[1], args[2])
    return sorted(s) if s else []


def _b_causal_identifiable(node, args, world, owner) -> bool:
    dag = _runtime_dag(world, args[0], "causal_identifiable")
    from ..causal import query_effect_identifiable
    ok, _ = query_effect_identifiable(dag, args[1], args[2])
    return bool(ok)


def _b_counterfactual(node, args, world, owner):
    """counterfactual(dagName, equations, observed, intervention, outcome).

    `equations` is a record {Child: {parents: [...], table: {"v1,v2": out}}},
    `observed` and `intervention` are records {var: value}. Returns the
    counterfactual value of `outcome`.
    """
    if len(args) < 5:
        raise _err("counterfactual(dag, equations, observed, intervention, outcome)")
    dag = _runtime_dag(world, args[0], "counterfactual")
    eqs_data, observed, intervention, outcome = args[1], args[2], args[3], args[4]
    from ..counterfactual import StructuralEquation, counterfactual_query
    equations: dict[str, StructuralEquation] = {}
    for child, spec in dict(eqs_data).items():
        parents = list(spec.get("parents", []))
        table: dict[tuple, Any] = {}
        for k, v in dict(spec.get("table", {})).items():
            key = tuple(p.strip() for p in k.split(",")) if isinstance(k, str) else (k,)
            table[key] = v
        equations[child] = StructuralEquation(child=child, parents=parents, table=table)
    iv_items = list(dict(intervention).items())
    iv = iv_items[0] if iv_items else (outcome, None)
    factual, cf = counterfactual_query(dag, equations, dict(observed), iv, outcome)
    world.record(owner, "counterfactual:query", outcome=outcome,
                 factual=factual, counterfactual=cf, intervention=f"{iv[0]}={iv[1]}")
    return cf


# ----- bayesian posterior (operates on a declared tsr:bayesian block) -----

def _calibrate_dist(dist: dict, mc, world, owner, label: str) -> dict:
    """Temperature-scale a probability distribution (metacognition), emit audit."""
    from ..calibration import softmax, temperature_scale
    keys = list(dist)
    logits = [math.log(max(min(dist[k], 0.999), 1e-9)) for k in keys]
    scaled = softmax(temperature_scale(logits, mc.temperature))
    out = {k: s for k, s in zip(keys, scaled)}
    if mc.track_ece:
        world.record(owner, "metacog:calibrated", label=label, temperature=mc.temperature,
                     before={k: round(dist[k], 4) for k in keys},
                     after={k: round(out[k], 4) for k in keys})
    return out


def _b_bayesian_posterior(node, args, world, owner) -> dict:
    """bayesian_posterior(latent, observed, value) → {latent_value: probability}.

    Exact discrete Bayes on the declared tsr:bayesian model. If tsr:metacognition
    is declared, the posterior is temperature-calibrated before returning.
    """
    if len(args) < 3:
        raise _err("bayesian_posterior(latent, observed, value)")
    latent, observed, value = args[0], args[1], args[2]
    decl = world.module.bayesian
    if decl is None:
        raise _err("bayesian_posterior: no tsr:bayesian model declared")
    var = next((v for v in decl.variables if v.name == latent), None)
    lik = next((l for l in decl.likelihoods
                if l.latent == latent and l.observed == observed), None)
    if var is None or lik is None:
        raise _err(f"bayesian_posterior: model lacks latent {latent!r} "
                   f"or likelihood {observed!r}|{latent!r}")
    unnorm = {lv: prior * lik.rows.get(lv, {}).get(value, 0.0)
              for lv, prior in zip(var.values, var.prior)}
    total = sum(unnorm.values()) or 1.0
    post = {k: v / total for k, v in unnorm.items()}
    if world.module.metacognition is not None:
        post = _calibrate_dist(post, world.module.metacognition, world, owner,
                               label=f"bayesian:{latent}")
    world.record(owner, "bayesian:posterior", latent=latent,
                 observed=f"{observed}={value}",
                 posterior={k: round(v, 4) for k, v in post.items()})
    if post:
        _set_auto_confidence(world, owner, max(post.values()))
    return post


def _b_calibrate(node, args, world, owner) -> float:
    """calibrate(confidence) → temperature-scaled confidence (metacognition)."""
    conf = float(args[0])
    mc = world.module.metacognition
    T = mc.temperature if mc is not None else 1.0
    p = max(min(conf, 0.999), 1e-9)
    logit = math.log(p / (1 - p))
    scaled = 1 / (1 + math.exp(-(logit / T)))
    if mc is not None and mc.track_ece:
        world.record(owner, "metacog:calibrate", before=round(conf, 4),
                     after=round(scaled, 4), temperature=T)
    return scaled


# ----- abductive: inference to the best explanation -----

def _b_abductive(node, args, world, owner) -> str:
    """abductive(hypotheses, observations) → name of the best explanation.

    `hypotheses` is a list of records {name, prior, complexity,
    likelihood: {observation: P(obs|h)}}; `observations` is a list of strings.
    Returns "none" when no hypothesis clears the confidence threshold.
    """
    if len(args) < 2:
        raise _err("abductive(hypotheses, observations)")
    from ..abductive import Hypothesis, best_explanation
    hyps = []
    for h in list(args[0]):
        lk = dict(h.get("likelihood", {}))
        hyps.append(Hypothesis(
            name=h.get("name", "?"),
            prior=float(h.get("prior", 0.5)),
            likelihood=(lambda o, _lk=lk: float(_lk.get(o, 0.5))),
            complexity=float(h.get("complexity", 1.0)),
        ))
    observations = list(args[1]) if isinstance(args[1], list) else [args[1]]
    best, ranked = best_explanation(hyps, observations)
    world.record(owner, "abductive:rank", best=best.name if best else None,
                 ranked=[(r.name, round(r.posterior, 4)) for r in ranked])
    if best:
        _set_auto_confidence(world, owner, best.posterior)
    return best.name if best else "none"


# ----- analogy: structure-mapping -----

def _b_analogy(node, args, world, owner) -> dict:
    """analogy(source, target) → object bindings {source_obj: target_obj}.

    Each domain is a record {name, objects: [...], relations: [{pred, args: [...]}]}.
    """
    if len(args) < 2:
        raise _err("analogy(source, target)")
    from ..analogy import Domain, Relation, find_best_mapping

    def _domain(d):
        d = dict(d)
        rels = [Relation(predicate=r.get("pred"), args=tuple(r.get("args", [])))
                for r in list(d.get("relations", []))]
        return Domain(name=d.get("name", "?"), objects=list(d.get("objects", [])),
                      relations=rels)

    mapping = find_best_mapping(_domain(args[0]), _domain(args[1]))
    if mapping is None:
        return {}
    world.record(owner, "analogy:map", bindings=dict(mapping.bindings),
                 score=round(mapping.score, 4))
    return dict(mapping.bindings)


BUILTINS = {
    "__list__": _b_list,
    "__record__": _b_record,
    "causal_backdoor": _b_causal_backdoor,
    "causal_identifiable": _b_causal_identifiable,
    "counterfactual": _b_counterfactual,
    "bayesian_posterior": _b_bayesian_posterior,
    "calibrate": _b_calibrate,
    "abductive": _b_abductive,
    "analogy": _b_analogy,
}
