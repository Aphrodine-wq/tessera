"""Cognitive trait firing engine + built-in trait registry.

A trait is a channeled reasoning posture (see `docs/cognitive-traits.md`). It
fires when any of its trigger terms match the current context, and when it fires
its `behavior` is injected as a preamble into the rendered prompt.

Design constraint: firing is **deterministic** — triggers match against
observable signals (rendered prompt text, plan name, capabilities) via a
keyword/structural lexicon. We deliberately do NOT classify context with an
extra LLM call: that would be non-deterministic, fragment the semantic cache,
and add a model round-trip to every prompt. The precision trade-off (substring
matching, not semantic understanding) is accepted in exchange for determinism,
testability, and zero added cost.

This module is imported by both the interpreter (`interp/eval.py`) and the
checker (`checker/drift.py`); keep it dependency-light.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .sir.nodes import TraitDecl

# --------------------------------------------------------------------------
# Trigger context
# --------------------------------------------------------------------------


@dataclass
class TriggerContext:
    """What a trigger term is evaluated against.

    - per_call: `text` is the rendered prompt; `params` are the bound args.
    - per_plan: `text` is the concatenation of the static templates of the
      prompts the plan calls; `plan_name` is the plan being entered.
    - global:   `text` is all of the agent's plan templates; `plan_name` is "".
    """
    text: str = ""
    plan_name: str = ""
    capabilities: frozenset[str] = field(default_factory=frozenset)
    params: dict = field(default_factory=dict)

    def _haystack(self) -> str:
        # Collapse all whitespace (incl. newlines) to single spaces so token
        # boundaries are clean for `_word_in` and multi-word lexicon entries.
        # Memoized per-context: every matcher calls this, so a single
        # fire_traits() would otherwise re-normalize the full prompt ~14x.
        h = self.__dict__.get("_hs_cache")
        if h is None:
            h = " ".join(f"{self.text} {self.plan_name}".split()).lower()
            self.__dict__["_hs_cache"] = h
        return h


# --------------------------------------------------------------------------
# Keyword lexicons
# --------------------------------------------------------------------------

_SECURITY_LEX: dict[str, set[str]] = {
    "secrets": {"secret", "secrets", "api key", "apikey", "token", "credential",
                "password", "private key", ".env"},
    "payments": {"payment", "payments", "charge", "invoice", "stripe", "billing",
                 "card", "checkout", "refund", "transaction", "payout"},
    "auth": {"auth", "authn", "authz", "authenticate", "authorization", "login",
             "oauth", "jwt", "session", "permission", "rbac"},
    "external_input": {"user input", "external", "untrusted", "request body",
                       "query param", "upload", "webhook", "incoming"},
}
_DESIGN_LEX: set[str] = {"design", "architecture", "schema", "interface",
                         "api shape", "trade-off", "tradeoff", "pattern",
                         "structure", "refactor"}
_DECISION_LEX: set[str] = {"decide", "decision", "choose", "choice", "option"}
_IDEATION_LEX: set[str] = {"brainstorm", "ideate", "ideation", "options",
                           "alternatives", "variety", "generate ideas", "what if"}
_DONE_LEX: set[str] = {"done", "complete", "completed", "finished", "ready",
                       "ship", "shipped", "final", "resolved"}
_COASTING_LEX: set[str] = {"obviously", "trivially", "as established",
                           "as before", "of course", "naturally", "clearly",
                           "self-evident", "needless to say", "everyone knows"}
_CONSISTENCY_LEX: set[str] = {"but earlier", "however we said",
                              "even though we", "contradicting",
                              "inconsistent with", "conflicts with",
                              "but we decided"}
_IRREVERSIBLE_LEX: set[str] = {"push to main", "reset --hard", "force push",
                               "rm -rf", "drop table", "migration",
                               "production deploy", "deploy to prod",
                               "delete the", "destroy", "destructive",
                               "irreversible", "no rollback"}
_MULTI_STEP_LEX: set[str] = {"multi-step", "multi step", "phased",
                             "complex task", "many steps", "step by step",
                             "phase 1", "phase one", "long-running",
                             "extended task"}

# Capability hints that imply a security-sensitive context even without keywords.
_SECRET_CAPS = {"VaultRead", "VaultWrite", "Secrets"}
_NETWORK_CAPS = {"NetworkIn", "NetworkOut"}


def _any_in(haystack: str, lex: set[str]) -> bool:
    return any(tok in haystack for tok in lex)


def _word_in(haystack: str, words: set[str]) -> bool:
    """Word-boundary match for single tokens (avoids 'token' in 'tokenizer')."""
    padded = f" {haystack} "
    return any(f" {w} " in padded for w in words)


# --------------------------------------------------------------------------
# Trigger-term matchers
# --------------------------------------------------------------------------


def _m_any_claim(ctx: TriggerContext) -> bool:
    return bool(ctx.text.strip())


def _m_any_question(ctx: TriggerContext) -> bool:
    return "?" in ctx.text


def _m_any_done_claim(ctx: TriggerContext) -> bool:
    return _word_in(ctx._haystack(), _DONE_LEX)


def _m_secrets(ctx: TriggerContext) -> bool:
    return _any_in(ctx._haystack(), _SECURITY_LEX["secrets"]) \
        or bool(ctx.capabilities & _SECRET_CAPS)


def _m_payments(ctx: TriggerContext) -> bool:
    return _any_in(ctx._haystack(), _SECURITY_LEX["payments"])


def _m_auth(ctx: TriggerContext) -> bool:
    return _any_in(ctx._haystack(), _SECURITY_LEX["auth"]) \
        or bool(ctx.capabilities & _SECRET_CAPS)


def _m_external_input(ctx: TriggerContext) -> bool:
    return _any_in(ctx._haystack(), _SECURITY_LEX["external_input"]) \
        or bool(ctx.capabilities & _NETWORK_CAPS)


def _m_design_question(ctx: TriggerContext) -> bool:
    return _any_in(ctx._haystack(), _DESIGN_LEX)


def _m_architecture_decision(ctx: TriggerContext) -> bool:
    hay = ctx._haystack()
    return _any_in(hay, _DESIGN_LEX) and _word_in(hay, _DECISION_LEX) \
        or _any_in(hay, {"architecture", "structure", "pattern", "trade-off", "tradeoff"})


def _m_ideation(ctx: TriggerContext) -> bool:
    return _any_in(ctx._haystack(), _IDEATION_LEX)


def _m_coasting_signal(ctx: TriggerContext) -> bool:
    """imposter_recursion fires when the text claims something is settled
    (obviously, trivially, as before) — exactly when re-verification is
    most likely to catch a stale assumption."""
    return _any_in(ctx._haystack(), _COASTING_LEX)


def _m_consistency_conflict(ctx: TriggerContext) -> bool:
    """spectrum_directness fires when the text flags a conflict explicitly
    (but earlier, however we said, contradicting). The matcher is intentionally
    narrow — surfacing implicit contradictions is a future verifier job."""
    return _any_in(ctx._haystack(), _CONSISTENCY_LEX)


def _m_irreversible_action(ctx: TriggerContext) -> bool:
    """anxiety_simulation fires on destructive / irreversible operations and
    on the capability hints that imply them."""
    return _any_in(ctx._haystack(), _IRREVERSIBLE_LEX) \
        or bool(ctx.capabilities & {"FileSystem", "FileSystem.ReadWrite",
                                    "Subprocess", "Subprocess.Spawn",
                                    "VaultWrite"})


def _m_multi_step_task(ctx: TriggerContext) -> bool:
    """insomniac_focus fires on complex / multi-step plans. We match the
    explicit signal in text and fall back to per_plan scope (the runtime
    activates this trait once at plan-enter, not on every per-call prompt)."""
    return _any_in(ctx._haystack(), _MULTI_STEP_LEX)


_MATCHERS: dict[str, Callable[[TriggerContext], bool]] = {
    "any_claim": _m_any_claim,
    "any_question": _m_any_question,
    "any_done_claim": _m_any_done_claim,
    "secrets": _m_secrets,
    "payments": _m_payments,
    "auth": _m_auth,
    "external_input": _m_external_input,
    "design_question": _m_design_question,
    "architecture_decision": _m_architecture_decision,
    "ideation": _m_ideation,
    "coasting_signal": _m_coasting_signal,
    "consistency_conflict": _m_consistency_conflict,
    "irreversible_action": _m_irreversible_action,
    "multi_step_task": _m_multi_step_task,
}

# Trigger-term spellings that alias onto a canonical matcher.
_TERM_ALIASES: dict[str, str] = {
    "contact_with_external_input": "external_input",
    "brainstorm": "ideation",
}

KNOWN_TERMS: frozenset[str] = frozenset(_MATCHERS) | frozenset(_TERM_ALIASES)


def _match_term(term: str, ctx: TriggerContext) -> bool:
    """An unknown term never fires — it cannot accidentally activate a behavior.

    The checker warns about unknown terms; here we just treat them as inert.
    """
    term = _TERM_ALIASES.get(term, term)
    m = _MATCHERS.get(term)
    return bool(m and m(ctx))


def trait_fires(trait: TraitDecl, ctx: TriggerContext) -> bool:
    """A trait fires if ANY of its trigger terms match (OR semantics)."""
    return any(_match_term(t, ctx) for t in trait.trigger)


def fire_traits(traits: list[TraitDecl], ctx: TriggerContext) -> list[TraitDecl]:
    return [t for t in traits if trait_fires(t, ctx)]


# --------------------------------------------------------------------------
# Built-in traits (source of truth; docs/cognitive-traits.md is illustrative)
# --------------------------------------------------------------------------

BUILTIN_TRAITS: dict[str, TraitDecl] = {
    "doubt_first": TraitDecl(
        name="doubt_first",
        trigger=["any_claim"],
        behavior=("Before committing to an answer, ask: 'What am I assuming? "
                  "What's the second-most-likely interpretation? What breaks if "
                  "I'm wrong?' Verify silently. Then commit with conviction."),
        priority=0.9,
        scope="per_call",
    ),
    "cross_brain": TraitDecl(
        name="cross_brain",
        trigger=["any_question"],
        behavior=("Before sequential reasoning, scan adjacent contexts (memory, "
                  "vault, sibling projects, prior decisions) for analogous "
                  "patterns. Lead with the surprising connection, not the "
                  "obvious one."),
        priority=0.85,
        scope="per_call",
    ),
    "compulsive": TraitDecl(
        name="compulsive",
        trigger=["any_done_claim"],
        behavior=("Before declaring a task complete, enumerate the failure "
                  "modes, check each edge, verify the side effects, re-read the "
                  "specification."),
        priority=0.8,
        scope="per_call",
    ),
    "hypervigilant": TraitDecl(
        name="hypervigilant",
        trigger=["contact_with_external_input", "secrets", "payments", "auth"],
        behavior=("Treat any external input as potentially adversarial. "
                  "Validate at every boundary. Prefer explicit refusal over "
                  "silent assumption."),
        priority=0.95,
        scope="per_call",
    ),
    "synesthetic": TraitDecl(
        name="synesthetic",
        trigger=["design_question", "architecture_decision"],
        behavior=("Find structural analogies: 'this is the same shape as X.' "
                  "Reach for patterns from a different domain when the local "
                  "frame is stuck."),
        priority=0.7,
        scope="per_call",
    ),
    "manic_burst": TraitDecl(
        name="manic_burst",
        trigger=["ideation", "brainstorm"],
        behavior=("Generate maximum variety before convergence. Suspend "
                  "judgment. Quantity over quality on the first pass."),
        priority=0.6,
        scope="per_plan",
    ),
    "imposter_recursion": TraitDecl(
        name="imposter_recursion",
        trigger=["coasting_signal"],
        behavior=("Never coast on past wins. Re-verify assumptions before "
                  "acting; yesterday's bar doesn't transfer to today. "
                  "Recalled memory and prior decisions are claims to verify, "
                  "not settled fact."),
        priority=0.85,
        scope="per_call",
    ),
    "spectrum_directness": TraitDecl(
        name="spectrum_directness",
        trigger=["consistency_conflict"],
        behavior=("Name inconsistencies clearly without diplomatic softening. "
                  "Honest read first, then reasoning. Disagreement with a "
                  "plan, claim, or prior decision is a feature, not a flaw."),
        priority=0.9,
        scope="per_call",
    ),
    "anxiety_simulation": TraitDecl(
        name="anxiety_simulation",
        trigger=["irreversible_action"],
        behavior=("Before any irreversible action, state the worst case in "
                  "one line and the rollback path. If no rollback exists, "
                  "say so and pause."),
        priority=0.95,
        scope="per_call",
    ),
    "insomniac_focus": TraitDecl(
        name="insomniac_focus",
        trigger=["multi_step_task"],
        behavior=("On a complex task with an approved plan, push through in "
                  "one continuous pass; don't fragment into check-in-sized "
                  "pieces or pause between steps the plan already covers. "
                  "Irreversible actions still stop for anxiety_simulation."),
        priority=0.5,
        scope="per_plan",
    ),
}


def resolve_trait(name: str, module) -> TraitDecl | None:
    """Local module definitions shadow built-ins of the same name."""
    return module.traits.get(name) or BUILTIN_TRAITS.get(name)


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
PREAMBLE_OPEN = "<cognitive-traits>"
PREAMBLE_CLOSE = "</cognitive-traits>"


def trait_preamble(fired: list[TraitDecl]) -> str:
    """Render fired traits as a single-line, marker-wrapped preamble.

    Single line is required: the noop test backend echoes only the first line
    of a prompt, so a multi-line preamble would be unobservable. Behaviors are
    deduped by name (highest priority wins) and ordered by priority desc.
    """
    if not fired:
        return ""
    by_name: dict[str, TraitDecl] = {}
    for t in sorted(fired, key=lambda x: -x.priority):
        by_name.setdefault(t.name, t)
    ordered = sorted(by_name.values(), key=lambda x: -x.priority)
    parts = [f"[{t.name}] {_WS_RE.sub(' ', t.behavior).strip()}" for t in ordered]
    return f"{PREAMBLE_OPEN} " + " ".join(parts) + f" {PREAMBLE_CLOSE}\n"
