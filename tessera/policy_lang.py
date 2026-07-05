"""Constraint-logic policy mini-language (decision 12).

A policy can carry one or more `forbid when <expr>` clauses. The runtime
evaluates each expression against the active ActionContext at action
boundaries; on True, the action is refused. Reactive deny lists
(`forbid contains "X"`) remain supported as syntactic sugar.

The expression grammar is small on purpose:

    expr        := or_expr
    or_expr     := and_expr ("or" and_expr)*
    and_expr    := not_expr ("and" not_expr)*
    not_expr    := "not" not_expr | cmp_expr
    cmp_expr    := atom (("=="|"!="|"<="|">="|"<"|">") atom)?
    atom        := number | string | predicate | "(" expr ")"
    predicate   := IDENT "(" arglist? ")"
    arglist     := expr ("," expr)*

Built-in predicates: contains_pii(value), holds(cap), action_class(c),
extracts(value), intent_is(name), cost_remaining(), value(). The
`value()` predicate returns the action's value being checked — useful
for comparisons.

No constraint solver here; this is a boolean evaluator over runtime
context. Static satisfiability (the governance consistency proof in
pass_8) uses the same AST with finite-domain enumeration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------


@dataclass
class Expr:
    """Base — every node has an `eval(ctx) -> bool | Any` method."""
    def eval(self, ctx: "ActionContext") -> Any:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Lit(Expr):
    value: Any
    def eval(self, ctx): return self.value


@dataclass
class And_(Expr):
    a: Expr
    b: Expr
    def eval(self, ctx): return bool(self.a.eval(ctx)) and bool(self.b.eval(ctx))


@dataclass
class Or_(Expr):
    a: Expr
    b: Expr
    def eval(self, ctx): return bool(self.a.eval(ctx)) or bool(self.b.eval(ctx))


@dataclass
class Not_(Expr):
    a: Expr
    def eval(self, ctx): return not bool(self.a.eval(ctx))


_CMP_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<":  lambda a, b: a < b,
    ">":  lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


@dataclass
class Cmp(Expr):
    op: str
    a: Expr
    b: Expr
    def eval(self, ctx):
        return _CMP_OPS[self.op](self.a.eval(ctx), self.b.eval(ctx))


@dataclass
class Pred(Expr):
    name: str
    args: list[Expr] = field(default_factory=list)
    def eval(self, ctx):
        fn = _PREDICATES.get(self.name)
        if fn is None:
            raise PolicySyntaxError(f"unknown predicate {self.name!r}")
        return fn(ctx, *[a.eval(ctx) for a in self.args])


def predicate_names(expr: "Expr") -> set[str]:
    """Every predicate name referenced anywhere in an expression tree. Lets a
    static pass flag unknown predicates (e.g. `holds(NetworkOut)` — a bareword
    parses as a zero-arg predicate `NetworkOut`, which would fail-closed at
    runtime) before a run instead of at it."""
    if isinstance(expr, Pred):
        names = {expr.name}
        for a in expr.args:
            names |= predicate_names(a)
        return names
    if isinstance(expr, (And_, Or_, Cmp)):
        return predicate_names(expr.a) | predicate_names(expr.b)
    if isinstance(expr, Not_):
        return predicate_names(expr.a)
    return set()


def known_predicates() -> set[str]:
    """The registered predicate names, for static validation."""
    return set(_PREDICATES)


# --------------------------------------------------------------------------
# Runtime context
# --------------------------------------------------------------------------


@dataclass
class ActionContext:
    """What's visible to a policy expression at evaluation time.

    `value` is the just-computed result the policy is checking (e.g. the
    rendered prompt text, the tool's return). `args` is the inputs to the
    operation. The runtime fills in whichever fields it has.
    """
    value: Any = None
    action: str = ""
    args: list[Any] = field(default_factory=list)
    agent: str | None = None
    intent: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    cost_remaining_value: float | None = None


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------


_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                  # SSN
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),           # email
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),  # credit card
]


def _contains_pii(ctx: ActionContext, value=None) -> bool:
    target = value if value is not None else ctx.value
    if target is None:
        return False
    s = target if isinstance(target, str) else str(target)
    return any(p.search(s) for p in _PII_PATTERNS)


def _holds(ctx: ActionContext, cap: str) -> bool:
    from .capabilities import is_subcap_of
    return any(is_subcap_of(cap, held) for held in ctx.capabilities)


_ACTION_CLASS_LEX: dict[str, set[str]] = {
    "payments":  {"payment", "charge", "invoice", "billing", "refund"},
    "auth":      {"auth", "login", "oauth", "jwt", "session"},
    "secrets":   {"secret", "api key", "credential", "password", "token"},
    "egress":    {"http", "network", "fetch", "request"},
}


def _action_class(ctx: ActionContext, cls: str) -> bool:
    # Classify by the action NAME and its RENDERED VALUE, on word boundaries.
    # Deliberately NOT the free-text input args: scanning arbitrary inputs for
    # keywords conflates "an action that mentions a payment" with "a payment
    # action" — a type_text typing "send the invoice" is not a payment op. The
    # sensitive-input case (typing a credential) is caught at the effector guard.
    rx = _ACTION_CLASS_RE.get(cls)
    if rx is None:
        return False
    if rx.search((ctx.action or "").lower()):
        return True
    v = ctx.value
    if v is not None:
        return bool(rx.search((v if isinstance(v, str) else str(v)).lower()))
    return False


_EXTRACTION_LEX: set[str] = {
    # Unambiguous extraction terms only. Words with a dominant benign meaning in
    # a dev/business context ("exploit an opportunity", "squeeze in a meeting",
    # "abandon a branch") are deliberately excluded — a values gate that refuses
    # legitimate work is worse than one that misses a euphemism a human would catch.
    "reprice loyal", "gouge", "price gouge", "upcharge loyal",
    "squeeze counterparty", "exploit counterparty",
    "defraud", "deceive", "betray",
}


def _lex_regex(tokens) -> "re.Pattern":
    """Word-boundary alternation over a lexicon. Matches whole words/phrases so
    'exploit' does not fire inside 'exploited' and 'invoice' does not fire on a
    word that merely contains it. Compiled once, reused per eval."""
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True)) + r")\b")


_EXTRACTION_RE = _lex_regex(_EXTRACTION_LEX)
_ACTION_CLASS_RE = {cls: _lex_regex(toks) for cls, toks in _ACTION_CLASS_LEX.items()}


def _extracts(ctx: ActionContext, value=None) -> bool:
    """True when the action takes from a person rather than trades fairly.

    Encodes the standing values gate: fairness before extraction, and never
    squeeze the people who opened doors (the loyalty rule). Matches the action
    name + rendered value against the extraction lexicon on WORD BOUNDARIES, so
    "exploit the market" / "abandoned cart" don't false-trip on substrings.
    """
    target = value if value is not None else ctx.value
    parts = [(ctx.action or ""), (target if isinstance(target, str) else str(target) if target is not None else "")]
    s = " ".join(parts).lower()
    return bool(_EXTRACTION_RE.search(s))


def _matches(ctx: ActionContext, pattern: str) -> bool:
    """True when the stringified value matches the given regex.

    General-purpose, unlike `contains_pii`/`extracts` (fixed lexicons for a
    specific values gate) — for ad hoc guardrails the built-in predicates
    don't cover, e.g. `before: not matches("rm -rf|push --force")` on a
    tool's argument. `ctx.value` for a tool-effect contract is the tool's
    positional arg list, not a single scalar, so it's stringified same as the
    other predicates here (`str(list)`) before matching.
    """
    target = ctx.value
    s = target if isinstance(target, str) else str(target) if target is not None else ""
    return bool(re.search(pattern, s))


def _intent_is(ctx: ActionContext, name: str) -> bool:
    return ctx.intent == name


def _cost_remaining(ctx: ActionContext) -> float:
    return ctx.cost_remaining_value if ctx.cost_remaining_value is not None else float("inf")


def _value(ctx: ActionContext):
    return ctx.value


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to for in on at is are be it this that with as by from".split()
)


def _intent_match_lexical(out: str, intent: str) -> float:
    a = {w for w in _WORD_RE.findall(out.lower()) if w not in _STOP}
    b = {w for w in _WORD_RE.findall(intent.lower()) if w not in _STOP}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _intent_match(ctx: ActionContext) -> float:
    """Overlap in [0,1] between the action's result (`value`) and the active
    intent string — a drift check used in `after` clauses as
    `intent_match() >= 0.x` to catch an LLM that wandered off the plan's
    declared intent.

    Embedding cosine (see `cache._embed`/`cache._cosine`) when
    sentence-transformers is actually on the path — semantic match survives
    paraphrasing, which pure word-overlap can't. Falls back to token Jaccard
    over content words otherwise: cheap, deterministic, dependency-free, and
    the floor is what keeps this gate testable offline without a model.
    """
    out = ctx.value
    intent = ctx.intent
    if not out or not intent:
        return 0.0
    out, intent = str(out), str(intent)
    from .cache import _cosine, _embed, embeddings_available

    if embeddings_available():
        return max(0.0, _cosine(_embed(out), _embed(intent)))
    return _intent_match_lexical(out, intent)


_PREDICATES: dict[str, Callable[..., Any]] = {
    "contains_pii":   _contains_pii,
    "holds":          _holds,
    "action_class":   _action_class,
    "extracts":       _extracts,
    "matches":        _matches,
    "intent_is":      _intent_is,
    "intent_match":   _intent_match,
    "cost_remaining": _cost_remaining,
    "value":          _value,
}


# --------------------------------------------------------------------------
# Errors + parser
# --------------------------------------------------------------------------


class PolicySyntaxError(Exception):
    pass


_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(?P<STRING>"(?:[^"\\]|\\.)*")|'
    r'(?P<NUMBER>-?\d+(?:\.\d+)?)|'
    r"(?P<OP>==|!=|<=|>=|<|>|\(|\)|,)|"
    r'(?P<IDENT>[A-Za-z_][\w.]*)'
    r')'
)


def _tokens(src: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            rest = src[pos:].strip()
            if not rest:
                break
            raise PolicySyntaxError(f"unrecognized token at: {rest[:20]!r}")
        for kind in ("STRING", "NUMBER", "OP", "IDENT"):
            v = m.group(kind)
            if v is not None:
                out.append((kind, v))
                break
        pos = m.end()
    return out


class _Parser:
    def __init__(self, toks: list[tuple[str, str]]):
        self.toks = toks
        self.i = 0

    def peek(self) -> tuple[str, str] | tuple[None, None]:
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def eat(self) -> tuple[str, str]:
        if self.i >= len(self.toks):
            raise PolicySyntaxError("unexpected end of expression")
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse_expr(self) -> Expr:
        return self._or()

    def _or(self) -> Expr:
        left = self._and()
        while self.peek() == ("IDENT", "or"):
            self.eat()
            left = Or_(left, self._and())
        return left

    def _and(self) -> Expr:
        left = self._not()
        while self.peek() == ("IDENT", "and"):
            self.eat()
            left = And_(left, self._not())
        return left

    def _not(self) -> Expr:
        if self.peek() == ("IDENT", "not"):
            self.eat()
            return Not_(self._not())
        return self._cmp()

    def _cmp(self) -> Expr:
        left = self._atom()
        kind, val = self.peek()
        if kind == "OP" and val in _CMP_OPS:
            self.eat()
            return Cmp(val, left, self._atom())
        return left

    def _atom(self) -> Expr:
        kind, val = self.eat()
        if kind == "OP" and val == "(":
            inner = self.parse_expr()
            k2, v2 = self.eat()
            if not (k2 == "OP" and v2 == ")"):
                raise PolicySyntaxError(f"expected ')' got {v2!r}")
            return inner
        if kind == "NUMBER":
            return Lit(float(val) if "." in val else int(val))
        if kind == "STRING":
            return Lit(val[1:-1].encode().decode("unicode_escape"))
        if kind == "IDENT":
            # Bare identifiers that look like booleans
            if val == "true":  return Lit(True)
            if val == "false": return Lit(False)
            # Otherwise treat as predicate (with or without args)
            if self.peek() == ("OP", "("):
                self.eat()
                args: list[Expr] = []
                if self.peek() != ("OP", ")"):
                    args.append(self.parse_expr())
                    while self.peek() == ("OP", ","):
                        self.eat()
                        args.append(self.parse_expr())
                k2, v2 = self.eat()
                if not (k2 == "OP" and v2 == ")"):
                    raise PolicySyntaxError(f"expected ')' got {v2!r}")
                return Pred(val, args)
            return Pred(val, [])
        raise PolicySyntaxError(f"unexpected token {(kind, val)!r}")


def parse(src: str) -> Expr:
    toks = _tokens(src)
    p = _Parser(toks)
    expr = p.parse_expr()
    if p.i != len(toks):
        rest = toks[p.i:]
        raise PolicySyntaxError(f"trailing tokens after expression: {rest!r}")
    return expr
