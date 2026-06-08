---
agent: QuoteExplainer
capabilities_requested: []
tessera_version: 0.2
---

# Runtime contracts — guarantees the interpreter enforces while it runs

Tessera's other gates are static: the `verify/` pass system checks substrate
and capability boundaries *before* a run. A `tsr:contract` is the runtime half —
author-declared before/after assertions bound to a named effect, enforced at the
moment that effect fires.

A contract is the **inverse of a `tsr:policy`**. A policy is a prohibition:
`forbid when <expr>` refuses when the expression is *true*. A contract is a
**guarantee**: every `before`/`after` clause is an assertion that *must hold*,
and a clause evaluating *false* is the violation. Read `before: not
contains_pii(value())` as "this must be true to proceed."

This agent explains a construction quote to a homeowner. The contract guarantees:

- **`before`** — the homeowner's message carries no PII before we ever spend a
  token on it (`not contains_pii(value())`).
- **`after`** — the explanation stayed on the intent it was asked to serve
  (`intent_match() >= 0.2`) and didn't drift into an extractive pitch
  (`not extracts(value())`).
- **`on_violation: retry(2) then refuse`** — a drifting answer is re-driven up
  to twice; if it still won't land, the agent refuses rather than ship it.

```tsr:intent
intent explain_quote {
  goal: "Explain a construction quote so a homeowner can act on it"
  success: explanation_present
  why: "Homeowners make real money decisions on this — it must be honest and never upsell"
}
```

```tsr:contract
contract honest_explanation on prompt:explain {
  before: not contains_pii(value())
  after: intent_match() >= 0.2
  after: not extracts(value())
  on_violation: retry(2) then refuse
}
```

```tsr:prompt
prompt explain(quote: String) -> String = "Explain this quote plainly to a homeowner: {quote}"
```

```tsr:agent
agent QuoteExplainer intends explain_quote {
  beliefs:
    @last_write quote: String
  intentions:
    plan walk_through serves explain_quote {
      let explanation = explain(quote)
      return explanation
    }
}
```

The `before` clause is checkable without an LLM, so a contract refusal is a
declared, testable guarantee — `tessera eval` asserts it directly:

```tsr:eval
eval {
  case "PII in the quote is refused before any cost" {
    input quote = "bill SSN 123-45-6789 for the reroof"
    expect_refusal = true
  }
}
```

Run it:

```bash
# A clean quote passes the before-gate and runs.
tessera compile examples/contracts.t.md --run QuoteExplainer \
    --set quote="2 squares of architectural shingles, tear-off included"

# A quote carrying PII is refused before any LLM cost — the before-clause holds
# the boundary the homeowner never asked us to cross.
tessera compile examples/contracts.t.md --run QuoteExplainer \
    --set quote="bill SSN 123-45-6789 for the reroof"

# Inspect what the contract did — every check lands in the audit graph.
tessera audit query --action contract:refuse
tessera audit query --action contract:retry
```
