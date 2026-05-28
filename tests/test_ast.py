"""Tests for the AST (Attention Schema Theory) substrate (research C2)."""
import math
import pytest

from tessera.ast_substrate import AttentionSchema, ASTConfig
from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail


def test_attention_schema_default_fidelity_is_vacuously_honest():
    """An untouched schema has no claims to test → fidelity = 1.0."""
    schema = AttentionSchema()
    assert schema.fidelity() == 1.0


def test_attention_schema_perfect_fidelity():
    schema = AttentionSchema()
    schema.update_from_workspace("apple")
    schema.record_truth("apple")
    schema.update_from_workspace("banana")
    schema.record_truth("banana")
    assert schema.fidelity() == 1.0


def test_attention_schema_partial_fidelity():
    schema = AttentionSchema()
    schema.update_from_workspace("apple")
    schema.record_truth("orange")  # mismatch
    schema.update_from_workspace("banana")
    schema.record_truth("banana")  # match
    assert math.isclose(schema.fidelity(), 0.5)


def test_attention_schema_report_returns_current_state():
    schema = AttentionSchema(current_focus="x", confidence=0.85)
    rep = schema.report()
    assert rep["focus"] == "x"
    assert math.isclose(rep["confidence"], 0.85)


def test_ast_substrate_parses_default_fields():
    src = """---
agent: Mind
tessera_version: 0.2
---

```tsr:ast
ast {
  min_fidelity: 0.6
  refuse_below_threshold: true
}
```

```tsr:agent
agent Mind {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert module.ast is not None
    assert math.isclose(module.ast.min_fidelity, 0.6)
    assert module.ast.refuse_below_threshold is True


def test_ast_substrate_rejects_invalid_fidelity():
    src = """---
agent: Bad
tessera_version: 0.2
---

```tsr:ast
ast {
  min_fidelity: 1.5
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
    with pytest.raises(SyntaxFail, match="\\[0, 1\\]"):
        lower(pm)
