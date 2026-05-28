"""Tests for pass_9_consciousness_claim_check (Phase 5)."""
import pytest

from tessera.parser.module import parse_source
from tessera.sir.build import lower, SyntaxFail
from tessera.verify.passes import run_local


def test_pass_9_no_iit_no_welfare_no_diagnostic():
    """A module without consciousness-adjacent substrates isn't gated."""
    src = """---
agent: Plain
tessera_version: 0.2
---

```tsr:agent
agent Plain {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    diags = run_local(module)
    e1100 = [d for d in diags if d.code == "E1100"]
    assert e1100 == []


def test_pass_9_iit_with_clean_module_passes():
    """An iit module without forbidden claims compiles clean."""
    src = """---
agent: Mind
tessera_version: 0.2
---

```tsr:iit
iit {
  emit_phi_audit: true
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
    diags = run_local(module)
    e1100 = [d for d in diags if d.code == "E1100"]
    assert e1100 == []


def test_pass_9_iit_body_already_rejects_at_lower_time():
    """The per-substrate check (iit's _lower_iit) catches forbidden
    claims BEFORE pass_9 ever runs. pass_9 is the cross-module belt."""
    src = """---
agent: Bad
tessera_version: 0.2
---

```tsr:iit
iit {
  emit_phi_audit: true
  // this agent is conscious when phi > 0
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
    with pytest.raises(SyntaxFail, match="forbidden consciousness claim"):
        lower(pm)
