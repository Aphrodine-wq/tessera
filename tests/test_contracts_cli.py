"""Tests for the contract verification surfaces: `tessera contracts`,
`tessera doctor`, contract-aware `tessera eval`, and the audit-tier routing of
contract events.

These drive the real CLI entry point (`tessera.cli.main`) so the argparse
wiring and exit codes are covered, not just the underlying functions. The
conftest auto-fixture points the audit stores at per-test temp files, so
`run` + `query` see a clean, isolated graph.
"""
from tessera.cli import main, _is_refusal
from tessera.adapters.audit import _classify
from tessera.interp.eval import Refusal


_CONTRACT_FILE = '''---
agent: QuoteExplainer
capabilities_requested: []
---

```tsr:contract
contract honest on prompt:explain {
  before: not contains_pii(value())
  on_violation: refuse
}
```

```tsr:intent
intent explain_quote {
  goal: "Explain a quote a homeowner can act on"
  success: explanation_present
  why: "Real money decisions ride on it"
}
```

```tsr:prompt
prompt explain(q: String) -> String = "Explain plainly: {q}"
```

```tsr:agent
agent QuoteExplainer intends explain_quote {
  beliefs:
    @last_write quote: String
  intentions:
    plan walk serves explain_quote { let e = explain(quote) return e }
}
```

```tsr:eval
eval {
  case "PII is refused before cost" {
    input quote = "bill SSN 123-45-6789"
    expect_refusal = true
  }
}
```
'''


def _write(tmp_path, body=_CONTRACT_FILE):
    p = tmp_path / "agent.t.md"
    p.write_text(body)
    return str(p)


# --------------------------------------------------------------- _is_refusal

def test_is_refusal_matches_objects_and_sentinels():
    assert _is_refusal(Refusal(reason="x", policy="y"))
    assert _is_refusal("[contract-refused: c — before: ...]")
    assert _is_refusal("[approval-required: spend]")
    assert _is_refusal("[precaution-refused: irreversible]")
    assert not _is_refusal("a normal answer")
    assert not _is_refusal("[noop:abcd]")        # not a refusal sentinel
    assert not _is_refusal(42)


# --------------------------------------------------------------- audit tier

def test_contract_refuse_and_error_are_governance_tier():
    # Refusals/errors are permanent proof-trail; retry/audit are routine.
    assert _classify({"action": "contract:refuse"}) == "governance"
    assert _classify({"action": "contract:error"}) == "governance"
    assert _classify({"action": "contract:retry"}) == "operational"
    assert _classify({"action": "contract:audit"}) == "operational"


# --------------------------------------------------------------- tessera eval

def test_eval_catches_contract_refusal(tmp_path, capsys):
    rc = main(["eval", _write(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1/1 cases passed" in out
    assert "[PASS]" in out


# --------------------------------------------------------------- tessera contracts

def test_contracts_lists_and_counts(tmp_path, capsys):
    f = _write(tmp_path)
    # Run once so there's a contract:refuse in the audit graph.
    main(["compile", f, "--run", "QuoteExplainer", "--set", "quote=bill SSN 123-45-6789"])
    capsys.readouterr()                          # drain compile output
    rc = main(["contracts", f])
    out = capsys.readouterr().out
    assert rc == 0
    assert "honest" in out and "on prompt:explain" in out
    assert "before: not contains_pii(value())" in out
    assert "on_violation: refuse" in out
    assert "refuse=" in out                      # live audit-derived count


def test_contracts_json(tmp_path, capsys):
    import json
    rc = main(["contracts", _write(tmp_path), "--json"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    obj = json.loads(out.splitlines()[0])
    assert obj["name"] == "honest"
    assert obj["target"] == "prompt:explain"
    assert obj["on_violation"] == ["refuse", 0, ""]


def test_contracts_flags_unknown_predicate(tmp_path, capsys):
    """A bareword capability (holds(NetworkOut)) is an unknown predicate — the
    contracts view surfaces the E833 warning so the trap is visible."""
    body = _CONTRACT_FILE.replace(
        "before: not contains_pii(value())", "before: holds(NetworkOut)")
    rc = main(["contracts", _write(tmp_path, body)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "E833" in out


# --------------------------------------------------------------- tessera doctor

def test_doctor_healthy(tmp_path, capsys):
    rc = main(["doctor", _write(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "health:     OK" in out
    assert "1/1 cases passed" in out
    assert "honest→prompt:explain" in out


def test_doctor_reports_errors(tmp_path, capsys):
    """A contract on a phantom target (E830 error) makes doctor exit non-zero."""
    body = _CONTRACT_FILE.replace("on prompt:explain", "on prompt:nonexistent")
    rc = main(["doctor", _write(tmp_path, body)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NEEDS ATTENTION" in out
    assert "E830" in out


def test_doctor_json(tmp_path, capsys):
    import json
    rc = main(["doctor", _write(tmp_path), "--json"])
    obj = json.loads(capsys.readouterr().out.strip())
    assert rc == 0
    assert obj["healthy"] is True
    assert obj["contracts"] == {"honest": "prompt:explain"}
    assert obj["eval"]["passed"] == 1
