# AEON Bridge — Tessera Verification Adapter

How Tessera's SIR is verified by AEON's engine pipeline.

## AEON surface (as of 2026-05-21)

AEON lives at `~/Projects/walt/aeon/`. Programmatic API:

```python
from aeon.language_adapter import verify_file, verify

result = verify_file(filepath, deep_verify=True, prove_kwargs=None)
# result: VerificationResult
#   .errors: List[Dict] — id, severity, category, location{file,line,column}, message, cwe?
#   .warnings: List[Dict]
#   .verified: bool
#   .summary: str
```

Engines (~73) are hardcoded in `aeon/engines/`. **No runtime plugin system** — extensions live in AEON's tree.

`prove_kwargs` selects engine subsets:
```python
prove_kwargs = {"taint_analysis": True, "secret_detection": True, "auth_check": True, ...}
```

## Adapter strategy: AEON-side language target

Two-file change inside AEON:

1. **`aeon/adapters/tessera_adapter.py`** — parses `.sir.txt` (textual SIR per RFC §9.1) into AEON's Program AST. Registers as a new language in `language_adapter.py`'s dispatch.
2. **`aeon/engines/tessera_*.py`** — thin engine wrappers that implement the 8 SIR verification passes (RFC §7) by walking the AEON AST produced from SIR.

Tessera-side, the adapter is one file:

```python
# tessera/adapters/aeon/__init__.py
from aeon.language_adapter import verify_file

def verify_sir(sir_path: str) -> list[Diagnostic]:
    result = verify_file(sir_path, deep_verify=True, prove_kwargs=TESSERA_PROVE_KWARGS)
    return [_map_to_diagnostic(e) for e in result.errors + result.warnings]
```

## Pass mapping (RFC §7 → AEON engines)

| RFC Pass | AEON engine | Error codes |
|---|---|---|
| 1. Substrate adjacency | `tessera_substrate_adjacency` (new) | E001 |
| 2. Effect inference + capability check | `tessera_effect_capability` (new) + `auth_access_control` (existing) | E101–E103 |
| 3. Cost aggregation | `tessera_cost` (new) + `resource_logic` (existing) | E201, W202 |
| 4. PII flow analysis | `taint_analysis` (existing, repurposed) | E301, E302 |
| 5. Revision policy check | `tessera_revision_policy` (new) | E401, E402 |
| 6. Higher-order depth check | `tessera_meta_depth` (new) | E501 |
| 7. Quantum dimension check | `tessera_quantum_dim` (new) | W601, E602 |
| 8. Determinism check | `tessera_determinism` (new) | E701 |

MVP implements passes 1–2 only. Rest stubbed.

## Diagnostic round-trip

AEON returns:
```json
{"id": "AUTH-014", "severity": "error", "category": "auth_access_control",
 "location": {"file": "...", "line": 42, "column": 7},
 "message": "...", "cwe": "CWE-862"}
```

Tessera maps to its own RFC §Appendix-C error code via the `category` field. Unmapped diagnostics surface verbatim with a `[aeon]` prefix.

## Tradeoffs we picked

- **Add adapter to AEON's tree, not fork.** Additive — no existing engine touched. James owns both repos so this is friction-free.
- **Shell out is the fallback.** If the in-tree adapter regresses, `subprocess.run(["aeon", "check", sir_path, "--format", "json"])` is the panic button.
- **Reuse existing engines where possible.** PII flow = `taint_analysis`. Capability = `auth_access_control`. Don't reinvent.
