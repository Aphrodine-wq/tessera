"""Test fixtures.

Both auto-fixtures point Tessera's persistent stores at per-test temp
files so tests don't pollute (or read from) the developer's real
`~/.tessera/` directory.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_tessera_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERA_SEMANTIC_DB", str(tmp_path / "semantic.db"))
    # Tiered audit: governance + operational each get their own tmp file
    # so tests can verify the routing without polluting the dev's stores.
    monkeypatch.setenv("TESSERA_AUDIT_GOV_DB", str(tmp_path / "audit_gov.db"))
    monkeypatch.setenv("TESSERA_AUDIT_OPS_DB", str(tmp_path / "audit_ops.db"))
    # Force the deterministic backend in tests so prompts don't hit the
    # network and so output is reproducible across machines.
    monkeypatch.setenv("TESSERA_LLM_BACKEND", "noop")
