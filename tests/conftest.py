"""Test fixtures.

Both auto-fixtures point Tessera's persistent stores at per-test temp
files so tests don't pollute (or read from) the developer's real
`~/.tessera/` directory.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_tessera_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERA_SEMANTIC_DB", str(tmp_path / "semantic.db"))
    monkeypatch.setenv("TESSERA_AUDIT_DB", str(tmp_path / "audit.db"))
