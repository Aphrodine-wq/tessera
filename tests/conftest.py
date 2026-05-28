"""Test fixtures.

`isolate_semantic_db` points the semantic adapter at a per-test temp
SQLite file so tests don't pollute (or read from) the developer's real
`~/.tessera/semantic.db`.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_semantic_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERA_SEMANTIC_DB", str(tmp_path / "semantic.db"))
