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
    # Checkpoints + corpora isolated per test as well.
    monkeypatch.setenv("TESSERA_CHECKPOINTS_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("TESSERA_CORPORA_DIR", str(tmp_path / "corpora"))
    monkeypatch.setenv("TESSERA_RL_DIR", str(tmp_path / "rl"))
    # Parse/verify/semantic caches → per-test dir so a run never reads from or
    # writes to the dev's real ~/.cache/tessera.
    monkeypatch.setenv("TESSERA_CACHE_DIR", str(tmp_path / "cache"))
    # Force the deterministic backend in tests so prompts don't hit the
    # network and so output is reproducible across machines.
    monkeypatch.setenv("TESSERA_LLM_BACKEND", "noop")
    # Reset the process-global cache singletons so in-memory state from a prior
    # test can never bleed into this one (the caches are keyed by path+mtime,
    # but resetting is the cheap, certain guarantee of isolation).
    from tessera import cache as _cache
    _cache._SEM_MEM = None
    _cache._VERIFY_MEM = None
    _cache.invalidate_parse_cache()
