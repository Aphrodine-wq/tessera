"""Cold-vs-warm benchmarks — evidence for the cache speedups, not micro-tuning.

Each asserts that the warm path (in-memory / cached) beats the cold path
(file read + parse + index build) it replaces. Bounds are loose by design so
the suite stays green on slow CI; the point is warm < cold, reliably, because
cold does real I/O the warm path skips. Timings print under `pytest -s`.
"""
from __future__ import annotations

import json
import time

from tessera import cache as cmod
from tessera.cache import (
    parse_file_cached, semantic_cache_lookup, semantic_cache_put,
)


def test_parse_cache_warm_beats_cold(tmp_path):
    f = tmp_path / "a.t.md"
    f.write_text(
        "---\nagent: A\ntessera_version: 0.2\n---\n\n"
        "```tsr:agent\nagent A {\n  beliefs:\n    @last_write q: String\n"
        "  intentions:\n    plan go { return q }\n}\n```\n"
    )
    cmod.invalidate_parse_cache()
    t0 = time.perf_counter()
    parse_file_cached(f)            # cold: real parse
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(50):
        parse_file_cached(f)        # warm: lru hit, no parse
    warm = (time.perf_counter() - t0) / 50

    print(f"\nparse cold={cold*1e6:.1f}us warm={warm*1e6:.1f}us")
    assert warm < cold


def test_semantic_cache_warm_lookup_beats_cold(tmp_path, monkeypatch):
    # conftest already points TESSERA_CACHE_DIR at tmp; seed a sizeable cache.
    for i in range(500):
        semantic_cache_put(f"prompt number {i}", f"answer {i}")
    probe = "prompt number 250"

    # cold: force a reload from disk (build the in-memory index over 500 rows)
    cmod._SEM_MEM = None
    t0 = time.perf_counter()
    hit = semantic_cache_lookup(probe)
    cold = time.perf_counter() - t0
    assert hit is not None and hit["text"] == "answer 250"

    # warm: exact-hash hit against the in-memory index
    t0 = time.perf_counter()
    for _ in range(100):
        semantic_cache_lookup(probe)
    warm = (time.perf_counter() - t0) / 100

    print(f"semantic cold={cold*1e6:.1f}us warm={warm*1e6:.1f}us "
          f"(speedup {cold/max(warm,1e-9):.0f}x)")
    assert warm < cold


def test_semantic_cache_file_unchanged_on_warm(tmp_path):
    """Warm lookups must not rewrite the JSONL (no accidental I/O amplification)."""
    semantic_cache_put("hello", "world")
    path = cmod._semantic_cache_path()
    before = path.read_text()
    for _ in range(20):
        semantic_cache_lookup("hello")
    assert path.read_text() == before
    # and the row is intact
    assert any(json.loads(line)["prompt"] == "hello"
               for line in before.splitlines())
