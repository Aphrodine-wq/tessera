"""Tessera cache — speed up parse + verify on warm runs.

Two independent caches:

  - **Parse cache** keyed by ``(file_path, file_mtime)`` — short-circuits
    Markdown parsing when the underlying file is unchanged. The value is the
    full ParsedModule (in-memory; LRU-bounded).

  - **Verify cache** keyed by the textual SIR's blake2b hash — short-circuits
    AEON's ``verify_file`` when we've already verified the same SIR. Persisted
    to ``~/.cache/tessera/verify.jsonl`` so it survives across runs.

Both caches degrade gracefully: a cache miss just runs the underlying work.
A cache-write failure is logged at warn level and the run continues.

Disable either via env vars:
  - ``TESSERA_NO_PARSE_CACHE=1``
  - ``TESSERA_NO_VERIFY_CACHE=1``
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from functools import lru_cache
from pathlib import Path


_CACHE_DIR = Path(os.environ.get("TESSERA_CACHE_DIR")
                  or (Path.home() / ".cache" / "tessera"))


def _ensure_cache_dir() -> Path:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        warnings.warn(f"could not create cache dir {_CACHE_DIR}: {e}")
    return _CACHE_DIR


# ---------- parse cache ----------


def parse_cache_disabled() -> bool:
    return os.environ.get("TESSERA_NO_PARSE_CACHE") == "1"


@lru_cache(maxsize=512)
def _parse_cached(path_str: str, mtime_ns: int):
    """Inner cached function — keyed by path + mtime, so any edit invalidates."""
    from .parser.module import parse_file
    return parse_file(path_str)


def parse_file_cached(path: str | Path):
    """Parse with mtime-keyed cache. Returns the same ParsedModule on warm hits."""
    p = Path(path)
    if parse_cache_disabled() or not p.exists():
        from .parser.module import parse_file
        return parse_file(p)
    mtime_ns = p.stat().st_mtime_ns
    return _parse_cached(str(p), mtime_ns)


def invalidate_parse_cache() -> None:
    _parse_cached.cache_clear()


def parse_cache_stats() -> dict:
    info = _parse_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "max": info.maxsize,
    }


# ---------- verify cache ----------


def verify_cache_disabled() -> bool:
    return os.environ.get("TESSERA_NO_VERIFY_CACHE") == "1"


def _verify_cache_path() -> Path:
    return _ensure_cache_dir() / "verify.jsonl"


def _sir_hash(sir_text: str) -> str:
    h = hashlib.blake2b(sir_text.encode("utf-8"), digest_size=16)
    return h.hexdigest()


def verify_cache_get(sir_text: str) -> list | None:
    if verify_cache_disabled():
        return None
    key = _sir_hash(sir_text)
    p = _verify_cache_path()
    if not p.exists():
        return None
    try:
        with p.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("hash") == key:
                    return row.get("diagnostics") or []
    except OSError as e:
        warnings.warn(f"verify cache read failed: {e}")
    return None


def verify_cache_put(sir_text: str, diagnostics: list) -> None:
    if verify_cache_disabled():
        return
    key = _sir_hash(sir_text)
    p = _verify_cache_path()
    try:
        with p.open("a") as f:
            f.write(json.dumps({"hash": key, "diagnostics": diagnostics}) + "\n")
    except OSError as e:
        warnings.warn(f"verify cache write failed: {e}")


def clear_verify_cache() -> None:
    p = _verify_cache_path()
    if p.exists():
        p.unlink()


# ---------- semantic prompt cache ----------


def semantic_cache_disabled() -> bool:
    return os.environ.get("TESSERA_NO_SEMANTIC_CACHE") == "1"


def _semantic_cache_path() -> Path:
    return _ensure_cache_dir() / "semantic_prompts.jsonl"


def _embed(text: str) -> list[float] | None:
    """Try sentence-transformers first; fall back to hashed-bag-of-tokens.

    Hashed fallback is intentionally crude — it only helps for IDENTICAL or
    near-identical prompts. The real semantic win lands when
    sentence-transformers (or any embedding lib) is on the path.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        global _ST_MODEL  # noqa: PLW0603
        if "_ST_MODEL" not in globals():
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # 384-d
        vec = _ST_MODEL.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]
    except ImportError:
        pass
    # Hashed-bag fallback — same string → same vector, paraphrases miss
    import hashlib
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=48).digest()
    return [b / 255.0 for b in h]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_cache_lookup(prompt: str, *, threshold: float = 0.95) -> dict | None:
    """Return a cached completion if a previously-seen prompt is semantically close.

    threshold default 0.95 — tight enough to avoid hallucinated matches, loose
    enough to catch paraphrases when a real embedding model is on path.
    """
    if semantic_cache_disabled():
        return None
    p = _semantic_cache_path()
    if not p.exists():
        return None
    q_vec = _embed(prompt)
    if q_vec is None:
        return None
    best = None
    best_sim = 0.0
    try:
        with p.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sim = _cosine(q_vec, row.get("embedding") or [])
                if sim > best_sim:
                    best_sim = sim
                    best = row
    except OSError:
        return None
    if best and best_sim >= threshold:
        return {
            "text": best.get("text", ""),
            "backend": best.get("backend", "cache"),
            "model": best.get("model", "cache"),
            "similarity": best_sim,
            "cached_prompt": best.get("prompt", ""),
        }
    return None


def semantic_cache_put(prompt: str, text: str, backend: str = "", model: str = "") -> None:
    if semantic_cache_disabled():
        return
    embedding = _embed(prompt)
    if embedding is None:
        return
    try:
        with _semantic_cache_path().open("a") as f:
            f.write(json.dumps({
                "prompt": prompt,
                "text": text,
                "backend": backend,
                "model": model,
                "embedding": embedding,
            }) + "\n")
    except OSError as e:
        warnings.warn(f"semantic cache write failed: {e}")


def clear_semantic_cache() -> None:
    p = _semantic_cache_path()
    if p.exists():
        p.unlink()
