"""Tessera cache — speed up parse + prompt resolution on warm runs.

Two independent caches:

  - **Parse cache** keyed by ``(file_path, file_mtime)`` — short-circuits
    Markdown parsing when the underlying file is unchanged. The value is the
    full ParsedModule (in-memory; LRU-bounded).

  - **Semantic prompt cache** keyed by the prompt text's hash — reuses prior
    LLM resolutions for identical or near-identical prompts.

Both caches degrade gracefully: a cache miss just runs the underlying work.
A cache-write failure is logged at warn level and the run continues.

Disable either via env vars:
  - ``TESSERA_NO_PARSE_CACHE=1``
  - ``TESSERA_NO_SEMANTIC_CACHE=1``
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from functools import lru_cache
from pathlib import Path


def _cache_dir() -> Path:
    """Resolve the cache dir on every call so a mid-process env change (tests
    isolating ``TESSERA_CACHE_DIR`` per case) takes effect."""
    return Path(os.environ.get("TESSERA_CACHE_DIR")
                or (Path.home() / ".cache" / "tessera"))


def _ensure_cache_dir() -> Path:
    d = _cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        warnings.warn(f"could not create cache dir {d}: {e}")
    return d


def _text_hash(text: str) -> str:
    """blake2b-16 hex of a UTF-8 string — used for exact cache keys."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


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


# In-memory prompt cache, single-slot, keyed by (path, mtime). Holds an exact
# hash→row map (the common case: an identical rendered prompt re-run) and the
# row list for the embedding-similarity fallback. Loaded once per file version
# instead of re-reading + re-embedding the whole JSONL on every prompt call.
_SEM_MEM: dict | None = None


def _row_result(row: dict, sim: float) -> dict:
    return {
        "text": row.get("text", ""),
        "backend": row.get("backend", "cache"),
        "model": row.get("model", "cache"),
        "similarity": sim,
        "cached_prompt": row.get("prompt", ""),
    }


def _load_sem_mem(p: Path) -> dict:
    global _SEM_MEM  # noqa: PLW0603
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        _SEM_MEM = {"path": str(p), "mtime": -1, "exact": {}, "rows": []}
        return _SEM_MEM
    if (_SEM_MEM is not None and _SEM_MEM["path"] == str(p)
            and _SEM_MEM["mtime"] == mtime):
        return _SEM_MEM
    exact: dict[str, dict] = {}
    rows: list[dict] = []
    try:
        with p.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(row)
                pr = row.get("prompt")
                if pr is not None:
                    exact[_text_hash(pr)] = row
    except OSError:
        pass
    _SEM_MEM = {"path": str(p), "mtime": mtime, "exact": exact, "rows": rows}
    return _SEM_MEM


def semantic_cache_lookup(prompt: str, *, threshold: float = 0.95) -> dict | None:
    """Return a cached completion if a previously-seen prompt matches.

    Exact-hash hit returns in O(1) with no embedding. On an exact miss we fall
    back to embedding cosine similarity over the in-memory rows; threshold
    default 0.95 — tight enough to avoid hallucinated matches, loose enough to
    catch paraphrases when a real embedding model is on path.
    """
    if semantic_cache_disabled():
        return None
    p = _semantic_cache_path()
    if not p.exists():
        return None
    mem = _load_sem_mem(p)
    # exact fast path — no embedding, no scan
    row = mem["exact"].get(_text_hash(prompt))
    if row is not None:
        return _row_result(row, 1.0)
    # similarity fallback over the in-memory rows
    q_vec = _embed(prompt)
    if q_vec is None:
        return None
    best = None
    best_sim = 0.0
    for row in mem["rows"]:
        sim = _cosine(q_vec, row.get("embedding") or [])
        if sim > best_sim:
            best_sim = sim
            best = row
    if best and best_sim >= threshold:
        return _row_result(best, best_sim)
    return None


def semantic_cache_put(prompt: str, text: str, backend: str = "", model: str = "") -> None:
    if semantic_cache_disabled():
        return
    embedding = _embed(prompt)
    if embedding is None:
        return
    row = {
        "prompt": prompt,
        "text": text,
        "backend": backend,
        "model": model,
        "embedding": embedding,
    }
    p = _semantic_cache_path()
    try:
        with p.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        warnings.warn(f"semantic cache write failed: {e}")
        return
    global _SEM_MEM  # noqa: PLW0603
    if _SEM_MEM is not None and _SEM_MEM["path"] == str(p):
        _SEM_MEM["rows"].append(row)
        _SEM_MEM["exact"][_text_hash(prompt)] = row
        try:
            _SEM_MEM["mtime"] = p.stat().st_mtime_ns
        except OSError:
            pass


def clear_semantic_cache() -> None:
    global _SEM_MEM  # noqa: PLW0603
    _SEM_MEM = None
    p = _semantic_cache_path()
    if p.exists():
        p.unlink()
