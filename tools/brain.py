"""Shared helpers for the Brain worker crew. Stdlib + Tessera adapters only.

Every worker is the same shape: read input, pull knowledge, think, hand back
something useful. These helpers are the common machinery so each worker stays
~30 lines and they all behave the same.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

VAULT = Path.home() / "Desktop" / "TheVault"
PROJECTS = Path.home() / "Projects"


def llm(prompt: str) -> str:
    """One LLM call via Tessera's adapter (TESSERA_LLM_BACKEND/_OLLAMA_MODEL)."""
    from tessera.adapters.llm import get_backend
    return get_backend().complete(prompt).text.strip()


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    ).stdout


def facts(contains: str | None = None, schema: str = "VaultNote", limit: int = 40):
    try:
        from tessera.adapters.semantic import query_facts
    except Exception:
        return []
    return query_facts(schema=schema, contains=contains, limit=limit)


_STOP = {
    "what", "about", "did", "does", "the", "with", "that", "this", "from",
    "decided", "decide", "should", "would", "could", "have", "were", "will",
    "your", "our", "into", "they", "them", "then", "than", "when", "where",
    "which", "while", "there", "their", "been", "over", "just", "like", "make",
}


def keywords(text: str, n: int = 5, minlen: int = 4) -> list[str]:
    seen: list[str] = []
    for w in re.findall(rf"[A-Za-z]{{{minlen},}}", (text or "").lower()):
        if w not in _STOP and w not in seen:
            seen.append(w)
    return seen[:n]


def gather_notes(query: str, limit: int = 12, schemas=("VaultNote", "FTWRule")) -> list[dict]:
    """Fact-fields that share a salient word with the query, best matches first.

    Searches multiple schemas and ranks by how many query keywords each note
    hits, so a focused answer beats whatever happened to match one word.
    """
    kws = keywords(query) or ["the"]
    scored: dict[str, tuple[int, dict]] = {}
    for schema in schemas:
        for kw in kws:
            for r in facts(contains=kw, schema=schema, limit=25):
                f = r["fields"]
                key = f.get("source_path") or f.get("title") or repr(f)
                hits, _ = scored.get(key, (0, f))
                scored[key] = (hits + 1, f)
    ranked = sorted(scored.values(), key=lambda t: -t[0])
    return [f for _, f in ranked[:limit]]


def git_repos() -> list[Path]:
    """Project repos under ~/Projects (and one level into monorepos like walt/)."""
    repos: list[Path] = []
    if not PROJECTS.is_dir():
        return repos
    for p in sorted(PROJECTS.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / ".git").is_dir():
            repos.append(p)
        else:
            for c in sorted(p.iterdir()):
                if c.is_dir() and (c / ".git").is_dir():
                    repos.append(c)
    return repos


def input_text(argv: list[str]) -> str:
    """Joined args, or piped stdin if no args were given."""
    t = " ".join(argv).strip()
    if t:
        return t
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""
