#!/usr/bin/env python3
"""Capture Clerk — turn a one-line brain-dump into a clean, filed vault note.

    capture "charge a setup fee separate from monthly so churn doesn't eat it"

The judgment (which folder, what title, which tags, what to link) is the LLM's,
working off the real vault folder taxonomy and existing note titles. The
mechanics and the guardrails are code, so it cannot misfile:

  - folder MUST be one of the vault's real top-level folders, else → 000 Inbox
  - writes ONLY inside the vault; never the Desktop root, never another vault
  - frontmatter matches the vault's convention (title/created/type/tags)
  - a zettel gets a Connections section with [[wikilinks]] to related notes

Uses Tessera's LLM adapter (TESSERA_LLM_BACKEND / TESSERA_OLLAMA_MODEL) for the
thinking and the ingested fact store to surface link candidates. Cloud models
can't be grammar-constrained, so the JSON is parsed with a repair fallback and
then validated in code — belt and suspenders.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

VAULT = Path.home() / "Desktop" / "TheVault"


def vault_folders() -> list[str]:
    """Real top-level folders (numbered), the only legal filing targets."""
    return sorted(
        p.name for p in VAULT.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name[:1].isdigit()
    )


def link_candidates(thought: str, limit: int = 25) -> list[str]:
    """Existing note titles that share a salient word with the thought."""
    try:
        from tessera.adapters.semantic import query_facts
    except Exception:
        return []
    words = [w for w in re.findall(r"[A-Za-z]{5,}", thought.lower())]
    seen: dict[str, None] = {}
    for w in words[:4]:
        for r in query_facts(schema="VaultNote", contains=w, limit=40):
            t = r["fields"].get("title")
            if t and t not in seen:
                seen[t] = None
            if len(seen) >= limit:
                break
    return list(seen)


PROMPT = """You are James's vault filing clerk. Turn his raw thought into one clean note.

His thought:
{thought}

Legal folders (choose EXACTLY one, copy it verbatim; if unsure use "000 Inbox"):
{folders}

Existing note titles you may link to (pick 0-3 that are genuinely related, copy verbatim):
{candidates}

Return ONLY a JSON object, no prose, no code fence:
{{"folder": "<one legal folder>",
  "title": "<short title, no quotes, no slashes>",
  "type": "<zettelkasten|idea|decision|reference|inbox>",
  "tags": ["<3-5 lowercase-hyphen tags>"],
  "links": ["<exact existing titles>"],
  "body": "<2-4 sentence clean writeup of the idea in James's plain, direct voice>"}}"""


def extract_json(text: str) -> dict | None:
    text = text.strip()
    # Strip a ```json fence if present, then grab the outermost {...}.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def sanitize_title(t: str) -> str:
    t = re.sub(r"[\\/:*?\"<>|]", "", t).strip()
    return (t or "Untitled")[:80]


def slug_tags(tags) -> list[str]:
    out = []
    for t in (tags or []):
        s = re.sub(r"[^a-z0-9]+", "-", str(t).lower()).strip("-")
        if s:
            out.append(s)
    return out[:5] or ["inbox"]


def build_note(decision: dict, today: str) -> str:
    title = sanitize_title(decision.get("title", "Untitled"))
    ntype = decision.get("type") or "inbox"
    tags = slug_tags(decision.get("tags"))
    body = (decision.get("body") or "").strip()
    fm = [
        "---",
        f'title: "{title}"',
        f"created: {today}",
        f"type: {ntype}",
        f"tags: [{', '.join(tags)}]",
    ]
    if ntype == "zettelkasten":
        fm.append("status: seedling")
    fm.append("---")

    parts = ["\n".join(fm), "", f"# {title}", "", body]
    links = [l for l in (decision.get("links") or []) if isinstance(l, str)][:3]
    if links:
        parts += ["", "## Connections", *[f"- [[{l}]]" for l in links]]
    return "\n".join(parts) + "\n"


def main(argv: list[str]) -> int:
    thought = " ".join(argv).strip()
    if not thought:
        print('usage: capture "your thought"', file=sys.stderr)
        return 1
    if not VAULT.is_dir():
        print(f"vault not found: {VAULT}", file=sys.stderr)
        return 1

    folders = vault_folders()
    candidates = link_candidates(thought)
    candidate_set = set(candidates)

    from tessera.adapters.llm import get_backend
    backend = get_backend()
    prompt = PROMPT.format(
        thought=thought,
        folders="\n".join(folders),
        candidates="\n".join(candidates) or "(none)",
    )
    raw = backend.complete(prompt).text
    decision = extract_json(raw) or {}

    # ---- guardrails (code, not trust) ----
    folder = decision.get("folder", "")
    if folder not in folders:
        folder = "000 Inbox"
        decision.setdefault("type", "inbox")
    if not decision.get("body"):
        decision["body"] = thought          # never lose the raw thought
    if not decision.get("title"):
        decision["title"] = thought[:60]
    # Links can only point at notes that actually exist — no invented links.
    decision["links"] = [l for l in (decision.get("links") or []) if l in candidate_set]

    today = datetime.now().strftime("%Y-%m-%d")
    note = build_note(decision, today)
    title = sanitize_title(decision["title"])

    target_dir = VAULT / folder
    target_dir.mkdir(parents=True, exist_ok=True)   # only ever inside the vault
    path = target_dir / f"{title}.md"
    n = 2
    while path.exists():
        path = target_dir / f"{title} ({n}).md"
        n += 1
    path.write_text(note)

    print(f"filed → {folder}/{path.name}")
    print("-" * 60)
    print(note, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
