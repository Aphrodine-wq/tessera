"""Assemble training corpora from audit-store provenance (decision 16 follow-up).

A procedural skill declared `promote_to: neural { threshold: N }` emits a
`skill_promotion_pending` audit event when its call count crosses N
(shipped in `03ab1f7`). This module reads the operational audit store,
gathers the prompt/tool/fn outputs that the skill produced, and writes
them as a JSONL training corpus at
`~/.tessera/training_corpora/<skill>.jsonl`.

Future training jobs (vast.ai fine-tune, etc.) consume the corpus.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CORPORA_DIR = Path.home() / ".tessera" / "training_corpora"
ENV_CORPORA_DIR = "TESSERA_CORPORA_DIR"


def _resolve_corpora_dir() -> Path:
    env = os.environ.get(ENV_CORPORA_DIR)
    return Path(env) if env else DEFAULT_CORPORA_DIR


def corpus_path(skill_name: str) -> Path:
    return _resolve_corpora_dir() / f"{skill_name}.jsonl"


def assemble_for_skill(skill_name: str) -> tuple[Path, int]:
    """Gather every recorded call of the named skill from the audit store
    and write (input, output) pairs to disk.

    Skill invocations appear in audit as `skill:<name>` actions, and the
    underlying prompt/tool that the skill dispatched to lands as a
    `prompt:<binding>` (or `tool:<binding>`) action immediately after.
    The corpus pairs each skill call with its underlying output.

    Returns (path, pairs_written).
    """
    from .adapters.audit import query_events
    skill_events = query_events(action=f"skill:{skill_name}", limit=10_000)
    if not skill_events:
        # Nothing to assemble — write an empty file so callers can detect.
        d = _resolve_corpora_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{skill_name}.jsonl"
        path.write_text("")
        return path, 0

    # Pair each skill event with the next event in the same run that
    # produced an output. We don't have a strict join key today, so use
    # `seq` ordering as a proxy: the underlying call lands with seq + 1
    # within the same agent. This is approximate; a real implementation
    # would propagate a correlation id from skill invocation to underlying
    # call. Good enough for the MVP.
    all_events = query_events(limit=10_000)
    by_seq = sorted(all_events, key=lambda r: (r.get("seq") or 0, r.get("created_at") or ""))

    pairs: list[dict] = []
    for i, evt in enumerate(by_seq):
        if evt.get("action") != f"skill:{skill_name}":
            continue
        # Look ahead for the underlying prompt/tool call from the same agent
        for j in range(i + 1, min(i + 5, len(by_seq))):
            nxt = by_seq[j]
            if nxt.get("agent") != evt.get("agent"):
                continue
            action = nxt.get("action") or ""
            if action.startswith("prompt:") or action.startswith("tool:"):
                # Best effort: record what we have. Real inputs/outputs would
                # need the runtime to log them on the underlying event.
                pairs.append({
                    "skill": skill_name,
                    "agent": evt.get("agent"),
                    "intent": evt.get("intent"),
                    "underlying": action,
                    "skill_seq": evt.get("seq"),
                    "underlying_seq": nxt.get("seq"),
                    "created_at": nxt.get("created_at"),
                })
                break

    d = _resolve_corpora_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{skill_name}.jsonl"
    with open(path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return path, len(pairs)
