"""Vault filesystem tools callable from Tessera `tsr:tool` blocks.

Each function is a string-in / string-out callable that the Tessera
interpreter can dispatch to. Side effects are real — these read and
write under ~/Desktop/TheVault.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

VAULT = Path.home() / "Desktop" / "TheVault"
DECISIONS_DIR = VAULT / "300 Decisions"
MAILBOX_TO_TWIN = VAULT / "999 Mailbox" / "to-twin"

_DATE_RE = re.compile(r"^date:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
_UPDATED_RE = re.compile(r"^updated:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)


def _newest_dated(text: str) -> datetime | None:
    """Return the newest YYYY-MM-DD date in frontmatter (updated wins over date)."""
    candidates = []
    if m := _UPDATED_RE.search(text):
        candidates.append(m.group(1))
    if m := _DATE_RE.search(text):
        candidates.append(m.group(1))
    for c in candidates:
        try:
            return datetime.strptime(c, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def scan_stale_decisions(threshold_days_str: str) -> str:
    """List decision notes older than N days, by frontmatter date.

    Falls back to file mtime if no date is in the frontmatter. Returns a
    pipe-delimited string: "path|days_old" per stale entry, newline-joined.
    Empty string if none are stale or the dir doesn't exist.
    """
    try:
        threshold_days = int(threshold_days_str)
    except (TypeError, ValueError):
        threshold_days = 90

    if not DECISIONS_DIR.exists():
        return ""

    now = datetime.now(timezone.utc)
    stale: list[str] = []
    for md in sorted(DECISIONS_DIR.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ts = _newest_dated(text)
        if ts is None:
            try:
                ts = datetime.fromtimestamp(md.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
        age_days = (now - ts).days
        if age_days >= threshold_days:
            rel = md.relative_to(VAULT)
            stale.append(f"{rel}|{age_days}")
    return "\n".join(stale)


def write_mailbox(message: str) -> str:
    """Append a timestamped message to 999 Mailbox/to-twin/.

    Returns the absolute path of the file written.
    """
    MAILBOX_TO_TWIN.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    path = MAILBOX_TO_TWIN / f"DecisionAuditor-{stamp}.md"
    path.write_text(message + "\n", encoding="utf-8")
    return str(path)
