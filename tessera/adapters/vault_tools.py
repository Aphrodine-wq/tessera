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
    """Scan decision notes older than N days and return a markdown report.

    Reads frontmatter `updated:`/`date:`, falls back to file mtime. Output is
    a ready-to-write markdown body — agents pass this straight to write_mailbox
    without further formatting, dodging escape-sequence issues in tsr:logic.
    """
    try:
        threshold_days = int(threshold_days_str)
    except (TypeError, ValueError):
        threshold_days = 90

    header = f"# Stale decisions (>= {threshold_days} days)"

    if not DECISIONS_DIR.exists():
        return f"{header}\n\n_300 Decisions/ does not exist._\n"

    now = datetime.now(timezone.utc)
    stale: list[tuple[Path, int]] = []
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
            stale.append((md, age_days))

    if not stale:
        return f"{header}\n\n_No stale decisions found._\n"

    lines = [header, "", f"Found {len(stale)} stale decision(s):", ""]
    for md, age in sorted(stale, key=lambda x: -x[1]):
        rel = md.relative_to(VAULT)
        lines.append(f"- [[{rel.stem}]] — {age} days old (`{rel}`)")
    return "\n".join(lines) + "\n"


def write_mailbox(message: str) -> str:
    """Append a timestamped message to 999 Mailbox/to-twin/.

    Returns the absolute path of the file written.
    """
    MAILBOX_TO_TWIN.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    path = MAILBOX_TO_TWIN / f"DecisionAuditor-{stamp}.md"
    path.write_text(message + "\n", encoding="utf-8")
    return str(path)
