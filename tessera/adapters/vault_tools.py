"""Vault filesystem tools callable from Tessera `tsr:tool` blocks.

Each function is a string-in / string-out callable that the Tessera
interpreter can dispatch to. Side effects are real — these read and
write under ~/Desktop/TheVault.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

VAULT = Path.home() / "Desktop" / "TheVault"
DECISIONS_DIR = VAULT / "300 Decisions"
PROJECTS_DIR = VAULT / "200 Projects"
DAILY_NOTES_DIR = VAULT / "600 Daily Notes"
MAILBOX_TO_TWIN = VAULT / "999 Mailbox" / "to-twin"
GOALS_FILE = Path.home() / ".claude" / "goals" / "goals.json"

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

    Returns the absolute path of the file written. Sender is inferred from
    the first markdown heading so all agents share one tool.
    """
    MAILBOX_TO_TWIN.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    sender = _infer_sender(message)
    path = MAILBOX_TO_TWIN / f"{sender}-{stamp}.md"
    path.write_text(message + "\n", encoding="utf-8")
    return str(path)


def _infer_sender(message: str) -> str:
    """Derive a sender slug from the first markdown heading."""
    for line in message.splitlines():
        line = line.strip()
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            # Take first 3 words, slugify
            words = re.findall(r"\w+", stripped)[:3]
            if words:
                return "".join(w.capitalize() for w in words)
    return "VaultAgent"


# ----- GoalAuditor ----------------------------------------------------------

def _parse_date_str(s: str) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    # Accept YYYY-MM-DD or full ISO
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def scan_stagnant_goals(threshold_days_str: str) -> str:
    """Flag active goals not touched in >= N days. Returns markdown."""
    try:
        threshold_days = int(threshold_days_str)
    except (TypeError, ValueError):
        threshold_days = 14

    header = f"# Stagnant goals (>= {threshold_days} days since last touch)"

    if not GOALS_FILE.exists():
        return f"{header}\n\n_goals.json not found._\n"

    try:
        data = json.loads(GOALS_FILE.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return f"{header}\n\n_failed to read goals.json: {e}_\n"

    now = datetime.now(timezone.utc)
    stagnant: list[tuple[dict, int]] = []
    for g in data.get("goals", []):
        if g.get("status") not in (None, "active", "in_progress"):
            continue
        last_str = g.get("last_touched") or g.get("last_worked") or g.get("started") or g.get("created")
        last = _parse_date_str(last_str)
        if last is None:
            continue
        age = (now - last).days
        if age >= threshold_days:
            stagnant.append((g, age))

    if not stagnant:
        return f"{header}\n\n_No stagnant goals._\n"

    lines = [header, "", f"Found {len(stagnant)} stagnant goal(s):", ""]
    for g, age in sorted(stagnant, key=lambda x: -x[1]):
        title = g.get("title", "(untitled)")
        proj = g.get("project", "?")
        pri = g.get("priority", "?")
        progress = g.get("progress", 0.0)
        deadline = g.get("deadline", "")
        deadline_str = f" · deadline {deadline}" if deadline else ""
        lines.append(
            f"- **{pri}** {title} ({proj}) — {age} days stagnant · "
            f"{int(progress * 100)}% complete{deadline_str}"
        )
    return "\n".join(lines) + "\n"


# ----- StaleProjectFlagger --------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_PROJECT_STATUS_RE = re.compile(r"^status:\s*(\w+)", re.MULTILINE)
_PROJECT_UPDATED_RE = re.compile(r"^updated:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
_PROJECT_DATE_RE = re.compile(r"^date:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)


def scan_stale_projects(threshold_days_str: str) -> str:
    """Flag active projects with no frontmatter updates in >= N days."""
    try:
        threshold_days = int(threshold_days_str)
    except (TypeError, ValueError):
        threshold_days = 30

    header = f"# Stale active projects (>= {threshold_days} days since update)"

    if not PROJECTS_DIR.exists():
        return f"{header}\n\n_200 Projects/ not found._\n"

    now = datetime.now(timezone.utc)
    stale: list[tuple[Path, int, str]] = []
    for md in sorted(PROJECTS_DIR.glob("*.md")):
        if md.name.endswith(".t.md"):
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        fm_match = _FRONTMATTER_RE.search(text)
        if not fm_match:
            continue
        fm = fm_match.group(1)
        status_m = _PROJECT_STATUS_RE.search(fm)
        status = status_m.group(1) if status_m else None
        if status != "active":
            continue
        date_str = None
        if m := _PROJECT_UPDATED_RE.search(fm):
            date_str = m.group(1)
        elif m := _PROJECT_DATE_RE.search(fm):
            date_str = m.group(1)
        if date_str is None:
            continue
        ts = _parse_date_str(date_str)
        if ts is None:
            continue
        age = (now - ts).days
        if age >= threshold_days:
            title = md.stem
            stale.append((md, age, title))

    if not stale:
        return f"{header}\n\n_No stale active projects._\n"

    lines = [header, "", f"Found {len(stale)} stale project(s):", ""]
    for md, age, title in sorted(stale, key=lambda x: -x[1]):
        rel = md.relative_to(VAULT)
        lines.append(f"- [[{title}]] — {age} days since update (`{rel}`)")
    return "\n".join(lines) + "\n"


# ----- MorningSynth ---------------------------------------------------------

SYNTH_SECTION_MARKER = "## Vault Agents — Morning Sweep"


def synthesize_morning_brief(hours_str: str) -> str:
    """Compose a single morning brief from recent mailbox alerts.

    Reads ~/Desktop/TheVault/999 Mailbox/to-twin/ for messages written
    within the last N hours, groups by inferred sender, and returns one
    consolidated markdown section. Empty agents are mentioned as healthy.
    """
    try:
        hours = float(hours_str)
    except (TypeError, ValueError):
        hours = 12.0

    if not MAILBOX_TO_TWIN.exists():
        return f"{SYNTH_SECTION_MARKER}\n\n_No mailbox dir yet._\n"

    cutoff_ts = datetime.now(timezone.utc).timestamp() - (hours * 3600)
    by_sender: dict[str, list[Path]] = {}
    for f in sorted(MAILBOX_TO_TWIN.glob("*.md")):
        try:
            if f.stat().st_mtime < cutoff_ts:
                continue
        except OSError:
            continue
        # Filename pattern: <Sender>-<timestamp>.md
        sender = f.stem.split("-", 1)[0]
        by_sender.setdefault(sender, []).append(f)

    if not by_sender:
        return (
            f"{SYNTH_SECTION_MARKER}\n\n"
            f"_No recent agent activity (last {hours:.0f}h)._\n"
        )

    lines: list[str] = [SYNTH_SECTION_MARKER, ""]
    lines.append(f"_Synthesised at {datetime.now().strftime('%H:%M')} from {sum(len(v) for v in by_sender.values())} agent message(s)._")
    lines.append("")
    for sender in sorted(by_sender):
        files = sorted(by_sender[sender], key=lambda p: p.stat().st_mtime)
        latest = files[-1]
        body = latest.read_text(encoding="utf-8", errors="ignore").strip()
        # Re-render the agent's existing markdown verbatim under a sender subhead.
        lines.append(f"### {sender}")
        lines.append("")
        # Skip the agent's own top-level # heading since we're nesting under ###
        body_lines = body.splitlines()
        if body_lines and body_lines[0].startswith("# "):
            body_lines = body_lines[1:]
        # Trim leading blank lines
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        lines.extend(body_lines)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def append_to_daily_note(content: str) -> str:
    """Write `content` into today's daily note, replacing any prior
    SYNTH_SECTION_MARKER section. Creates the daily note if missing.
    Returns the daily note path.
    """
    DAILY_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    note = DAILY_NOTES_DIR / f"{today}.md"

    if note.exists():
        existing = note.read_text(encoding="utf-8")
    else:
        existing = f"# {today}\n"

    # Find existing section by marker; replace it. Section ends at the next
    # `## ` at the same level or EOF.
    pattern = re.compile(
        rf"^{re.escape(SYNTH_SECTION_MARKER)}.*?(?=^## |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    if pattern.search(existing):
        new_text = pattern.sub(content.rstrip() + "\n\n", existing)
    else:
        sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        new_text = existing + sep + content.rstrip() + "\n"

    note.write_text(new_text, encoding="utf-8")
    return str(note)
