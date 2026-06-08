"""Local SQLite audit store — the queryable provenance graph.

Every meaningful runtime action is persisted as one row, routed by
event class into one of two stores:

- ~/.tessera/audit_governance.db  — retained forever. Refusals, ethics
  applications, autonomy approval gates, capability narrowing,
  plan_enter when intent is bound, escalations, governance-error codes.
  This is the provenance proof trail. Never auto-purged.
- ~/.tessera/audit_operational.db — rolling window (default 30 days,
  override via TESSERA_AUDIT_RETENTION_DAYS). Routine tool/prompt/skill
  calls, cache events, latency-only events.

`query_events` reads both and merges by created_at. `purge_operational`
drops rows older than the rolling window (governance is sacrosanct).

Override paths via env vars or explicit db_path arg (tests use the
conftest fixture to point both at tmp paths).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

DEFAULT_GOV_DB = Path.home() / ".tessera" / "audit_governance.db"
DEFAULT_OPS_DB = Path.home() / ".tessera" / "audit_operational.db"
ENV_GOV_DB = "TESSERA_AUDIT_GOV_DB"
ENV_OPS_DB = "TESSERA_AUDIT_OPS_DB"
# Single-tier legacy env name still respected as a fallback so older
# scripts that set TESSERA_AUDIT_DB keep working — when set, BOTH stores
# point at the same file (governance + operational coalesce). The new
# split env vars take precedence when both are set.
ENV_LEGACY_DB = "TESSERA_AUDIT_DB"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  seq        INTEGER NOT NULL,
  agent      TEXT,
  plan       TEXT,
  intent     TEXT,
  action     TEXT NOT NULL,
  detail_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_agent  ON events(agent);
CREATE INDEX IF NOT EXISTS idx_events_intent ON events(intent);
CREATE INDEX IF NOT EXISTS idx_events_action ON events(action);
"""

# Action prefixes that always classify as governance (proof trail).
_GOVERNANCE_PREFIXES: tuple[str, ...] = (
    "refuse",
    "refusal",
    "approval_blocked",
    "caps_narrowed",
    "spawn:",          # capability-grant decisions live here
    "escalate",
    "policy_violation",
    "capability_violation",
    "evolve:",
)

EventClass = Literal["governance", "operational"]


def _resolve_paths(db_path: str | Path | None = None) -> tuple[Path, Path]:
    """Returns (governance_path, operational_path).

    Resolution order, applied per-store:
      1. Explicit db_path arg → BOTH stores write to that one file
         (intended for tests that want a single tmp DB).
      2. TESSERA_AUDIT_GOV_DB / TESSERA_AUDIT_OPS_DB env vars.
      3. Legacy TESSERA_AUDIT_DB env (coalesces gov+ops into one file).
      4. Defaults under ~/.tessera/.
    """
    if db_path is not None:
        p = Path(db_path)
        return p, p
    gov_env = os.environ.get(ENV_GOV_DB)
    ops_env = os.environ.get(ENV_OPS_DB)
    legacy = os.environ.get(ENV_LEGACY_DB)
    gov = Path(gov_env) if gov_env else (Path(legacy) if legacy else DEFAULT_GOV_DB)
    ops = Path(ops_env) if ops_env else (Path(legacy) if legacy else DEFAULT_OPS_DB)
    return gov, ops


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_SQL)
    return conn


def _classify(event: dict) -> EventClass:
    """Route an event to governance or operational. Governance wins on any
    match — the proof trail is permissive on inclusion, strict on dropping.
    """
    action = event.get("action") or ""
    if action.startswith(_GOVERNANCE_PREFIXES):
        return "governance"
    # A contract refusal/error is a denial decision — part of the permanent
    # proof trail, like any other refusal (not a routine operational event).
    # contract:retry / contract:audit stay operational (routine churn).
    if action in ("contract:refuse", "contract:error"):
        return "governance"
    # ethics_applied with content makes any action governance-relevant
    if event.get("ethics_applied"):
        return "governance"
    # An intent-bound plan_enter is governance (proves the agent committed to a goal)
    if action.startswith("plan_enter") and event.get("intent_served"):
        return "governance"
    return "operational"


def record_event(event: dict, *, db_path: str | Path | None = None) -> None:
    """Persist one audit event into the appropriate tier."""
    cls = _classify(event)
    gov, ops = _resolve_paths(db_path)
    target = gov if cls == "governance" else ops
    seq = event.get("seq")
    agent = event.get("agent")
    plan = event.get("plan")
    intent = event.get("intent")
    action = event.get("action") or ""
    detail = {k: v for k, v in event.items()
              if k not in {"seq", "agent", "plan", "intent", "action"}}
    now = datetime.now(timezone.utc).isoformat()
    with _connect(target) as conn:
        conn.execute(
            "INSERT INTO events "
            "(seq, agent, plan, intent, action, detail_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (seq, agent, plan, intent, action, json.dumps(detail), now),
        )
        conn.commit()


def _query_one(
    path: Path,
    *,
    where: list[str],
    params: list[Any],
    limit: int,
) -> list[tuple]:
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT seq, agent, plan, intent, action, detail_json, created_at "
        f"FROM events{clause} ORDER BY id LIMIT ?"
    )
    with _connect(path) as conn:
        return conn.execute(sql, [*params, limit]).fetchall()


def query_events(
    *,
    agent: str | None = None,
    intent: str | None = None,
    action: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
    tier: EventClass | None = None,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Read both audit stores and merge by created_at. `tier` restricts
    to one store ('governance' or 'operational') if needed."""
    where: list[str] = []
    params: list[Any] = []
    if agent is not None:
        where.append("agent = ?")
        params.append(agent)
    if intent is not None:
        where.append("intent = ?")
        params.append(intent)
    if action is not None:
        where.append("action LIKE ?")
        params.append(action if "%" in action else f"%{action}%")
    if since is not None:
        where.append("created_at >= ?")
        params.append(since)
    if until is not None:
        where.append("created_at <= ?")
        params.append(until)

    gov, ops = _resolve_paths(db_path)
    rows: list[tuple] = []
    if tier in (None, "governance"):
        rows.extend(_query_one(gov, where=where, params=params, limit=limit))
    if tier in (None, "operational") and ops != gov:
        rows.extend(_query_one(ops, where=where, params=params, limit=limit))

    out: list[dict] = []
    for seq, agent_, plan, intent_, action_, detail_json, created in rows:
        record = {
            "seq": seq,
            "agent": agent_,
            "plan": plan,
            "intent": intent_,
            "action": action_,
            "created_at": created,
        }
        try:
            record.update(json.loads(detail_json))
        except json.JSONDecodeError:
            pass
        out.append(record)
    out.sort(key=lambda r: r.get("created_at") or "")
    return out[:limit]


def purge_operational(
    before: str | None = None,
    *,
    retention_days: int | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Drop operational events older than `before` (ISO date) or older than
    retention_days. Returns the row count deleted. Governance untouched."""
    _, ops = _resolve_paths(db_path)
    if before is None:
        if retention_days is None:
            retention_days = int(os.environ.get("TESSERA_AUDIT_RETENTION_DAYS", "30"))
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        before = cutoff.isoformat()
    with _connect(ops) as conn:
        cur = conn.execute("DELETE FROM events WHERE created_at < ?", (before,))
        conn.commit()
        return cur.rowcount
