"""Local SQLite audit store — the queryable provenance graph.

Every meaningful runtime action is persisted as one row in `events`. Query
later with `tessera audit query --agent X --intent Y --since DATE`. The
single-table schema is intentionally minimal; tiered retention (separate
governance vs operational stores) is deferred to a follow-up — see decision
2026-05-28 Tessera audit retention is tiered by event type.

Default path: ~/.tessera/audit.db. Override via TESSERA_AUDIT_DB env var or
explicit `db_path=` (tests do this through the conftest fixture).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path.home() / ".tessera" / "audit.db"
ENV_DB_PATH = "TESSERA_AUDIT_DB"

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


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is not None:
        return Path(db_path)
    env = os.environ.get(ENV_DB_PATH)
    if env:
        return Path(env)
    return DEFAULT_DB_PATH


def _connect(db_path: str | Path | None) -> sqlite3.Connection:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_SQL)
    return conn


def record_event(event: dict, *, db_path: str | Path | None = None) -> None:
    """Persist one audit event. `event` is the dict shape produced by
    `AuditEvent.to_dict()` — has `seq`, `agent`, `plan`, `intent`, `action`,
    plus arbitrary detail fields."""
    seq = event.get("seq")
    agent = event.get("agent")
    plan = event.get("plan")
    intent = event.get("intent")
    action = event.get("action") or ""
    detail = {k: v for k, v in event.items()
              if k not in {"seq", "agent", "plan", "intent", "action"}}
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO events "
            "(seq, agent, plan, intent, action, detail_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (seq, agent, plan, intent, action, json.dumps(detail), now),
        )
        conn.commit()


def query_events(
    *,
    agent: str | None = None,
    intent: str | None = None,
    action: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Filter audit events. Returns a list of fully-rehydrated event dicts
    (agent/plan/intent/action plus the detail fields)."""
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
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT seq, agent, plan, intent, action, detail_json, created_at "
        f"FROM events{clause} ORDER BY id LIMIT ?"
    )
    params.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
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
    return out
