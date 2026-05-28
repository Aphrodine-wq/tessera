"""Local SQLite fact store for the `memory:semantic` substrate.

Tessera-owned. No external dependencies. Default location is
`~/.tessera/semantic.db`; override via the `TESSERA_SEMANTIC_DB` env var
or pass an explicit `db_path=` (tests do this with `tmp_path`).

The schema is one table — `facts` — keyed by a UUID, with the substrate's
`schema` name, the field dict serialized as JSON, and optional agent/plan
provenance. No edges. If a graph layer is needed later, add it then.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path.home() / ".tessera" / "semantic.db"
ENV_DB_PATH = "TESSERA_SEMANTIC_DB"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
  id          TEXT PRIMARY KEY,
  schema      TEXT NOT NULL,
  fields_json TEXT NOT NULL,
  agent_id    TEXT,
  plan_id     TEXT,
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_schema ON facts(schema);
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


def remember_fact(
    schema: str,
    fields: dict[str, Any],
    *,
    db_path: str | Path | None = None,
    agent_id: str | None = None,
    plan_id: str | None = None,
) -> str:
    """Insert a fact row. Returns the new fact id."""
    fact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO facts (id, schema, fields_json, agent_id, plan_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fact_id, schema, json.dumps(fields), agent_id, plan_id, now),
        )
        conn.commit()
    return fact_id


def lookup_facts(
    schema: str,
    *,
    where_field: str | None = None,
    where_value: Any = None,
    db_path: str | Path | None = None,
    limit: int = 100,
) -> list[dict]:
    """Return facts of the given schema. Optionally filter by a single field == value."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, schema, fields_json, agent_id, plan_id, created_at "
            "FROM facts WHERE schema = ? ORDER BY created_at LIMIT ?",
            (schema, limit),
        ).fetchall()

    out: list[dict] = []
    for fid, sch, fjson, aid, pid, created in rows:
        fields = json.loads(fjson)
        if where_field is not None and fields.get(where_field) != where_value:
            continue
        out.append({
            "id": fid,
            "schema": sch,
            "fields": fields,
            "agent_id": aid,
            "plan_id": pid,
            "created_at": created,
        })
    return out
