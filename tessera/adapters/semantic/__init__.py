"""Local SQLite fact store for the `memory:semantic` substrate.

Tessera-owned. No external dependencies. Default location is
`~/.tessera/semantic.db`; override via the `TESSERA_SEMANTIC_DB` env var
or pass an explicit `db_path=` (tests do this with `tmp_path`).

The `facts` table is keyed by a UUID, with the substrate's `schema` name, the
field dict serialized as JSON, and optional agent/plan provenance. Typed fields
are validated on insert when the caller passes `field_types` (backward compatible
— omit it and behavior is unchanged).

The graph layer the original design deferred now exists: a `relations` table of
typed edges (subject_fact_id -predicate-> object_fact_id) turns the flat store
into a knowledge graph. It is purely additive — old `facts` rows are untouched
and remain fully relatable by their existing ids.
"""
from __future__ import annotations

import json
import os
import re
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
CREATE TABLE IF NOT EXISTS relations (
  id              TEXT PRIMARY KEY,
  subject_fact_id TEXT NOT NULL,
  predicate       TEXT NOT NULL,
  object_fact_id  TEXT NOT NULL,
  fields_json     TEXT NOT NULL DEFAULT '{}',
  agent_id        TEXT,
  plan_id         TEXT,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject_fact_id, predicate);
CREATE INDEX IF NOT EXISTS idx_rel_object  ON relations(object_fact_id, predicate);
"""

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UUID_SHAPE = re.compile(r"^[0-9a-fA-F-]{8,}$")


class FactTypeError(ValueError):
    """A fact field (or relation endpoint) violated its declared type."""


def _coerce_scalar(type_str: str, value: Any) -> Any:
    """Validate/coerce one value against a declared type. Returns the coerced
    value or raises FactTypeError. `Any` (and unknown types) pass through, which
    keeps every existing String-only / untyped schema valid."""
    t = (type_str or "Any").strip()
    base = t[4:-1].strip() if t.startswith("Ref<") and t.endswith(">") else None
    if t == "Any" or t == "":
        return value
    if t == "String":
        if not isinstance(value, str):
            raise FactTypeError(f"expected String, got {type(value).__name__}")
        return value
    if t == "Int":
        if isinstance(value, bool) or not isinstance(value, int):
            try:
                return int(str(value))
            except (TypeError, ValueError):
                raise FactTypeError(f"expected Int, got {value!r}")
        return value
    if t == "Float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                return float(str(value))
            except (TypeError, ValueError):
                raise FactTypeError(f"expected Float, got {value!r}")
        return value
    if t == "Bool":
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "false"):
            return str(value).lower() == "true"
        raise FactTypeError(f"expected Bool, got {value!r}")
    if t == "Date":
        if isinstance(value, str) and _ISO_DATE.match(value):
            return value
        raise FactTypeError(f"expected Date (YYYY-MM-DD), got {value!r}")
    if base is not None:  # Ref<Schema> — a fact id of that schema
        if isinstance(value, str) and _UUID_SHAPE.match(value):
            return value
        raise FactTypeError(f"expected Ref<{base}> (a fact id), got {value!r}")
    return value  # unknown declared type — be permissive


def validate_fields(
    schema: str,
    fields: dict[str, Any],
    field_types: "list[tuple[str, str]] | None",
    *,
    db_path: str | Path | None = None,
    ref_check: bool = True,
) -> dict:
    """Coerce each declared field against its type. Fields not in the schema pass
    through (forward-compat); missing declared fields are allowed (no required
    enforcement in v1). Optionally verify Ref<S> targets exist with schema S."""
    if not field_types:
        return fields
    typed = {name: typ for name, typ in field_types}
    out = dict(fields)
    for name, value in fields.items():
        typ = typed.get(name)
        if typ is None:
            continue
        coerced = _coerce_scalar(typ, value)
        out[name] = coerced
        if ref_check and typ.startswith("Ref<") and typ.endswith(">"):
            want = typ[4:-1].strip()
            got = fact_schema(coerced, db_path=db_path)
            if got is not None and got != want:
                raise FactTypeError(
                    f"{schema}.{name}: Ref<{want}> points at a {got} fact"
                )
    return out


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
    field_types: "list[tuple[str, str]] | None" = None,
    db_path: str | Path | None = None,
    agent_id: str | None = None,
    plan_id: str | None = None,
) -> str:
    """Insert a fact row. Returns the new fact id. When `field_types` is given,
    declared fields are validated/coerced first (raises FactTypeError on
    violation); omitting it is byte-for-byte the old behavior."""
    fields = validate_fields(schema, fields, field_types, db_path=db_path)
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


def fact_schema(fact_id: str, *, db_path: str | Path | None = None) -> str | None:
    """The schema name of a stored fact, or None if it isn't in the store."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT schema FROM facts WHERE id = ?", (fact_id,)).fetchone()
    return row[0] if row else None


def relate_facts(
    subject_fact_id: str,
    predicate: str,
    object_fact_id: str,
    *,
    fields: dict[str, Any] | None = None,
    db_path: str | Path | None = None,
    agent_id: str | None = None,
    plan_id: str | None = None,
) -> str:
    """Add a typed edge subject -predicate-> object. Returns the relation id."""
    rel_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO relations (id, subject_fact_id, predicate, object_fact_id, "
            "fields_json, agent_id, plan_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rel_id, subject_fact_id, predicate, object_fact_id,
             json.dumps(fields or {}), agent_id, plan_id, now),
        )
        conn.commit()
    return rel_id


def related_facts(
    fact_id: str,
    *,
    predicate: str | None = None,
    direction: str = "out",
    db_path: str | Path | None = None,
    limit: int = 100,
) -> list[dict]:
    """Facts linked to `fact_id`. direction='out' → objects where it is subject;
    'in' → subjects where it is object; 'both' → the union. Each result carries a
    `_via` predicate key. Joins back to `facts` so callers get full fact dicts."""
    edges: list[tuple[str, str]] = []  # (other_fact_id, predicate)
    with _connect(db_path) as conn:
        if direction in ("out", "both"):
            q = "SELECT object_fact_id, predicate FROM relations WHERE subject_fact_id = ?"
            params: list[Any] = [fact_id]
            if predicate:
                q += " AND predicate = ?"; params.append(predicate)
            edges += [(r[0], r[1]) for r in conn.execute(q, params).fetchall()]
        if direction in ("in", "both"):
            q = "SELECT subject_fact_id, predicate FROM relations WHERE object_fact_id = ?"
            params = [fact_id]
            if predicate:
                q += " AND predicate = ?"; params.append(predicate)
            edges += [(r[0], r[1]) for r in conn.execute(q, params).fetchall()]
        out: list[dict] = []
        for other_id, pred in edges[:limit]:
            row = conn.execute(
                "SELECT id, schema, fields_json, agent_id, plan_id, created_at "
                "FROM facts WHERE id = ?", (other_id,)).fetchone()
            if row:
                d = _row_to_dict(row)
                d["_via"] = pred
                out.append(d)
    return out


def neighbors(
    fact_id: str, *, predicate: str | None = None,
    direction: str = "out", db_path: str | Path | None = None,
) -> list[str]:
    """Just the linked fact ids (used by traverse)."""
    return [f["id"] for f in related_facts(
        fact_id, predicate=predicate, direction=direction, db_path=db_path)]


def traverse(
    start_fact_id: str,
    predicate: str,
    *,
    depth: int = 2,
    direction: str = "out",
    db_path: str | Path | None = None,
) -> list[dict]:
    """BFS up to `depth` hops following one predicate. Dedups by fact id and
    terminates on cycles. Each result carries a `_hops` distance. depth is capped
    at 8 so a pathological graph can't run away."""
    depth = max(1, min(int(depth), 8))
    seen = {start_fact_id}
    frontier = [start_fact_id]
    out: list[dict] = []
    for hop in range(1, depth + 1):
        nxt: list[str] = []
        for fid in frontier:
            for f in related_facts(fid, predicate=predicate, direction=direction, db_path=db_path):
                if f["id"] in seen:
                    continue
                seen.add(f["id"])
                f["_hops"] = hop
                out.append(f)
                nxt.append(f["id"])
        if not nxt:
            break
        frontier = nxt
    return out


def clear_relations(
    *, predicate: str | None = None, all: bool = False,
    db_path: str | Path | None = None,
) -> int:
    """Delete relations. Refuses a full wipe without all=True or a predicate filter."""
    if predicate is None and not all:
        raise ValueError("refusing to clear all relations without all=True or a predicate")
    where, params = ("", [])
    if predicate is not None:
        where, params = (" WHERE predicate = ?", [predicate])
    with _connect(db_path) as conn:
        cur = conn.execute(f"DELETE FROM relations{where}", params)
        conn.commit()
        return cur.rowcount


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


def _row_to_dict(row: tuple) -> dict:
    fid, sch, fjson, aid, pid, created = row
    return {
        "id": fid,
        "schema": sch,
        "fields": json.loads(fjson),
        "agent_id": aid,
        "plan_id": pid,
        "created_at": created,
    }


def query_facts(
    *,
    schema: str | None = None,
    agent_id: str | None = None,
    plan_id: str | None = None,
    contains: str | None = None,
    db_path: str | Path | None = None,
    limit: int = 100,
) -> list[dict]:
    """General-purpose fact query for inspection. All filters optional.

    `contains` is a substring match against the serialized field JSON
    (SQLite LIKE — ASCII case-insensitive). Returns the same row shape as
    `lookup_facts`.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if schema is not None:
        clauses.append("schema = ?")
        params.append(schema)
    if agent_id is not None:
        clauses.append("agent_id = ?")
        params.append(agent_id)
    if plan_id is not None:
        clauses.append("plan_id = ?")
        params.append(plan_id)
    if contains is not None:
        clauses.append("fields_json LIKE ?")
        params.append(f"%{contains}%")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, schema, fields_json, agent_id, plan_id, created_at "
            f"FROM facts{where} ORDER BY created_at LIMIT ?",
            params,
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def schema_summary(*, db_path: str | Path | None = None) -> list[tuple[str, int]]:
    """Return (schema, count) pairs, most-populated first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT schema, COUNT(*) FROM facts GROUP BY schema ORDER BY COUNT(*) DESC"
        ).fetchall()
    return [(sch, n) for sch, n in rows]


_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "you", "are", "was", "what",
    "from", "have", "has", "but", "not", "all", "any", "can", "will", "your",
    "about", "into", "out", "how", "why", "who", "when", "where", "which",
})


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def rank_facts(facts: list[dict], query: str, *, limit: int = 5) -> list[dict]:
    """Rank fact dicts by keyword overlap of their fields against `query`.

    Pure function (no DB) so it's unit-testable and works on facts from either
    the persistent store or an in-World ephemeral shadow. Facts that share no
    keyword with the query are kept only if nothing scores — then the most
    recent win (by `created_at`, blank sorts last). Returns at most `limit`.
    """
    kws = _keywords(query)

    def score(f: dict) -> int:
        blob = json.dumps(f.get("fields", {})).lower()
        return sum(1 for k in kws if k in blob)

    scored = [(score(f), f.get("created_at") or "", f) for f in facts]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    positive = [t for t in scored if t[0] > 0]
    chosen = positive or scored
    return [t[2] for t in chosen[:limit]]


def clear_facts(
    *,
    schema: str | None = None,
    agent_id: str | None = None,
    before: str | None = None,
    all: bool = False,
    db_path: str | Path | None = None,
) -> int:
    """Delete facts. Returns the number of rows removed.

    Destructive and unrecoverable — the store is a plain local SQLite file
    with no history. To guard against an accidental full wipe, this refuses
    to delete everything unless `all=True` is passed explicitly; any of
    `schema` / `agent_id` / `before` scopes the deletion to a subset.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if schema is not None:
        clauses.append("schema = ?")
        params.append(schema)
    if agent_id is not None:
        clauses.append("agent_id = ?")
        params.append(agent_id)
    if before is not None:
        clauses.append("created_at < ?")
        params.append(before)
    if not clauses and not all:
        raise ValueError(
            "refusing to clear all facts without all=True or a filter"
        )
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with _connect(db_path) as conn:
        cur = conn.execute(f"DELETE FROM facts{where}", params)
        conn.commit()
        return cur.rowcount
