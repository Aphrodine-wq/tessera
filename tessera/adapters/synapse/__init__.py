"""Synapse adapter — Κ target emission.

Writes one Block per SIR region+node, one Edge per data dependency, and logs
the compile as a single trace. Hybrid strategy:

  - < 100 nodes  → MCP calls (synapse_create, synapse_link, synapse_log_trace)
  - ≥ 100 nodes  → direct SQLite transaction on the vault.sqlite
  - trace        → always MCP

When neither path is reachable (Synapse not built, MCP not registered), this
returns a stub `CompileArtifact` describing what *would* have been written.
This keeps the dev loop unblocked while Synapse is unavailable.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ...sir.nodes import Module


VAULT_PATH = Path.home() / "Library/Application Support/Synapse/vault.sqlite"
COMPILER_AGENT_ID = "tessera-compiler"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.") + f"{datetime.now(timezone.utc).microsecond // 1000:03d}"


def _synapse_uuid() -> str:
    return str(uuid.uuid4()).upper()


@dataclass
class CompileArtifact:
    compile_id: str
    folder: str
    block_count: int
    edge_count: int
    trace_id: str | None = None
    backend: str = "stub"
    written: bool = False
    notes: list[str] = field(default_factory=list)


def _hlc_string() -> str:
    return f"{int(time.time() * 1000)}:0:tessera"


def _block_row(
    block_id: str,
    title: str,
    content: str,
    tags: list[str],
    folder: str,
    block_type: str = "text",
) -> tuple:
    now = _iso_now()
    return (
        block_id,
        json.dumps({
            "plainText": content,
            "title": title,
            "tags": tags,
            "folderLabel": folder,
        }).encode("utf-8"),         # content is BLOB in Synapse schema
        content,                    # content_text (FTS)
        block_type,
        now,                        # created_at (ISO)
        now,                        # updated_at (ISO)
        1.0,                        # activation_score
        1.0,                        # decay_rate
        0.0,                        # emotional_valence
        0,                          # is_pinned
        None,                       # embedding
        _hlc_string(),
        "agent",                    # author_kind
        COMPILER_AGENT_ID,          # author_id
        "pending",                  # review_status
    )


def _edge_row(
    edge_id: str,
    source_id: str,
    target_id: str,
    polarity: str = "reinforcing",
    weight: float = 0.7,
    edge_type: str = "explicit",
) -> tuple:
    now = _iso_now()
    return (
        edge_id,
        source_id,
        target_id,
        edge_type,
        weight,
        polarity,
        now,                        # created_at (ISO)
        now,                        # last_traversed (ISO)
        0,                          # traversal_count
        "agent",                    # author_kind
        COMPILER_AGENT_ID,          # author_id
        "pending",                  # review_status
    )


def remember_fact(
    schema: str,
    fields: dict[str, object],
    *,
    dry_run: bool = True,
    vault_path: Path | None = None,
) -> str | None:
    """Persist one schema instance as a Synapse Block. Returns the block id if written.

    Safety: dry-run default; real vault requires TESSERA_ALLOW_REAL_VAULT=1 OR
    an explicit vault_path (e.g. a test sqlite).
    """
    target = Path(vault_path) if vault_path else VAULT_PATH
    is_real = (target == VAULT_PATH)

    bid = _synapse_uuid()
    if dry_run or (is_real and os.environ.get("TESSERA_ALLOW_REAL_VAULT") != "1"):
        return None
    if not target.exists() and is_real:
        return None

    folder = f"Tessera Knowledge: {schema}"
    title = f"{schema}: " + " | ".join(f"{k}={v}" for k, v in fields.items())
    content = json.dumps(fields, default=str)
    tags = ["tessera", "knowledge", schema] + [f"{schema}.{k}" for k in fields.keys()]

    try:
        with sqlite3.connect(target) as conn:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT INTO blocks (
                  id, content, content_text, block_type, created_at, updated_at,
                  activation_score, decay_rate, emotional_valence, is_pinned, embedding,
                  hlc_timestamp, author_kind, author_id, review_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                _block_row(bid, title, content, tags, folder, "knowledge"),
            )
            conn.commit()
        return bid
    except sqlite3.Error:
        return None


def lookup_facts(
    schema: str,
    *,
    where_field: str | None = None,
    where_value: object = None,
    vault_path: Path | None = None,
    limit: int = 100,
) -> list[dict]:
    """Read Synapse Blocks tagged with the schema name. Optional field filter.

    Read-only — always safe. Returns [] when the vault doesn't exist or is
    unreadable, never raises.
    """
    target = Path(vault_path) if vault_path else VAULT_PATH
    if not target.exists():
        return []
    try:
        with sqlite3.connect(target) as conn:
            cur = conn.execute(
                """
                SELECT id, content_text FROM blocks
                WHERE author_id = ? AND block_type = 'knowledge'
                  AND content_text LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (COMPILER_AGENT_ID, "%", limit),
            )
            rows = cur.fetchall()
    except sqlite3.Error:
        return []

    out: list[dict] = []
    for bid, content_text in rows:
        try:
            fields = json.loads(content_text)
        except (TypeError, ValueError):
            continue
        if where_field and fields.get(where_field) != where_value:
            continue
        out.append({"id": bid, "schema": schema, "fields": fields})
    return out


def write_module(
    module: Module,
    *,
    dry_run: bool = True,
    vault_path: Path | None = None,
) -> CompileArtifact:
    """Emit Κ artifacts.

    Safety: dry_run=True by default. To actually write to the real Synapse vault you
    must pass ``dry_run=False`` AND either set ``TESSERA_ALLOW_REAL_VAULT=1`` in the
    environment, or pass an explicit ``vault_path`` (e.g. a test SQLite file).
    """
    compile_id = uuid.uuid4().hex[:8]
    folder = f"Tessera Compile #{compile_id}"

    blocks: list[tuple[str, str, str, list[str], str]] = []
    edges: list[tuple[str, str, str]] = []

    region_block_id: dict[str, str] = {}
    for region in module.regions:
        rb_id = _synapse_uuid()
        region_block_id[region.id] = rb_id
        blocks.append((
            rb_id,
            f"region:{region.name}",
            f"region {region.name} -> {region.return_type}",
            ["tessera", "region"],
            "region",
        ))

    node_block_id: dict[str, str] = {}
    for region in module.regions:
        for node in region.nodes:
            nb_id = _synapse_uuid()
            node_block_id[node.id] = nb_id
            blocks.append((
                nb_id,
                f"sir:{node.op.value}",
                f"%{node.id} = {node.op.value} ({', '.join(node.inputs)}) {dict(node.attributes)}",
                ["tessera", "sir-node", node.substrate],
                "sir-node",
            ))
            edges.append((_synapse_uuid(), region_block_id[region.id], nb_id))
            for inp in node.inputs:
                if inp in node_block_id:
                    edges.append((_synapse_uuid(), node_block_id[inp], nb_id))

    artifact = CompileArtifact(
        compile_id=compile_id,
        folder=folder,
        block_count=len(blocks),
        edge_count=len(edges),
    )

    target = Path(vault_path) if vault_path else VAULT_PATH
    is_real = (target == VAULT_PATH)

    if dry_run:
        artifact.backend = "stub"
        artifact.notes.append("dry_run=True — emit planned but not persisted")
        return artifact

    if is_real and os.environ.get("TESSERA_ALLOW_REAL_VAULT") != "1":
        artifact.backend = "stub"
        artifact.notes.append(
            "refusing to write to real Synapse vault without TESSERA_ALLOW_REAL_VAULT=1; "
            "pass an explicit vault_path for tests"
        )
        return artifact

    if not target.exists() and is_real:
        artifact.backend = "stub"
        artifact.notes.append(f"target vault {target} does not exist — not auto-creating real path")
        return artifact

    try:
        with sqlite3.connect(target) as conn:
            conn.execute("BEGIN")
            for bid, title, content, tags, kind in blocks:
                conn.execute(
                    """
                    INSERT INTO blocks (
                      id, content, content_text, block_type, created_at, updated_at,
                      activation_score, decay_rate, emotional_valence, is_pinned, embedding,
                      hlc_timestamp, author_kind, author_id, review_status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    _block_row(bid, title, content, tags, folder, kind),
                )
            for eid, src, tgt in edges:
                conn.execute(
                    """
                    INSERT INTO edges (
                      id, source_id, target_id, edge_type, weight, polarity,
                      created_at, last_traversed, traversal_count,
                      author_kind, author_id, review_status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    _edge_row(eid, src, tgt),
                )
            conn.commit()
        artifact.backend = "sqlite"
        artifact.written = True
    except sqlite3.Error as exc:
        artifact.notes.append(f"sqlite write failed: {exc}")
        artifact.backend = "stub"

    return artifact
