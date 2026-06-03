"""Non-destructive Obsidian vault → Tessera ingest.

Walks an Obsidian vault and routes every note one of three ways:

- **knowledge** (the bulk: zettels, projects, people, references, daily
  notes, curated conversations) → a fact in the ``memory:semantic`` store,
  so Tessera agents can ``lookup`` it.
- **agent-shaped** (the small, intentional set: standing decisions, core
  theses, wisdom rules) → a real, compiling ``.t.md`` agent emitted to the
  output dir.
- **transcript** (raw ``conversation-transcript`` notes) → a lightweight
  index fact (no body) so they're catalogued and recoverable without
  flooding the searchable layer.

The source vault is NEVER mutated. Output goes to the semantic db (via
``remember_fact``, every row tagged ``agent_id="vault-ingest"``) and a
separate ``out`` directory (generated agents + report). Fully reversible::

    tessera facts clear --agent vault-ingest
    rm -rf <out>

Zero external dependencies — frontmatter is read with a small stdlib parser
(``_read_frontmatter``) rather than PyYAML/python-frontmatter, matching
Tessera's dependency-free design. The parser only needs the handful of
fields the classifier and fact mapper consume; on anything it can't parse it
degrades to "no frontmatter, body = whole file" and logs the note.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ...adapters.semantic import query_facts, remember_fact
from . import _WIKILINK_RE

# ---- constants -------------------------------------------------------------

FACT_AGENT_ID = "vault-ingest"   # provenance tag → enables one-command reversal
NOTE_SCHEMA = "VaultNote"
TRANSCRIPT_SCHEMA = "VaultTranscriptIndex"

SUMMARY_CAP = 1200               # max chars of body stored per note
RULE_GATE_BYTES = 6_000          # R4: only short notes can be agent-shaped by heading
DEFAULT_MAX_AGENTS = 80          # runaway-classifier backstop

# A heading that smells like a standing rule (R4 gate).
_RULE_HEADING_RE = re.compile(
    r"^#+\s*(Decision|Rule|Policy|Principle|Always|Never|Trigger|When)\b",
    re.M | re.I,
)
# A "## Decision" section, used to distill an agent's principle.
_DECISION_SECTION_RE = re.compile(
    r"^#+\s*Decision\s*\n+(.+?)(?:\n#|\Z)", re.M | re.I | re.S
)

_AGENT_STATUSES = {"permanent", "evergreen", "accepted", "active", "decided"}
_NEVER_AGENT_TYPES = {"section-readme", "moc", "map-of-content"}
_TRANSCRIPT_TYPES = {"conversation-transcript"}


# ---- frontmatter (stdlib, robust for the vault's real patterns) ------------


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _scalar_or_list(val: str):
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_unquote(x) for x in inner.split(",") if x.strip()]
    return _unquote(val)


def _read_frontmatter(text: str) -> tuple[dict, str]:
    """Split a note into (frontmatter dict, body). Best-effort, never raises.

    Handles: ``key: value`` scalars, inline ``[a, b]`` lists, and block lists
    (``key:`` then indented ``- item`` lines). Quoted scalars are unquoted.
    Nested maps are ignored (not present in vault notes). Returns ``({}, text)``
    when there is no parseable frontmatter block.
    """
    m = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?(.*)\Z", text, re.DOTALL)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    out: dict = {}
    lines = raw.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if line[0] in (" ", "\t") or ":" not in line:
            i += 1  # stray continuation / malformed — skip
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            # Possible block list on following indented lines.
            items: list[str] = []
            j = i + 1
            while j < len(lines) and (lines[j][:1] in (" ", "\t")):
                s = lines[j].strip()
                if s.startswith("- "):
                    items.append(_unquote(s[2:]))
                elif s == "":
                    pass
                else:
                    break
                j += 1
            out[key] = items if items else ""
            i = j if items else i + 1
            continue
        out[key] = _scalar_or_list(val)
        i += 1
    return out, body


# ---- note model + classifier -----------------------------------------------


@dataclass
class Note:
    path: Path
    rel: str
    domain: str          # top-level folder, e.g. "100 Zettelkasten"
    fm: dict
    body: str
    raw_bytes: bytes
    degraded: bool = False

    @property
    def title(self) -> str:
        t = self.fm.get("title")
        return str(t).strip() if t else self.path.stem

    @property
    def note_type(self) -> str:
        return str(self.fm.get("type") or "").strip().lower()

    @property
    def status(self) -> str:
        return str(self.fm.get("status") or "").strip().lower()

    @property
    def vault_hash(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()[:16]


def classify(note: Note) -> str:
    """Return one of: 'transcript', 'agent', 'knowledge'. Default knowledge."""
    nt = note.note_type
    if nt in _TRANSCRIPT_TYPES:
        return "transcript"
    if nt in _NEVER_AGENT_TYPES or note.path.stem.lower() in ("readme", "index"):
        return "knowledge"

    # AGENT-SHAPED gates — conservative, high precision.
    # R1 keys on `type: thesis` ONLY, not the folder: the Core Theses folder
    # also holds strategy docs (type: weapon/strategic-asset/index) that are
    # knowledge, not principles. Folder membership is too broad a signal.
    if nt == "thesis":                                                # R1
        return "agent"
    if nt == "decision" and note.status in _AGENT_STATUSES:           # R2
        return "agent"
    if nt == "wisdom" or note.domain == "360 Wisdom":                 # R3
        return "agent"
    # R4 — catch agent-shaped notes that lack a clean `type:`. Only fires on
    # UNtyped notes: a typed note that fell through R1–R3 (e.g. a draft
    # decision rejected by its status) was intentionally left as knowledge,
    # so R4 must not override that with a heading match.
    if (
        not nt
        and len(note.raw_bytes) < RULE_GATE_BYTES
        and _RULE_HEADING_RE.search(note.body)
    ):
        return "agent"
    return "knowledge"


# ---- string helpers for agent emission -------------------------------------


def _pascal(s: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", s)
    name = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not name or not name[0].isalpha():
        name = "Agent" + name
    return name[:60]


def _ident(s: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = "p_" + slug
    return slug[:48] or "principle"


def _sanitize_rule(s: str) -> str:
    """One clean line safe inside a ``rule: "..."`` string."""
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace('"', "'").replace("{", "(").replace("}", ")")
    s = s.replace("\\", "/")
    if len(s) > 220:
        s = s[:217].rstrip() + "..."
    return s or "uphold this principle"


def _distill_rule(note: Note) -> str:
    """Best one-line distillation of a note's standing rule."""
    m = _DECISION_SECTION_RE.search(note.body)
    if m:
        return _sanitize_rule(m.group(1))
    # First non-heading, non-blockquote paragraph.
    for para in re.split(r"\n\s*\n", note.body):
        line = para.strip()
        if line and not line.startswith("#") and not line.startswith(">"):
            return _sanitize_rule(line)
    return _sanitize_rule(note.title)


def _first_paragraph(note: Note) -> str:
    for para in re.split(r"\n\s*\n", note.body):
        line = para.strip()
        if line and not line.startswith("#"):
            return _sanitize_rule(line)[:300]
    return note.title


def build_agent_text(note: Note, agent_name: str) -> str:
    """Emit a compiling ethics+agent .t.md for an agent-shaped note."""
    principle = _ident(note.title)
    rule = _distill_rule(note)
    prose = _first_paragraph(note)
    return f"""---
agent: {agent_name}
tessera_version: 0.2
capabilities_requested: []
max_cost: {{ dollars: 0.00, tokens: 0 }}
---

# {note.title}

{prose}

> Generated from vault note `{note.rel}` by `tessera vault ingest`.

```tsr:ethics
ethics {{
  principle {principle} {{ weight: 1.0  rule: "{rule}" }}
  on_conflict: highest_weight
  on_violation: refuse
}}
```

```tsr:agent
agent {agent_name} {{
  beliefs:
    @last_write situation: String
  intentions:
    plan apply {{
      return situation
    }}
}}
```
"""


def _agent_validates(text: str) -> list[str]:
    """Return a list of error messages; empty means it compiles clean."""
    from ...parser.module import parse_source
    from ...sir.build import lower
    from ...verify.passes import run_local

    try:
        pm = parse_source(text)
        module = lower(pm)
        diags = run_local(module)
    except Exception as e:  # noqa: BLE001 — any lowering failure is a reject
        return [f"{type(e).__name__}: {e}"]
    return [str(d) for d in diags if getattr(d, "severity", "") == "error"]


# ---- fact field mapping -----------------------------------------------------


def _note_fields(note: Note) -> dict:
    return {
        "title": note.title,
        "domain": note.domain,
        "note_type": note.note_type or "unknown",
        "tags": note.fm.get("tags") or [],
        "status": note.status or None,
        "created": str(note.fm.get("created") or note.fm.get("date") or ""),
        "wikilinks": sorted(set(_WIKILINK_RE.findall(note.body))),
        "summary": note.body.strip()[:SUMMARY_CAP],
        "source_path": note.rel,
        "vault_hash": note.vault_hash,
        "degraded": note.degraded,
    }


def _transcript_fields(note: Note) -> dict:
    return {
        "title": note.title,
        "domain": note.domain,
        "note_type": note.note_type or "conversation-transcript",
        "created": str(note.fm.get("created") or note.fm.get("date") or ""),
        "size_bytes": len(note.raw_bytes),
        "source_path": note.rel,
        "vault_hash": note.vault_hash,
    }


# ---- report -----------------------------------------------------------------


@dataclass
class IngestReport:
    vault: str
    out: str
    db: str
    transcripts_mode: str
    dry_run: bool
    scanned: int = 0
    facts_new: int = 0
    facts_existing: int = 0
    transcripts_indexed: int = 0
    transcripts_existing: int = 0
    agents_emitted: list = field(default_factory=list)   # (filename, src, kind)
    skipped: list = field(default_factory=list)          # (rel, reason)
    degraded: list = field(default_factory=list)         # (rel, reason)
    per_folder: dict = field(default_factory=dict)       # folder -> counts
    aborted: str | None = None

    def _bump(self, folder: str, key: str) -> None:
        d = self.per_folder.setdefault(
            folder, {"facts": 0, "agents": 0, "transcripts": 0}
        )
        d[key] += 1

    def to_markdown(self) -> str:
        lines = [
            "# Vault Ingest Report",
            "",
            f"- Source: `{self.vault}`",
            f"- DB: `{self.db}`",
            f"- Output: `{self.out}`",
            f"- Transcripts mode: `{self.transcripts_mode}`",
            f"- Dry run: {self.dry_run}",
            "",
            "## Totals",
            f"- Scanned: {self.scanned} .md files",
            f"- Facts ingested (new): {self.facts_new}",
            f"- Facts already present: {self.facts_existing}",
            f"- Transcripts indexed (new): {self.transcripts_indexed}",
            f"- Transcripts already present: {self.transcripts_existing}",
            f"- Agents emitted: {len(self.agents_emitted)}",
            f"- Skipped: {len(self.skipped)}",
            f"- Degraded frontmatter: {len(self.degraded)}",
        ]
        if self.aborted:
            lines += ["", f"## ABORTED\n{self.aborted}"]
        if self.agents_emitted:
            lines += ["", "## Agents emitted"]
            for fn, src, kind in sorted(self.agents_emitted):
                lines.append(f"- `{fn}`  ←  `{src}`  ({kind})")
        lines += ["", "## Per-folder breakdown"]
        for folder in sorted(self.per_folder):
            d = self.per_folder[folder]
            lines.append(
                f"- {folder}: {d['facts']} facts, {d['agents']} agents, "
                f"{d['transcripts']} transcripts"
            )
        if self.degraded:
            lines += ["", "## Degraded frontmatter"]
            lines += [f"- `{rel}` — {why}" for rel, why in sorted(self.degraded)]
        if self.skipped:
            lines += ["", "## Skipped"]
            lines += [f"- `{rel}` — {why}" for rel, why in sorted(self.skipped)]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return {
            "vault": self.vault, "out": self.out, "db": self.db,
            "transcripts_mode": self.transcripts_mode, "dry_run": self.dry_run,
            "scanned": self.scanned, "facts_new": self.facts_new,
            "facts_existing": self.facts_existing,
            "transcripts_indexed": self.transcripts_indexed,
            "transcripts_existing": self.transcripts_existing,
            "agents_emitted": self.agents_emitted, "skipped": self.skipped,
            "degraded": self.degraded, "per_folder": self.per_folder,
            "aborted": self.aborted,
        }


# ---- walk + load ------------------------------------------------------------


def _iter_notes(vault_root: Path, out_root: Path | None, limit: int | None):
    count = 0
    for path in sorted(vault_root.rglob("*.md")):
        rel_parts = path.relative_to(vault_root).parts
        if any(p.startswith(".") for p in rel_parts):
            continue
        if out_root is not None and out_root in path.parents:
            continue
        if limit is not None and count >= limit:
            return
        count += 1
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="ignore")
        except OSError as e:
            yield ("error", path, f"read failed: {e}")
            continue
        rel = str(path.relative_to(vault_root))
        domain = rel_parts[0] if len(rel_parts) > 1 else "(root)"
        try:
            fm, body = _read_frontmatter(text)
            degraded = False
        except Exception as e:  # noqa: BLE001
            fm, body, degraded = {}, text, True
        yield (
            "note",
            Note(path=path, rel=rel, domain=domain, fm=fm, body=body,
                 raw_bytes=raw, degraded=degraded),
            None,
        )


# ---- main entrypoint --------------------------------------------------------


def ingest_vault(
    vault_root: str | Path,
    out_dir: str | Path,
    *,
    db_path: str | Path | None = None,
    dry_run: bool = False,
    transcripts: str = "skip",        # skip | stub | full
    limit: int | None = None,
    max_agents: int = DEFAULT_MAX_AGENTS,
    verbose: bool = False,
) -> IngestReport:
    vault = Path(vault_root).expanduser().resolve()
    if not vault.is_dir():
        raise FileNotFoundError(f"vault not found: {vault}")
    out = Path(out_dir).expanduser().resolve()
    agents_dir = out / "agents"

    from ...adapters.semantic import _resolve_db_path
    resolved_db = _resolve_db_path(db_path)

    report = IngestReport(
        vault=str(vault), out=str(out), db=str(resolved_db),
        transcripts_mode=transcripts, dry_run=dry_run,
    )

    # Pre-load existing hashes for idempotent re-runs (one query, not per-row).
    existing: set[str] = set()
    for schema in (NOTE_SCHEMA, TRANSCRIPT_SCHEMA):
        for row in query_facts(
            schema=schema, agent_id=FACT_AGENT_ID, db_path=db_path, limit=10_000_000
        ):
            h = row["fields"].get("vault_hash")
            if h:
                existing.add(h)

    # Pass 1: load + classify everything (no writes yet).
    notes: list[tuple[str, Note]] = []   # (kind, note)
    for kind, payload, err in _iter_notes(vault, out, limit):
        if kind == "error":
            report.scanned += 1
            report.skipped.append((str(payload), err))
            continue
        note: Note = payload
        report.scanned += 1
        if note.degraded:
            report.degraded.append((note.rel, "unparseable frontmatter"))
        notes.append((classify(note), note))

    agent_notes = [n for k, n in notes if k == "agent"]
    if len(agent_notes) > max_agents:
        report.aborted = (
            f"agent-shaped set is {len(agent_notes)} (> --max-agents {max_agents}). "
            "Review the classifier before emitting. Candidates:\n"
            + "\n".join(f"  - {n.rel}" for n in agent_notes)
        )
        if verbose:
            print(report.aborted)
        return report

    used_names: set[str] = set()

    def _unique_name(base: str) -> str:
        name = base
        n = 2
        while name in used_names:
            name = f"{base}_{n}"
            n += 1
        used_names.add(name)
        return name

    # Pass 2: act.
    for kind, note in notes:
        if kind == "transcript":
            if transcripts == "full":
                _ingest_note_fact(note, report, existing, db_path, dry_run)
            else:
                fields = _transcript_fields(note)
                if transcripts == "stub":
                    fields["summary"] = note.body.strip()[:SUMMARY_CAP]
                if note.vault_hash in existing:
                    report.transcripts_existing += 1
                else:
                    if not dry_run:
                        remember_fact(TRANSCRIPT_SCHEMA, fields,
                                      db_path=db_path, agent_id=FACT_AGENT_ID)
                    existing.add(note.vault_hash)
                    report.transcripts_indexed += 1
                    report._bump(note.domain, "transcripts")
            continue

        if kind == "agent":
            agent_name = _unique_name(_pascal(note.title))
            text = build_agent_text(note, agent_name)
            errs = _agent_validates(text)
            if errs:
                report.skipped.append(
                    (note.rel, f"agent failed validation: {errs[0]}")
                )
                # Fall back: still capture the knowledge as a fact.
                _ingest_note_fact(note, report, existing, db_path, dry_run)
                continue
            fn = f"{agent_name}.t.md"
            if not dry_run:
                agents_dir.mkdir(parents=True, exist_ok=True)
                (agents_dir / fn).write_text(text)
            report.agents_emitted.append((fn, note.rel, f"type={note.note_type or '?'}"))
            report._bump(note.domain, "agents")
            continue

        # knowledge
        _ingest_note_fact(note, report, existing, db_path, dry_run)

    # Write report (even on dry-run, so you can inspect before committing).
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "INGEST_REPORT.md").write_text(report.to_markdown())
        (out / "manifest.json").write_text(json.dumps(report.to_dict(), indent=2))
    if verbose:
        print(report.to_markdown())
    return report


def _ingest_note_fact(note, report, existing, db_path, dry_run) -> None:
    if note.vault_hash in existing:
        report.facts_existing += 1
        return
    if not dry_run:
        remember_fact(NOTE_SCHEMA, _note_fields(note),
                      db_path=db_path, agent_id=FACT_AGENT_ID)
    existing.add(note.vault_hash)
    report.facts_new += 1
    report._bump(note.domain, "facts")
