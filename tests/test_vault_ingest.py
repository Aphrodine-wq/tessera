"""Coverage for `tessera vault ingest` — the non-destructive vault converter.

The autouse `isolate_tessera_stores` fixture (conftest) redirects the
semantic store to a tmp db, so facts written here stay isolated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from tessera.adapters.obsidian.ingest import (
    Note,
    _read_frontmatter,
    build_agent_text,
    classify,
    ingest_vault,
)
from tessera.adapters.semantic import query_facts


# ---- fixture vault ----------------------------------------------------------


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _make_vault(tmp_path: Path) -> Path:
    v = tmp_path / "TheVault"
    # knowledge: a zettel
    _write(v, "100 Zettelkasten/timeouts.md", """---
title: Real timeouts beat artificial budgets
created: 2026-03-23
type: zettelkasten
tags: [systems, infra]
---
Adaptive systems need real timeouts. See [[Adaptive systems]] and [[Budgets]].
""")
    # agent-shaped: a core thesis
    _write(v, "050 Core Theses/automation.md", """---
title: Automation IS Freedom
type: thesis
status: permanent
---
Owning your tools is owning your time. The point of automation is not idleness.
""")
    # agent-shaped: a standing decision
    _write(v, "700 Decisions/no-browse.md", """---
title: Homeowners cannot browse contractors
type: decision
status: accepted
---
## Decision
Homeowners post a job and receive bids; there is no contractor directory.
""")
    # NOT agent-shaped: a draft decision stays knowledge
    _write(v, "700 Decisions/draft-thing.md", """---
title: Maybe charge per seat
type: decision
status: draft
---
## Decision
Undecided — exploring per-seat pricing.
""")
    # transcript
    _write(v, "800 Conversations/Transcripts/sess1.md", """---
title: Claude Session 2026-05-01
type: conversation-transcript
date: 2026-05-01
---
""" + ("blah blah transcript line\n" * 500))
    # a dotfile dir that must be skipped
    _write(v, ".obsidian/app.md", "should be ignored")
    return v


# ---- frontmatter parser -----------------------------------------------------


def test_read_frontmatter_inline_and_block_lists():
    fm, body = _read_frontmatter(
        '---\ntitle: "Quoted Title"\ntags: [a, b, c]\nstatus: active\n'
        "owners:\n  - alice\n  - bob\n---\nThe body.\n"
    )
    assert fm["title"] == "Quoted Title"
    assert fm["tags"] == ["a", "b", "c"]
    assert fm["status"] == "active"
    assert fm["owners"] == ["alice", "bob"]
    assert body.strip() == "The body."


def test_read_frontmatter_absent():
    fm, body = _read_frontmatter("# Just a heading\n\ntext")
    assert fm == {}
    assert body.startswith("# Just a heading")


# ---- classifier -------------------------------------------------------------


def _note(tmp_path, rel, fm, body, raw=None):
    domain = rel.split("/")[0]
    return Note(
        path=tmp_path / rel, rel=rel, domain=domain, fm=fm, body=body,
        raw_bytes=raw if raw is not None else body.encode(),
    )


def test_classify_thesis_is_agent(tmp_path):
    n = _note(tmp_path, "050 Core Theses/x.md", {"type": "thesis"}, "principle text")
    assert classify(n) == "agent"


def test_classify_strategy_doc_in_theses_folder_stays_knowledge(tmp_path):
    # A non-thesis note in the Core Theses folder is a strategy doc, not a
    # principle — it must stay knowledge (R1 keys on type, not folder).
    n = _note(tmp_path, "050 Core Theses/Strategic Analysis/oracle.md",
              {"type": "weapon", "title": "The Construction Oracle"}, "strategy")
    assert classify(n) == "knowledge"


def test_classify_accepted_decision_is_agent(tmp_path):
    n = _note(tmp_path, "700 Decisions/x.md",
              {"type": "decision", "status": "accepted"}, "## Decision\nDo X.")
    assert classify(n) == "agent"


def test_classify_draft_decision_is_knowledge(tmp_path):
    n = _note(tmp_path, "700 Decisions/x.md",
              {"type": "decision", "status": "draft"}, "## Decision\nMaybe.")
    assert classify(n) == "knowledge"


def test_classify_transcript(tmp_path):
    n = _note(tmp_path, "800 Conversations/x.md",
              {"type": "conversation-transcript"}, "chat")
    assert classify(n) == "transcript"


def test_classify_zettel_is_knowledge(tmp_path):
    n = _note(tmp_path, "100 Zettelkasten/x.md", {"type": "zettelkasten"}, "idea")
    assert classify(n) == "knowledge"


def test_classify_big_rule_note_not_agent(tmp_path):
    # R4 only fires on small notes — a long essay with a rule heading stays knowledge.
    body = "## Rule\n" + ("filler " * 2000)
    n = _note(tmp_path, "000 Inbox/x.md", {}, body, raw=body.encode())
    assert classify(n) == "knowledge"


# ---- agent emission validates ----------------------------------------------


def test_emitted_agent_compiles(tmp_path):
    from tessera.parser.module import parse_source
    from tessera.sir.build import lower
    from tessera.verify.passes import run_local

    n = _note(tmp_path, "700 Decisions/no-browse.md",
              {"title": "Homeowners cannot browse contractors", "type": "decision"},
              "## Decision\nHomeowners post a job and receive bids.")
    text = build_agent_text(n, "HomeownersCannotBrowse")
    module = lower(parse_source(text))
    errors = [d for d in run_local(module) if d.severity == "error"]
    assert errors == [], errors


def test_emitted_agent_sanitizes_quotes_and_braces(tmp_path):
    n = _note(tmp_path, "050 Core Theses/x.md", {"title": 'Use "real" {tools}'},
              'Owning your "tools" means {freedom}.')
    text = build_agent_text(n, "Tools")
    # No raw double-quote or brace inside the rule string would break the parser.
    rule_line = [l for l in text.splitlines() if "rule:" in l][0]
    assert '"real"' not in rule_line
    # the rule value is wrapped in exactly one pair of double quotes
    assert rule_line.count('"') == 2


# ---- end-to-end ingest ------------------------------------------------------


def test_ingest_end_to_end(tmp_path):
    v = _make_vault(tmp_path)
    out = tmp_path / "out"
    report = ingest_vault(v, out, transcripts="skip")

    # 2 knowledge facts (zettel + draft decision), 2 agents, 1 transcript indexed.
    assert report.facts_new == 2
    assert len(report.agents_emitted) == 2
    assert report.transcripts_indexed == 1
    # dotfile skipped → never scanned as a note path part
    assert all(".obsidian" not in s for s, _ in report.skipped)

    # agents written and named
    agent_files = sorted(p.name for p in (out / "agents").glob("*.t.md"))
    assert len(agent_files) == 2

    # facts landed in the store with provenance tag
    notes = query_facts(schema="VaultNote", agent_id="vault-ingest")
    assert len(notes) == 2
    titles = {n["fields"]["title"] for n in notes}
    assert "Real timeouts beat artificial budgets" in titles
    # wikilinks were extracted, not inlined
    z = next(n for n in notes if n["fields"]["domain"] == "100 Zettelkasten")
    assert "Adaptive systems" in z["fields"]["wikilinks"]

    # transcript index has no body
    tx = query_facts(schema="VaultTranscriptIndex", agent_id="vault-ingest")
    assert len(tx) == 1
    assert "summary" not in tx[0]["fields"]
    assert tx[0]["fields"]["size_bytes"] > 0

    # report files written
    assert (out / "INGEST_REPORT.md").exists()
    assert (out / "manifest.json").exists()


def test_ingest_idempotent(tmp_path):
    v = _make_vault(tmp_path)
    out = tmp_path / "out"
    first = ingest_vault(v, out, transcripts="skip")
    assert first.facts_new == 2

    second = ingest_vault(v, out, transcripts="skip")
    assert second.facts_new == 0
    assert second.facts_existing == 2
    assert second.transcripts_indexed == 0
    assert second.transcripts_existing == 1

    # store did not grow
    assert len(query_facts(schema="VaultNote", agent_id="vault-ingest")) == 2


def test_dry_run_writes_nothing(tmp_path):
    v = _make_vault(tmp_path)
    out = tmp_path / "out"
    report = ingest_vault(v, out, transcripts="skip", dry_run=True)
    assert report.facts_new == 2          # counted
    assert not out.exists()               # but nothing written
    assert query_facts(schema="VaultNote", agent_id="vault-ingest") == []


def test_transcripts_full_mode_stores_summary(tmp_path):
    v = _make_vault(tmp_path)
    out = tmp_path / "out"
    ingest_vault(v, out, transcripts="full")
    # in full mode the transcript becomes a VaultNote with a (capped) summary
    notes = query_facts(schema="VaultNote", agent_id="vault-ingest")
    tx = [n for n in notes if n["fields"]["domain"] == "800 Conversations"]
    assert len(tx) == 1
    assert 0 < len(tx[0]["fields"]["summary"]) <= 1200


def test_max_agents_abort(tmp_path):
    v = _make_vault(tmp_path)
    out = tmp_path / "out"
    report = ingest_vault(v, out, max_agents=1)
    assert report.aborted is not None
    assert "agent-shaped set is 2" in report.aborted
    # aborted before writing anything
    assert query_facts(schema="VaultNote", agent_id="vault-ingest") == []
