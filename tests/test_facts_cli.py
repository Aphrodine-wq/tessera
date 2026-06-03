"""CLI coverage for `tessera facts` (list / search / clear).

The autouse `isolate_tessera_stores` fixture in conftest redirects the
semantic store to a tmp path, so `remember_fact` here lands in an isolated
db and the CLI reads it back through the same env override.
"""
from __future__ import annotations

import json

from tessera.adapters.semantic import remember_fact
from tessera.cli import main


def _seed():
    remember_fact("note", {"text": "hello world"})
    remember_fact("note", {"text": "goodbye moon"})
    remember_fact("person", {"name": "josh"}, agent_id="A1")


def test_list_summary_when_unfiltered(capsys):
    _seed()
    rc = main(["facts", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "note" in out and "person" in out
    assert "3 fact(s) across 2 schema(s)" in out


def test_list_json_returns_rows(capsys):
    _seed()
    rc = main(["facts", "list", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    rows = [json.loads(ln) for ln in out.splitlines()]
    assert len(rows) == 3
    assert {r["schema"] for r in rows} == {"note", "person"}


def test_list_filter_by_schema(capsys):
    _seed()
    rc = main(["facts", "list", "--schema", "note", "--json"])
    rows = [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert len(rows) == 2
    assert all(r["schema"] == "note" for r in rows)


def test_list_limit_caps(capsys):
    _seed()
    rc = main(["facts", "list", "--schema", "note", "--limit", "1", "--json"])
    rows = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert len(rows) == 1


def test_search_hits_and_misses(capsys):
    _seed()
    rc = main(["facts", "search", "hello", "--json"])
    rows = [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert len(rows) == 1
    assert rows[0]["fields"]["text"] == "hello world"

    rc = main(["facts", "search", "nonexistent", "--json"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_clear_bare_refuses(capsys):
    _seed()
    rc = main(["facts", "clear"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "refusing to clear all facts" in err
    # store untouched
    assert main(["facts", "list", "--json"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_clear_by_schema(capsys):
    _seed()
    rc = main(["facts", "clear", "--schema", "note"])
    assert rc == 0
    assert "cleared 2 fact(s)" in capsys.readouterr().out
    main(["facts", "list", "--json"])
    rows = [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]
    assert len(rows) == 1 and rows[0]["schema"] == "person"


def test_clear_all_wipes(capsys):
    _seed()
    rc = main(["facts", "clear", "--all"])
    assert rc == 0
    assert "cleared 3 fact(s)" in capsys.readouterr().out
    main(["facts", "list", "--json"])
    assert capsys.readouterr().out.strip() == ""
