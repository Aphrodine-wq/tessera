"""Unit tests for the bare-callable coding tools (read_file/edit_file/bash)."""
from tessera.adapters.coding import bash, edit_file, read_file


def test_read_file_returns_contents(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    assert read_file(str(p)) == "hello world"


def test_read_file_missing_path_returns_error_string():
    out = read_file("/no/such/path/xyz")
    assert out.startswith("error:")


def test_edit_file_replaces_unique_match(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("foo bar baz")
    out = edit_file(str(p), "bar", "QUX")
    assert out == f"edited {p}"
    assert p.read_text() == "foo QUX baz"


def test_edit_file_rejects_ambiguous_match(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("bar bar")
    out = edit_file(str(p), "bar", "QUX")
    assert "not unique" in out
    assert p.read_text() == "bar bar"  # unchanged


def test_edit_file_rejects_missing_match(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("foo")
    out = edit_file(str(p), "bar", "QUX")
    assert "not found" in out


def test_bash_runs_and_captures_output():
    out = bash("echo hi")
    assert out.startswith("exit=0")
    assert "hi" in out


def test_bash_captures_nonzero_exit():
    out = bash("exit 7")
    assert out.startswith("exit=7")


def test_bash_times_out(monkeypatch):
    import tessera.adapters.coding as coding
    monkeypatch.setattr(coding, "_BASH_TIMEOUT_SECONDS", 1)
    out = coding.bash("sleep 5")
    assert "timed out" in out
