"""Bare-callable coding tools — read_file / edit_file / bash.

Resolved via dotted import path same as any Tessera tool (see
`adapters.langchain.resolve_callable`), so these need no special wiring: a
`tsr:tool` declaration pointing `from tessera.adapters.coding.read_file` (etc.)
just works, no LangChain or other dependency required.

Each function catches its own errors and returns a descriptive string rather
than raising — an agent's plan has no exception handling, so a raised
exception would just crash the run. A returned error string lets the agent
see what went wrong and adapt (e.g. re-read a path it got wrong).
"""
from __future__ import annotations

import subprocess

_MAX_READ_BYTES = 1_500      # cap what lands in a prompt — sized for a small
                             # local model's context window (a coding agent
                             # driven by a cheap on-device model, not a
                             # frontier API with a huge window), and it
                             # compounds: this text gets re-embedded in every
                             # loop iteration's prompt, not read once
_MAX_OUTPUT_CHARS = 500      # same reasoning for bash output
_BASH_TIMEOUT_SECONDS = 30


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(_MAX_READ_BYTES + 1)
    except OSError as e:
        return f"error: {e}"
    if len(text) > _MAX_READ_BYTES:
        text = text[:_MAX_READ_BYTES] + f"\n...(truncated at {_MAX_READ_BYTES} bytes)"
    return text


def edit_file(path: str, old: str, new: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return f"error: {e}"
    count = text.count(old)
    if count == 0:
        return f"error: old_string not found in {path}"
    if count > 1:
        return f"error: old_string is not unique in {path} ({count} matches) — add more context"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
    except OSError as e:
        return f"error: {e}"
    return f"edited {path}"


def bash(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=_BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {_BASH_TIMEOUT_SECONDS}s"
    output = (result.stdout or "") + (result.stderr or "")
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + f"\n...(truncated at {_MAX_OUTPUT_CHARS} chars)"
    return f"exit={result.returncode}\n{output}"
