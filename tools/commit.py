#!/usr/bin/env python3
"""Commit Scribe — read the diff, write the message in your style, flag risk.

    commit                # write a message for the staged diff, print it
    commit --commit       # ...and actually commit (blocked if risk is found)
    commit --commit --force   # commit anyway, risk acknowledged

The message is the LLM's, learned from this repo's recent commit style. The
RISK SCAN is deterministic code — secrets, migrations, deletions, credential
files — because you never trust a model to be the thing that catches a leaked
key. If the scan trips, `--commit` refuses unless you pass `--force`.

Uses Tessera's LLM adapter (TESSERA_LLM_BACKEND / TESSERA_OLLAMA_MODEL).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DIFF_CHAR_CAP = 12_000   # keep the prompt sane on huge diffs

# --- deterministic risk patterns (scanned on ADDED lines only) ---
_SECRET_RES = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|passwd|password|auth[_-]?token|access[_-]?token|bearer)\b\s*[:=]\s*['\"]?[^\s'\"]{6,}"), "hardcoded credential"),
    (re.compile(r"(?i)gh[pousr]_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI-style key"),
]
_MIGRATION_RE = re.compile(r"(?i)(/migrations?/|\.sql$|schema\.prisma$|/alembic/|flyway)")
_CRED_FILE_RE = re.compile(r"(?i)(^|/)(\.env|.*\.pem$|.*\.key$|credentials(\.json)?$|id_rsa)")


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)
    return out.stdout


def scan_risk(repo: Path, diff: str) -> list[str]:
    flags: list[str] = []
    added = [ln[1:] for ln in diff.splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    body = "\n".join(added)
    for rx, label in _SECRET_RES:
        if rx.search(body):
            flags.append(f"possible {label} in added lines")

    # changed file paths from the name-status
    names = git(repo, "diff", "--cached", "--name-status")
    if not names.strip():
        names = git(repo, "diff", "HEAD", "--name-status")
    deleted, migrations, credfiles = [], [], []
    for line in names.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        if status.startswith("D"):
            deleted.append(path)
        if _MIGRATION_RE.search(path):
            migrations.append(path)
        if _CRED_FILE_RE.search(path):
            credfiles.append(path)
    if deleted:
        flags.append(f"{len(deleted)} file(s) DELETED: {', '.join(deleted[:5])}")
    if migrations:
        flags.append(f"DB migration touched: {', '.join(migrations[:5])}")
    if credfiles:
        flags.append(f"credential/secret file touched: {', '.join(credfiles[:5])}")
    return flags


PROMPT = """Write ONE git commit message for this diff, matching the repo's style exactly.

Recent commits in this repo (match this style — conventional prefix, tight subject, file-level body bullets):
{style}

The diff:
{diff}

Rules:
- First line: <type>: <imperative subject>, <=72 chars. Types: feat, fix, perf, refactor, test, docs, chore.
- Blank line, then 2-5 bullets describing what changed, by file or area.
- No fences, no preamble. Output only the commit message.
"""


def get_diff(repo: Path, staged: bool) -> tuple[str, str]:
    """Return (diff, source-label)."""
    if staged:
        return git(repo, "diff", "--cached"), "staged"
    cached = git(repo, "diff", "--cached")
    if cached.strip():
        return cached, "staged"
    return git(repo, "diff", "HEAD"), "tracked (unstaged)"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="commit")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--staged", action="store_true",
                    help="force using the staged diff only")
    ap.add_argument("--commit", action="store_true",
                    help="actually create the commit (blocked on risk unless --force)")
    ap.add_argument("--force", action="store_true",
                    help="commit even when the risk scan trips")
    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    if not (repo / ".git").exists() and "true" not in git(repo, "rev-parse", "--is-inside-work-tree"):
        print(f"not a git repo: {repo}", file=sys.stderr)
        return 1

    diff, src = get_diff(repo, args.staged)
    if not diff.strip():
        print("nothing to commit (no staged or tracked changes). `git add` first?",
              file=sys.stderr)
        return 1

    untracked = [l for l in git(repo, "ls-files", "--others", "--exclude-standard").splitlines() if l]
    risks = scan_risk(repo, diff)

    style = git(repo, "log", "--format=%s%n%b%n---", "-n", "5").strip() or "(no history)"
    sent_diff = diff[:DIFF_CHAR_CAP] + ("\n…[diff truncated]" if len(diff) > DIFF_CHAR_CAP else "")

    from tessera.adapters.llm import get_backend
    msg = get_backend().complete(
        PROMPT.format(style=style, diff=sent_diff)).text.strip()
    # strip an accidental code fence
    msg = re.sub(r"^```[a-z]*\n?|\n?```$", "", msg).strip()

    print(f"# commit message  (diff source: {src})")
    print("=" * 64)
    print(msg)
    print("=" * 64)
    if untracked:
        print(f"note: {len(untracked)} untracked file(s) NOT included — `git add` them to commit.")
    if risks:
        print("\n  RISK — review before committing:")
        for r in risks:
            print(f"   ⚠  {r}")

    if args.commit:
        if risks and not args.force:
            print("\nrefusing to auto-commit: risk scan tripped. Re-run with --force to override.",
                  file=sys.stderr)
            return 2
        res = subprocess.run(["git", "-C", str(repo), "commit", "-m", msg],
                             capture_output=True, text=True)
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        return res.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
