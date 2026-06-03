#!/usr/bin/env python3
"""focus — the single highest-leverage thing to do right now. No option list.

    focus
"""
import sys

from brain import VAULT, git, git_repos, llm


def main(argv):
    loops_file = VAULT / "010 Working Memory" / "Open Loops.md"
    loops = loops_file.read_text()[:2000] if loops_file.exists() else "(no open-loops file)"
    uncommitted = []
    for repo in git_repos():
        s = git(repo, "status", "--short").strip()
        if s:
            uncommitted.append(f"{repo.name}: {len(s.splitlines())} uncommitted")
    ans = llm(
        "James needs the ONE thing to work on right now — the highest-leverage "
        "next action, not a list.\n\n"
        f"Open loops:\n{loops}\n\n"
        f"Repos with uncommitted work:\n{chr(10).join(uncommitted) or 'none'}\n\n"
        "Answer in two lines: the single action, then one line on why it's the lever."
    )
    print(ans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
