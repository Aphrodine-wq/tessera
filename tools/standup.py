#!/usr/bin/env python3
"""standup — what you shipped across every repo in the last 24h, plus what's next.

    standup            last 24 hours
    standup 3          last 3 days
"""
import sys

from brain import git, git_repos, llm


def main(argv):
    days = int(argv[0]) if argv and argv[0].isdigit() else 1
    since = f"{days * 24} hours ago"
    blocks = []
    for repo in git_repos():
        log = git(repo, "log", f"--since={since}", "--format=%h %s").strip()
        if log:
            blocks.append(f"## {repo.name}\n" + "\n".join("  " + l for l in log.splitlines()))
    if not blocks:
        print(f"no commits across your repos in the last {days}d.")
        return 0
    raw = "\n".join(blocks)
    summary = llm(
        f"These are James's git commits from the last {days}d across projects. "
        "Write a 3-line standup: what shipped, what's mid-flight, what to tackle "
        f"next. Plain and direct, no fluff.\n\n{raw}"
    )
    print(summary)
    print("\n--- raw commits ---\n" + raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
