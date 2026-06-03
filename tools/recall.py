#!/usr/bin/env python3
"""recall — ask your brain a question, get an answer grounded in your notes.

    recall "what did we decide about marketplace payments?"
"""
import sys

from brain import gather_notes, llm


def main(argv):
    q = " ".join(argv).strip()
    if not q:
        print('usage: recall "question"', file=sys.stderr)
        return 1
    notes = gather_notes(q, limit=12)
    if not notes:
        print("nothing in the vault touches that — first time we've thought about it.")
        return 0
    def line(n: dict) -> str:
        where = n.get("domain") or n.get("area") or "?"
        label = n.get("title") or n.get("rule", "")[:60] or "(note)"
        text = (n.get("summary") or n.get("rule") or "").strip()[:180]
        return f"- [{where}] {label}: {text}"

    ctx = "\n".join(line(n) for n in notes)
    ans = llm(
        "Answer James's question using ONLY these vault notes. Cite note titles "
        "in [brackets]. If the notes don't actually answer it, say so plainly "
        f"instead of guessing.\n\nNOTES:\n{ctx}\n\nQUESTION: {q}"
    )
    print(ans)
    print("\nsources:")
    for n in notes[:6]:
        print(f"  - {n.get('title') or n.get('rule','')[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
