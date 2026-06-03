#!/usr/bin/env python3
"""gardener — find notes that should be linked but aren't (vault gardening).

    gardener pricing
    gardener            (broad pass)
"""
import sys

from brain import facts, keywords, llm


def main(argv):
    topic = " ".join(argv).strip() or "ideas"
    notes: dict[str, str] = {}
    for kw in keywords(topic) or ["the"]:
        for r in facts(contains=kw, limit=25):
            f = r["fields"]
            notes.setdefault(f.get("title", ""), f.get("domain", "?"))
    items = [(t, d) for t, d in notes.items() if t][:20]
    if len(items) < 2:
        print(f"not enough notes on '{topic}' to suggest links.")
        return 0
    lst = "\n".join(f"- {t} [{d}]" for t, d in items)
    out = llm(
        "These vault notes share a topic. Suggest 3-5 PAIRS that should be linked "
        "but probably aren't, each with one sentence on why the connection "
        "matters. Only pair notes from this list; quote titles exactly.\n\n"
        f"NOTES:\n{lst}"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
