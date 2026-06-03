#!/usr/bin/env python3
"""proposal — draft a tight client proposal / SOW in your voice.

    proposal "Casey's Diner | new website + local SEO, 5 pages"
"""
import sys
from datetime import datetime

from brain import llm


def main(argv):
    arg = " ".join(argv).strip()
    if not arg:
        print('usage: proposal "Client | scope of work"', file=sys.stderr)
        return 1
    client, _, scope = arg.partition("|")
    today = datetime.now().strftime("%B %-d, %Y")
    out = llm(
        "Draft a short, professional proposal / SOW in James's plain, direct "
        "voice for Walt Builds (websites, local SEO, AI consulting for small "
        "businesses). Sections: Overview, Scope of Work (bullets), Timeline, "
        "Investment (a price range), Next Steps. Tight, concrete, zero "
        f"buzzwords. Use this exact date, do not invent one: {today}. "
        "End with a contact line:\n"
        "jamesburge.mcm@gmail.com · (662) 292-5533\n\n"
        f"CLIENT: {client.strip() or 'the client'}\n"
        f"SCOPE: {scope.strip() or arg}"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
