#!/usr/bin/env python3
"""triage — sort an incoming message and draft a reply, holding money for approval.

    triage "Hey James, can you get the invoice over for the kitchen job?"
    pbpaste | triage
"""
import sys

from brain import input_text, llm


def main(argv):
    msg = input_text(argv)
    if not msg:
        print('usage: triage "message"  (or pipe text in)', file=sys.stderr)
        return 1
    out = llm(
        "Triage this incoming message for James. Output exactly:\n"
        "PRIORITY: urgent | client | normal | noise\n"
        "SUMMARY: one line\n"
        "DRAFT REPLY: in James's plain, direct voice\n\n"
        "BUT if it involves money, payment, invoices, contracts, or pricing: do "
        'NOT draft a reply — output "HOLD — needs James\'s approval before '
        'sending" and one line on why.\n\n'
        f"MESSAGE:\n{msg}"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
