#!/usr/bin/env python3
"""explain — paste code or an error, get a plain explanation and the fix.

    explain "TypeError: cannot unpack non-iterable NoneType object"
    pbpaste | explain
"""
import sys

from brain import input_text, llm


def main(argv):
    blob = input_text(argv)
    if not blob:
        print('usage: explain "code or error"  (or pipe it in)', file=sys.stderr)
        return 1
    out = llm(
        "Explain this in plain English for a sharp, self-taught builder — what "
        "it does, or what the error means and why it happens — then the concrete "
        "fix. Short, no lecturing, no restating the obvious.\n\n"
        f"{blob}"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
