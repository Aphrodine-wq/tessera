#!/usr/bin/env python3
"""estimate — a rough construction estimate from a job description.

    estimate "reframe a 400 sqft fire-damaged garage, replace roof decking"

Honest note: this runs on the configured general model. Point it at
ConstructionAI (TESSERA_LLM_BACKEND/_OLLAMA_MODEL) for domain-trained numbers.
"""
import sys

from brain import llm


def main(argv):
    job = " ".join(argv).strip()
    if not job:
        print('usage: estimate "job description"', file=sys.stderr)
        return 1
    out = llm(
        "You are a seasoned construction estimator. Give a ROUGH estimate for "
        "this job using real trade knowledge. Format:\n"
        "- Scope (1 line)\n"
        "- Line items: materials and labor with rough costs\n"
        "- Subtotal, 15% overhead+profit, TOTAL as a range\n"
        "- Unknowns that would move the price\n"
        "Be realistic, not falsely precise.\n\n"
        f"JOB: {job}"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
