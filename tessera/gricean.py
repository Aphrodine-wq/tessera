"""Gricean cooperative-communication maxims (research 4.5).

Primary reference: Grice, H. P. (1975). Logic and conversation. In
P. Cole & J. Morgan (eds.), Syntax and Semantics 3: Speech Acts.
Academic Press, 41-58.

Grice's Cooperative Principle has four maxims:
  - QUANTITY: be as informative as required, no more.
  - QUALITY: don't say what you believe false, or what you lack
    adequate evidence for.
  - RELATION: be relevant.
  - MANNER: avoid obscurity / ambiguity / unnecessary prolixity.

This substrate ships a checker per maxim. An outgoing message is
scored against each; violations land in audit as gricean:violation
events. Author declares which maxims gate (refuse on violation) vs.
which warn (audit only).

Honest scope: maxims are DEFEASIBLE in actual human conversation
(irony, polite indirection, deniability). The substrate makes them
pluggable per agent — a customer-facing agent might disable Manner's
prolixity check; an investigator agent might dial Quality strict.

Pure Python; no NLP dependency. The checkers use simple
length/repetition/evidence heuristics. A follow-up can swap in
LLM-based scoring per maxim while keeping the substrate interface
stable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MaximResult:
    """Per-maxim verdict on a candidate message."""
    maxim: str            # "quantity" | "quality" | "relation" | "manner"
    violated: bool
    score: float          # in [0, 1] — 1 = fully compliant, 0 = full violation
    reason: str           # human-readable explanation


def check_quantity(
    message: str,
    *,
    min_words: int = 1,
    max_words: int = 200,
) -> MaximResult:
    """QUANTITY: be informative as required, no more.

    Violation conditions:
      - shorter than min_words (under-informative)
      - longer than max_words (over-informative)
    """
    n = len(message.split())
    if n < min_words:
        return MaximResult(
            maxim="quantity", violated=True,
            score=max(0.0, n / max(1, min_words)),
            reason=f"under-informative: {n} words < min {min_words}",
        )
    if n > max_words:
        return MaximResult(
            maxim="quantity", violated=True,
            score=max(0.0, 1 - (n - max_words) / max_words),
            reason=f"over-informative: {n} words > max {max_words}",
        )
    return MaximResult(
        maxim="quantity", violated=False,
        score=1.0,
        reason=f"OK ({n} words within [{min_words}, {max_words}])",
    )


def check_quality(
    message: str,
    evidence_keywords: list[str] | None = None,
    *,
    hedging_markers: tuple[str, ...] = ("i think", "maybe", "possibly",
                                        "i believe", "likely", "uncertain"),
) -> MaximResult:
    """QUALITY: don't claim without evidence.

    Heuristic: when the message makes a strong assertion (no hedging
    marker) but contains NONE of the declared evidence keywords, that's
    a quality violation. Author declares evidence_keywords to inject
    domain-relevant evidence terms ("according to", "per the spec",
    "sec citation:", etc.).
    """
    evidence_keywords = evidence_keywords or []
    lower = message.lower()
    has_hedge = any(h in lower for h in hedging_markers)
    has_evidence = any(kw.lower() in lower for kw in evidence_keywords)
    # If hedged, the agent is appropriately uncertain — not a violation.
    if has_hedge:
        return MaximResult(
            maxim="quality", violated=False, score=1.0,
            reason="hedged claim — quality preserved",
        )
    if not evidence_keywords:
        # No evidence terms declared — can't check; pass.
        return MaximResult(
            maxim="quality", violated=False, score=1.0,
            reason="no evidence keywords declared",
        )
    if has_evidence:
        return MaximResult(
            maxim="quality", violated=False, score=1.0,
            reason="assertion backed by declared evidence keyword",
        )
    return MaximResult(
        maxim="quality", violated=True, score=0.3,
        reason="unhedged assertion without evidence keyword",
    )


def check_relation(
    message: str,
    topic_keywords: list[str],
) -> MaximResult:
    """RELATION: be relevant to the topic.

    Heuristic: at least one topic_keyword should appear in the message.
    Author declares topic_keywords (e.g. plan name + agent intent +
    important nouns from the prompt).
    """
    if not topic_keywords:
        return MaximResult(
            maxim="relation", violated=False, score=1.0,
            reason="no topic keywords declared",
        )
    lower = message.lower()
    hits = sum(1 for kw in topic_keywords if kw.lower() in lower)
    if hits == 0:
        return MaximResult(
            maxim="relation", violated=True, score=0.0,
            reason=f"no topic keywords matched in message; topics: {topic_keywords}",
        )
    return MaximResult(
        maxim="relation", violated=False,
        score=min(1.0, hits / len(topic_keywords)),
        reason=f"matched {hits}/{len(topic_keywords)} topic keywords",
    )


def check_manner(
    message: str,
    *,
    max_repeat_ratio: float = 0.3,
    max_clause_words: int = 40,
) -> MaximResult:
    """MANNER: avoid obscurity, ambiguity, unnecessary prolixity.

    Heuristics:
      - high word-repetition ratio = prolix
      - long single clauses (no comma/semicolon for max_clause_words
        words) = obscure
    """
    words = message.lower().split()
    if not words:
        return MaximResult(
            maxim="manner", violated=False, score=1.0,
            reason="empty message",
        )
    unique = len(set(words))
    repeat_ratio = 1 - unique / len(words)
    # Find longest run of words without comma or semicolon
    clauses = re.split(r"[,;]+", message)
    longest_clause = max((len(c.split()) for c in clauses), default=0)
    issues = []
    score = 1.0
    if repeat_ratio > max_repeat_ratio:
        issues.append(f"repetition ratio {repeat_ratio:.2f} > {max_repeat_ratio}")
        score = min(score, 1 - repeat_ratio)
    if longest_clause > max_clause_words:
        issues.append(f"longest clause {longest_clause} words > {max_clause_words}")
        score = min(score, max(0.0, 1 - (longest_clause - max_clause_words) / max_clause_words))
    if issues:
        return MaximResult(
            maxim="manner", violated=True, score=score,
            reason="; ".join(issues),
        )
    return MaximResult(
        maxim="manner", violated=False, score=1.0,
        reason=f"OK (repeat {repeat_ratio:.2f}, longest clause {longest_clause})",
    )


def check_all_maxims(
    message: str,
    *,
    min_words: int = 1,
    max_words: int = 200,
    evidence_keywords: list[str] | None = None,
    topic_keywords: list[str] | None = None,
) -> list[MaximResult]:
    """Run all four maxim checks and return the results in canonical order."""
    return [
        check_quantity(message, min_words=min_words, max_words=max_words),
        check_quality(message, evidence_keywords=evidence_keywords),
        check_relation(message, topic_keywords=topic_keywords or []),
        check_manner(message),
    ]
