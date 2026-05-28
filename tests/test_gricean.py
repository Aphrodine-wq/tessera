"""Tests for the Gricean substrate (research 4.5)."""
from tessera.gricean import (
    check_quantity,
    check_quality,
    check_relation,
    check_manner,
    check_all_maxims,
)


# ---------- Quantity ----------


def test_quantity_under_informative_violates():
    r = check_quantity("hi", min_words=5)
    assert r.violated
    assert "under-informative" in r.reason


def test_quantity_over_informative_violates():
    r = check_quantity(" ".join(["word"] * 300), max_words=200)
    assert r.violated
    assert "over-informative" in r.reason


def test_quantity_inside_range_passes():
    r = check_quantity("a reasonable message of moderate length",
                       min_words=3, max_words=50)
    assert not r.violated


# ---------- Quality ----------


def test_quality_hedged_claim_passes():
    r = check_quality("I think the budget might be tight",
                      evidence_keywords=["budget", "according to"])
    assert not r.violated
    assert "hedged" in r.reason


def test_quality_unhedged_with_evidence_passes():
    r = check_quality("According to the spec, the cap is 5",
                      evidence_keywords=["according to", "spec", "rfc"])
    assert not r.violated


def test_quality_unhedged_no_evidence_violates():
    r = check_quality("The cap is definitely 5",
                      evidence_keywords=["according to", "spec"])
    assert r.violated
    assert "without evidence" in r.reason


def test_quality_no_evidence_keywords_declared_passes():
    """When the agent doesn't declare evidence keywords, we can't enforce."""
    r = check_quality("The cap is 5", evidence_keywords=[])
    assert not r.violated


# ---------- Relation ----------


def test_relation_topic_match_passes():
    r = check_relation("The retainage on this job is 10%",
                       topic_keywords=["retainage", "job"])
    assert not r.violated


def test_relation_no_topic_match_violates():
    r = check_relation("Let's talk about the weather",
                       topic_keywords=["retainage", "invoice"])
    assert r.violated


# ---------- Manner ----------


def test_manner_repetition_violates():
    """Repeating the same word over and over trips the repeat ratio."""
    msg = "very very very very very very long very very very"
    r = check_manner(msg, max_repeat_ratio=0.3)
    assert r.violated
    assert "repetition" in r.reason


def test_manner_long_clause_violates():
    long_clause = " ".join(["word"] * 50)  # no commas at all
    r = check_manner(long_clause, max_clause_words=40)
    assert r.violated
    assert "longest clause" in r.reason


def test_manner_clean_passes():
    r = check_manner("Short, clean, well-punctuated, easy to read.")
    assert not r.violated


# ---------- check_all_maxims ----------


def test_check_all_returns_four_results_in_order():
    results = check_all_maxims(
        "A short message about the topic",
        evidence_keywords=["topic"],
        topic_keywords=["topic"],
    )
    assert [r.maxim for r in results] == ["quantity", "quality", "relation", "manner"]
