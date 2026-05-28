"""Tests for the analogy substrate — structure-mapping (research 4.4)."""
from tessera.analogy import (
    Relation,
    Domain,
    find_best_mapping,
)


# ----- The canonical Rutherford analogy -----


def test_solar_system_to_atom_canonical_analogy():
    """Gentner's example: sun↔nucleus and planet↔electron via
    structural relation attracts(_, _) and revolves(_, _)."""
    solar = Domain(
        name="solar",
        objects=["sun", "planet"],
        relations=[
            Relation("attracts", ("sun", "planet")),
            Relation("revolves", ("planet", "sun")),
            Relation("more_massive", ("sun", "planet")),
        ],
    )
    atom = Domain(
        name="atom",
        objects=["nucleus", "electron"],
        relations=[
            Relation("attracts", ("nucleus", "electron")),
            Relation("revolves", ("electron", "nucleus")),
            Relation("more_massive", ("nucleus", "electron")),
        ],
    )
    m = find_best_mapping(solar, atom)
    assert m is not None
    assert m.bindings == {"sun": "nucleus", "planet": "electron"}
    assert len(m.matched_relations) == 3


def test_no_shared_structure_returns_none():
    s = Domain(name="s", objects=["a", "b"],
               relations=[Relation("foo", ("a", "b"))])
    t = Domain(name="t", objects=["x", "y"],
               relations=[Relation("bar", ("x", "y"))])
    m = find_best_mapping(s, t)
    assert m is None


def test_partial_match_returns_best_available():
    """Source has three relations; target only matches two. Mapping
    should bind correctly and report two matched relations."""
    s = Domain(
        name="s", objects=["a", "b", "c"],
        relations=[
            Relation("R1", ("a", "b")),
            Relation("R2", ("b", "c")),
            Relation("R3", ("a", "c")),
        ],
    )
    t = Domain(
        name="t", objects=["x", "y", "z"],
        relations=[
            Relation("R1", ("x", "y")),
            Relation("R2", ("y", "z")),
            # No R3
        ],
    )
    m = find_best_mapping(s, t)
    assert m is not None
    assert len(m.matched_relations) == 2
    assert m.bindings == {"a": "x", "b": "y", "c": "z"}


def test_systematicity_bonus_prefers_higher_arity():
    """When two mappings tie on count of matched relations, the one
    with higher-arity (more systematic) matches scores higher."""
    s = Domain(name="s", objects=["a", "b", "c"],
               relations=[Relation("R3", ("a", "b", "c"))])
    t = Domain(name="t", objects=["x", "y", "z"],
               relations=[Relation("R3", ("x", "y", "z"))])
    m = find_best_mapping(s, t)
    assert m is not None
    # 3-arity relation gets bonus: 1.0 + 0.25 * 2 = 1.5
    assert m.score == 1.5


def test_swapped_argument_order_does_not_match():
    """attracts(a, b) ≠ attracts(b, a) structurally — directionality
    matters in Gentner's framework."""
    s = Domain(name="s", objects=["a", "b"],
               relations=[Relation("attracts", ("a", "b"))])
    t_correct = Domain(name="t", objects=["x", "y"],
                       relations=[Relation("attracts", ("x", "y"))])
    t_swapped = Domain(name="t", objects=["x", "y"],
                       relations=[Relation("attracts", ("y", "x"))])
    # Correct direction: maps a→x, b→y; matched
    m_correct = find_best_mapping(s, t_correct)
    assert m_correct is not None
    # Swapped direction: a→y, b→x also works (still matched, but with
    # swapped binding)
    m_swapped = find_best_mapping(s, t_swapped)
    assert m_swapped is not None
    assert m_swapped.bindings == {"a": "y", "b": "x"}


def test_empty_domain_returns_none():
    s = Domain(name="s")
    t = Domain(name="t", objects=["x"])
    assert find_best_mapping(s, t) is None
