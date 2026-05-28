"""Causal inference substrate (research D1, Pearl 2009).

Primary references:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.).
  Cambridge University Press.
- Pearl, J., & Mackenzie, D. (2018). The Book of Why.

A `tsr:causal` block declares a causal DAG over named variables:

    causal MarketDAG {
      var price: Float
      var demand: Float
      var season: String
      edge season -> price
      edge season -> demand
      edge price -> demand
    }

The substrate ships Pearl's backdoor criterion and adjustment-set search.
The MVP supports identifiability of do(X=x) → Y queries via the backdoor
adjustment: given DAG G, treatment X, outcome Y, find a set Z that
(a) contains no descendants of X, (b) blocks every backdoor path from X
to Y. If such Z exists, the causal effect P(Y | do(X)) is identifiable
from observational data via the adjustment formula:

    P(Y | do(X=x)) = sum over z of P(Y | X=x, Z=z) * P(Z=z)

Honest scope: this MVP handles single-treatment / single-outcome
identifiability. Front-door adjustment, instrumental variables, and the
full do-calculus rules (Pearl 2009, ch. 3) are explicit follow-ups.
Real numerical estimation of the adjustment integral requires data; we
ship the structural identifiability check only.

Pure Python — no networkx dependency. If `networkx` is installed, a
follow-up commit can substitute its DAG algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class CausalDAG:
    """A declared causal DAG. `variables` is the node set; `edges` is the
    edge list (parent, child). No cycle check — the lower-pass refuses
    cycles at compile time."""
    name: str
    variables: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def parents(self, node: str) -> set[str]:
        return {p for p, c in self.edges if c == node}

    def children(self, node: str) -> set[str]:
        return {c for p, c in self.edges if p == node}

    def descendants(self, node: str) -> set[str]:
        seen: set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            for c in self.children(n):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def has_cycle(self) -> bool:
        """Standard Kahn-style cycle detection."""
        indeg = {v: 0 for v in self.variables}
        for p, c in self.edges:
            if c in indeg:
                indeg[c] += 1
        queue = [v for v, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for c in self.children(n):
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        return seen != len(self.variables)


def _all_paths(dag: CausalDAG, src: str, dst: str) -> list[list[tuple[str, str, str]]]:
    """Return every undirected path from src to dst as a list of edges.

    Each edge is a (u, v, direction) triple where direction is 'out' if
    u -> v in the DAG and 'in' if v -> u (we traversed against the arrow).
    """
    paths: list[list[tuple[str, str, str]]] = []
    visited = {src}

    def dfs(current: str, trail: list[tuple[str, str, str]]):
        if current == dst:
            paths.append(list(trail))
            return
        # All neighbors, with the direction we traverse the edge
        nbrs: list[tuple[str, str]] = []
        for c in dag.children(current):
            if c not in visited:
                nbrs.append((c, "out"))
        for p in dag.parents(current):
            if p not in visited:
                nbrs.append((p, "in"))
        for nxt, direction in nbrs:
            visited.add(nxt)
            trail.append((current, nxt, direction))
            dfs(nxt, trail)
            trail.pop()
            visited.discard(nxt)

    dfs(src, [])
    return paths


def _path_starts_into(path: list[tuple[str, str, str]]) -> bool:
    """A backdoor path is one that starts with an arrow INTO the treatment."""
    if not path:
        return False
    _, _, direction = path[0]
    return direction == "in"


def _is_path_blocked_by(path: list[tuple[str, str, str]], Z: set[str], dag: CausalDAG) -> bool:
    """d-separation: a path is blocked by conditioning set Z when, traversing
    the path, every node falls into one of:
      - chain (A -> B -> C) or fork (A <- B -> C): B blocks the path iff B in Z
      - collider (A -> B <- C): B blocks unless B (or any descendant of B) is in Z
    """
    if not path:
        return True
    # Reconstruct the sequence of nodes along the path
    nodes = [path[0][0]]
    for _, v, _ in path:
        nodes.append(v)

    # Walk internal nodes (everything except endpoints) and classify each
    for i in range(1, len(nodes) - 1):
        # Determine the kinds of incoming/outgoing edges at node nodes[i]
        # Look at edge (i-1) -> (i) and edge (i) -> (i+1).
        # In our path encoding, an edge u→v with direction='out' means u→v in DAG;
        # direction='in' means v→u in DAG (so the path traversed v's arrow backward).
        e_in = path[i - 1]   # (prev_node, this_node, dir)
        e_out = path[i]      # (this_node, next_node, dir)
        # Arrow head at nodes[i] from the left edge?
        left_arrow_at_i = (e_in[2] == "out")   # prev -> this
        right_arrow_at_i = (e_out[2] == "in")  # this <- next
        is_collider = left_arrow_at_i and right_arrow_at_i
        if is_collider:
            # Collider blocks unless the collider OR any descendant is in Z.
            descendants_plus_self = dag.descendants(nodes[i]) | {nodes[i]}
            if not (descendants_plus_self & Z):
                return True  # collider blocks the path
        else:
            # Chain or fork: blocks iff conditioned-on.
            if nodes[i] in Z:
                return True
    return False


def find_backdoor_adjustment_set(
    dag: CausalDAG, treatment: str, outcome: str
) -> set[str] | None:
    """Search for a minimal admissible Z satisfying Pearl's backdoor criterion.

    Returns the smallest Z found (lexicographically among smallest sets),
    or None if no admissible Z exists in the declared variables.

    Pearl's criterion: Z is admissible iff
      (a) no node in Z is a descendant of treatment, and
      (b) Z blocks every backdoor path from treatment to outcome.
    """
    if treatment not in dag.variables or outcome not in dag.variables:
        return None
    descendants_of_T = dag.descendants(treatment)
    candidates = [v for v in dag.variables
                  if v not in {treatment, outcome} and v not in descendants_of_T]
    backdoor_paths = [p for p in _all_paths(dag, treatment, outcome)
                      if _path_starts_into(p)]
    if not backdoor_paths:
        return set()  # no backdoor paths — effect identifiable with empty Z

    # Brute-force from smallest subset up. Real implementations use
    # Tian's algorithm or van der Zander's; brute force is fine for ≤12 vars.
    for size in range(0, len(candidates) + 1):
        for combo in combinations(candidates, size):
            Z = set(combo)
            if all(_is_path_blocked_by(p, Z, dag) for p in backdoor_paths):
                return Z
    return None


def query_effect_identifiable(
    dag: CausalDAG, treatment: str, outcome: str
) -> tuple[bool, set[str] | None]:
    """Return (identifiable, adjustment_set) for P(outcome | do(treatment))."""
    Z = find_backdoor_adjustment_set(dag, treatment, outcome)
    return (Z is not None, Z)
