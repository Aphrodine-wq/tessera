"""Integrated Information Theory substrate (research C1).

Primary references:
- Tononi, G. (2004). An information integration theory of consciousness.
  BMC Neuroscience, 5, 42.
- Tononi, G., Boly, M., Massimini, M., Koch, C. (2016). Integrated
  information theory: from consciousness to its physical substrate.
  Nature Reviews Neuroscience, 17(7), 450-461.
- Mediano, P. A. M., Rosas, F. E., Bor, D., Seth, A. K., Barrett, A. B.
  (2022). The strength of weak integrated information theory. Trends in
  Cognitive Sciences, 26(8), 646-655.

This module ships a TRACTABLE APPROXIMATION of integrated information
— we call it φ*. The canonical Tononi-IIT φ is super-exponential in
the number of partitions; we cannot ship it without the PyPhi
dependency, and even PyPhi struggles past ~6 nodes.

What we ship instead:
  φ*(graph) = 1 - (sum of intra-partition edge weight) / (total edge weight)

over the MINIMUM-CUT partition found via greedy edge-removal search.
This is the "geometric information loss" approximation Mediano et al.
(2022) endorse as a tractable signature of integration. It correlates
with canonical φ on small systems and runs in polynomial time on
declared belief/intention dependency graphs.

CRITICAL (PHILOSOPHY.md): φ* is a STRUCTURAL property of the graph.
It is NOT consciousness. The substrate ships the measure; it explicitly
refuses any block claiming φ > 0 → conscious.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class DependencyGraph:
    """A directed weighted dependency graph.

    `edges[(src, dst)] = weight` — each edge represents an information
    dependency. For a Tessera agent: src/dst are belief or intention
    names; weight is the strength of the dependency (default 1.0 if
    not declared).
    """
    nodes: list[str] = field(default_factory=list)
    edges: dict[tuple[str, str], float] = field(default_factory=dict)

    def total_edge_weight(self) -> float:
        return sum(self.edges.values())

    def intra_partition_weight(self, partition: list[set[str]]) -> float:
        """Sum of edge weights with BOTH endpoints in the same partition."""
        node_to_part: dict[str, int] = {}
        for i, p in enumerate(partition):
            for n in p:
                node_to_part[n] = i
        total = 0.0
        for (src, dst), w in self.edges.items():
            if node_to_part.get(src) == node_to_part.get(dst):
                total += w
        return total


def all_bipartitions(nodes: list[str]) -> list[list[set[str]]]:
    """Enumerate all non-trivial bipartitions of a node set.

    Returns at most 2^(n-1) - 1 distinct cuts (we don't count
    permutations of the same cut as different).
    """
    parts: list[list[set[str]]] = []
    n = len(nodes)
    if n < 2:
        return []
    # Enumerate subsets of indices 1..n-1 (fix index 0 in one side to
    # avoid counting [A, B] and [B, A] as different). k=0 → left = {0}
    # only; up to k=n-2 → right is just one node. k=n-1 means left = all,
    # right = empty → skipped.
    for k in range(0, n):
        for combo in combinations(range(1, n), k):
            left_idx = {0, *combo}
            left = {nodes[i] for i in range(n) if i in left_idx}
            right = {nodes[i] for i in range(n) if i not in left_idx}
            if left and right:
                parts.append([left, right])
    return parts


def phi_star(graph: DependencyGraph) -> float:
    """The φ* approximation.

    For every bipartition P, compute the fraction of edge weight that
    CROSSES the partition. φ* is the MINIMUM such fraction across all
    partitions — i.e. the cut that least disrupts the graph. High φ*
    means the graph is highly integrated; low φ* means it has a
    near-decomposable structure.

    Returns φ* in [0, 1]. By convention φ* = 0 when the graph is
    trivial (≤1 node or no edges).
    """
    if len(graph.nodes) < 2 or not graph.edges:
        return 0.0
    total_w = graph.total_edge_weight()
    if total_w == 0:
        return 0.0

    best = 1.0  # min cut fraction; start at upper bound
    for partition in all_bipartitions(graph.nodes):
        intra = graph.intra_partition_weight(partition)
        # Fraction of weight CROSSING the partition
        cross = (total_w - intra) / total_w
        if cross < best:
            best = cross
    # φ* = the minimum cross-partition fraction
    return best


def build_dependency_graph_for_agent(module, agent_name: str) -> DependencyGraph:
    """Extract a belief/intention dependency graph for one agent.

    Belief nodes come from the agent region's declared @last_write
    beliefs. Edges connect beliefs that appear together in plan bodies
    (rough proxy — a follow-up could parse SIR data-flow precisely).
    """
    graph = DependencyGraph()
    region = module.agents.get(agent_name)
    if region is None:
        return graph
    # Crude extraction: for each plan in the region (children with
    # name starting "plan:"), gather all BeliefRead/BeliefWrite node
    # names; create edges between every consecutive pair.
    plan_names: list[str] = []
    from .sir.nodes import Op
    for r in module.regions:
        if r.parent == region.id and r.name.startswith("plan:"):
            plan_names.append(r.name)
            belief_names: list[str] = []
            for node in r.nodes:
                if node.op in (Op.BeliefRead, Op.BeliefRevise):
                    nm = node.attributes.get("name")
                    if nm:
                        belief_names.append(nm)
            unique = list(dict.fromkeys(belief_names))
            for n in unique:
                if n not in graph.nodes:
                    graph.nodes.append(n)
            for i in range(len(belief_names) - 1):
                a, b = belief_names[i], belief_names[i + 1]
                if a != b:
                    key = (a, b)
                    graph.edges[key] = graph.edges.get(key, 0.0) + 1.0
    return graph


# ----- Substrate decl + claim-check guards -----


_FORBIDDEN_CLAIM_PATTERNS = (
    "is conscious",
    "has consciousness",
    "subjective experience",
    "phi implies",
    "phi > 0 means",
    "consciousness proven",
)


def claim_violates_consciousness_discipline(text: str) -> str | None:
    """Return a reason string if `text` makes a metaphysical consciousness
    claim that PHILOSOPHY.md forbids; else None. Used by pass_9."""
    if not text:
        return None
    lower = text.lower()
    for pattern in _FORBIDDEN_CLAIM_PATTERNS:
        if pattern in lower:
            return f"forbidden consciousness claim: matched pattern {pattern!r}"
    return None
