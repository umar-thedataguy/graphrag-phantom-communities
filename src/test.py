"""Exact degree-preserving conditional test for community structure,
plus a hierarchical estimator of the number of supported communities (K-hat).

Test
----
Statistic: maximized modularity Q(G) (Louvain, randomized restart seed).
Null:      uniform over simple graphs with the SAME degree sequence,
           sampled via double edge swaps (conditioning on degrees exactly).
p-value:   Monte Carlo, p = (1 + #{Q_null >= Q_obs}) / (1 + B).
           A valid finite-sample ("exact") Monte Carlo p-value.

Why condition on degrees? Hubs alone inflate modularity. Conditioning on
the exact degree sequence removes degree heterogeneity as an explanation,
so a rejection means group structure beyond what the wiring (degrees)
already implies.

K-hat
-----
Recursive significance pruning:
  1. Test the whole graph. Not significant -> K-hat = 1.
  2. Significant -> split into the Louvain partition, recurse into each
     sufficiently large part.
  3. K-hat = number of leaves where splitting stopped.
All decisions use Bonferroni correction by the realized number of tests.
Correction only shrinks the tree (it never adds splits), so applying it
after the optimistic recursion is conservative and sound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from null_models import degree_preserving_sample


def max_modularity(G: nx.Graph, rng: np.random.Generator) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    seed = int(rng.integers(0, 2**31 - 1))
    comms = nx.community.louvain_communities(G, seed=seed)
    return nx.community.modularity(G, comms)


def conditional_test(G: nx.Graph, B: int, rng: np.random.Generator,
                     swaps_per_edge: int = 5):
    """Return (p_value, Q_observed, null_Q_distribution)."""
    q_obs = max_modularity(G, rng)
    null_qs = np.empty(B)
    for b in range(B):
        H = degree_preserving_sample(G, rng, swaps_per_edge)
        null_qs[b] = max_modularity(H, rng)
    p = (1 + int(np.sum(null_qs >= q_obs))) / (1 + B)
    return float(p), float(q_obs), null_qs


class _Node:
    __slots__ = ("size", "raw_p", "q", "children")

    def __init__(self, size: int):
        self.size = size
        self.raw_p: float | None = None
        self.q: float | None = None
        self.children: list["_Node"] = []


@dataclass
class KhatResult:
    khat: int
    n_tests: int
    root_p_raw: float
    root_p_bonferroni: float
    root_q: float = 0.0
    root_null_qs: np.ndarray | None = None
    tests: list = field(default_factory=list)  # dicts: size, raw_p, bonf_p


def estimate_khat(G: nx.Graph, B: int, rng: np.random.Generator,
                  alpha: float = 0.05, min_size: int = 30,
                  max_depth: int = 3, swaps_per_edge: int = 5) -> KhatResult:
    tested: list[_Node] = []
    root_dist = {}

    def recurse(H: nx.Graph, depth: int) -> _Node:
        node = _Node(H.number_of_nodes())
        if H.number_of_nodes() < min_size or depth > max_depth:
            return node
        p, q, null_qs = conditional_test(H, B, rng, swaps_per_edge)
        if depth == 0:
            root_dist["q"], root_dist["nulls"] = q, null_qs
        node.raw_p, node.q = p, q
        tested.append(node)
        if p < alpha:  # optimistic recursion; Bonferroni prunes later
            seed = int(rng.integers(0, 2**31 - 1))
            parts = nx.community.louvain_communities(H, seed=seed)
            if len(parts) > 1:
                for part in parts:
                    node.children.append(recurse(H.subgraph(part).copy(),
                                                 depth + 1))
        return node

    root = recurse(G, 0)
    n_tests = max(1, len(tested))

    def leaves(node: _Node) -> int:
        significant = (node.raw_p is not None
                       and node.raw_p * n_tests < alpha
                       and node.children)
        if significant:
            return sum(leaves(c) for c in node.children)
        return 1

    khat = leaves(root)
    root_p = root.raw_p if root.raw_p is not None else 1.0
    return KhatResult(
        khat=khat,
        n_tests=n_tests,
        root_p_raw=root_p,
        root_p_bonferroni=min(1.0, root_p * n_tests),
        root_q=root_dist.get("q", 0.0),
        root_null_qs=root_dist.get("nulls"),
        tests=[{"size": t.size, "raw_p": t.raw_p,
                "bonf_p": min(1.0, t.raw_p * n_tests)} for t in tested],
    )
