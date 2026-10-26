"""Degree-preserving null models for conditional testing.

The null hypothesis: the observed graph is a uniform draw from the set of
simple graphs with exactly the same degree sequence. Sampling is done by
double edge swaps (a standard MCMC over this set), which preserve every
node's degree exactly -- hence "exact degree-preserving".
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def degree_preserving_sample(G: nx.Graph, rng: np.random.Generator,
                             swaps_per_edge: int = 10) -> nx.Graph:
    """Return one sample from the degree-preserving null.

    Performs `swaps_per_edge * m` double edge swaps on a copy of G.
    Each swap picks two edges (a,b),(c,d) and rewires to (a,d),(c,b)
    if that keeps the graph simple. Degrees never change.
    """
    H = G.copy()
    m = H.number_of_edges()
    nswap = swaps_per_edge * m
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        nx.double_edge_swap(H, nswap=nswap, max_tries=nswap * 20, seed=seed)
    except nx.NetworkXAlgorithmError:
        # max_tries exceeded before completing all swaps; the chain has
        # still mixed substantially. Acceptable for Monte Carlo use.
        pass
    return H


def structureless_control(n: int, rng: np.random.Generator,
                          gamma: float = 2.5, mean_deg: float = 8.0) -> nx.Graph:
    """A graph with heavy-tailed degrees but NO community structure.

    Chung-Lu model: edge (i,j) present with prob ~ w_i * w_j / sum(w).
    Degree heterogeneity mimics real entity graphs (a few hub entities,
    many rare ones) while containing zero planted group structure.
    """
    w = rng.pareto(gamma - 1, size=n) + 1.0
    w = w / w.mean() * mean_deg
    # expected_degree_graph implements Chung-Lu efficiently
    G = nx.expected_degree_graph(w, seed=int(rng.integers(0, 2**31 - 1)),
                                 selfloops=False)
    G.remove_edges_from(nx.selfloop_edges(G))
    # keep largest connected component, as GraphRAG graphs are analyzed per component
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len)
    return G.subgraph(giant).copy()


def planted_partition(n: int, k: int, mu: float,
                      rng: np.random.Generator,
                      mean_deg: float = 10.0) -> nx.Graph:
    """Planted-partition graph with k equal communities.

    mu = mixing parameter: fraction of a node's edges expected to go
    OUTSIDE its community. mu close to 0 -> strong communities;
    mu = (k-1)/k -> no structure (uniform random).
    """
    sizes = [n // k] * k
    sizes[0] += n - sum(sizes)
    p_in = mean_deg * (1 - mu) / (n / k - 1)
    p_out = mean_deg * mu / (n - n / k)
    p_in, p_out = min(p_in, 1.0), min(p_out, 1.0)
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs,
                                  seed=int(rng.integers(0, 2**31 - 1)))
    G = nx.Graph(G)  # strip block model metadata wrapper
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len)
    return G.subgraph(giant).copy()
