"""Wrapper around graspologic.partition.hierarchical_leiden -- the SAME
function Microsoft GraphRAG calls to build its community hierarchy
(graphrag/index/operations/cluster_graph.py uses graspologic).

GraphRAG then pays for one LLM "community report" per community, across
the nested levels. So the per-level community counts below are exactly
the number of summaries a GraphRAG indexing run would bill.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
from graspologic.partition import hierarchical_leiden

# GraphRAG's default
MAX_CLUSTER_SIZE = 10


def graphrag_communities(G: nx.Graph, max_cluster_size: int = MAX_CLUSTER_SIZE,
                         seed: int = 7) -> dict:
    """Run GraphRAG's clusterer; return per-level and leaf community counts."""
    H = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    partitions = hierarchical_leiden(H, max_cluster_size=max_cluster_size,
                                     random_seed=seed)
    by_level = defaultdict(set)
    leaf_of_node = {}
    for p in partitions:
        by_level[p.level].add(p.cluster)
        if p.is_final_cluster:
            leaf_of_node[p.node] = p.cluster
    level_counts = {lvl: len(cs) for lvl, cs in sorted(by_level.items())}
    leaf_clusters = set(leaf_of_node.values())
    return {
        "level_counts": level_counts,
        "n_levels": len(level_counts),
        "n_leaf_communities": len(leaf_clusters),
        "n_total_communities": sum(level_counts.values()),
    }
