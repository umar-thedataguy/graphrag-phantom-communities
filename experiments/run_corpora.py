"""Experiment 3 — three real corpora.

Pipeline per corpus (all public-domain books via NLTK's Gutenberg corpus):
  raw text -> regex entity co-occurrence graph -> GraphRAG's clusterer
  -> exact conditional test + K-hat -> summaries billed vs supported.

"Fineness ratio" = Leiden leaf communities / K-hat: how much finer the
partition GraphRAG pays to summarize is than the structure the wiring
statistically supports.
"""
import json
import pathlib
import sys
import time

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import nltk  # noqa: E402

nltk.download("gutenberg", quiet=True)
from nltk.corpus import gutenberg  # noqa: E402

from cost import estimate as cost_estimate  # noqa: E402
from extract_regex import extract_graph  # noqa: E402
from leiden_wrap import graphrag_communities  # noqa: E402
from test import estimate_khat  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 20251022
B = 199
CORPORA = ["austen-emma.txt", "melville-moby_dick.txt",
           "chesterton-thursday.txt"]


def main():
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for fid in CORPORA:
        t0 = time.time()
        text = gutenberg.raw(fid)
        G = extract_graph(text, min_count=3, min_edge_weight=2)
        # GraphRAG clusters (and bills) the weighted graph it builds:
        leiden = graphrag_communities(G)
        # The statistical audit runs on the unweighted backbone, because
        # the degree-preserving swap null is exact for simple unweighted
        # graphs (double_edge_swap does not preserve edge weights).
        import networkx as nx
        G_uw = nx.Graph()
        G_uw.add_nodes_from(G.nodes())
        G_uw.add_edges_from(G.edges())
        kh = estimate_khat(G_uw, B=B, rng=rng, min_size=40, max_depth=2)
        p, q_obs, null_qs = kh.root_p_raw, kh.root_q, kh.root_null_qs
        cost = cost_estimate(leiden["n_total_communities"], kh.khat)
        row = {
            "corpus": fid,
            "chars": len(text),
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "q_observed": q_obs, "null_q_mean": float(null_qs.mean()),
            "p_value": p,
            "khat": kh.khat,
            "bonferroni_root_p": kh.root_p_bonferroni,
            "n_tests": kh.n_tests,
            "leiden_level_counts": leiden["level_counts"],
            "leiden_leaf_communities": leiden["n_leaf_communities"],
            "leiden_total_communities": leiden["n_total_communities"],
            "fineness_ratio": (round(leiden["n_leaf_communities"] / kh.khat, 2)
                               if kh.khat else None),
            "summaries_billed": cost.summaries_billed,
            "summaries_supported": cost.summaries_supported,
            "excess_summaries": cost.excess_summaries,
            "usd_billed_at_ref_price": cost.usd_billed,
            "usd_excess_at_ref_price": cost.usd_excess,
            "runtime_sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    (RESULTS / "corpora.json").write_text(json.dumps(
        {"seed": SEED + 2, "B": B, "rows": rows}, indent=2))
    print("saved results/corpora.json")


if __name__ == "__main__":
    main()
