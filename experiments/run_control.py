"""Experiment 1 — structureless negative control.

Generate a graph with heavy-tailed degrees (like a real entity graph)
but ZERO community structure. Then:
  (a) run GraphRAG's own clusterer (hierarchical_leiden) and count the
      communities it would pay an LLM to summarize;
  (b) run the exact degree-preserving conditional test and K-hat.

Expected: (a) resolves many communities, (b) says "nothing here".
"""
import json
import pathlib
import sys
import time

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from leiden_wrap import graphrag_communities  # noqa: E402
from null_models import structureless_control  # noqa: E402
from test import estimate_khat  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 20251022
B = 199

def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    G = structureless_control(n=800, rng=rng, gamma=2.5, mean_deg=7.0)
    print(f"control graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges (largest component)")

    leiden = graphrag_communities(G)
    print("GraphRAG clusterer output:", leiden)

    kh = estimate_khat(G, B=B, rng=rng, min_size=40, max_depth=2)
    p, q_obs, null_qs = kh.root_p_raw, kh.root_q, kh.root_null_qs
    print(f"conditional test (khat root): Q_obs={q_obs:.4f}, p={p:.3f}")
    print(f"K-hat={kh.khat}, Bonferroni p={kh.root_p_bonferroni:.3f} "
          f"({kh.n_tests} tests)")

    out = {
        "seed": SEED, "B": B,
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "leiden": leiden,
        "q_observed": q_obs,
        "null_q_mean": float(null_qs.mean()),
        "null_q_max": float(null_qs.max()),
        "p_value": p,
        "khat": kh.khat,
        "bonferroni_p": kh.root_p_bonferroni,
        "n_tests": kh.n_tests,
    }
    print(f"elapsed {time.time() - t0:.1f}s")
    (RESULTS / "control.json").write_text(json.dumps(out, indent=2))
    np.savez(RESULTS / "control_nulldist.npz",
             null_qs=null_qs, q_obs=q_obs)
    print("saved results/control.json")


if __name__ == "__main__":
    main()
