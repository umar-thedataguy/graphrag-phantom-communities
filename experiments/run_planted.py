"""Experiment 2 — planted-partition positive control (power check).

Graphs with k=5 planted communities at increasing mixing (noise) levels.
A useful test must (a) detect the structure when it is there and
(b) recover roughly the right K, not just reject.
"""
import json
import pathlib
import sys
import time

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from leiden_wrap import graphrag_communities  # noqa: E402
from null_models import planted_partition  # noqa: E402
from test import estimate_khat  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 20251022
B = 199
K_TRUE = 5


def main():
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for mu in (0.10, 0.25, 0.40):
        t0 = time.time()
        G = planted_partition(n=500, k=K_TRUE, mu=mu, rng=rng, mean_deg=10.0)
        leiden = graphrag_communities(G)
        kh = estimate_khat(G, B=B, rng=rng, min_size=40, max_depth=2)
        p, q_obs, null_qs = kh.root_p_raw, kh.root_q, kh.root_null_qs
        row = {
            "mu": mu, "k_true": K_TRUE,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "q_observed": q_obs, "null_q_mean": float(null_qs.mean()),
            "p_value": p,
            "khat": kh.khat, "bonferroni_root_p": kh.root_p_bonferroni,
            "n_tests": kh.n_tests,
            "leiden_leaf_communities": leiden["n_leaf_communities"],
            "leiden_total_communities": leiden["n_total_communities"],
            "runtime_sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(row)

    (RESULTS / "planted.json").write_text(json.dumps(
        {"seed": SEED + 1, "B": B, "rows": rows}, indent=2))
    print("saved results/planted.json")


if __name__ == "__main__":
    main()
