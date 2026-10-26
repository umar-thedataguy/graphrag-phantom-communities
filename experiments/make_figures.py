"""Generate figures from saved results."""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def fig_control():
    d = np.load(RESULTS / "control_nulldist.npz")
    c = json.loads((RESULTS / "control.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(d["null_qs"], bins=25, color="#9ecae1", edgecolor="white",
            label=f"null: {len(d['null_qs'])} degree-preserving rewirings")
    ax.axvline(float(d["q_obs"]), color="#d62728", lw=2,
               label=f"observed graph (p = {c['p_value']:.3f})")
    ax.set_xlabel("maximized modularity Q")
    ax.set_ylabel("count")
    leaves = c["leiden"]["n_leaf_communities"]
    ax.set_title(
        "Structureless control: GraphRAG's clusterer resolves "
        f"{leaves} leaf communities;\nthe exact conditional test finds "
        f"nothing (K-hat = {c['khat']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_control.png", dpi=150)
    print("wrote fig_control.png")


def fig_billed_vs_supported():
    c = json.loads((RESULTS / "corpora.json").read_text())
    rows = c["rows"]
    labels = [r["corpus"].replace(".txt", "") for r in rows]
    billed = [r["summaries_billed"] for r in rows]
    supported = [r["summaries_supported"] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w / 2, billed, w, label="summaries GraphRAG bills",
           color="#fdae6b")
    ax.bar(x + w / 2, supported, w,
           label="communities the wiring supports (K-hat)", color="#74c476")
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylabel("count")
    ax.set_title("Per corpus: LLM summaries billed vs statistically "
                 "supported communities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_billed_vs_supported.png", dpi=150)
    print("wrote fig_billed_vs_supported.png")


if __name__ == "__main__":
    fig_control()
    try:
        fig_billed_vs_supported()
    except FileNotFoundError:
        sys.exit("corpora.json not found; run run_corpora.py first")
