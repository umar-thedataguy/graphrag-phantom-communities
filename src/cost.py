"""Cost model: GraphRAG bills one LLM community report per community,
across all hierarchy levels. Convert community counts into summaries
billed and an indicative dollar estimate.

Token assumptions (conservative, documented):
  input  ~ 4000 tokens per report (entity/relationship context stuffed
           into the prompt, GraphRAG default max_input_length is larger)
  output ~ 500 tokens per report

Prices are parameters, not facts baked into results: pass the current
per-million-token prices of whatever model you'd use. Defaults below use
gpt-4o-mini-era pricing ($0.15 in / $0.60 out per 1M tokens) purely as
an illustrative reference point.
"""
from __future__ import annotations

from dataclasses import dataclass

IN_TOK = 4000
OUT_TOK = 500


@dataclass
class CostEstimate:
    summaries_billed: int
    summaries_supported: int
    excess_summaries: int
    usd_billed: float
    usd_excess: float


def estimate(n_total_communities: int, khat_supported: int,
             usd_per_m_in: float = 0.15,
             usd_per_m_out: float = 0.60) -> CostEstimate:
    per_report = (IN_TOK * usd_per_m_in + OUT_TOK * usd_per_m_out) / 1e6
    billed = n_total_communities
    supported = khat_supported
    excess = max(0, billed - supported)
    return CostEstimate(
        summaries_billed=billed,
        summaries_supported=supported,
        excess_summaries=excess,
        usd_billed=round(billed * per_report, 4),
        usd_excess=round(excess * per_report, 4),
    )
