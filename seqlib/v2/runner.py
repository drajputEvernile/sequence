"""v2 -- order a chart with the trained pair scorer and the decoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from seqlib.io.chart import Chart
from seqlib.v1.runner import EXCLUDE, OrderResult, run_v1

from .decode import decode
from .features import START, ChartFeatures, candidates, chart_features, pair_matrix
from .model import PairScorer


@dataclass
class ScoredChart:
    """A chart's pages, their scan nodes (START = 0, pages 1..n in scan order) and the model's
    probability for every candidate link (node a -> node b)."""

    features: ChartFeatures
    node: dict[int, int]  # page number (START = -1) -> node
    edge: dict[tuple[int, int], float]

    def page_of(self, k: int) -> int:
        return self.features.pages[k - 1].page_number


def score_pairs(chart: Chart, scorer: PairScorer, pages: Sequence[int], embedder=None) -> ScoredChart:
    cf = chart_features(chart, pages, embedder)
    s = scorer.settings
    pairs = candidates(cf, s.window, s.top_k)
    probs = scorer.predict(pair_matrix(cf, pairs)) if pairs else []
    node = {p.page_number: p.rank + 1 for p in cf.pages}
    node[START] = 0
    edge = {(node[a], node[b]): float(pr) for (a, b), pr in zip(pairs, probs)}
    return ScoredChart(cf, node, edge)


def v1_start(chart: Chart, pages: Sequence[int]) -> list[int]:
    """v1's order of ``pages`` (its page-number repairs), any page it left out kept in scan order."""
    keep = set(pages)
    order = [p for p in run_v1(chart).main_order if p in keep]
    for p in sorted(keep - set(order)):
        k = next((i for i, q in enumerate(order) if q > p), len(order))
        order.insert(k, p)
    return order


def run_v2(chart: Chart, scorer: PairScorer, embedder=None, pages: Optional[Sequence[int]] = None,
           start: str = "scan") -> OrderResult:
    """``pages``: the pages to order (default: the pipeline's Main pages; pass the truth's Main
    pages to measure ordering alone). ``start``: "scan" or "v1" -- the order the search begins
    from (documents of up to 11 pages are solved exactly either way)."""
    junk = [p.page_number for p in chart.pages if p.group in EXCLUDE]
    if pages is None:
        pages = [p.page_number for p in chart.pages if p.group not in EXCLUDE]
    else:
        junk = [p.page_number for p in chart.pages if p.page_number not in set(pages)]
    scored = score_pairs(chart, scorer, pages, embedder)
    begin = None
    if start == "v1":
        begin = [scored.node[p] for p in v1_start(chart, [p.page_number for p in scored.features.pages])]
    order_nodes = decode(len(scored.features.pages), scored.edge, max_block=scorer.settings.max_block, start=begin)
    order = [scored.page_of(k) for k in order_nodes]
    detail = {}
    for prev, cur in zip([START, *order], order):
        p = scored.edge.get((scored.node[prev], scored.node[cur]))
        detail[cur] = {"confidence": round(p, 4) if p is not None else 0.0, "after": prev if prev != START else "start"}
    return OrderResult(chart.name, order, junk, f"v2:{start}", detail)
