"""v2 -- order a chart with the trained pair scorer and the decoder."""
from __future__ import annotations

from typing import Optional, Sequence

from seqlib.io.chart import Chart
from seqlib.v1.runner import EXCLUDE, OrderResult

from .decode import decode
from .features import START, candidates, chart_features, pair_matrix
from .model import PairScorer


def run_v2(chart: Chart, scorer: PairScorer, embedder=None, pages: Optional[Sequence[int]] = None) -> OrderResult:
    """``pages``: the pages to order (default: the pipeline's Main pages; pass the truth's Main
    pages to measure ordering alone)."""
    junk = [p.page_number for p in chart.pages if p.group in EXCLUDE]
    if pages is None:
        pages = [p.page_number for p in chart.pages if p.group not in EXCLUDE]
    else:
        junk = [p.page_number for p in chart.pages if p.page_number not in set(pages)]
    cf = chart_features(chart, pages, embedder)
    s = scorer.settings
    pairs = candidates(cf, s.window, s.top_k)
    probs = scorer.predict(pair_matrix(cf, pairs)) if pairs else []
    node = {p.page_number: p.rank + 1 for p in cf.pages}
    node[START] = 0
    edge = {(node[a], node[b]): float(pr) for (a, b), pr in zip(pairs, probs)}
    order_nodes = decode(len(cf.pages), edge, max_block=s.max_block)
    order = [cf.pages[k - 1].page_number for k in order_nodes]
    detail = {}
    for prev, cur in zip([START, *order], order):
        p = edge.get((node[prev], node[cur]))
        detail[cur] = {"confidence": round(p, 4) if p is not None else 0.0, "after": prev if prev != START else "start"}
    return OrderResult(chart.name, order, junk, "v2", detail)
