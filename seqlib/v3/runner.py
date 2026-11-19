"""v3 and the ladder versions -- order a chart with the evidence model, the move margin and a solver."""
from __future__ import annotations

from typing import Optional, Sequence

from seqlib.io.chart import Chart
from seqlib.v1.runner import EXCLUDE, OrderResult

from .model import V3Model, chart_pairs, order_with


def run_v3(chart: Chart, model: V3Model, embedder=None, pages: Optional[Sequence[int]] = None,
           margin: Optional[float] = None, solver: Optional[str] = None, seam=None) -> OrderResult:
    """``pages``: the pages to order (default: the pipeline's Main pages). ``margin`` / ``solver``
    override the trained model's (for experiments). ``seam``: the seam scorer for v6."""
    junk = [p.page_number for p in chart.pages if p.group in EXCLUDE]
    if pages is None:
        pages = [p.page_number for p in chart.pages if p.group not in EXCLUDE]
    else:
        junk = [p.page_number for p in chart.pages if p.page_number not in set(pages)]
    version = getattr(model.settings, "version", "v3")
    if len(pages) < 2:
        return OrderResult(chart.name, sorted(pages), junk, version)
    rows = chart_pairs(chart, pages, model.settings, embedder=embedder, seam=seam)
    m = model.margin if margin is None else margin
    s = solver or model.settings.solver
    order = order_with(rows, model.evidence.predict(rows.X), m, s)
    return OrderResult(chart.name, order, junk, f"{version}:{s}:m{m}")
