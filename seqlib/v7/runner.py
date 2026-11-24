"""The gated versions (v8, v9): order a chart -- the gate picks the move margin, then the solver orders."""
from __future__ import annotations

from typing import Optional, Sequence

from seqlib.io.chart import Chart
from seqlib.v1.runner import EXCLUDE, OrderResult
from seqlib.v3.model import chart_pairs, decode_options
from seqlib.v3.tuning import decode_pages, graph

from .model import V7Model, gate_features


def run_v7(chart: Chart, model: V7Model, pages: Optional[Sequence[int]] = None,
           seam=None) -> OrderResult:
    junk = [p.page_number for p in chart.pages if p.group in EXCLUDE]
    if pages is None:
        pages = [p.page_number for p in chart.pages if p.group not in EXCLUDE]
    else:
        junk = [p.page_number for p in chart.pages if p.page_number not in set(pages)]
    version = model.settings.version
    if len(pages) < 2:
        return OrderResult(chart.name, sorted(pages), junk, version)
    rows = chart_pairs(chart, pages, model.settings, seam=seam)
    g = graph(rows, model.evidence.predict(rows.X))
    solver = model.settings.solver
    options = decode_options(model.settings)
    order0 = decode_pages(g, 0.0, solver, options)
    p = model.gate_prob(gate_features(rows, g, model.settings, order0))
    margin = model.margin_for(p)
    order = order0 if margin == 0 else decode_pages(g, margin, solver, options)
    return OrderResult(chart.name, order, junk, f"{version}:gate{p:.2f}:m{margin:g}")
