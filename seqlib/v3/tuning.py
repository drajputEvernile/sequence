"""Tuning shared by v3 to v8: out-of-fold evidence scores, decoding at each margin, the policy.

* **Cross-fitting** -- the training documents are split into folds by name; each fold is scored
  by an evidence model fitted on the others. Every document is tuned on (not a 20% hold-out), and
  always by a model that did not learn from it.
* **The view at prediction time** -- training rows carry every true pair; tuning drops the ones
  the candidate step (and the anchors) would not have proposed, as for an unseen document.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import numpy as np

from seqlib.v2.features import START, candidates

logger = logging.getLogger(__name__)


def prediction_view(r, settings):
    """The chart's pairs as the model sees them when ordering: candidates, anchors and v1's
    jumps only (training rows also carry every true pair, which an unseen document would not)."""
    from .model import ChartPairs

    keep = set(candidates(r.cf, settings.window, settings.top_k)) | set(r.anchors) | set(r.extra_candidates)
    idx = [i for i, p in enumerate(r.pairs) if p in keep]
    return ChartPairs(r.chart, r.cf, [r.pairs[i] for i in idx], r.X[idx], r.y[idx], r.anchors, r.extra_candidates)


def graph(r, probs: np.ndarray):
    """(page numbers in scan order, node edge probabilities, node anchors)."""
    pages = [p.page_number for p in r.cf.pages]
    node = {p: k + 1 for k, p in enumerate(pages)}
    node[START] = 0
    edge = {(node[a], node[b]): float(p) for (a, b), p in zip(r.pairs, probs)}
    anchors = [(node[a], node[b]) for a, b in r.anchors if a in node and b in node]
    return pages, edge, anchors


def decode_pages(g, margin: float, solver: str) -> list[int]:
    from .solve import decode_v3

    pages, edge, anchors = g
    return [pages[k - 1] for k in decode_v3(len(pages), edge, margin, solver, anchors=anchors)]


def fold_of(name: str, folds: int) -> int:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % folds


def cross_fit(rows: list, settings, folds: int) -> dict:
    """chart name -> (its prediction-time pairs, out-of-fold probabilities)."""
    from .model import _fit

    out = {}
    folds = max(2, min(folds, len(rows)))
    for k in range(folds):
        fit = [r for r in rows if fold_of(r.chart, folds) != k]
        part = [r for r in rows if fold_of(r.chart, folds) == k]
        if not part or not fit:
            continue
        model = _fit(fit, settings.params)
        for r in part:
            view = prediction_view(r, settings)
            out[r.chart] = (view, model.predict(view.X))
        logger.info("%s: fold %d/%d scored (%d documents)", settings.version, k + 1, folds, len(part))
    return out


def margin_table(oof: dict, truths: dict, margins: Sequence[float], solver: str):
    """(names, documents x margins successor accuracy, {name: {margin: order}}, {name: graph})."""
    from .model import successor_accuracy

    names = sorted(oof)
    succ = np.zeros((len(names), len(margins)))
    orders, graphs = {}, {}
    for i, name in enumerate(names):
        view, probs = oof[name]
        g = graph(view, probs)
        orders[name] = {m: decode_pages(g, m, solver) for m in margins}
        graphs[name] = g
        succ[i] = [successor_accuracy(orders[name][m], truths[name]) for m in margins]
        if (i + 1) % 50 == 0:
            logger.info("%d/%d documents decoded at %d margins", i + 1, len(names), len(margins))
    return names, succ, orders, graphs


def choose_margin(succ: np.ndarray, in_order: np.ndarray, margins: Sequence[float], harm_floor: float):
    """One margin for every document (v3 to v6): the best repair among the margins that keep
    in-order documents at least ``harm_floor`` right; when none does, the least harm, then the
    most repair. Returns (margin, curve)."""
    moved = ~in_order
    curve = []
    for j, m in enumerate(margins):
        curve.append({"margin": m,
                      "held_out_shuffled_successor": round(float(succ[moved, j].mean()), 4) if moved.any() else None,
                      "held_out_in_order_successor": round(float(succ[in_order, j].mean()), 4) if in_order.any() else None,
                      "documents": [int(moved.sum()), int(in_order.sum())]})
    ok = [p for p in curve if p["held_out_in_order_successor"] is None or p["held_out_in_order_successor"] >= harm_floor]
    if ok:
        best = max(ok, key=lambda p: (p["held_out_shuffled_successor"] or 0.0, p["held_out_in_order_successor"] or 0.0,
                                      p["margin"]))
    else:
        best = max(curve, key=lambda p: (p["held_out_in_order_successor"] or 0.0, p["held_out_shuffled_successor"] or 0.0,
                                         p["margin"]))
    return best["margin"], curve


def choose_policy(succ: np.ndarray, gate_p: np.ndarray, in_order: np.ndarray, margins: Sequence[float],
                  harm_floor: float) -> dict:
    """The (threshold, low margin, high margin) with the best repair on out-of-order documents
    while in-order documents keep at least ``harm_floor`` successor accuracy; when no choice
    meets the floor, the least harm, then the most repair. ``succ``: documents x margins. With a
    gate of zeros this is a single margin (margin_high)."""
    thresholds = sorted(set(np.round(np.quantile(gate_p, np.linspace(0, 1, 41)), 4).tolist()) | {1.01})
    moved = ~in_order
    best, fallback = None, None
    for hi in range(len(margins)):
        for lo in range(hi + 1):
            for t in thresholds:
                pick = np.where(gate_p >= t, lo, hi)
                s = succ[np.arange(len(succ)), pick]
                harm = float(s[in_order].mean()) if in_order.any() else None
                repair = float(s[moved].mean()) if moved.any() else 0.0
                point = {"threshold": float(t), "margin_low": margins[lo], "margin_high": margins[hi],
                         "repair": round(repair, 4), "in_order": None if harm is None else round(harm, 4),
                         "flagged_out_of_order": round(float((gate_p[moved] >= t).mean()), 4) if moved.any() else None,
                         "flagged_in_order": round(float((gate_p[in_order] >= t).mean()), 4) if in_order.any() else None}
                key = (repair, harm or 0.0, margins[hi], -t)
                if harm is None or harm >= harm_floor:
                    if best is None or key > best[0]:
                        best = (key, point)
                fkey = (harm or 0.0, repair, margins[hi], -t)
                if fallback is None or fkey > fallback[0]:
                    fallback = (fkey, point)
    return {**(best or fallback)[1], "meets_floor": best is not None}
