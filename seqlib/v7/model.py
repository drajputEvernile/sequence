"""v7 / v8: training (cross-fitted evidence, the gate, the two margins) and the model."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from seqlib.io.chart import Chart
from seqlib.io.ground_truth import ChartTruth
from seqlib.v2.decode import cost_matrix, path_cost
from seqlib.v3.model import ChartPairs, EvidenceModel, V3Settings, _fit, chart_pairs, decode_options
from seqlib.v3.tuning import choose_margin, choose_policy, cross_fit, decode_pages, graph, margin_table

logger = logging.getLogger(__name__)

MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GATE_FEATURES = (
    "log_pages",
    "anchors",                # printed "n of m" / fax-counter links found
    "anchors_broken",         # ... that the scan order does not follow
    "anchors_broken_share",
    "scan_cost_per_link",     # mean -log P(evidence) along the scan order
    "gain_per_link",          # how much cheaper the best order (no margin) is than the scan order
    "moved_share",            # share of links the best order (no margin) changes
    "weak_scan_share",        # scan links the evidence gives P < 0.1
    "min_scan_p",
    "strong_alternatives",    # pages whose scan successor is weak (< 0.2) and another is strong (> 0.8)
    "strong_alternative_share",
    "date_back_jumps",        # scan neighbours whose own dates go back more than 30 days
    "banner_breaks",          # scan neighbours in one fax transmission whose counter does not step by 1
    "v1_jumps",               # links v1's rules make against the scan order (v8)
    "marker_breaks",          # scan neighbours printing numbers in one style that do not step by 1 (v9)
)


def gate_features(r: ChartPairs, g, settings: V3Settings, order0: Sequence[int]) -> np.ndarray:
    """Document-level signs that the scan order is wrong (GATE_FEATURES). ``order0``: the order
    decoded with no margin. A sign the version has no evidence for is NaN."""
    pages, edge, anchors = g
    n = len(pages)
    node = {p: k + 1 for k, p in enumerate(pages)}
    links = n + 1
    nxt = lambda k: (k + 1) % links  # noqa: E731 - the scan successor of node k (n -> START)
    cost = cost_matrix(n, edge)
    scan_p = np.array([edge.get((k, nxt(k)), 1e-6) for k in range(links)])
    scan_cost = path_cost(list(range(1, n + 1)), cost)
    dec = [node[p] for p in order0]
    best_cost = path_cost(dec, cost)
    succ0 = dict(zip([0, *dec], [*dec, 0]))
    broken = sum(1 for a, b in anchors if b != nxt(a))
    best_alt: dict[int, float] = defaultdict(float)
    for (a, b), p in edge.items():
        if b != nxt(a):
            best_alt[a] = max(best_alt[a], p)
    strong = sum(1 for k in range(links) if best_alt.get(k, 0.0) > 0.8 and scan_p[k] < 0.2)

    names = settings.feature_names
    row_of = {pair: i for i, pair in enumerate(r.pairs)}
    scan_rows = [row_of[(a, b)] for a, b in zip(pages, pages[1:]) if (a, b) in row_of]

    def column(name):
        return r.X[scan_rows, names.index(name)] if name in names and scan_rows else None

    days = column("dt_days")
    date_back = float(np.sum(np.nan_to_num(days, nan=0.0) < -30)) if days is not None else np.nan
    same_tx, step = column("bn_same_tx"), column("bn_step")
    banner = (float(np.sum((np.nan_to_num(same_tx, nan=0.0) == 1) & (np.nan_to_num(step, nan=1.0) != 1)))
              if same_tx is not None else np.nan)
    v1 = float(len(r.extra_candidates)) if "v1" in settings.groups else np.nan
    marks = getattr(r, "markers", None) or {}
    marker_breaks = (float(sum(1 for a, b in zip(pages, pages[1:]) if a in marks and b in marks
                               and marks[a].style == marks[b].style and marks[b].number != marks[a].number + 1))
                     if "markers" in settings.groups else np.nan)
    return np.array([
        np.log1p(n), len(anchors), broken, broken / len(anchors) if anchors else 0.0,
        scan_cost / links, (scan_cost - best_cost) / links,
        sum(succ0[k] != nxt(k) for k in range(links)) / links,
        float(np.mean(scan_p < 0.1)), float(scan_p.min()), strong, strong / links,
        date_back, banner, v1, marker_breaks,
    ], dtype=np.float64)


class GateModel:
    """A small gradient-boosted classifier on the gate signs that vary in training."""

    def __init__(self):
        self.active: list[int] = []
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GateModel":
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.active = [k for k in range(X.shape[1])
                       if np.isfinite(X[:, k]).any() and np.nanmin(X[:, k]) != np.nanmax(X[:, k])]
        if not self.active:
            self.active = [0]
        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                                    min_samples_leaf=10, l2_regularization=1.0,
                                                    random_state=0).fit(X[:, self.active], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, self.active])[:, 1]


def gate_out_of_fold(X: np.ndarray, y: np.ndarray, folds: int = 5) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    out = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=int(min(folds, y.sum(), (~y).sum())), shuffle=True, random_state=0)
    for fit, part in cv.split(X, y):
        out[part] = GateModel().fit(X[fit], y[fit]).predict(X[part])
    return out


@dataclass
class V7Model:
    evidence: EvidenceModel
    gate: Optional[GateModel]  # None: every document gets margin_high
    threshold: float
    margin_low: float
    margin_high: float
    settings: V3Settings
    features: tuple[str, ...] = ()
    report: dict = field(default_factory=dict)

    def gate_prob(self, x: np.ndarray) -> float:
        if self.gate is None:
            return 0.0
        return float(self.gate.predict(x.reshape(1, -1))[0])

    def margin_for(self, p: float) -> float:
        return self.margin_low if p >= self.threshold else self.margin_high

    def save(self, path: Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "V7Model":
        import joblib

        model = joblib.load(Path(path))
        if tuple(model.features) != model.settings.feature_names:
            raise ValueError(f"{path} was trained on other features; retrain it")
        return model


def train_v7(examples: Iterable[tuple[Chart, ChartTruth]], settings: V3Settings, seam=None,
             fit_exclude: Optional[set[str]] = None, folds: int = 5, margins: Sequence[float] = MARGINS) -> V7Model:
    """``fit_exclude``: documents the evidence model must not learn from (the seam model's half);
    they are left out of tuning too (the seam has seen them)."""
    margins = tuple(sorted(set(margins) | {0.0}))
    fit_exclude = fit_exclude or set()
    from seqlib.v3.model import true_pairs
    from seqlib.v3.tuning import prediction_view

    rows: list[ChartPairs] = []
    truths: dict[str, list[int]] = {}
    recall = [0, 0]
    anchor_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # source -> [anchors, true]
    for chart, truth in examples:
        order = [p for p in truth.main_order if p in chart.by_number()]
        if len(order) < 2:
            continue
        r = chart_pairs(chart, order, settings, truth_order=order, seam=seam)
        pos = true_pairs(order)
        recall[0] += len(pos & set(prediction_view(r, settings).pairs))
        recall[1] += len(pos)
        for a, b in r.anchors:
            ma, mb = r.markers.get(a), r.markers.get(b)
            source = (f"marker:{ma.kind}" if ma is not None and mb is not None and ma.style == mb.style
                      and mb.number == ma.number + 1 else "n of m / banner")
            anchor_stats[source][0] += 1
            anchor_stats[source][1] += (a, b) in pos
        rows.append(r)
        truths[chart.name] = order
    usable = [r for r in rows if r.chart not in fit_exclude]
    if len(usable) < 2:
        raise ValueError(f"{settings.version} needs two or more training documents with two or more Main pages")
    logger.info("%s: %d documents (%d to fit and tune), %d folds", settings.version, len(rows), len(usable), folds)

    # 1. out-of-fold evidence, every document decoded at every margin, its gate signs
    oof = cross_fit(usable, settings, folds)
    objective = getattr(settings, "objective", "successor")
    names, succ, orders, graphs = margin_table(oof, truths, margins, settings.solver, objective,
                                               decode_options(settings))
    feats = np.array([gate_features(oof[n][0], graphs[n], settings, orders[n][0.0]) for n in names])
    moved = np.array([truths[n] != sorted(truths[n]) for n in names])
    in_order = ~moved

    # 2. the gate: out-of-fold probabilities to tune on, then the final gate on all of them
    gate, gate_p, auc = None, np.zeros(len(names)), None
    if min(moved.sum(), in_order.sum()) >= 3:
        from sklearn.metrics import roc_auc_score

        gate_p = gate_out_of_fold(feats, moved)
        auc = round(float(roc_auc_score(moved, gate_p)), 4)
        gate = GateModel().fit(feats, moved)

    # 3. the policy, and for comparison the best single margin (no gate) on the same documents
    policy = choose_policy(succ, gate_p, in_order, margins, settings.harm_floor)
    single, curve = choose_margin(succ, in_order, margins, settings.harm_floor)
    single_point = next(p for p in curve if p["margin"] == single)

    final = _fit(usable, settings.params)
    report = {
        "version": settings.version, "evidence": list(settings.groups), "anchors": settings.anchors,
        "documents": len(rows), "tuned_on_documents": len(names),
        "out_of_order_documents": int(moved.sum()), "in_order_documents": int(in_order.sum()),
        "folds": folds, "solver": settings.solver, "harm_floor": settings.harm_floor,
        "tuned_on": objective, "decode": decode_options(settings),
        "candidate_recall": round(recall[0] / recall[1], 4) if recall[1] else None,
        "anchors_found": sum(v[0] for v in anchor_stats.values()),
        "anchors_correct": (round(sum(v[1] for v in anchor_stats.values()) / sum(v[0] for v in anchor_stats.values()), 4)
                            if anchor_stats else None),
        "anchors_by_source": {k: {"anchors": v[0], "correct": round(v[1] / v[0], 4)} for k, v in sorted(anchor_stats.items())},
        "gate_auc_out_of_fold": auc,
        "gate_mean_p": {"out_of_order": round(float(gate_p[moved].mean()), 4) if moved.any() else None,
                        "in_order": round(float(gate_p[in_order].mean()), 4) if in_order.any() else None},
        "gate_signs_used": [GATE_FEATURES[k] for k in gate.active] if gate is not None else [],
        "chosen": policy,
        "without_gate": {"margin": single, "repair": single_point["held_out_shuffled_successor"],
                         "in_order": single_point["held_out_in_order_successor"]},
        "margin_curve_out_of_fold": curve,
        "unused_evidence": [settings.feature_names[k] for k in range(len(settings.feature_names))
                            if k not in final.active],
    }
    if gate is None:
        policy = {**policy, "threshold": 1.01}
    logger.info("%s trained: threshold %s, margins %s / %s", settings.version, policy["threshold"],
                policy["margin_low"], policy["margin_high"])
    return V7Model(final, gate, policy["threshold"], policy["margin_low"], policy["margin_high"], settings,
                   features=settings.feature_names, report=report)
