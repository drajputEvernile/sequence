"""v3 and the versions built on it: evidence and scan order kept apart (design doc RC1, FR6, 7.5.1).

v2 learned a shortcut: with the scan position among its inputs, "the scan neighbour is the next
page" explained almost every training pair, so the evidence (page numbers, text continuing, page
facts) was hardly used and nothing ever moved (diagnose_v2: the true next page is the model's
first choice for 6% of moved pages).

v3 splits the two:

* **Evidence model** -- P(b directly follows a | the pages' content), trained *without* any scan
  input (scan distance, scan neighbour, relative position). It cannot lean on where a page was
  scanned, so it has to learn what the pages say.
* **Scan prior** -- a single number, the *move margin*: the cost added to any link that is not
  the scan order's (a -> its scan neighbour, START -> the first page, the last page -> END). A page
  leaves its scan neighbour only when the evidence beats the margin. The margin is tuned on
  held-out training documents: shuffled ones measure repair, the ones left in order measure harm.

The later versions are v3 with more evidence (seqlib/ladder): ``groups`` add pair features
(banner, dates, hash, seam) and ``anchors`` force unambiguous page-number / banner links.

Probabilities are not class-weighted (each page's candidates weigh the same in total), so
-log P is a fair cost to add a margin to.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from seqlib.io.chart import Chart
from seqlib.io.ground_truth import ChartTruth
from seqlib.ladder.extra import extra_matrix, feature_names, find_anchors
from seqlib.v2.features import FEATURES, START, candidates, chart_features, pair_matrix
from seqlib.v2.model import true_pairs

logger = logging.getLogger(__name__)

# Inputs that say where a page was scanned: left out of the evidence model.
SCAN_FEATURES = frozenset({"scan_delta", "abs_scan_delta", "is_scan_next", "is_scan_prev", "a_rel_pos", "b_rel_pos"})
EVIDENCE = tuple(f for f in FEATURES if f not in SCAN_FEATURES)
_EVIDENCE_IDX = [FEATURES.index(f) for f in EVIDENCE]


@dataclass
class V3Settings:
    version: str = "v3"
    groups: tuple[str, ...] = ()  # extra evidence (seqlib/ladder/extra.py)
    anchors: bool = False  # force unambiguous page-number / banner links
    banner_anchors: bool = True
    window: int = 8  # scan neighbours each side that are candidates (a candidate source only)
    top_k: int = 10  # most similar text continuations that are candidates
    margins: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    # tuning rule: the margin with the best repair on held-out shuffled documents among those that
    # keep held-out in-order documents at least this right (successor accuracy)
    harm_floor: float = 0.998
    validation_share: float = 0.2  # no longer used (tuning is cross-fitted); kept for saved models
    folds: int = 5  # cross-fitting folds for tuning (seqlib/v3/tuning.py)
    solver: str = "search"  # "search" (exact <= 11 pages, else block moves) or "cover"
    params: dict = field(default_factory=lambda: {
        "max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0, "random_state": 0,
    })

    @property
    def feature_names(self) -> tuple[str, ...]:
        return EVIDENCE + feature_names(self.groups)


def _evidence(X: np.ndarray) -> np.ndarray:
    return X[:, _EVIDENCE_IDX]


@dataclass
class ChartPairs:
    chart: str
    cf: object
    pairs: list[tuple[int, int]]
    X: np.ndarray  # evidence columns (+ the version's extra groups)
    y: np.ndarray
    anchors: list[tuple[int, int]] = field(default_factory=list)  # page numbers
    extra_candidates: list[tuple[int, int]] = field(default_factory=list)  # v1's jumps (v8)


def chart_pairs(chart: Chart, pages: Sequence[int], settings: V3Settings, truth_order: Optional[list[int]] = None,
                embedder=None, seam=None) -> ChartPairs:
    cf = chart_features(chart, pages, embedder)
    pairs = set(candidates(cf, settings.window, settings.top_k))
    facts = None
    anchors: list[tuple[int, int]] = []
    if settings.groups or settings.anchors:
        from seqlib.ladder.facts import chart_facts

        facts = chart_facts(chart, [g for g in settings.groups if g not in ("seam", "v1")]
                            + (["banner"] if settings.anchors else []))
        keep = {p.page_number for p in cf.pages}
        facts.dos = {k: v for k, v in facts.dos.items() if k in keep}
        facts.banner = {k: v for k, v in facts.banner.items() if k in keep}
        facts.header_hash = {k: v for k, v in facts.header_hash.items() if k in keep}
    if settings.anchors:
        printed = {p.page_number: p.printed for p in cf.pages if p.printed}
        anchors = find_anchors(printed, facts, settings.banner_anchors)
        pairs |= set(anchors)
    jumps: list[tuple[int, int]] = []
    if "v1" in settings.groups:
        from seqlib.ladder.extra import v1_jumps

        jumps = v1_jumps(chart, [p.page_number for p in cf.pages])
        pairs |= set(jumps)
    positives = true_pairs(truth_order) if truth_order else set()
    pairs = sorted(pairs | positives)  # training sees every true pair
    X = _evidence(pair_matrix(cf, pairs)) if pairs else np.zeros((0, len(EVIDENCE)))
    if settings.groups:
        seam_probs = None
        if "seam" in settings.groups and seam is not None:
            from seqlib.ladder.seam import seam_for_pairs

            seam_probs = seam_for_pairs(seam, cf, pairs)
        X = np.hstack([X, extra_matrix(pairs, facts, settings.groups, seam_probs, set(jumps))])
    y = np.array([p in positives for p in pairs], dtype=np.int8)
    return ChartPairs(chart.name, cf, pairs, X, y, anchors, jumps)


def per_page_weights(pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Each page's candidate links weigh 1 in total, whatever their number (design doc 8.2)."""
    from collections import Counter

    count = Counter(a for a, _ in pairs)
    return np.array([1.0 / count[a] for a, _ in pairs])


class EvidenceModel:
    """HistGradientBoosting on the evidence columns that carry information in training."""

    def __init__(self, params: dict):
        self.params = dict(params)
        self.active: list[int] = []
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "EvidenceModel":
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.active = [k for k in range(X.shape[1])
                       if np.isfinite(X[:, k]).any() and np.nanmin(X[:, k]) != np.nanmax(X[:, k])]
        self.model = HistGradientBoostingClassifier(**self.params).fit(X[:, self.active], y, sample_weight=w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.model.predict_proba(X[:, self.active])[:, 1]


@dataclass
class V3Model:
    evidence: EvidenceModel
    margin: float
    settings: V3Settings
    features: tuple[str, ...] = EVIDENCE
    report: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "V3Model":
        import joblib

        model = joblib.load(Path(path))
        settings = getattr(model, "settings", None)
        for name, default in (("version", "v3"), ("groups", ()), ("anchors", False), ("banner_anchors", True)):
            if settings is not None and not hasattr(settings, name):
                setattr(settings, name, default)  # a model saved before these settings existed
        if tuple(model.features) != model.settings.feature_names:
            raise ValueError(f"{path} was trained on other features; retrain it")
        return model


def _fit(rows: list[ChartPairs], params: dict) -> EvidenceModel:
    X = np.vstack([r.X for r in rows])
    y = np.concatenate([r.y for r in rows])
    w = np.concatenate([per_page_weights(r.pairs) for r in rows])
    return EvidenceModel(params).fit(X, y, w)


def order_with(rows: ChartPairs, probs: np.ndarray, margin: float, solver: str) -> list[int]:
    """The decoded order (page numbers) for one chart."""
    from .solve import decode_v3

    pages = [p.page_number for p in rows.cf.pages]
    node = {p: k + 1 for k, p in enumerate(pages)}
    node[START] = 0
    edge = {(node[a], node[b]): float(p) for (a, b), p in zip(rows.pairs, probs)}
    anchors = [(node[a], node[b]) for a, b in rows.anchors if a in node and b in node]
    order = decode_v3(len(pages), edge, margin, solver, anchors=anchors)
    return [pages[k - 1] for k in order]


def successor_accuracy(order: list[int], truth: list[int]) -> float:
    if len(truth) < 2:
        return 1.0
    nxt = {a: b for a, b in zip(order, order[1:])}
    return sum(nxt.get(a) == b for a, b in zip(truth, truth[1:])) / (len(truth) - 1)


def train(examples: Iterable[tuple[Chart, ChartTruth]], settings: Optional[V3Settings] = None,
          controls: Optional[set[str]] = None, seam=None, fit_exclude: Optional[set[str]] = None) -> V3Model:
    """``controls``: names of the training documents left in scan order (shuffle manifest), used to
    measure harm while tuning the margin; without them, documents whose truth is in scan order are.
    ``fit_exclude``: documents the evidence model must not fit on (the seam model's own half);
    they are left out of tuning too (the seam has seen them)."""
    settings = settings or V3Settings()
    rows: list[ChartPairs] = []
    truths: dict[str, list[int]] = {}
    recall = [0, 0]
    anchor_stats = [0, 0]  # anchors, anchors that are true links
    for chart, truth in examples:
        order = [p for p in truth.main_order if p in chart.by_number()]
        if len(order) < 2:
            continue
        r = chart_pairs(chart, order, settings, truth_order=order, seam=seam)
        cand = set(candidates(r.cf, settings.window, settings.top_k)) | set(r.anchors) | set(r.extra_candidates)
        pos = true_pairs(order)
        recall[0] += len(pos & cand)
        recall[1] += len(pos)
        anchor_stats[0] += len(r.anchors)
        anchor_stats[1] += len(set(r.anchors) & pos)
        rows.append(r)
        truths[r.chart] = order
    if not rows:
        raise ValueError("no training document with two or more Main pages")
    in_order = controls if controls is not None else {n for n, t in truths.items() if t == sorted(t)}
    fit_exclude = fit_exclude or set()

    # 1. tune the margin on out-of-fold scores of every document the evidence model may fit on
    #    (seqlib/v3/tuning.py); with the seam, its own half is left out (its seam scores are too good)
    from .tuning import choose_margin, cross_fit, margin_table

    usable = [r for r in rows if r.chart not in fit_exclude] or rows
    margins = tuple(sorted(settings.margins))
    oof = cross_fit(usable, settings, getattr(settings, "folds", 5))
    names, succ, _, _ = margin_table(oof, truths, margins, settings.solver)
    flags = np.array([n in in_order for n in names])
    margin, curve = choose_margin(succ, flags, margins, settings.harm_floor)

    # 2. the final evidence model on every training document it may fit on
    final = _fit(usable, settings.params)
    names_all = settings.feature_names
    report = {
        "version": settings.version, "groups": list(settings.groups), "anchors": settings.anchors,
        "documents": len(rows), "fitted_on_documents": len(usable),
        "pairs": int(sum(len(r.y) for r in rows)), "positive_pairs": int(sum(r.y.sum() for r in rows)),
        "candidate_recall": round(recall[0] / recall[1], 4) if recall[1] else None,
        "anchors_found": anchor_stats[0],
        "anchors_correct": round(anchor_stats[1] / anchor_stats[0], 4) if anchor_stats[0] else None,
        "tuned_on_documents": len(names), "tuned_in_order": int(flags.sum()), "folds": getattr(settings, "folds", 5),
        "margin_curve": curve, "chosen_margin": margin, "solver": settings.solver,
        "harm_floor": settings.harm_floor,
        "unused_evidence": [names_all[k] for k in range(len(names_all)) if k not in final.active],
    }
    logger.info("%s trained: margin %s", settings.version, margin)
    return V3Model(final, margin, settings, features=names_all, report=report)
