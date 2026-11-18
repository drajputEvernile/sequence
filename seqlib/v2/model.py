"""The v2 pair scorer: P(page B directly follows page A).

A gradient-boosted tree model (scikit-learn ``HistGradientBoostingClassifier``: no extra
dependency, handles missing values, trains in seconds on a CPU) over ``features.FEATURES``.
Trained on the ground truth's true neighbour pairs against every other candidate pair the
candidate step proposes, so training sees exactly the pairs inference will score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from seqlib.io.chart import Chart
from seqlib.io.ground_truth import ChartTruth

from .features import FEATURES, START, candidates, chart_features, pair_matrix

logger = logging.getLogger(__name__)


@dataclass
class V2Settings:
    window: int = 8  # scan neighbours each side that are always candidates
    top_k: int = 5  # most similar tail/head matches that are candidates
    max_block: int = 40  # largest block the decoder moves at once
    params: dict = field(default_factory=lambda: {
        "max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0,
        "class_weight": "balanced", "random_state": 0,
    })


def true_pairs(order: list[int]) -> set[tuple[int, int]]:
    full = [START, *order, START]
    return set(zip(full, full[1:]))


def chart_training_rows(chart: Chart, truth: ChartTruth, settings: V2Settings, embedder=None):
    """(X, y, candidate recall) for one chart; pages = the truth's Main pages."""
    order = [p for p in truth.main_order if p in chart.by_number()]
    if len(order) < 2:
        return None
    cf = chart_features(chart, order, embedder)
    pairs = candidates(cf, settings.window, settings.top_k)
    positives = true_pairs(order)
    found = positives & set(pairs)
    pairs = sorted(set(pairs) | positives)  # the model must see every true pair
    X = pair_matrix(cf, pairs)
    y = np.array([p in positives for p in pairs], dtype=np.int8)
    return X, y, (len(found), len(positives))


@dataclass
class PairScorer:
    model: object
    settings: V2Settings
    features: tuple[str, ...] = FEATURES
    trained_on: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    # Columns that had any value in training (an all-missing column -- the embedding one with
    # embeddings off, a CSV this machine lacks -- is left out; scikit-learn cannot bin it).
    active: list[int] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, self.active])[:, 1]

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PairScorer":
        import joblib

        scorer = joblib.load(Path(path))
        if tuple(scorer.features) != FEATURES:
            raise ValueError(f"{path} was trained on other features; retrain with scripts/train_v2.py")
        return scorer


def train(examples: Iterable[tuple[Chart, ChartTruth]], settings: Optional[V2Settings] = None,
          embedder=None) -> PairScorer:
    from sklearn.ensemble import HistGradientBoostingClassifier

    settings = settings or V2Settings()
    xs, ys, names = [], [], []
    found = total = 0
    for chart, truth in examples:
        rows = chart_training_rows(chart, truth, settings, embedder)
        if rows is None:
            continue
        X, y, (f, t) = rows
        xs.append(X)
        ys.append(y)
        names.append(chart.name)
        found += f
        total += t
    if not xs:
        raise ValueError("no training chart had two or more labelled Main pages")
    X, y = np.vstack(xs), np.concatenate(ys)
    active = [k for k in range(X.shape[1]) if np.isfinite(X[:, k]).any() and np.nanmin(X[:, k]) != np.nanmax(X[:, k])]
    model = HistGradientBoostingClassifier(**settings.params)
    model.fit(X[:, active], y)
    report = {
        "charts": len(names),
        "pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        # share of true neighbour pairs the candidate step proposes -- the ceiling on successor
        # accuracy; raise window / top_k when it is low
        "candidate_recall": round(found / total, 4) if total else None,
        "unused_features": [FEATURES[k] for k in range(len(FEATURES)) if k not in active],
    }
    logger.info("v2 trained: %s", report)
    return PairScorer(model=model, settings=settings, trained_on=names, report=report, active=active)
