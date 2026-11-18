"""Which pages are set aside (the J series): a model trained on the truth's own labels.

Why it matters more than ordering: on the labelled documents the scan order is already right for
~98% of them, but the pipeline's blank / junk / duplicate verdict disagrees with the truth on many
pages, and every disagreement shifts the review-ui number of every later page in its document.

Inputs per page, all produced by the pipeline (no truth):

* the pipeline's own verdict (``blank_junk_flag``, ``is_duplicate``, ``duplicate_of_page_id``) and
  its confidences (``confidences`` JSON: blank_junk, page_type, member, dos ...);
* page type (``model_type`` L1 / L2, document ``page_type``, ``page_type_source``, codability,
  encounter type), quality / handwriting, signature, whether the document has a real DOS;
* where the page sits: first / last page, its place in the chart, its continuity document's
  first / last page, the neighbours' verdicts;
* the OCR text: length, and an out-of-fold text model's probability (TF-IDF over the page's
  first and last words -> logistic regression).

A gradient-boosted tree model (scikit-learn) combines them. Trained on the train split; folds
are grouped by document so no document is ever scored by a model that saw it.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from seqlib.io import pipeline_csv
from seqlib.io.chart import Chart
from seqlib.io.ground_truth import ChartTruth

logger = logging.getLogger(__name__)

CONFIDENCE_KEYS = ("blank_junk", "page_type", "member", "dos", "provider", "codability", "handwriting",
                   "encounter_type", "specialty")
NUMERIC = (
    "pipe_set_aside", "pipe_blank", "pipe_junk", "pipe_duplicate", "is_duplicate", "has_duplicate_of",
    *[f"conf_{k}" for k in CONFIDENCE_KEYS],
    "handwritten_pct", "signature_present", "needs_review", "has_dos", "non_visit_dos",
    "words", "lines", "chars", "digits_share", "upper_share",
    "rel_pos", "is_first", "is_last", "n_pages",
    "doc_first", "doc_last", "doc_size", "has_doc",
    "prev_set_aside", "next_set_aside", "prev_words", "next_words",
    "text_prob",
)
CATEGORICAL = (
    "blank_junk_flag", "model_l1", "model_l2", "page_type", "page_subtype", "page_type_source",
    "codability", "encounter_type", "quality_tag", "printed_or_handwritten", "dos_source", "continuation_rule",
)
HEAD_WORDS, TAIL_WORDS = 250, 60


def _view(text: str) -> str:
    words = (text or "").split()
    if len(words) > HEAD_WORDS + TAIL_WORDS:
        words = words[:HEAD_WORDS] + words[-TAIL_WORDS:]
    return " ".join(words)


def _num(value) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan


def _confidences(row: dict) -> dict:
    try:
        data = json.loads(row.get("confidences") or "{}")
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


@dataclass
class PageRows:
    """One chart's pages as feature rows (text kept apart for the text model)."""

    chart: str
    pages: list[int]
    numeric: list[list[float]]
    categorical: list[list[str]]
    texts: list[str]


def chart_rows(chart: Chart) -> PageRows:
    pages = chart.pages
    n = len(pages)
    docs: dict = {}
    for i, page in enumerate(pages):
        d = page.pipeline.get("document_seq") if page.pipeline else None
        if d not in (None, ""):
            docs.setdefault(str(d), []).append(i)
    pipe_aside = [pipeline_csv.group(p.pipeline) != "main" if p.pipeline else False for p in pages]
    words = [len(p.text.split()) for p in pages]
    numeric, categorical, texts = [], [], []
    for i, page in enumerate(pages):
        row = page.pipeline or {}
        group = pipeline_csv.group(row)
        conf = _confidences(row)
        text = page.text or ""
        letters = [c for c in text if c.isalpha()]
        doc_pages = docs.get(str(row.get("document_seq") or ""), [])
        path = pipeline_csv.type_path(row)
        dos_raw = str(row.get("dos_from") or "")
        values = {
            "pipe_set_aside": float(group != "main"), "pipe_blank": float(group == "blank"),
            "pipe_junk": float(group == "junk"), "pipe_duplicate": float(group == "duplicate"),
            "is_duplicate": float(pipeline_csv.flag(row.get("is_duplicate"))),
            "has_duplicate_of": float(bool(str(row.get("duplicate_of_page_id") or "").strip())),
            **{f"conf_{k}": _num(conf.get(k)) for k in CONFIDENCE_KEYS},
            "handwritten_pct": _num(row.get("handwritten_area_pct")),
            "signature_present": float(pipeline_csv.flag(row.get("signature_present"))),
            "needs_review": float(pipeline_csv.flag(row.get("needs_review"))),
            "has_dos": float(bool(pipeline_csv.iso_date(dos_raw))),
            "non_visit_dos": float(dos_raw.endswith("2222")),
            "words": float(words[i]), "lines": float(len([l for l in text.splitlines() if l.strip()])),
            "chars": float(len(text)),
            "digits_share": sum(c.isdigit() for c in text) / max(1, len(text)),
            "upper_share": sum(c.isupper() for c in letters) / max(1, len(letters)),
            "rel_pos": i / max(1, n - 1), "is_first": float(i == 0), "is_last": float(i == n - 1),
            "n_pages": float(n),
            "doc_first": float(bool(doc_pages) and doc_pages[0] == i),
            "doc_last": float(bool(doc_pages) and doc_pages[-1] == i),
            "doc_size": float(len(doc_pages)), "has_doc": float(bool(doc_pages)),
            "prev_set_aside": float(pipe_aside[i - 1]) if i > 0 else math.nan,
            "next_set_aside": float(pipe_aside[i + 1]) if i + 1 < n else math.nan,
            "prev_words": float(words[i - 1]) if i > 0 else math.nan,
            "next_words": float(words[i + 1]) if i + 1 < n else math.nan,
            "text_prob": math.nan,  # filled by the text model
        }
        numeric.append([values[k] for k in NUMERIC])
        cats = {
            "blank_junk_flag": str(row.get("blank_junk_flag") or ""),
            "model_l1": path[0] if path else "", "model_l2": path[1] if len(path) > 1 else "",
            "page_type": str(row.get("page_type") or ""), "page_subtype": str(row.get("page_subtype") or ""),
            "page_type_source": str(row.get("page_type_source") or ""),
            "codability": str(row.get("codability") or ""), "encounter_type": str(row.get("encounter_type") or ""),
            "quality_tag": str(row.get("quality_tag") or ""),
            "printed_or_handwritten": str(row.get("printed_or_handwritten") or ""),
            "dos_source": str(row.get("dos_source") or ""), "continuation_rule": str(row.get("continuation_rule") or ""),
        }
        categorical.append([cats[k] for k in CATEGORICAL])
        texts.append(_view(text))
    return PageRows(chart.name, [p.page_number for p in pages], numeric, categorical, texts)


def labels_for(rows: PageRows, truth: ChartTruth) -> tuple[list[int], list[int]]:
    """(indexes of the rows the truth labels, 1 = set aside / 0 = Main)."""
    junk = set(truth.junk_pages)
    keep, y = [], []
    for k, page in enumerate(rows.pages):
        value = truth.labels.get(page)
        if not value:
            continue
        keep.append(k)
        y.append(int(page in junk))
    return keep, y


def _text_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    return Pipeline([
        ("tfidf", FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=80000,
                                  sublinear_tf=True)),
        ])),
        ("lr", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")),
    ])


class _GBM:
    """HistGradientBoostingClassifier on the columns that carry information in the rows it is fitted
    on: a column with no value, or one value, is dropped (scikit-learn cannot bin an all-missing
    column, and a rare input can be absent from a whole fold)."""

    def __init__(self, categorical_mask: Sequence[bool]):
        self.categorical_mask = list(categorical_mask)
        self.active: list[int] = []
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_GBM":
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.active = [k for k in range(X.shape[1])
                       if np.isfinite(X[:, k]).any() and np.nanmin(X[:, k]) != np.nanmax(X[:, k])]
        mask = [self.categorical_mask[k] for k in self.active]
        self.model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            categorical_features=mask if any(mask) else None, random_state=0)
        self.model.fit(X[:, self.active], y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, self.active])


@dataclass
class SetAsideModel:
    text_model: object
    encoder: object
    model: object
    threshold: float = 0.5
    report: dict = field(default_factory=dict)

    def _matrix(self, numeric: np.ndarray, categorical: list[list[str]], text_prob: np.ndarray) -> np.ndarray:
        numeric = numeric.copy()
        numeric[:, NUMERIC.index("text_prob")] = text_prob
        cats = self.encoder.transform(np.asarray(categorical, dtype=object)) if categorical else \
            np.zeros((0, len(CATEGORICAL)))
        return np.hstack([numeric, cats])

    def proba(self, rows: PageRows) -> np.ndarray:
        if not rows.pages:
            return np.zeros(0)
        text_prob = self.text_model.predict_proba(rows.texts)[:, 1]
        X = self._matrix(np.asarray(rows.numeric, dtype=float), rows.categorical, text_prob)
        return self.model.predict_proba(X)[:, 1]

    def set_aside(self, chart: Chart) -> dict[int, bool]:
        rows = chart_rows(chart)
        return {p: bool(v >= self.threshold) for p, v in zip(rows.pages, self.proba(rows))}

    def apply(self, chart: Chart) -> Chart:
        """Sets each page's group_override from the model (keeps the pipeline's kind of set-aside
        page when it agrees: blank / junk / duplicate; else "junk")."""
        decided = self.set_aside(chart)
        for page in chart.pages:
            if decided.get(page.page_number):
                own = pipeline_csv.group(page.pipeline) if page.pipeline else "junk"
                page.group_override = own if own != "main" else "junk"
            else:
                page.group_override = "main"
        return chart

    def save(self, path: Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "SetAsideModel":
        import joblib

        return joblib.load(path)


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for t in np.arange(0.1, 0.91, 0.02):
        best = max(best, (float(((p >= t) == (y == 1)).mean()), round(float(t), 2)))
    return best[1]


def train(examples: Iterable[tuple[Chart, ChartTruth]], folds: int = 5) -> SetAsideModel:
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import OrdinalEncoder

    numeric, categorical, texts, y, groups = [], [], [], [], []
    for chart, truth in examples:
        rows = chart_rows(chart)
        keep, labels = labels_for(rows, truth)
        for k, label in zip(keep, labels):
            numeric.append(rows.numeric[k])
            categorical.append(rows.categorical[k])
            texts.append(rows.texts[k])
            y.append(label)
            groups.append(chart.name)
    if not y:
        raise ValueError("no labelled page to train on")
    y_arr, num = np.asarray(y), np.asarray(numeric, dtype=float)
    n_groups = len(set(groups))
    k = max(2, min(folds, n_groups))

    # 1. the text model's out-of-fold probability (grouped by document)
    text_oof = np.zeros(len(y_arr))
    for tr, te in GroupKFold(n_splits=k).split(texts, y_arr, groups):
        if len(set(y_arr[tr])) < 2:
            text_oof[te] = y_arr[tr].mean()
            continue
        m = _text_model().fit([texts[i] for i in tr], y_arr[tr])
        text_oof[te] = m.predict_proba([texts[i] for i in te])[:, 1]
    text_model = _text_model().fit(texts, y_arr)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan, min_frequency=5, max_categories=200)
    encoder.fit(np.asarray(categorical, dtype=object))
    holder = SetAsideModel(text_model, encoder, None)
    X = holder._matrix(num, categorical, text_oof)
    cat_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)

    def gbm():
        return _GBM(cat_mask)

    # 2. out-of-fold probabilities of the whole model, to pick the threshold and report honestly
    oof = np.zeros(len(y_arr))
    for tr, te in GroupKFold(n_splits=k).split(X, y_arr, groups):
        oof[te] = gbm().fit(X[tr], y_arr[tr]).predict_proba(X[te])[:, 1]
    threshold = _best_threshold(y_arr, oof)
    model = gbm().fit(X, y_arr)

    pipe = num[:, NUMERIC.index("pipe_set_aside")] >= 0.5
    pred = oof >= threshold

    def stats(pred_mask):
        tp = int((pred_mask & (y_arr == 1)).sum())
        fp = int((pred_mask & (y_arr == 0)).sum())
        fn = int((~pred_mask & (y_arr == 1)).sum())
        return {"accuracy": round(float((pred_mask == (y_arr == 1)).mean()), 4),
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                "main_pages_set_aside": fp, "set_aside_pages_kept": fn}

    report = {"documents": n_groups, "pages": int(len(y_arr)), "set_aside_in_truth": int(y_arr.sum()),
              "threshold": threshold, "pipeline_verdict (train, for comparison)": stats(pipe),
              "model_out_of_fold (train)": stats(pred)}
    holder.model, holder.threshold, holder.report = model, threshold, report
    return holder


def importance(model: SetAsideModel, examples: Sequence[tuple[Chart, ChartTruth]], top: int = 15) -> list:
    """Permutation importance on the given documents (which features the model leans on)."""
    from sklearn.inspection import permutation_importance

    numeric, categorical, texts, y = [], [], [], []
    for chart, truth in examples:
        rows = chart_rows(chart)
        keep, labels = labels_for(rows, truth)
        numeric += [rows.numeric[k] for k in keep]
        categorical += [rows.categorical[k] for k in keep]
        texts += [rows.texts[k] for k in keep]
        y += labels
    if not y:
        return []
    X = model._matrix(np.asarray(numeric, dtype=float), categorical, model.text_model.predict_proba(texts)[:, 1])
    gbm = model.model
    result = permutation_importance(gbm.model, X[:, gbm.active], np.asarray(y), n_repeats=3, random_state=0,
                                    scoring="accuracy")
    names = [(list(NUMERIC) + list(CATEGORICAL))[k] for k in gbm.active]
    order = np.argsort(-result.importances_mean)[:top]
    return [(names[i], round(float(result.importances_mean[i]), 4)) for i in order]
