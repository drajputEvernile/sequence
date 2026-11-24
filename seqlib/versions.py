"""The sequencing versions kept in this repo, and where their trained models live.

Every version is the evidence engine (seqlib/v3: an evidence model that never sees where a page
was scanned, a move margin that keeps the scan order unless the evidence beats it, a solver):

    v5  page-number / fax-banner anchors + banner and own-date evidence; one tuned margin.
        The safe reference: the best score on real documents (98.95% position).
    v8  v5 + header image fingerprint + the seam text model + v1's rules as evidence; a gate
        picks the margin per document (P(out of order)); both solvers. The best repair so far
        on the shuffle test (83.9% successor, 61.8% position).
    v9  v8 + page numbers in every printed form (markers: "n of m" any total, "Page n", a bare
        number) as evidence and anchors + content clues (section order, numbered items, encounter
        id / date, "continued", tables carried over) + the chart's direction in time; reversed
        runs and swaps are cheap to undo; the margin and gate are tuned on *position* accuracy.

``trainer``: "single" (seqlib/v3/model.train, one margin) or "gated" (seqlib/v7/model.train_v7).
Saved models: config.LADDER_MODEL_DIR/<version>/model.joblib.
"""
from __future__ import annotations

from pathlib import Path

V8_GROUPS = ("banner", "dates", "hash", "seam", "v1")

VERSIONS: dict[str, dict] = {
    "v5": {"trainer": "single", "groups": ("banner", "dates"), "anchors": True},
    "v8": {"trainer": "gated", "groups": V8_GROUPS, "anchors": True, "solver": "best"},
    "v9": {"trainer": "gated", "groups": (*V8_GROUPS, "markers", "content", "pattern"), "anchors": True,
           "solver": "best", "reverse_share": 0.25, "swaps": True, "objective": "position", "harm_floor": 0.995},
}


def settings_for(version: str, **overrides):
    from seqlib.v3.model import V3Settings

    if version not in VERSIONS:
        raise ValueError(f"unknown version {version}: one of {sorted(VERSIONS)}")
    spec = {k: v for k, v in VERSIONS[version].items() if k != "trainer"}
    return V3Settings(version=version, **{**spec, **overrides})


def is_gated(version: str) -> bool:
    return VERSIONS[version]["trainer"] == "gated"


def model_path(version: str) -> Path:
    from seqlib.settings import cfg

    return Path(getattr(cfg, "LADDER_MODEL_DIR", Path(cfg.REPO) / "models")) / version / "model.joblib"


def load_model(version: str):
    if is_gated(version):
        from seqlib.v7.model import V7Model

        return V7Model.load(model_path(version))
    from seqlib.v3.model import V3Model

    return V3Model.load(model_path(version))
