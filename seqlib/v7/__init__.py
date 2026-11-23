"""v7 and v8: the gated versions (docs/EXPERIMENTS.md).

What v3 to v6 showed: the evidence repairs more with a low move margin (v6 at 1.5) but then also
breaks documents that were in order; the tuned margin (3.0) protects them and repairs less. v7
decides per document *first* whether it looks out of order, then uses the margin that suits:

* a **gate** -- P(this document's scan order is wrong), from document-level signs read on its
  scan order: printed page numbers / fax counters the scan order breaks, scan links the evidence
  finds weak, how much cheaper the best order is than the scan order, dates jumping back;
* two margins -- a low one for documents the gate flags, a high one for the rest.

Versions:

    v7   v5's evidence (page-number / banner anchors, banner and own-date features) + the gate
    v7s  v6's evidence (+ header fingerprint, seam cross-encoder) + the gate
    v8   everything: v6's evidence + v1's rules as evidence (its departures from scan order, as a
         pair feature, a candidate source and a gate sign) + the gate + both solvers (search and
         cover; the cheaper order wins)

Everything is tuned on out-of-fold scores of the training documents (cross-fitting,
seqlib/v3/tuning.py), so the gate, its threshold and both margins are chosen on documents the
evidence model did not learn from.
"""
from __future__ import annotations

VERSIONS: dict[str, dict] = {
    "v7": {"groups": ("banner", "dates"), "anchors": True},
    "v7s": {"groups": ("banner", "dates", "hash", "seam"), "anchors": True},
    "v8": {"groups": ("banner", "dates", "hash", "seam", "v1"), "anchors": True, "solver": "best"},
}


def settings_for(version: str, **overrides):
    from seqlib.v3.model import V3Settings

    if version not in VERSIONS:
        raise ValueError(f"unknown version {version}: one of {sorted(VERSIONS)}")
    return V3Settings(version=version, **{**VERSIONS[version], **overrides})


def model_path(version: str):
    from pathlib import Path

    from seqlib.settings import cfg

    return Path(getattr(cfg, "LADDER_MODEL_DIR", Path(cfg.REPO) / "models")) / version / "model.joblib"
