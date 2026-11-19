"""The experiment ladder: v3 plus one more kind of evidence per version (docs/EXPERIMENTS.md).

Every version is the v3 engine (evidence model with no scan inputs + the tuned move margin +
a solver) with these switches:

    v3  nothing more
    v4  + page-number / fax-banner anchors (forced links) and banner features
    v5  + each page's own date of service read from its OCR (date features)
    v6  + header image fingerprint and the seam cross-encoder (text continuation)
"""
from __future__ import annotations

VERSIONS: dict[str, dict] = {
    "v3": {"groups": (), "anchors": False},
    "v4": {"groups": ("banner",), "anchors": True},
    "v5": {"groups": ("banner", "dates"), "anchors": True},
    "v6": {"groups": ("banner", "dates", "hash", "seam"), "anchors": True},
}


def settings_for(version: str, **overrides):
    from seqlib.v3.model import V3Settings

    if version not in VERSIONS:
        raise ValueError(f"unknown version {version}: one of {sorted(VERSIONS)}")
    return V3Settings(version=version, **VERSIONS[version], **overrides)


def model_path(version: str):
    """Where a version's model lives (config.V3_MODEL_PATH for v3, LADDER_MODEL_DIR/<v>/ else)."""
    from pathlib import Path

    from seqlib.settings import cfg

    if version == "v3":
        return Path(cfg.V3_MODEL_PATH)
    return Path(getattr(cfg, "LADDER_MODEL_DIR", Path(cfg.REPO) / "models")) / version / "model.joblib"


def load_seam(required: bool = False):
    """The seam scorer from config.SEAM_MODEL_PATH, or None (logged) when it cannot load."""
    import logging

    from seqlib.settings import cfg

    path = getattr(cfg, "SEAM_MODEL_PATH", None)
    try:
        if path is None or not (path / "config.json").is_file():
            raise FileNotFoundError(f"no seam model at {path} (scripts/train_seam.py)")
        from .seam import SeamScorer

        return SeamScorer(path, getattr(cfg, "SEAM_DEVICE", "cpu"))
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        logging.getLogger(__name__).warning("seam model not used: %s", exc)
        return None
