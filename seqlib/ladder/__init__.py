"""Extra evidence for the engine (seqlib/v3): page facts, pair features and anchors, page
markers, content clues, the seam text model. Which version uses what: seqlib/versions.py."""
from __future__ import annotations


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
