"""Loads config.py (the repo root's, or the file SEQ_CONFIG names) as ``cfg``."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    path = Path(os.environ.get("SEQ_CONFIG") or REPO_ROOT / "config.py")
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path} (set SEQ_CONFIG or create config.py)")
    spec = importlib.util.spec_from_file_location("seq_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.__file__ = str(path)
    return module


cfg = _load()


def input_path(chart_dir: Path, key: str) -> Path:
    """The file ``key`` (config.INPUT_FILES) for the chart in ``chart_dir``."""
    return Path(chart_dir) / cfg.INPUT_FILES[key].format(chart=Path(chart_dir).name)
