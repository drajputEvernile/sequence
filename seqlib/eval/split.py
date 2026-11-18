"""Train / test split by chart, stratified so both halves see the same mix.

Strata: whether the truth is plain scan order (most charts may need no reordering, and a test
set without the hard ones would flatter every method) x chart size (small / medium / large).
The split is written once (config.SPLIT_FILE) and read by every later run.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from seqlib.io.ground_truth import ChartTruth


def _stratum(truth: ChartTruth) -> str:
    order = truth.main_order
    size = "s" if len(order) <= 10 else "m" if len(order) <= 40 else "l"
    return f"{'scan' if order == sorted(order) else 'moved'}-{size}"


def make_split(truths: dict[str, ChartTruth], test_fraction: float, seed: int) -> dict:
    usable = {n: t for n, t in truths.items() if not t.problems()}
    groups: dict[str, list[str]] = defaultdict(list)
    for name, truth in sorted(usable.items()):
        groups[_stratum(truth)].append(name)
    rng = random.Random(seed)
    train, test = [], []
    for key in sorted(groups):
        names = groups[key]
        rng.shuffle(names)
        k = round(len(names) * test_fraction)
        if len(names) >= 2:
            k = max(1, k)
        test += names[:k]
        train += names[k:]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "test_fraction": test_fraction,
        "strata": {k: len(v) for k, v in sorted(groups.items())},
        "excluded": {n: t.problems() for n, t in truths.items() if t.problems()},
        "train": sorted(train),
        "test": sorted(test),
    }


def read_split(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no split at {path}: run scripts/make_split.py first")
    return json.loads(path.read_text(encoding="utf-8"))
