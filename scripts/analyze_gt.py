"""What kind of reordering the ground truth needs (run this first; it decides the v2 plan).

    python scripts/analyze_gt.py              # every labelled chart
    python scripts/analyze_gt.py --split train

Writes <OUTPUT_ROOT>/analysis/gt_charts.csv (one row per chart) and gt_overview.json.
"""
import argparse
import csv
import json

import _common  # noqa: F401

from seqlib.eval.analyze import describe, overview
from seqlib.eval.split import read_split
from seqlib.io.chart import load_chart
from seqlib.io.ground_truth import load_ground_truth
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("all", "train", "test"), default="all")
    ap.add_argument("--no-charts", action="store_true", help="ground truth only (skip reading OCR)")
    args = ap.parse_args()
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = sorted(truths) if args.split == "all" else read_split(cfg.SPLIT_FILE)[args.split]
    rows = []
    for name in names:
        chart = None
        if not args.no_charts:
            try:
                chart = load_chart(name)
            except Exception:  # noqa: BLE001
                chart = None
        rows.append(describe(truths[name], chart))
    out = cfg.OUTPUT_ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gt_charts.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = overview([r for r in rows if not r["problems"]])
    (out / "gt_overview.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
