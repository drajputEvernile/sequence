"""Every method's latest result, side by side, for the experiment ladder (scan, v1, v2, v3 ...).

Reads <root>/eval/<tag>_<split>_<time>/summary.json for each output root given and prints one
row per (root, method): pages, review-UI accuracy (strict), position, successor (adjacency),
fully right documents, Kendall tau. With the roots of a shuffle set and the real data together,
repair and no-harm sit next to each other.

    python scripts/compare.py
    python scripts/compare.py --roots outputs workspace\\shuffle_eval\\outputs

Writes <first root>/compare.csv. Scores only -- safe to bring back.
"""
import argparse
import csv
import json
from pathlib import Path

import _common  # noqa: F401

from seqlib.settings import cfg

COLUMNS = ("strict_page_accuracy", "position_accuracy", "successor_accuracy", "perfect_charts", "mean_kendall_tau")


def latest(root: Path) -> dict[str, dict]:
    found: dict[str, tuple[str, dict]] = {}
    folder = root / "eval"
    for d in sorted(folder.iterdir()) if folder.is_dir() else []:
        parts = d.name.rsplit("_", 2)
        summary = d / "summary.json"
        if len(parts) != 3 or not summary.is_file():
            continue
        tag, split, stamp = parts
        key = f"{tag} ({split})"
        if key not in found or stamp > found[key][0]:
            found[key] = (stamp, json.loads(summary.read_text(encoding="utf-8")))
    return {k: v for k, (_, v) in found.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roots", nargs="*", default=None,
                    help="output roots (default: the main OUTPUT_ROOT and every workspace\\shuffle_*\\outputs)")
    args = ap.parse_args()
    roots = [Path(r) for r in args.roots] if args.roots else (
        [Path(cfg.OUTPUT_ROOT)] + sorted((Path(cfg.REPO) / "workspace").glob("shuffle_*/outputs")))
    rows = []
    for root in roots:
        label = root.parent.name if root.name == "outputs" and root.parent.name.startswith("shuffle") else "real data"
        for method, s in sorted(latest(root).items()):
            rows.append({"data": label, "method": method, "documents": s.get("charts"), "pages": s.get("pages"),
                         **{c: s.get(c) for c in COLUMNS}})
    if not rows:
        print("no evaluation found")
        return 1
    width = max(len(r["method"]) for r in rows)
    print(f"{'data':14} {'method':{width}}  {'docs':>5} {'strict':>7} {'position':>8} {'successor':>9} {'perfect':>7} {'tau':>6}")
    for r in rows:
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-"  # noqa: E731
        print(f"{r['data'][:14]:14} {r['method']:{width}}  {r['documents'] or 0:>5} {fmt(r['strict_page_accuracy']):>7} "
              f"{fmt(r['position_accuracy']):>8} {fmt(r['successor_accuracy']):>9} {fmt(r['perfect_charts']):>7} "
              f"{fmt(r['mean_kendall_tau']):>6}")
    out = roots[0] / "compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
