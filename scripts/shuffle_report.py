"""Shuffle test results by kind of damage and document size (run with the shuffle config active).

Joins <shuffle folder>/shuffle_manifest.json with the latest evaluation of each method
(<OUTPUT_ROOT>/eval/<method>_<split>_<time>/per_chart.csv) and prints, per method:

* by operation kind (documents whose shuffle included a swap / moved page / moved block /
  reversed block): mean position and successor accuracy, share of documents fully repaired;
* by the number of operations, by document size (Main pages) and by how many pages the shuffle moved.

    python scripts/shuffle_report.py
    python scripts/shuffle_report.py --methods scan-setaside_truth v9-setaside_truth

Writes <OUTPUT_ROOT>/shuffle_report.json. Scores and counts only -- safe to bring back.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401

from seqlib.settings import cfg

KINDS = ("swap", "move page", "move block", "reverse block", "unchanged")


def kind_of(op: str) -> str:
    for kind in KINDS:
        if op.startswith(kind):
            return kind
    return op.split()[0]


def size_bucket(n: int) -> str:
    return "4-10" if n <= 10 else "11-30" if n <= 30 else "31-60" if n <= 60 else "61+"


def moved_bucket(share: float) -> str:
    return "<=10% pages moved" if share <= 0.1 else "10-30%" if share <= 0.3 else "30-60%" if share <= 0.6 else ">60%"


def latest_evals(root: Path) -> dict[str, Path]:
    """method tag -> its latest evaluation folder (named <tag>_<split>_<YYYYmmdd-HHMMSS>)."""
    out: dict[str, tuple[str, Path]] = {}
    folder = root / "eval"
    for d in sorted(folder.iterdir()) if folder.is_dir() else []:
        parts = d.name.rsplit("_", 2)
        if len(parts) != 3 or not (d / "per_chart.csv").is_file():
            continue
        tag, _split, stamp = parts
        if tag not in out or stamp > out[tag][0]:
            out[tag] = (stamp, d)
    return {tag: d for tag, (_, d) in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="*", help="evaluation tags (folder names without _<split>_<time>)")
    args = ap.parse_args()
    manifest_path = Path(cfg.GROUND_TRUTH_PATH).parent / "shuffle_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no shuffle manifest at {manifest_path}: set SEQ_CONFIG to the shuffle test's config")
    manifest = {d["doc"]: d for d in json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]}
    evals = latest_evals(Path(cfg.OUTPUT_ROOT))
    tags = args.methods or sorted(evals)
    report = {}
    for tag in tags:
        if tag not in evals:
            print(f"no evaluation found for {tag}")
            continue
        with open(evals[tag] / "per_chart.csv", newline="", encoding="utf-8") as fh:
            rows = {r["chart"]: r for r in csv.DictReader(fh)}
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for doc, info in manifest.items():
            r = rows.get(doc)
            if r is None:
                continue
            score = (float(r["position"] or 0), float(r["successor"] or 1), r["perfect"] == "True")
            groups["all"]["all"].append(score)
            for kind in sorted({kind_of(op) for op in info["operations"]}):
                groups["by operation (documents that had it)"][kind].append(score)
            ops = info["operations"]
            groups["by number of operations"][
                "0 (control)" if ops == ["unchanged (control)"] else f"{len(ops)} operation{'s' * (len(ops) > 1)}"
            ].append(score)
            groups["by size (Main pages)"][size_bucket(info["main_pages"])].append(score)
            groups["by pages moved"][moved_bucket(info["pages_moved"] / max(1, info["pages"]))].append(score)
        report[tag] = {
            dim: {value: {"documents": len(s), "position": round(sum(x[0] for x in s) / len(s), 4),
                          "successor": round(sum(x[1] for x in s) / len(s), 4),
                          "fully_repaired": round(sum(x[2] for x in s) / len(s), 4)}
                  for value, s in sorted(values.items())}
            for dim, values in groups.items()
        }
    out = Path(cfg.OUTPUT_ROOT) / "shuffle_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for tag, dims in report.items():
        print(f"\n== {tag}")
        for dim, values in dims.items():
            print(f"  {dim}")
            for value, s in values.items():
                print(f"    {value:22} docs {s['documents']:4}  position {s['position']:.3f}  "
                      f"successor {s['successor']:.3f}  fully repaired {s['fully_repaired']:.2f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
