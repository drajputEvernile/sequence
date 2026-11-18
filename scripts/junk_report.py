"""Where the pipeline's set-aside verdict and the truth disagree (train split by default).

For every labelled page: the pipeline's verdict (blank / junk / duplicate / main) against the
truth's (J series or Main), broken down by the pipeline's flag, the page's own type, the
document's type, page position and text length -- the patterns a better set-aside decision must
fix.

    python scripts/junk_report.py               # train split
    python scripts/junk_report.py --split all

Writes <OUTPUT_ROOT>/setaside/junk_report.json. Counts only (no page text): safe to bring back.
"""
import argparse
import json
from collections import Counter, defaultdict

import _common  # noqa: F401

from seqlib.eval.split import read_split
from seqlib.io import pipeline_csv
from seqlib.io.chart import load_chart
from seqlib.io.ground_truth import load_ground_truth
from seqlib.settings import cfg


def bucket(words: int) -> str:
    for limit, name in ((0, "0 words"), (10, "1-10"), (50, "11-50"), (150, "51-150"), (400, "151-400")):
        if words <= limit:
            return name
    return "400+"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("train", "test", "all"), default="train")
    args = ap.parse_args()
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    split = read_split(cfg.SPLIT_FILE)
    names = split["train"] + split["test"] if args.split == "all" else split[args.split]

    overall = Counter()
    by = defaultdict(lambda: defaultdict(Counter))  # dimension -> value -> outcome
    for name in names:
        truth = truths.get(name)
        if truth is None or truth.problems():
            continue
        chart = load_chart(name)
        junk = set(truth.junk_pages)
        n = len(chart.pages)
        for i, page in enumerate(chart.pages):
            if not truth.labels.get(page.page_number):
                continue
            row = page.pipeline or {}
            pipe = pipeline_csv.group(row)
            want = "set aside" if page.page_number in junk else "main"
            outcome = ("agree: " + want) if (pipe != "main") == (want == "set aside") else \
                ("pipeline sets aside, truth Main" if pipe != "main" else "pipeline Main, truth sets aside")
            overall[outcome] += 1
            path = pipeline_csv.type_path(row)
            dims = {
                "pipeline flag": pipe,
                "is_duplicate": str(pipeline_csv.flag(row.get("is_duplicate"))),
                "own type (model_type L1)": path[0] if path else "(none)",
                "own type (model_type L1 > L2)": " > ".join(path[:2]) if path else "(none)",
                "document type (page_type)": str(row.get("page_type") or "(none)"),
                "position": "first page" if i == 0 else "last page" if i == n - 1 else "middle",
                "text length": bucket(len(page.text.split())),
                "quality_tag": str(row.get("quality_tag") or "(none)"),
                "printed_or_handwritten": str(row.get("printed_or_handwritten") or "(none)"),
                "truth is_duplicate": str(truth.extra.get(page.page_number, {}).get("isduplicate", "") or "(empty)"),
            }
            for dim, value in dims.items():
                by[dim][value][outcome] += 1

    def table(values: dict) -> list:
        rows = []
        for value, c in values.items():
            wrong = c["pipeline sets aside, truth Main"] + c["pipeline Main, truth sets aside"]
            rows.append({"value": value, "pages": sum(c.values()), "disagree": wrong,
                         "pipeline sets aside, truth Main": c["pipeline sets aside, truth Main"],
                         "pipeline Main, truth sets aside": c["pipeline Main, truth sets aside"]})
        return sorted(rows, key=lambda r: -r["disagree"])[:25]

    report = {"split": args.split, "pages": sum(overall.values()), "overall": dict(overall),
              "breakdown (most disagreements first)": {dim: table(values) for dim, values in by.items()}}
    out = cfg.OUTPUT_ROOT / "setaside"
    out.mkdir(parents=True, exist_ok=True)
    (out / "junk_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pages": report["pages"], "overall": report["overall"]}, indent=2))
    for dim in ("pipeline flag", "own type (model_type L1)", "position", "text length"):
        print(f"\n{dim}:")
        for r in report["breakdown (most disagreements first)"][dim][:8]:
            print(f"  {r['value'][:40]:40} pages {r['pages']:6}  set aside but Main {r['pipeline sets aside, truth Main']:5}"
                  f"  Main but set aside {r['pipeline Main, truth sets aside']:5}")
    print(f"\n-> {out / 'junk_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
