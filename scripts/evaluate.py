"""Score a method against the ground truth.

    python scripts/evaluate.py --method scan            # baseline: scan order as is
    python scripts/evaluate.py --method v1_csv          # v1 as production wrote it (the outputs CSV's seq)
    python scripts/evaluate.py --method v1              # v1's engines re-run here on the pipeline's documents
    python scripts/evaluate.py --method v1 --documents recompute   # ... with continuity re-run too
    python scripts/evaluate.py --method v9 --set-aside truth       # a trained version (scripts/train.py)

    --set-aside pipeline (default) | truth   which pages are set aside (the J series): the pipeline's
        blank_junk_flag, or the truth's own -- blank / junk taken as perfect, so only the order counts
    --split test (default) | train | all     all = train + test (the usable documents)

Writes <OUTPUT_ROOT>/eval/<method>_<split>_<time>/: summary.json, per_chart.csv, errors.txt.
"""
import argparse
import csv
import json
import time
from datetime import datetime

import _common  # noqa: F401

from seqlib.eval.metrics import score_chart, summarize
from seqlib.eval.split import read_split
from seqlib.io.chart import load_chart
from seqlib.io.ground_truth import load_ground_truth
from seqlib.settings import cfg
from seqlib.v1.runner import run_scan_order, run_v1, run_v1_csv
from seqlib.versions import VERSIONS

METHODS = ("scan", "v1", "v1_csv", *VERSIONS)


def build_method(name: str, documents: str = "csv"):
    if name == "scan":
        return lambda chart: run_scan_order(chart)
    if name == "v1":
        return lambda chart: run_v1(chart, documents)
    if name == "v1_csv":
        return lambda chart: run_v1_csv(chart)
    from seqlib.ladder import load_seam
    from seqlib.versions import is_gated, load_model

    model = load_model(name)
    seam = load_seam(required=True) if "seam" in model.settings.groups else None
    if is_gated(name):
        from seqlib.v7.runner import run_v7

        print(f"{name}: gate threshold {model.threshold}, margins {model.margin_low} (flagged) / "
              f"{model.margin_high} (others), evidence {list(model.settings.groups)}")
        return lambda chart: run_v7(chart, model, seam=seam)
    from seqlib.v3.runner import run_v3

    print(f"{name}: margin {model.margin}, solver {model.settings.solver}, evidence {list(model.settings.groups)}")
    return lambda chart: run_v3(chart, model, seam=seam)


def set_aside_by_truth(chart, truth) -> None:
    junk = set(truth.junk_pages)
    for page in chart.pages:
        if truth.labels.get(page.page_number):
            page.group_override = "junk" if page.page_number in junk else "main"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--split", choices=("test", "train", "all"), default="test")
    ap.add_argument("--documents", choices=("csv", "recompute"), default="csv",
                    help="v1: where its continuity documents come from")
    ap.add_argument("--set-aside", choices=("pipeline", "truth"), default="pipeline",
                    help="which pages are set aside (J series)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.method == "v1_csv" and getattr(cfg, "SHUFFLE_TEST", False):
        raise SystemExit("v1_csv is production's answer for the original order: on a shuffle test use "
                         "--method scan / v1 / a version")
    if args.method == "v1_csv" and args.set_aside != "pipeline":
        raise SystemExit("v1_csv is production's own answer; use --method scan / v1 / a version with --set-aside")

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    split = read_split(cfg.SPLIT_FILE)
    names = split["train"] + split["test"] if args.split == "all" else split[args.split]
    names = [n for n in names if n in truths and not truths[n].problems()]
    if args.limit:
        names = names[: args.limit]
    method = build_method(args.method, args.documents)

    tag = (args.method
           + (f"-{args.documents}" if args.method == "v1" else "")
           + (f"-setaside_{args.set_aside}" if args.set_aside != "pipeline" else ""))
    out = cfg.OUTPUT_ROOT / "eval" / f"{tag}_{args.split}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    scores, failed, rows = [], {}, []
    errors = open(out / "errors.txt", "w", encoding="utf-8")
    started = time.time()
    for k, name in enumerate(names, start=1):
        try:
            chart = load_chart(name)
            if args.set_aside == "truth":
                set_aside_by_truth(chart, truths[name])
            result = method(chart)
        except Exception as exc:  # noqa: BLE001
            failed[name] = repr(exc)
            continue
        s = score_chart(truths[name], result.main_order, result.junk_pages)
        scores.append(s)
        rows.append({
            "chart": name, "pages": s.pages, "scan_order_correct": s.scan_order_correct,
            "strict": round(s.strict_correct / s.strict_total, 4) if s.strict_total else "",
            "position": round(s.position_correct / s.pages, 4) if s.pages else "",
            "successor": round(s.successor_correct / s.successor_total, 4) if s.successor_total else "",
            "in_place": round(s.in_place / s.pages, 4) if s.pages else "",
            "kendall": "" if s.kendall is None else round(s.kendall, 4),
            "perfect": s.perfect, "junk_fp": s.junk_fp, "junk_fn": s.junk_fn,
            "method_detail": result.method,
            "predicted": " ".join(map(str, result.main_order)),
            "truth": " ".join(map(str, truths[name].main_order)),
        })
        if not s.perfect:
            errors.write(f"{name}\n  truth:     {rows[-1]['truth']}\n  predicted: {rows[-1]['predicted']}\n")
            errors.writelines(f"  {e}\n" for e in s.errors)
        if k % 25 == 0:
            print(f"  {k}/{len(names)} charts")
    errors.close()

    summary = {"method": tag, "split": args.split, "seconds": round(time.time() - started, 1),
               **summarize(scores), "failed_charts": len(failed), "failures": failed}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with open(out / "per_chart.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
