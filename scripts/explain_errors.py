"""The documents whose correct order is not the scan order: what explains each one?

For every usable document (train + test) whose truth order of Main pages differs from the scan
order, tries the candidate rules in seqlib/eval/explain.py (date order, the pipeline's documents
moved as blocks, a flipped stack, printed page numbers) and reports which one reproduces the
truth. Every rule is also run on the documents already in scan order, to count how many it
would break -- a rule is only usable if it fixes the wrong ones without breaking the right ones.

    python scripts/explain_errors.py

Writes <OUTPUT_ROOT>/explain/explain_errors.json (per-document scores, run shapes and a page
table) and prints a summary. No page text; dates appear as D1, D2 ... in date order within a
document; nothing identifies a patient. Safe to bring back.
"""
import argparse
import json

import _common  # noqa: F401

import logging

logging.getLogger("seqlib.v1").setLevel(logging.WARNING)  # v1 logs every document

from seqlib.eval.explain import RULES, explain, rule_effect
from seqlib.eval.split import read_split
from seqlib.io.chart import load_chart
from seqlib.io.ground_truth import load_ground_truth
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-tables", action="store_true", help="leave the per-page tables out")
    args = ap.parse_args()
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    split = read_split(cfg.SPLIT_FILE)
    names = [n for n in split["train"] + split["test"] if n in truths and not truths[n].problems()]

    moved, in_order = [], []
    for name in names:
        truth = truths[name]
        try:
            chart = load_chart(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {name}: {exc}")
            continue
        is_moved = truth.main_order != sorted(truth.main_order)
        e = explain(chart, truth, with_table=is_moved and not args.no_tables)
        (moved if is_moved else in_order).append(e)

    effect = rule_effect(moved, in_order)
    report = {
        "documents": len(moved) + len(in_order),
        "out_of_scan_order": len(moved),
        "rules (fixed / improved among the out-of-order, broken among the in-order)": effect,
        "documents_out_of_scan_order": [
            {"chart": e.chart, "split": "test" if e.chart in split["test"] else "train", "main_pages": e.pages,
             "pages_out_of_place": e.pages_out_of_place, "truth_order_as_scan_runs": e.shape,
             "best_rule": e.best, "scores": e.scores, "pages": e.table}
            for e in sorted(moved, key=lambda e: -e.pages_out_of_place)
        ],
    }
    out = cfg.OUTPUT_ROOT / "explain"
    out.mkdir(parents=True, exist_ok=True)
    (out / "explain_errors.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{len(moved)} of {report['documents']} documents are not in scan order\n")
    print(f"{'rule':16} {'fixes':>6} {'improves':>9} {'breaks (of ' + str(len(in_order)) + ' in order)':>26}")
    for name in RULES:
        r = effect[name]
        print(f"{name:16} {r['moved_docs_fixed']:>6} {r['moved_docs_improved']:>9} {r['in_order_docs_broken']:>26}")
    print()
    for d in report["documents_out_of_scan_order"]:
        print(f"{d['chart']} ({d['split']}): {d['pages_out_of_place']}/{d['main_pages']} pages out of place; "
              f"truth = scan pages {d['truth_order_as_scan_runs']}")
        print(f"   best rule: {d['best_rule']}  " + "  ".join(f"{k} {v:.2f}" for k, v in d["scores"].items()))
    print(f"\n-> {out / 'explain_errors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
