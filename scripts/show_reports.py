"""Print the training report saved inside each trained model (seqlib/versions.py). Scores only -- safe to bring back.

    python scripts\\show_reports.py                 # the parts that matter, every trained version
    python scripts\\show_reports.py --full          # the whole report
    python scripts\\show_reports.py v8 v9           # some versions only
"""
import argparse
import json

import _common  # noqa: F401

from seqlib.versions import VERSIONS, is_gated, model_path

KEY = {
    "ladder": ("version", "groups", "documents", "tuned_on_documents", "tuned_in_order", "candidate_recall",
               "anchors_found", "anchors_correct", "chosen_margin", "margin_curve", "unused_evidence"),
    "gated": ("version", "evidence", "documents", "tuned_on_documents", "out_of_order_documents",
              "in_order_documents", "gate_auc_out_of_fold", "gate_mean_p", "gate_signs_used", "chosen",
              "tuned_on", "decode", "without_gate", "margin_curve_out_of_fold", "unused_evidence",
              "anchors_found", "anchors_correct", "anchors_by_source", "candidate_recall"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("versions", nargs="*", default=list(VERSIONS))
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    import joblib

    for v in args.versions:
        gated = is_gated(v)
        path = model_path(v)
        if not path.is_file():
            print(f"== {v}: not trained ({path})")
            continue
        report = joblib.load(path).report
        if not args.full:
            report = {k: report[k] for k in KEY["gated" if gated else "ladder"] if k in report}
        print(f"== {v}")
        print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
