"""Train the set-aside model (which pages are blank / junk / duplicate -> the J series) on the
split's train documents, using the truth's labels. Saves it to config.SETASIDE_MODEL_PATH.

    python scripts/train_setaside.py
    python scripts/train_setaside.py --importance      # also: which inputs it leans on (slower)

Then score any order with it:
    python scripts/evaluate.py --method scan --set-aside model

Prints counts and scores only (no page text) -- safe to bring back.
"""
import argparse
import json

import _common  # noqa: F401

from seqlib.eval.split import read_split
from seqlib.io.chart import load_chart
from seqlib.io.ground_truth import load_ground_truth
from seqlib.settings import cfg
from seqlib.v2.setaside import importance, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--importance", action="store_true")
    args = ap.parse_args()
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = [n for n in read_split(cfg.SPLIT_FILE)["train"] if n in truths and not truths[n].problems()]
    examples = []
    for name in names:
        try:
            examples.append((load_chart(name), truths[name]))
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {name}: {exc}")
    print(f"training on {len(examples)} documents")
    model = train(examples)
    model.save(cfg.SETASIDE_MODEL_PATH)
    report = dict(model.report)
    if args.importance:
        report["importance (train)"] = importance(model, examples)
    print(json.dumps(report, indent=2))
    out = cfg.OUTPUT_ROOT / "setaside"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {cfg.SETASIDE_MODEL_PATH}\n-> {out / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
