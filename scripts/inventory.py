"""Step 1: which documents have all four inputs (raw images, OCR, pipeline outputs, truth)?

Counts each source, the documents in all four (the usable ones: the only ones used from here
on), what is missing where, page counts that disagree, truth label problems and the OCR
formats found.

    python scripts/inventory.py

Writes <OUTPUT_ROOT>/inventory/: inventory.csv (one row per document), summary.json and
usable_documents.txt. None contains page text: only counts, file types and JSON key names --
safe to bring back.
"""
import argparse
import csv
import json

import _common  # noqa: F401

from seqlib.eval.inventory import build, rows, write_usable
from seqlib.io.ground_truth import load_ground_truth
from seqlib.io.pipeline_csv import load_pipeline_csv
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    for label in ("RAW_DOCS_ROOT", "OCR_DOCS_ROOT", "PIPELINE_OUTPUT_CSV", "GROUND_TRUTH_PATH"):
        path = getattr(cfg, label)
        print(f"{label:20} {'ok' if path.exists() else 'MISSING':8} {path}")
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH) if cfg.GROUND_TRUTH_PATH.exists() else {}
    pipeline = load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV)
    docs, summary = build(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, truths, cfg.DOC_KEY_REGEX, pipeline)

    folders = {d.doc for d in docs if d.has_images or d.has_ocr}
    if truths and folders and not (set(truths) & folders):
        summary["hint"] = (f"no truth document matches a folder: truth names look like {sorted(truths)[:3]}, "
                           f"folders like {sorted(folders)[:3]}. Check DOC_KEY_REGEX.")
    if truths and not any(t.main_order for t in truths.values()):
        summary["hint_sequence"] = f"no truth page has a numeric {cfg.GT_SEQUENCE_COLUMN!r}: check GT_SEQUENCE_COLUMN"

    out = cfg.OUTPUT_ROOT / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    table = rows(docs)
    if table:
        with open(out / "inventory.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    usable = summary.pop("usable_documents")
    write_usable(out / "usable_documents.txt", usable)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
