"""Step 2: bring each usable document's OCR into the workspace (no OCR is re-run).

Converts the OCR (Azure Document Intelligence JSON, the pipeline's own OCR files, per-page
JSON / text) into <CHARTS_ROOT>/<doc>/ocr/<doc>_final2.json and records each page's raw image.
Everything else comes from the pipeline outputs CSV.

    python scripts/import_ocr.py                  # every usable document (all four inputs)
    python scripts/import_ocr.py --limit 5        # try a few first
    python scripts/import_ocr.py --docs 70082444_56511414

Writes <OUTPUT_ROOT>/import/import_report.csv (counts per document) and failed.json.
"""
import argparse
import csv
import json
import time

import _common  # noqa: F401

from seqlib.eval.inventory import source_files, usable_documents
from seqlib.extract.import_ocr import import_document
from seqlib.io.ground_truth import load_ground_truth
from seqlib.io.pipeline_csv import load_pipeline_csv
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", default="", help="comma-separated document keys")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    images, ocr = source_files(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, cfg.DOC_KEY_REGEX)
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    keys = usable_documents(truths, cfg.DOC_KEY_REGEX, cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT,
                            load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV))
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",")}
        keys = [k for k in keys if k in wanted]
    if args.limit:
        keys = keys[: args.limit]
    print(f"{len(keys)} usable documents to import")

    out = cfg.OUTPUT_ROOT / "import"
    out.mkdir(parents=True, exist_ok=True)
    report, failed = [], {}
    started = time.time()
    for i, key in enumerate(keys, start=1):
        try:
            r = import_document(key, images.get(key, []), ocr.get(key, []))
            report.append({"doc": key, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}})
        except Exception as exc:  # noqa: BLE001 - one document must not stop the batch
            failed[key] = f"{type(exc).__name__}: {exc}"
            print(f"  {key}: FAILED {failed[key]}")
        if i % 25 == 0 or i == len(keys):
            print(f"  {i}/{len(keys)} ({time.time() - started:.0f}s)")
    if report:
        with open(out / "import_report.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    (out / "failed.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
    print(f"done: {len(report)} imported, {len(failed)} failed -> {out}")
    return 0 if report or not keys else 1


if __name__ == "__main__":
    raise SystemExit(main())
