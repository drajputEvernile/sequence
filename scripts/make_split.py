"""Write the train / test split (by chart) to config.SPLIT_FILE. Run once; later runs reuse it.

    python scripts/make_split.py              # refuses to overwrite an existing split
    python scripts/make_split.py --force      # make a new one (old numbers stop being comparable)
"""
import argparse
import json

import _common  # noqa: F401

from seqlib.eval.split import make_split
from seqlib.io.chart import chart_dirs
from seqlib.eval.inventory import usable_documents
from seqlib.io.ground_truth import load_ground_truth
from seqlib.io.pipeline_csv import load_pipeline_csv
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if cfg.SPLIT_FILE.exists() and not args.force:
        print(f"{cfg.SPLIT_FILE} exists; pass --force to replace it")
        return 1
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    # Only usable documents (images, OCR, pipeline outputs and truth) that import_ocr.py brought in.
    usable = set(usable_documents(truths, cfg.DOC_KEY_REGEX, cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT,
                                  load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV)))
    have = {d.name for d in chart_dirs()}
    truths = {n: t for n, t in truths.items() if n in usable and n in have}
    if usable - have:
        print(f"{len(usable - have)} usable documents not imported yet (run import_ocr.py): left out")
    split = make_split(truths, cfg.TEST_FRACTION, cfg.SPLIT_SEED)
    cfg.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg.SPLIT_FILE.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"train {len(split['train'])}, test {len(split['test'])}, excluded {len(split['excluded'])}")
    print(f"strata: {split['strata']}")
    print(f"-> {cfg.SPLIT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
