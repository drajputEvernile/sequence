"""Build a shuffle test: copies of real documents with their pages renamed out of order.

Picks documents at random (more than --min-pages Main pages), copies their images, OCR, pipeline
rows and truth rows to --out, renames pages there to put them out of order (swaps, moved pages,
moved blocks, reversed blocks), and writes:

    <out>/raw_images/<doc>/...          renamed images
    <out>/ocr/<doc>/ocr/...             renamed OCR (the pipeline's final2 / final1 / prelim files)
    <out>/imaging_output_full.csv       renamed, with the order-dependent columns blanked
    <out>/ground_truth.csv              renamed (page_sequence untouched)
    <out>/shuffle_manifest.json         per document: what was done, which page became which
    <out>/split.json                    every shuffled document is a test document
    <out>/config_shuffled.py            a config pointing every script at this folder

The original data is only read. Then:

    $env:SEQ_CONFIG = "<out>\\config_shuffled.py"
    python scripts\\import_ocr.py
    python scripts\\evaluate.py --method scan       # how much the shuffle broke
    python scripts\\evaluate.py --method v1         # what the current sequencer repairs
    Remove-Item Env:SEQ_CONFIG
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import _common  # noqa: F401

from seqlib.eval.inventory import source_files, usable_documents
from seqlib.eval.shuffle import CONFIG_TEMPLATE, build, choose
from seqlib.eval.split import read_split
from seqlib.io.ground_truth import load_ground_truth
from seqlib.io.pipeline_csv import load_pipeline_csv
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--min-pages", type=int, default=4, help="at least this many Main pages")
    ap.add_argument("--from", dest="source", choices=("all", "test", "train"), default="all",
                    help="which split to draw from (all = train + test)")
    ap.add_argument("--max-ops", type=int, default=3, help="up to this many shuffle operations per document")
    ap.add_argument("--unchanged", type=float, default=0.0,
                    help="share of the chosen documents left in order as a control (e.g. 0.1)")
    ap.add_argument("--role", choices=("test", "train"), default="test",
                    help="what the shuffled documents are for: a test set, or a set to train v2 on")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(cfg.REPO / "workspace" / "shuffle_test"))
    args = ap.parse_args()
    if getattr(cfg, "SHUFFLE_TEST", False):
        raise SystemExit("SEQ_CONFIG points at a shuffle test: build from the main config (Remove-Item Env:SEQ_CONFIG)")

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    pipeline = load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV)
    images, ocr = source_files(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, cfg.DOC_KEY_REGEX)
    usable = usable_documents(truths, cfg.DOC_KEY_REGEX, cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, pipeline)
    splits = {}
    try:
        split = read_split(cfg.SPLIT_FILE)
        splits = {n: "train" for n in split["train"]} | {n: "test" for n in split["test"]}
    except FileNotFoundError:
        pass
    candidates = [n for n in usable if args.source == "all" or splits.get(n) == args.source]
    docs = choose(truths, candidates, args.count, args.min_pages, args.seed)
    if not docs:
        raise SystemExit("no document qualifies")
    out = Path(args.out).resolve()
    made = build(out, docs, truths, pipeline, cfg.PIPELINE_OUTPUT_CSV, cfg.GROUND_TRUTH_PATH, images, ocr,
                 args.seed, args.max_ops, args.unchanged, splits, role=args.role)
    config_path = out / "config_shuffled.py"
    config_path.write_text(CONFIG_TEMPLATE.format(main=Path(cfg.__file__).resolve(), root=out, this=config_path),
                           encoding="utf-8")

    sizes = Counter("4-10" if m.main_pages <= 10 else "11-30" if m.main_pages <= 30 else "31-60"
                    if m.main_pages <= 60 else "61+" for m in made)
    kinds = Counter(" ".join(op.split()[:2]) if op.startswith(("move", "reverse")) else op.split()[0]
                    for m in made for op in m.operations)
    summary = {
        "documents": len(made), "from": args.source, "role": args.role, "seed": args.seed,
        "left_in_order (control)": sum(1 for m in made if m.operations == ["unchanged (control)"]),
        "main_pages_per_document": dict(sorted(sizes.items())),
        "operations": dict(kinds), "pages_renamed": sum(m.pages_moved for m in made),
        "pages_total": sum(m.pages for m in made),
        "documents_by_split": dict(Counter(splits.get(m.doc, "?") for m in made)),
    }
    print(json.dumps(summary, indent=2))
    then = ("  python scripts\\train_v2.py\n" if args.role == "train" else
            "  python scripts\\evaluate.py --method scan --set-aside truth\n"
            "  python scripts\\evaluate.py --method v1 --set-aside truth\n")
    print(f"\n-> {out}\n\nNext:\n  $env:SEQ_CONFIG = \"{config_path}\"\n  python scripts\\import_ocr.py\n"
          f"{then}  Remove-Item Env:SEQ_CONFIG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
