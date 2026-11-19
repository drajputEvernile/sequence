# sequence

Puts a medical chart's pages in their correct reading order. This repo holds the imaging
pipeline's page-sequencing logic (v1), separated out so it can be measured and replaced (v2).

**Read first:** [SEQUENCING_OVERVIEW.pdf](SEQUENCING_OVERVIEW.pdf) explains how v1 works, what it
can and cannot fix, how accuracy is measured, and the v2 plan. Source: `docs/SEQUENCING_OVERVIEW.md`.
Rebuild it with `python docs/build_pdf.py`.

## Ground rules

- **Local files only.** Nothing calls Azure, a database or an OCR service. It reads the OCR the
  pipeline already wrote (`ocr/<chart>_prelim.txt`, `_final1.json`, `_final2.json`) and the
  per-page CSVs its other stages wrote (`imaging/<chart>_*.csv`).
- **Every path is in [config.py](config.py)**: chart folders, ground truth, split file, models,
  Tesseract, outputs. On another machine, copy it and set `SEQ_CONFIG` to the copy.
- **Data stays on the machine that holds it.** Train and evaluate there; bring back numbers.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            (Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q tests         # 98 tests, including the pipeline's own v1 tests
```

## On the training machine

Four inputs, all already produced (nothing is re-run):

1. raw page images: one folder per document, `RecordID_DocumentID`
2. their OCR: same folder names; inside each, only the `ocr/` subfolder is read (`OCR_SUBFOLDER` in config.py)
3. the full pipeline outputs: one CSV, one row per page (the imaging_final export)
4. the ground truth: one CSV, one row per page (the page_ground_truth export)

Only documents in all four are used. Set the four paths in `config.py`, then:

```
python scripts/check_inputs.py            # paths found, pipeline CSV size
python scripts/inventory.py               # documents in each input, the usable ones, what is missing
python scripts/import_ocr.py              # each usable document's OCR -> workspace
python scripts/make_split.py              # once: 80/20 by document, stratified
python scripts/analyze_gt.py              # what reordering the documents actually need
python scripts/evaluate.py --method scan    # baseline: scan order
python scripts/evaluate.py --method v1_csv  # v1 exactly as production wrote it (the CSV's seq)
python scripts/evaluate.py --method v1      # v1's engines re-run here
python scripts/train_v2.py
python scripts/evaluate.py --method v2
python scripts/evaluate.py --method v2 --oracle-junk   # ordering alone, without blank/junk errors
```

## Shuffle test (can the sequencer fix an order?)

Most real documents are already in scan order, so their scores say little about repairing one.
`make_shuffle_test.py` copies a random sample (default 100 documents, more than 3 Main pages)
to a separate folder, renames pages there to put them out of order (swaps, moved pages, moved
blocks, reversed blocks) and records every renaming in `shuffle_manifest.json`. The originals
are only read. Pipeline values computed from the original order (`seq`, `document_seq`,
document-level type / DOS / confidences ...) are blanked in the copy so they cannot give the
answer away.

```
python scripts\make_shuffle_test.py                      # --count 100 --min-pages 4 --max-ops 3 --seed 2026
$env:SEQ_CONFIG = "<repo>\workspace\shuffle_test\config_shuffled.py"
python scripts\import_ocr.py
python scripts\evaluate.py --method scan                 # how much the shuffle broke
python scripts\evaluate.py --method v1                   # what the current sequencer repairs
python scripts\evaluate.py --method scan --set-aside truth   # the same, ordering alone
python scripts\evaluate.py --method v1 --set-aside truth
Remove-Item Env:SEQ_CONFIG                                # back to the real data
```

## Training and testing v2 on shuffles

v2 reads only per-page values (never the pipeline's order-dependent columns), so it sees the same
kind of input on real data, on a shuffle set and in production. Train it on shuffled *train*
documents, score it on shuffled *test* documents, and check it does no harm on the real ones:

```
Remove-Item Env:SEQ_CONFIG -ErrorAction SilentlyContinue
python scripts\make_shuffle_test.py --from train --role train --min-pages 2 --unchanged 0.3 --count 1000 --out workspace\shuffle_train
python scripts\make_shuffle_test.py --from test --role test --count 1000 --out workspace\shuffle_eval

$env:SEQ_CONFIG = "<repo>\workspace\shuffle_train\config_shuffled.py"
python scripts\import_ocr.py
python scripts\train_v2.py                     # the model goes to the main config's V2_MODEL_PATH

$env:SEQ_CONFIG = "<repo>\workspace\shuffle_eval\config_shuffled.py"
python scripts\import_ocr.py
python scripts\evaluate.py --method scan --set-aside truth
python scripts\evaluate.py --method v1 --set-aside truth
python scripts\evaluate.py --method v2 --set-aside truth
python scripts\shuffle_report.py               # by kind of damage and document size

Remove-Item Env:SEQ_CONFIG
python scripts\evaluate.py --method v2 --set-aside truth   # no harm on the real test documents (scan: 98.7%)
```

Each evaluation writes `summary.json`, `per_chart.csv` and `errors.txt` under `OUTPUT_ROOT/eval/`.

## Looking at one document

```
python scripts/order_chart.py <chart> --method v1 --truth
```

For a photographed document on a machine without the pipeline: put the photos in
`PHOTO_INPUT_ROOT/<case>/`, then

```
python scripts/ocr_photos.py <case>       # local Tesseract -> PHOTO_OUTPUT_ROOT/<case>/
python scripts/order_chart.py <case> --root <PHOTO_OUTPUT_ROOT> --method v1
```

## Layout

| Path | |
|---|---|
| `seqlib/io/` | readers: OCR files, the pipeline outputs CSV, the ground-truth CSV |
| `seqlib/extract/` | brings each document's OCR (Azure JSON, pipeline OCR files, per-page files) into the workspace |
| `seqlib/v1/` | the pipeline's continuity + sequencing code, unchanged except imports (`SOURCE_COMMIT`); `runner.py` feeds it local files; `_reference/` keeps the DB-bound originals for comparison |
| `seqlib/v2/` | pair features, pair-scorer training, order decoder |
| `seqlib/eval/` | metrics, split, ground-truth analysis |
| `seqlib/local_ocr/` | Tesseract OCR for photos |
| `scripts/` | command-line entry points |
| `tests/` | v1's ported unit tests + end-to-end tests on synthetic charts |
