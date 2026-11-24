# sequence

Puts a medical chart's pages in their correct reading order. This repo holds the imaging
pipeline's page-sequencing logic (v1), separated out so it can be measured and replaced.

**Read first:** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) (the goal, the versions, results) and
[TEAM_DISCUSSION.pdf](TEAM_DISCUSSION.pdf) (findings and plan; source `docs/TEAM_DISCUSSION.md`,
rebuild with `python docs/build_pdf.py`).

## Ground rules

- **Local files only.** Nothing calls Azure, a database or an OCR service. It reads the OCR the
  pipeline already wrote and the pipeline's per-page outputs CSV.
- **Every path is in [config.py](config.py).** On another machine, copy it and set `SEQ_CONFIG` to the copy.
- **Data stays on the machine that holds it.** Train and evaluate there; bring back numbers only.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            (Linux: source .venv/bin/activate)
pip install -r requirements.txt   (+ requirements-seam.txt for the seam text model)
python -m pytest -q tests
```

## On the training machine

Four inputs, all already produced (nothing is re-run): the raw page images and their OCR (one
folder per document, `RecordID_DocumentID`), the pipeline outputs CSV and the ground-truth CSV, one
row per page each. Only documents in all four are used. Set the paths in `config.py`, then once:

```
python scripts/check_inputs.py            # paths found, pipeline CSV size
python scripts/inventory.py               # documents in each input, the usable ones
python scripts/import_ocr.py              # each usable document's OCR -> workspace
python scripts/make_split.py              # 80/20 by document, stratified
python scripts/make_shuffle_test.py --from train --role train --min-pages 2 --unchanged 0.3 --count 1000 --out workspace\shuffle_train
python scripts/make_shuffle_test.py --from test --role test --count 1000 --out workspace\shuffle_eval
```

A shuffle set copies documents and renames their pages so they come out of order (swaps, moved
pages, moved blocks, reversed blocks; up to 3 per document), recording every renaming in
`shuffle_manifest.json`. The originals are only read. Pipeline values computed from the original
order are blanked in the copy so they cannot give the answer away.

Then train and score (the current commands are in [steps.txt](steps.txt)):

```
$env:SEQ_CONFIG = "<repo>\workspace\shuffle_train\config_shuffled.py"
python scripts\train.py --version v9
$env:SEQ_CONFIG = "<repo>\workspace\shuffle_eval\config_shuffled.py"
python scripts\evaluate.py --method v9 --set-aside truth
python scripts\link_report.py                    # which links each method repairs, by evidence
python scripts\shuffle_report.py                 # by kind of damage
Remove-Item Env:SEQ_CONFIG
python scripts\evaluate.py --method v9 --set-aside truth   # no harm on the real test documents
python scripts\compare.py
```

Each evaluation writes `summary.json`, `per_chart.csv` and `errors.txt` under `OUTPUT_ROOT/eval/`.

## Layout

| Path | |
|---|---|
| `seqlib/io/` | readers: OCR files, the pipeline outputs CSV, the ground-truth CSV |
| `seqlib/extract/` | brings each document's OCR into the workspace; the pipeline's date-of-service rules |
| `seqlib/eval/` | metrics, split, inventory, shuffle sets, error explanations |
| `seqlib/v1/` | the pipeline's continuity + sequencing code, unchanged except imports (`SOURCE_COMMIT`); `runner.py` feeds it local files |
| `seqlib/v2/` | page reading and pair features (`features.py`), the base solver (`decode.py`) |
| `seqlib/v3/` | the engine: evidence model, move margin, solvers, cross-fitted tuning |
| `seqlib/ladder/` | extra evidence: dates / banners / header hash, page markers, content clues, anchors, the seam model |
| `seqlib/v7/` | the gate (v8, v9) |
| `seqlib/versions.py` | the versions kept: v5, v8, v9 |
| `scripts/` | command-line entry points |
| `tests/` | v1's ported unit tests + end-to-end tests on synthetic charts |
