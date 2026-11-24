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
pip install -r requirements-seam.txt    (everything, incl. the seam text model; CPU only)
python -m pytest -q tests
```

## On the training machine

**Follow [steps.txt](steps.txt)** -- the full rebuild from a fresh folder, in order: packages, the
seam's base model, config, inputs and split, the shuffle sets, training, scoring, and the results
to bring back (small separate files in `outputs\share\`, from `scripts\collect_results.py`).

Four inputs, all already produced (nothing is re-run): the raw page images and their OCR (one
folder per document, `RecordID_DocumentID`), the pipeline outputs CSV and the ground-truth CSV,
one row per page each. Only documents in all four are used.

**Shuffle sets.** A shuffle set copies documents and renames their pages so they come out of order
(swaps, moved pages, moved blocks, reversed blocks), recording every renaming in
`shuffle_manifest.json`; the originals are only read, and pipeline values computed from the original
order are blanked in the copy. The models train on `shuffle_train` (train documents, 1-5 random
operations, 30% left in order) and are scored on six sets built from the same test documents:
exactly 5, 4, 3, 2 and 1 operations (`shuffle_ops5` ... `shuffle_ops1`) and 1-5 at random
(`shuffle_random`), then on the real, unshuffled test documents.

Each evaluation writes `summary.json`, `per_chart.csv` and `errors.txt` under `OUTPUT_ROOT/eval/`
(the last two name documents and stay on that machine).

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
