# Experiment ladder

Each version adds one idea from `page_assembly_v3_design_and_build.md` (the team's design), so
every gain can be attributed. Every version is scored the same three ways:

| Check | Data | Must |
|---|---|---|
| Repair | `workspace\shuffle_eval` (92 shuffled test documents), `--set-aside truth` | beat v1 (position 60.4%, successor 81.2%) |
| No harm | real test documents (97), `--set-aside truth` | stay at scan order's 98.7% position / 99.7% successor (design gate RG3) |
| Review-UI score | real test documents, `--set-aside pipeline` / `model` | the number the 90% target may refer to (74.5% today) |

`python scripts\compare.py` prints all of them side by side.

| Version | Idea (design section) | Status |
|---|---|---|
| scan | do nothing | measured |
| v1 | production: printed "page n of m" families, at most 5 places | measured: repair 60.4% / no harm 98.7% |
| v2 | one pair model with scan position among its inputs + search | measured: repair 56.6% / no harm 98.7%. Diagnosis: it learned "scan neighbour = next page" (first choice for moved pages 6%) |
| **v3** | evidence model with **no scan inputs** + scan order as a **tuned move margin** (RC1, FR6, 7.5.1); solver `search` or path `cover` with cycle repair (7.5) | built, to run |
| **v4** | v3 + page-number and fax/print **banner anchors** (forced links, 7.4.1) + banner features (7.1.1) | built, to run |
| **v5** | v4 + each page's **own DOS read from its OCR** (the pipeline's DOS rules); the model learns from the truth whether order follows dates | built, to run |
| **v6** | v5 + **header image fingerprint** (7.1.2) + **seam cross-encoder** (fine-tuned small language model, 8.3) | built, to run (needs a local base model; GPU recommended) |
| **v7** / **v7s** | v5's (v7) or v6's (v7s) evidence + a **gate**: P(document out of order) from document-level signs on its scan order (anchors the scan order breaks, weak scan links, gain of the best order over scan, dates jumping back, fax counters breaking). Flagged documents get a low margin, others a high one (`train_v7.py`) | built, to run |
| **v8** | everything: v6's evidence + **v1's rules as evidence** (its departures from scan order: a pair feature, extra candidates, a gate sign) + the gate + **both solvers** (search and cover, the cheaper order wins). Evaluate also with `--set-aside model` for the review-UI score | built, to run |
| set-aside | blank / junk / duplicate learned from the truth (review-UI score) | built (`train_setaside.py`), to run |

Differences from the design document, given the data on the training machine:

- **M1 vs the review UI.** The review UI compares each page's exact number (`groundTruth.ts`), not adjacency. Adjacency = `successor_accuracy`; the review-UI figure = `strict_page_accuracy`.
- **No encounter labels** (`encounter_review`) in the exported truth: the partition metrics (M2, M3, M5, M6) and the same-document model need a stand-in (pipeline documents, or DOS + type).
- **Dates.** Almost no page carries its own DOS in the outputs CSV (most are carried across the document, which v2+ ignores because it depends on the order). Date features, F1 and Step 7 need per-page DOS re-read from the OCR (v5).
- **Models.** scikit-learn's HistGradientBoosting stands in for LightGBM (same family); scipy is used for the assignment solver.

Not built, and why:

- **Start / end (boundary) models (7.3.2), same-document model, document merge (7.6) and document
  ordering by DOS (7.7)** need encounter labels; the exported truth has none. v5's date features
  let the evidence model learn any date ordering the truth has instead.
- **Forbidden edges F1, F2, F4, F5 (7.4.2)** belong to the design's multi-chain formulation. Here
  the chart is one path, where two different-date or demographics / visit pages may legitimately
  sit side by side at a document boundary; the page-number anchors carry the F4 / F5 logic.

## Tuning (all versions, since v7)

Every version's margin (and v7 / v8's gate) is tuned on **out-of-fold** scores: the training
documents are split into 5 folds, each scored by an evidence model fitted on the other four, so
all ~390 training documents count (before: a 20% hold-out, ~78 documents, ~23 of them in order).
Tuning sees only the pairs an unseen document would have (candidates, anchors, v1's jumps; before,
every true pair was among them, which flattered the evidence). With the seam (v6, v7s, v8) the
seam model's own half is left out of tuning. v3 to v6 trained before this change keep working;
retrain them (`train_ladder.py`) to compare like with like.

## Results so far (test documents; truth set-aside)

| method | shuffled: position | successor | perfect | real: position | successor | perfect |
|---|---|---|---|---|---|---|
| scan | 56.1% | 78.9% | 0.0% | 98.72% | 99.74% | 97.94% |
| v1 | 60.4% | 81.2% | 8.7% | 98.72% | 99.74% | 97.94% |
| v2 | 56.6% | 79.9% | 1.1% | 98.72% | 99.74% | 97.94% |
| v3 (m 6) | 56.1% | 78.9% | 0.0% | 98.72% | 99.74% | 97.94% |
| v4 (m 3) | 59.0% | 81.1% | 5.4% | 98.95% | 99.80% | 97.94% |
| v5 (m 3) | 59.9% | 81.2% | 6.5% | 98.95% | 99.80% | 97.94% |
| v6 (m 3) | 59.1% | 81.3% | 7.6% | 98.95% | 99.80% | 97.94% |
| v6 (m 1.5) | 61.2% | 82.2% | 9.8% | 98.84% | 99.71% | 96.91% |

Read: anchors (v4) and own dates (v5) help; the text evidence (v3, seam) adds little. A low margin
repairs more but breaks an in-order real document -- the trade-off v7's gate is built to remove.
On real documents the review-UI score (strict, pipeline set-aside: 74.5%) is limited by the
set-aside (blank / junk) disagreement, not by the order: the set-aside model is the larger lever.

`scripts/link_report.py` splits the shuffle test's true links into within-document and boundary
links (by date): boundary links between separate documents are largely beyond what page content
can tell, so their share of the broken links bounds what any version can repair.
