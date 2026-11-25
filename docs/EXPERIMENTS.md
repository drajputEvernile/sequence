# Sequencing experiments

The goal (agreed 2026-11-24): with blank / junk taken as perfect (`--set-aside truth`),

1. **≥ 95% position accuracy on the shuffle tests** -- every page at exactly its right place, on the
   test documents put out of order by exactly 5, 4, 3, 2 and 1 operations, and by 1-5 at random
   (swap, moved page, moved block, reversed block): six sets, the same documents in each;
2. then **≈ 99% position accuracy on the real (unshuffled) documents** -- no harm.

Clean rebuild (steps.txt): new split; every model trained on `shuffle_train` (train documents,
1-5 random operations, 30% left in order). CPU only.

Position accuracy is what the review UI counts: one page out of place shifts every page after it.

## Versions kept (seqlib/versions.py)

All share one engine (seqlib/v3): an **evidence model** that never sees where a page was scanned
(P(b directly follows a) from what the pages say), a **move margin** added to every link that is not
the scan order's (a page leaves its scan place only when the evidence beats it), and a **solver**.

| Version | What it is | Why kept |
|---|---|---|
| scan, v1, v1_csv | do nothing; production's rules re-run / as production wrote them | baselines |
| **v5** | page-number and fax-banner **anchors** (forced links) + banner features + each page's own **date** read from its OCR; one tuned margin | best on real documents (98.95%) -- the safe default |
| **v8** | v5 + header image fingerprint + the **seam** text model + **v1's rules** as evidence; a **gate** (P(document out of order)) picks a low or high margin per document; both solvers | best repair so far |
| **v9** | v8 + everything below; tuned on **position** | the attempt at the 95% target |

### What v9 adds

- **Page numbers in every printed form** (seqlib/ladder/markers.py). v1 and v5 read only "page n of m"
  with m ≤ 20. v9 also reads "n of m" with any total, "Page n" / "Pg. n" with no total, and a bare
  number alone on the first / last line ("3", "- 3 -", "[3]"). Each has a *style* (kind, band, the line
  with digits masked, the total), so numbers are compared within one way of numbering. They are pair
  features, candidates (the page with the next number is always considered), and anchors when
  number n and n + 1 of a style are each printed by one page only.
- **Content clues** (seqlib/ladder/content.py): section order in a note (chief complaint -> HPI -> ... ->
  assessment / plan -> signature), numbered items going on ("14." -> "15."), the running header's
  encounter id / visit date, "(continued)" at the top of the page, a table carried over the break.
- **The chart's pattern in time**: whether the chart runs forwards or backwards by date (read from
  its scan order, which a few operations leave mostly intact) and whether a -> b steps along it.
- **Reversed runs are cheap to undo.** A link to the page scanned just *before* pays a quarter of the
  margin: undoing a reversed block of k pages used to cost k + 1 margins, now 2 + (k - 1) / 4.
- **Swaps.** The search can swap two pages anywhere (a swap is two moves that only pay off together).
- **Tuned on position accuracy** (the margins and the gate), with in-order training documents kept
  at ≥ 99.5% position.
- **One more gate sign**: printed numbers of one style that the scan order does not step by one.

## Tuning

Every version's margin (and the gate for v8 / v9) is tuned on **out-of-fold** scores: the training
documents are split into 5 folds, each scored by an evidence model fitted on the other four.
Tuning sees only the pairs an unseen document would have (candidates, anchors, v1's jumps, next
printed number / list item). With the seam, the seam model's own half is left out of tuning.

## Diagnostics (counts only)

- `scripts/link_report.py` -- every true link of the shuffle test: within a document or between
  documents, anchored or not, and **the strongest clue the two pages give** (printed number, bare
  number, fax counter, list, sentence / table carried over, encounter id, same date, section order,
  or none). Per method: share of broken links repaired, of unbroken links kept. The broken links
  with no clue are the part no content model can be expected to repair.
- `scripts/shuffle_report.py` -- position / successor / fully repaired by operation kind, document
  size and pages moved.
- `scripts/explain_errors.py` -- real documents: how many are out of order, and which rules explain it.

## Results so far (test documents; truth set-aside)

| method | shuffled: position | successor | perfect | real: position | perfect |
|---|---|---|---|---|---|
| scan | 56.1% | 78.9% | 0.0% | 98.72% | 97.94% |
| v1 | 60.4% | 81.2% | 8.7% | 98.72% | 97.94% |
| v5 | 59.9% | 81.2% | 6.5% | 98.95% | 97.94% |
| v7 (dropped) | 61.9% | 82.0% | 10.9% | 98.21% | 96.91% |
| v8 | 61.8% | 83.9% | 16.3% | 97.96% | 96.91% |
| v9 | to run | | | | |

Dropped after measuring (code in git history): v2 (pair model with scan inputs: learned "scan
neighbour = next page"), v3 / v4 / v6 / v7 / v7s (steps of the ladder, superseded by v5 / v8),
the set-aside model and photo OCR (blank / junk is taken as perfect for now).

Seam diagnosis (2026-11-24, clean rebuild, `seam_diagnose.py`, 214 unseen training documents): the
first seam (wrong pages: scan neighbour, most similar-looking, random; lr 2e-5) reached AUC 0.62 --
0.50 against random pages and 0.51 against the scan neighbour (TF-IDF: 0.70 / 0.67), 0.76 against
the most similar page. It learned only "the most similar page is not the next one", which the TF-IDF
features already carry. Not over-fitting (0.67 on its own documents); text length fine (~42 words a
side, 1.6% cut off). Only 5.5% of true breaks visibly continue a sentence. Strongest clue of each true
link: printed page number 25%, bare number 23%, dates / encounter 15%, list / sentence / section 9%,
none 28%. Retrained with the scan neighbour + two random pages, all of the seam half, lr 5e-5, the
held-out AUC per epoch (best epoch kept). Bar to keep the seam: about 0.80 against random pages.

Retrained seam (5 epochs; epoch 5 kept: +0.009 over epoch 4 while the loss fell 0.51 -> 0.46): on
the 195 unseen documents, 0.77 against random pages and 0.74 against the scan neighbour (TF-IDF 0.69 /
0.67) -- a modest signal TF-IDF does not carry; kept in v8 / v9. Against the most similar page 0.39
(TF-IDF 0.17): neither tells a sibling page (same form, another visit) from the true next one. No help
at document boundaries (0.50-0.56). Its own documents: 0.90 against random pages -- more documents,
not more epochs, would be the next lever.

Link report (v8): of 3,424 true links the shuffle broke 721; only 36 had a page-number anchor (all
repaired since v4); within-document links: 32% repaired. v8's real-data loss is one in-order
document the gate flagged (0.99) and scrambled.
