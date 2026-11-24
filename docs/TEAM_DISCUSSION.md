# Page sequencing and blank / junk: findings and plan

For team discussion, 24 November 2026. All figures are from the labelled documents on the training machine (scores and counts only; no page content left that machine).

## 1. Summary

- **The review-UI score is 74.45% today and 84.77% with a correction layer on the blank / junk verdict.** With a perfect set-aside it would be 98.72%: the remaining gap is almost entirely which pages are set aside (the J series), not the page order.
- **Page order is already right for 98% of real documents** (480 of 489 are in scan order). The best safe sequencer (v5) moves real-document position accuracy from 98.72% to 98.95% and breaks nothing.
- **On deliberately shuffled documents** the best model (v8) gets 83.9% of next-page links right (scan order: 78.9%) and fully repairs 16% of documents. It also breaks one real document, so it is not ready for production.
- **The production blank / junk model is about 98.5% accurate per page**, but the review UI scores exact page numbers: one wrong set-aside page renumbers every later page in the document. The few remaining errors are therefore expensive, and most of them come from two places: the **duplicate rules** (76% of duplicate flags are Main in the truth) and **short pages called junk** (160 pages).
- **Proposal:** fix the blank / junk errors first (target: review UI above 90%), use v5 for ordering now, and continue ordering research on the patterns real documents actually show.

## 2. How the work is set up

- A separate repository (`sequence`) holds the sequencing code taken out of the pipeline, so it can be changed and tested without touching production. It calls no Azure service and runs no OCR: it reads the pipeline's existing outputs.
- Code is written on one machine; training and testing run on the machine that holds the labelled data. Only scores and counts come back.
- **Inputs per document:** the page images, the OCR output (the pipeline's `ocr` folder), the full pipeline outputs CSV (`imaging_output_full.csv`) and the reviewers' ground truth (`page_sequence`; blank / junk pages numbered in their own J series).
- **Documents:** 500 have all four inputs; 489 are usable (the rest have truth labelling problems). Split by document: **392 train, 97 test** (fixed seed).
- **Measures:**

| Measure | What it counts |
|---|---|
| Review-UI accuracy (strict) | pages whose number matches the reviewer's exactly, as the review UI compares them |
| Position accuracy | Main pages at the right place in the order |
| Successor accuracy | pages followed by the right next page (adjacency) |
| Perfect documents | documents with every page right |

## 3. Where the review-UI score is lost

Real test documents (97 documents, 3,527 pages):

| Set-aside decided by | Order | Review-UI accuracy |
|---|---|---|
| production pipeline (as today) | v1 (production) | **74.45%** |
| correction layer on top of production | scan order | **84.77%** |
| correction layer on top of production | v8 | 84.72% |
| the truth (perfect set-aside) | scan order | 98.72% |
| the truth (perfect set-aside) | v5 | 98.95% |

A perfect sequencer can add at most about 1.3 points; a perfect set-aside adds about 14. **The 90% goal for the review UI depends on the blank / junk verdict.**

## 4. Blank / junk

### 4.1 The production model is accurate page by page

On the 392 training documents (14,199 pages; 700 set aside by the reviewers):

| Production verdict | Pages | Truth agrees | Truth disagrees |
|---|---|---|---|
| blank | 31 | 31 | 0 |
| junk | 748 | 588 | 160 kept as Main by the reviewers |
| duplicate | 123 | 30 | **93 kept as Main by the reviewers (76%)** |
| main | 13,297 | 13,246 | 51 set aside by the reviewers |

- Page accuracy: **98.5%** on the pages not flagged duplicate, 97.9% overall. (Calling every page Main would already score 95.1%: most pages are Main.)
- Of the pages it sets aside, 72% are set aside in the truth (precision); of the truth's set-aside pages it finds 93% (recall).

### 4.2 Why 98–99% per page becomes 74% in the review UI

The review UI compares page numbers. A page wrongly set aside (or wrongly kept) shifts the number of every Main page after it. One error on page 10 of a 40-page document leaves 30 pages with wrong numbers: that document scores 75% from a single mistake. About 300 page errors spread over 391 documents therefore cost far more than 2% of the score. **An error early in a long document costs the most.**

### 4.3 Where the errors are

- **Duplicates (93 errors).** 76% of the pages the duplicate rules flag are Main for the reviewers. The truth marks only 37 pages as duplicates at all.
- **Short pages called junk (160 errors).** Almost all have no page type and 11–50 words. From the decision ladder in `modules/blank_junk/method.py` these look like the short-page rule (fewer than 5 new words -> Junk / Others) or the tiny-page rule, not DistilBERT.
- **Missed set-asides (51).** Spread across Progress Note, Lab Report, Operative Note and Discharge Summary pages.
- **Position.** 277 of the errors are mid-document, 22 on the last page, 5 on the first page.

### 4.4 The correction layer (built in this repository)

A gradient-boosted model per page, trained on the reviewers' set-aside labels. It takes the production verdict and confidences, page type, quality, place in the document, the neighbours' verdicts and the OCR text length / words as inputs.

| | Precision | Recall | Main pages set aside | Set-aside pages kept |
|---|---|---|---|---|
| production (train) | 72% | 93% | 253 | 51 |
| correction layer (train, out of fold) | 98% | 86% | 13 | 99 |
| correction layer (test) | 87% | 73% | – | – |

It removes most of production's false set-asides but now keeps too many pages that should be set aside. The test set is small (97 documents), so its figures move more than the training ones.

### 4.5 Points to discuss

1. **The duplicate rules.** What do reviewers count as a duplicate? 93 of 123 flags disagree with them.
2. **The short-page rule.** Should short pages with fewer than 5 new words be Junk? Reviewers keep most of them as Main (continuation or signature pages, for example).
3. **Definitions.** Production keeps invoice pages as Main; check how reviewers number them. Agree one definition of blank / junk / duplicate with the review team.
4. **Version.** Was `imaging_output_full.csv` produced with the current production rules (several rules changed on 8 and 15 November)? If not, re-run the current version on these documents before deciding.
5. **Traceability.** Add the rule that decided each page (R1–R5, short page, signature, duplicate rule) to the outputs, so every error can be traced to a rule.
6. **How to measure.** Report blank / junk accuracy against the reviewers' numbering, and weight each error by the pages it renumbers, next to page accuracy.
7. **Where to fix.** Fix the production rules (one place, everyone benefits), keep a correction layer on top, or both.

## 5. Page order

### 5.1 What real disorder looks like

9 of 489 documents (1.8%) are not in scan order. Typical cases: **one page out of place** (5 documents, e.g. a page that belongs earlier or first) or **a small block of 2–4 pages moved**; 2–3 are more complex, one with a short reversed run.

| Rule tried on the real documents | Fixes (of 9) | Breaks (of 480 in order) |
|---|---|---|
| sort by date of service, oldest first | 0 | 297 |
| sort by date of service, newest first | 0 | 349 |
| pipeline documents as blocks, by date | 0 | 297–349 |
| printed "page n of m" (v1's rules) | 2 | 1 |

Dates do not explain the reviewers' order.

### 5.2 The shuffle test

Because real documents are almost always in order, a separate test checks whether a method can repair an order. Copies of the test documents (92 documents, 3,516 pages) had their pages renamed out of order (pages swapped, a page moved, a block moved, a block reversed). The renaming is applied to every input (images, OCR, pipeline CSV, truth), the pipeline values that depend on the original order are blanked, and the truth order is unchanged. A training set was built the same way (277 shuffled documents and 114 unchanged controls). Set-aside pages are taken from the truth here, so only the order is scored.

### 5.3 The versions

| Version | What it adds |
|---|---|
| v1 | production: scan order, except confident printed "page n of m" families |
| v2 | learned pair scorer + order search (learned to copy scan order; superseded) |
| v3 | evidence model without scan-position inputs + a tuned "move margin" (how strong evidence must be to move a page) |
| v4 | + printed page-number and fax-banner anchors (forced links) |
| v5 | + each page's own date of service read from its OCR |
| v6 | + header image fingerprint + seam model (a small language model: does page b's text continue page a's?) |
| v7 | v5 + a gate: per document, P(out of order); flagged documents get a low margin |
| v8 | everything: v6 + v1's rules as evidence + the gate + two order solvers |

### 5.4 Results

| Method | Shuffled: successor | Shuffled: position | Shuffled: perfect | Real: position | Real: perfect |
|---|---|---|---|---|---|
| scan order | 78.9% | 56.1% | 0.0% | 98.72% | 97.94% |
| v1 (production) | 81.2% | 60.4% | 8.7% | 98.72% | 97.94% |
| v2 | 79.9% | 56.6% | 1.1% | 98.72% | 97.94% |
| v3 | 80.1% | 58.5% | 3.3% | 98.72% | 97.94% |
| v4 | 81.1% | 59.0% | 5.4% | 98.95% | 97.94% |
| **v5** | 81.2% | 59.9% | 6.5% | **98.95%** | **97.94%** |
| v6 | 81.3% | 59.1% | 7.6% | 98.95% | 97.94% |
| v6, low margin | 82.2% | 61.2% | 9.8% | 98.84% | 96.91% |
| v7 | 82.0% | 61.9% | 10.9% | 98.21% | 96.91% |
| v7s (v7 with the seam) | 82.0% | 60.4% | 9.8% | 98.84% | 96.91% |
| **v8** | **83.9%** | 61.8% | **16.3%** | 97.96% | 96.91% |

### 5.5 Which links get repaired

Of 3,424 true next-page links in the shuffled documents, the shuffle broke 721. "Within / between documents" uses the pipeline's own document grouping on the original order.

| Broken links | Count | Share | Repaired by v5 | Repaired by v8 |
|---|---|---|---|---|
| all | 721 | 100% | 11% | **27%** |
| with printed page numbers / fax counters | 36 | 5% | **100%** | **100%** |
| within one document | 502 | 70% | 14% | 32% |
| between documents, next one later | 82 | 11% | 1% | 12% |
| between documents, next one earlier | 54 | 7% | 6% | 11% |
| between documents, same date | 52 | 7% | 4% | 13% |
| between documents, undated | 31 | 4% | 13% | 29% |

Links that were not broken were kept: 99.9% by v5, 99.1% by v8, 98.7% by v7.

### 5.6 What worked and what did not

Worked:

- **Anchors:** every broken link with printed page numbers or fax counters was repaired, with no harm.
- **v1's rules as evidence (v8):** the largest repair gain (27% of broken links vs 11% for v5).
- **Dates read from each page's own OCR (v5):** a small gain.
- **Tuning on every training document, out of fold:** v3 now repairs something.

Did not work, or not yet:

- **Lower margins and the gate harm real documents.** v7 and v8 both scramble the same in-order real document whose pages have weak links to each other; the gate reads "weak links" as "shuffled". They were tuned on a set where 71% of documents are shuffled; in reality 1.8% are.
- **Block order.** On a truly out-of-order real document, v8 found the right chains of pages (95.6% of next-page links) but put the blocks in the wrong order, which the review UI counts as wrong.
- **Single moved pages.** No version fixes a real document with one page out of place: the evidence does not beat the margin.
- **The seam model and header fingerprint** add little (v6 is about the same as v5) and are slow: v8 took 55 minutes for 97 documents (about 34 seconds each).
- **Page content is a weak signal.** Within documents, even v8 repairs only a third of broken links; the seam model's AUC is 0.62 (0.5 is chance).

### 5.7 What it could be

- **Real documents:** v5 now (+0.23 points position, no harm). The ceiling for ordering alone is about +1.3 points.
- **Shuffled documents:** if every within-document link were repaired, successor accuracy would reach about 93.6%; today's best is 83.9%. Reaching 90% needs much stronger evidence within documents and a way to order blocks.

## 6. Plan

| Step | What | Expected effect |
|---|---|---|
| A | Blank / junk: an error report on the verdicts actually used (by rule, page kind, position, pages renumbered); fix the duplicate and short-page rules with the team, or extend the correction layer; tune the cut-off on the review-UI score | the largest gain: review UI from 84.8% towards 90%+ (ceiling 98.7%) |
| B | Ordering, safe: use v5 in production; tune the gate and margins against all 392 real training documents in their original order; teach the gate "weak links everywhere" (in order) vs "a few links clearly misplaced" (out of order) | keep real documents unharmed while repairing more |
| C | Block order: a separate step that orders the linked blocks (printed numbers, fax counters, v1's rules) | fixes cases like the real document above |
| D | Rebuild the shuffle sets to match real disorder (mostly single pages and small blocks moved, few in-order documents disordered) | tuning that reflects production |
| E | Speed: drop the seam model; run the costly steps only on flagged documents | production-ready run time |

## 7. Decisions needed from the team

1. Agree the priority: blank / junk first (step A), v5 for ordering now.
2. The duplicate rules and the short-page rule: change them, or keep them and correct on top?
3. One definition of blank / junk / duplicate (including invoices) agreed with the review team.
4. Confirm the production version behind `imaging_output_full.csv`, and whether a "deciding rule" column can be added to the outputs.
5. How disorder arises in practice (pages swapped, a page added late at the front or end, rescans inserted, faxes in pieces), so the shuffle sets match reality.
