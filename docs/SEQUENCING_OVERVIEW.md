# Page Sequencing: How v1 Works and the Plan for v2

Repo: F:\Projects\sequence. Source: model-repo commit 7ce9f8c (core-pipeline/modules/sequencing and modules/continuity). Updated 2026-11-18.

## 1. Summary

The imaging pipeline puts every chart's pages in a suggested reading order (stage 14, page_sequencing). It does not move files; it writes an actual_sequence number per page. This repo takes that logic out of the pipeline so it can be measured and improved on its own, from local files only: no Azure, no database, no OCR service.

What v1 actually does is narrower than its code suggests. **The order it writes is the scan order, except for pages that print a confident "Page n of m"**. Those pages are re-sorted among themselves, and at most 5 places from where they were scanned. A page without such a number never moves. The stream splitting, header grouping and builder code still runs, but its order is overwritten. Only its method labels survive.

So v1's accuracy is essentially the accuracy of the scan order plus the repairs page numbers allow. The first job is to measure both on the labelled documents: the pipeline outputs CSV holds v1's production answer (its seq column), so evaluate.py --method v1_csv scores v1 exactly as it ran, and --method scan scores the scan order. Every v2 gain should be reported against those two numbers.

Recommendation: build v2 as a new design and keep v1 as the baseline. v2 learns, from the labelled charts, how likely it is that page B directly follows page A. Then it searches for the most likely complete order, starting from the scan order. A first version of v2 is already in the repo and runs end to end. It still has to be trained and measured on the real data.

## 2. Where sequencing sits in the pipeline

Stages before it produce every input it uses. The repo reads them from four files, with no re-run of the pipeline: the raw page images, their OCR, the full pipeline outputs (the imaging_final table exported as one CSV, one row per page) and the ground truth (the page_ground_truth table exported as one CSV). The pipeline outputs columns that matter for sequencing:

| Column (pipeline outputs CSV) | Written by | Used for |
|---|---|---|
| blank_junk_flag | blank / junk / duplicate stage | pages set aside (blank, junk, duplicate: the J series) |
| model_type, page_subtype | page classification | the page's own type path |
| page_type, page_type_source | imaging_final (agreed across the document) | the document's type; "continuation" = carried from the page before |
| dos_from, dos_to | date of service, carried to the document | the document's DOS (2/22/2222 = non-visit) |
| signature_present | key/value extraction | signed pages |
| document_seq | 13 continuity | which document each page belongs to |
| seq | 14 page sequencing | v1's order, as production wrote it |

Only documents present in all four inputs are used; the others wait until their files arrive. Two inputs the pipeline used are not in the outputs CSV: the printed "page n of m" from its key/value extraction (v1 falls back to reading it from the text) and each page's own DOS before it was carried to the document.

Ground truth: the page_sequence column. Main pages are numbered 1..N in the correct order. Blank, junk and invoice pages run their own series, stored as J1, J2 and so on. review-ui marks a page correct when its predicted number equals the ground-truth number.

## 3. How v1 works

### 3.1 Continuity: splitting the chart into documents

Continuity scores every gap between two consecutive Main pages (in scan order) with a sum of points. Plus means break, minus means same document. Blank pages, single junk pages and duplicates are transparent. A run of two or more junk pages becomes its own junk segment.

| Signal | Points | Signal | Points |
|---|---|---|---|
| different DOS (label-found) | +5 decisive | same DOS (label-found) | -5 decisive |
| page numbers run on (n, n+1 of m) | -5 decisive | junk run begins / ends | +3 (+5 if 3+ pages) |
| previous page signed at its end | +3 | previous page signed anywhere | +2 |
| previous page is "page m of m" | +3 | title at the top | +3 |
| visit heading at the top | +3 | page type changes | +3 x confidence |
| says "continued" | -3 | sections run on in note order | -3 |
| same visit / FIN / CSN number | -3 | where the text ends (page fill) | -2..+2 |
| same header line | -2 | wording runs on across the gap (char) | -2 |
| same kind of junk | -2 | blank page between | +1 |
| same footer line | -1 | wording runs on (words) | -1 |
| ends mid-sentence / starts lower case | -1 each | same page type | -1 |

Hard rules then override the sum. A signed page always ends its document. A Demographics page, a summary page and a consent page never share a document with another kind of page. A visit heading always breaks unless the page numbers run on. "Section (continued)" always joins. A final pass merges weak breaks: same date, "continued", same type with no title, orphan pages, a Progress Note running on, and signature-only pages. It also splits a document that holds two label-found dates. Every weight is hand-set in continuity_canon.json. None is fitted to data; the canon itself says the confidence curve is "to be fitted on the labelled gaps".

Continuity never reorders pages. It only groups consecutive pages, so a document whose pages are scattered by the scan cannot be found as one document.

### 3.2 Sequencing: the order itself

1. Page markers. The printed page number comes from the key/value extraction (additional_page_details) when it found one. Otherwise regexes run over the header, footer and body: "Page 3 of 7", "Pg 3 of 7", "p. 3 of 7", "Page 3/7", "3 of 7", "3/7", "Page 3", "- 3 -", and a bare footer number. Markers are grouped into families by their total m. The largest family is taken as the record's pagination.
2. Streams, header groups and the builder (ported from advantmed-document-processing). These split combined uploads by total, find duplicate printouts and group unnumbered pages by header similarity. **Their order is computed and then overwritten by step 3.** They now only set the method label.
3. The fenced order (detect.fenced_order, 2026-10-25). Main pages keep their scan order. Pages whose marker is confident form families by total m: an explicit "page/pg/p." pattern, or the extraction's number. Each family's pages take the family's own scan slots, sorted by n. A page may move at most 5 places, unless every Main page of its documents is numbered (then a fully reversed document is put back in order). A family is held in scan order and flagged when two pages claim the same number or a move would exceed 5. When numbers repeat (two printouts of the same length), the family is split back by continuity document.
4. Blank / junk / duplicate pages are left out and numbered J1, J2 and so on, in scan order.
5. Confidence. A family-placed page gets its marker's confidence. Any other page gets the share of family-placed pages that did not move (a proxy for "scan order holds in this chart"). Held pages are flagged for review.

### 3.3 What v1 gets right, and what it cannot do

Right by design:

- Charts already scanned in the correct order. v1 leaves them alone, so it never makes a correct chart worse.
- A few pages swapped locally inside a document that prints "Page n of m".
- A numbered document scanned fully in reverse.

Cannot fix:

- Any misplaced page without a confident "page n of m". This covers most notes, labs and faxes, handwritten pages, and numbers the OCR misread. Bare "3 of 7", "3/7" and footer digits are never trusted.
- Whole documents or blocks out of place. Nothing orders documents relative to each other.
- A reversed or shuffled stack that is not numbered.
- Moves of more than 5 places inside partially numbered documents (held and flagged instead).
- Numbers that restart per printed note, where the same "1 of 2" appears many times. The family is held, or split only when continuity separated the documents correctly.
- Evidence that the next page continues the text: sentences running on, the same header or footer band, section order, dates. Continuity computes these for document breaks, but sequencing never uses them to choose an order.

Two more things limit the score no matter how good the ordering is:

- **Blank / junk errors count against strict accuracy.** A Main page wrongly junked has no number (the number is wrong). A junk page wrongly kept shifts every later number by one.
- **The strict measure compounds errors.** One page misplaced near the front makes every later page's number wrong, even when the rest of the order is perfect.

## 4. Measuring it

scripts/evaluate.py reports, for any method:

| Measure | Meaning |
|---|---|
| strict_page_accuracy | review-ui's definition: predicted number = ground-truth number, Main pages; a junked Main page counts as wrong |
| strict_scored_only | the same, leaving out pages the method set aside (what review-ui shows as unknown) |
| position_accuracy | page at exactly the right place, using the ground truth's blank/junk (ordering alone) |
| successor_accuracy | share of true "A then B" pairs that are adjacent in the prediction: does not compound |
| in_place | share of pages that need not move (longest common subsequence / pages) |
| perfect_charts | whole chart in the right order |
| junk_precision / recall | set-aside pages against the J series |

**Decision needed: which measure the 90% target refers to.** Strict page accuracy is the review-ui number, but it mixes blank/junk errors with ordering errors and it compounds. Successor accuracy measures ordering alone and is the standard measure for page ordering. Our suggestion: report strict for continuity with review-ui, and set the 90% target on successor accuracy (or on position accuracy with ground-truth junk), with strict tracked alongside.

scripts/analyze_gt.py runs on the ground truth alone, before any method. It answers what kind of reordering the charts need. How many are already in scan order? How many pages are out of place, and how far? Are runs reversed? How many misplaced pages print "page n of m"? (That last count is the most v1 could ever fix.) If, say, 85% of charts are already in scan order, the work is to find the few broken charts and leave the rest alone, not to rebuild every order. Run it first; it decides where v2 effort goes.

## 5. The new repo

| Path | What it is |
|---|---|
| config.py | every absolute path: the four inputs, workspace, split file, models, Tesseract, outputs. SEQ_CONFIG can point at a per-machine copy |
| seqlib/io/ | readers for the OCR files, the pipeline outputs CSV and the ground-truth CSV |
| seqlib/extract/ | brings each document's OCR into the workspace (Azure Document Intelligence JSON, the pipeline's OCR files, per-page JSON / text); no OCR is re-run |
| seqlib/v1/ | the pipeline's continuity and sequencing code, copied unchanged except for imports; runner.py feeds it the pipeline outputs and OCR |
| seqlib/v2/ | features.py (pair features), model.py (training), decode.py (order search), runner.py |
| seqlib/eval/ | metrics, stratified train/test split by chart, ground-truth analysis |
| seqlib/local_ocr/ | Tesseract OCR for photographed pages, written in the pipeline's file layout |
| scripts/ | check_inputs, inventory, import_ocr, make_split, analyze_gt, evaluate, train_v2, order_chart, ocr_photos |
| tests/ | the pipeline's v1 unit tests (71, passing unchanged) and end-to-end tests on synthetic charts |

Guarantees:

- No network calls and no pipeline run: no Azure, no database, no OCR service. The OCR is Azure's earlier output, read from disk; every other page fact comes from the pipeline outputs CSV.
- v1 is scored twice: v1_csv is exactly what production wrote; v1 re-runs the ported engines here on the pipeline's documents and the OCR text, so v1's rules can be changed and re-measured. The gap between the two is measured, not assumed.
- Every column of the pipeline outputs was produced from the pages, not from the truth, so all of them are fair inputs. When v2 goes into the pipeline it runs where those values exist (after imaging_final, or on the same values from the stages before it).

Working on the training machine (every path is in config.py):

1. Set RAW_DOCS_ROOT, OCR_DOCS_ROOT (folders named RecordID_DocumentID), PIPELINE_OUTPUT_CSV and GROUND_TRUTH_PATH. Run check_inputs.py.
2. inventory.py: documents in each of the four inputs, the usable ones (all four), what is missing where, page counts that disagree, truth label problems, the OCR format. Its report holds counts and JSON key names only, so it is safe to bring back.
3. import_ocr.py: brings each usable document's OCR into the workspace.
4. make_split.py (once; commit the split).
5. analyze_gt.py, then evaluate.py --method scan, --method v1_csv and --method v1: the baselines.
6. train_v2.py, then evaluate.py --method v2 (and --oracle-junk to separate ordering from blank/junk errors).
7. Read errors.txt and per_chart.csv. Bring back counts and page-level patterns, not documents. When a failure has to be seen, photograph the pages, run scripts/ocr_photos.py (local Tesseract) and scripts/order_chart.py on the result.

Splitting: the usable documents (about 500) become 80% train and 20% test, split by document, never by page. The split is stratified by "already in scan order" and chart size, so both halves contain the hard charts. Tune on train (or with cross-validation inside train). Look at test only to report.

## 6. v2: the design

Treat ordering as two steps: score every plausible pair, then find the best complete order.

**Pair scorer.** For each pair (A, B) it computes evidence that B directly follows A. A virtual START page stands for "B is first" and "A is last". The features include every v1 signal, computed for any pair instead of only scan neighbours: printed n/m and the step between them, DOS and days between, page type, signature, title, visit heading, header and footer band similarity, TF-IDF similarity of A's last lines to B's first lines, ends mid-sentence, starts lower case, "Section (continued)", visit numbers, and the scan distance; from the pipeline outputs: its document grouping (document_seq), the document's agreed type and whether the pipeline carried a page's type from the page before, and the encounter type. The scan distance is the prior: most pages are where they were scanned. A gradient-boosted tree model (scikit-learn, CPU, seconds to train) learns the weights from the labelled pairs, replacing the hand-set points.

**Candidates.** Only plausible pairs are scored: scan neighbours within 8, the 5 most similar text continuations, page-number successors, and START. Training reports candidate recall, the share of true pairs in the candidate set. That share caps successor accuracy, so widen the net if it is low.

**Decoder.** The best order is the cheapest path START, p1 ... pN, START under cost -log P. Charts of up to 11 Main pages are solved exactly (Held-Karp). Larger charts use local search that starts from the scan order and moves or reverses any block of up to 40 pages while the cost drops. Starting from the scan order and accepting only improvements keeps the conservative behaviour that makes v1 safe.

**Next steps, in order of expected value (decide after analyze_gt):**

1. Train and measure the baseline as built. Tune the window, top-k and model settings on cross-validation inside train.
2. Text-continuation model. Fine-tune a small open transformer as a cross-encoder on pairs: input A's last lines and B's first lines, output whether they follow. The training pairs come free from the ordered ground truth. Its probability becomes one more feature. Candidates under permissive licences: MiniLM / all-MiniLM-L6-v2 (Apache-2.0), ModernBERT-base (Apache-2.0), Bio_ClinicalBERT (MIT). Train on the GPU machine; this machine runs it on CPU. Set EMBEDDING_MODEL_PATH to try a frozen embedding first; it is wired in already.
3. Document-level order. If analyze_gt shows whole documents out of place, add an explicit document step: group pages (continuity, or v2's own links), then order the documents. Use DOS chronology only if the ground truth confirms charts are chronological.
4. Better page numbers. Learn which number patterns to trust (bare "3/7", footer digits) from labelled data, instead of v1's fixed list. Recover numbers OCR misread using the neighbouring pages' sequence.
5. Image features, only if text is exhausted: header and footer band image similarity, the same form template. Avoid LayoutLMv3 and similar models under non-commercial licences (CC BY-NC-SA).
6. Calibrated confidence and a review threshold. Send low-confidence charts to review rather than force them, which raises accuracy on the charts accepted.

**Risks.** 500 charts is enough for a tree model over these features. It is small for training deep models from scratch, which is why the plan fine-tunes small pre-trained models and keeps them as features. Strict accuracy is capped by blank/junk accuracy, so report ordering with and without ground-truth junk. Bring data back as numbers and patterns; photograph a document only when a failure cannot be understood otherwise.
