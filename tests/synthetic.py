"""Synthetic charts in the pipeline's file layout, for tests and smoke runs (no real data).

Each chart is a few documents; every document has its own header banner, a topic vocabulary
whose sentences run on across its pages, and (for some) a printed "Page i of k" footer. The scan
order is the true order with damage applied: a swapped pair, a document scanned in reverse, a
block moved. One fax cover page is junk.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

TOPICS = [
    "cardiac rhythm atrial fibrillation anticoagulation warfarin inr monitoring",
    "diabetes mellitus insulin glucose hemoglobin a1c neuropathy foot exam",
    "chronic kidney disease creatinine egfr dialysis nephrology referral",
    "asthma inhaler spirometry wheezing albuterol steroid taper",
    "hypertension lisinopril blood pressure sodium diet home readings",
    "depression sertraline phq9 counselling sleep appetite mood",
    "knee osteoarthritis injection physical therapy mobility pain scale",
    "copd oxygen saturation nebulizer pulmonary rehab exacerbation",
]


def _doc_pages(rng: random.Random, doc: int, n_pages: int, numbered: bool) -> list[list[str]]:
    words = TOPICS[doc % len(TOPICS)].split()
    banner = f"patient: test person {doc} | visit {1000 + doc} | dos 0{1 + doc % 9}/1{doc % 9}/2024"
    pages = []
    for i in range(n_pages):
        body = []
        if i == 0:
            body.append(f"PROGRESS NOTE {doc}")
        for _ in range(rng.randint(14, 20)):
            body.append(" ".join(rng.choice(words) for _ in range(9)) + ",")
        body[-1] = body[-1].rstrip(",") + (" continued" if i < n_pages - 1 else ".")
        footer = f"Page {i + 1} of {n_pages}" if numbered else f"clinic {doc} main street"
        pages.append([banner, *body, footer])
    return pages


def make_chart(root: Path, name: str, seed: int, info: dict | None = None) -> list[int]:
    """Writes the chart; returns the truth (scan page numbers in the right order, Main only).
    ``info``, when given, gets doc_of (page number -> document index) and cover (its page)."""
    rng = random.Random(seed)
    docs = [_doc_pages(rng, seed * 10 + d, rng.randint(2, 5), numbered=rng.random() < 0.6)
            for d in range(rng.randint(2, 4))]
    true_pages = [p for d in docs for p in d]
    doc_index = [k for k, d in enumerate(docs) for _ in d]
    order = list(range(len(true_pages)))  # true index at each scan slot
    damage = rng.choice(["swap", "reverse", "block", "none"])
    if damage == "swap" and len(order) > 3:
        i = rng.randrange(1, len(order) - 1)
        order[i], order[i + 1] = order[i + 1], order[i]
    elif damage == "reverse":
        k = len(docs[0])
        order[:k] = order[:k][::-1]
    elif damage == "block" and len(docs) > 1:
        k = len(docs[0])
        order = order[k:] + order[:k]
    cover_slot = rng.randrange(0, len(order) + 1)
    scan: list = list(order)
    scan.insert(cover_slot, "cover")

    chart = Path(root) / name
    (chart / "ocr").mkdir(parents=True, exist_ok=True)
    (chart / "imaging").mkdir(parents=True, exist_ok=True)
    pages_json, junk_rows, detail_rows, truth = [], [], [], {}
    for number, item in enumerate(scan, start=1):
        lines = ["FAX COVER SHEET", "to: records", "confidential"] if item == "cover" else true_pages[item]
        height = 1000.0
        step = 900.0 / max(1, len(lines))
        meta_lines = [{"content": t, "polygon": [50, 20 + k * step, 900, 20 + k * step, 900, 20 + k * step + 12, 50,
                                                   20 + k * step + 12]} for k, t in enumerate(lines)]
        if item != "cover":
            meta_lines[-1]["polygon"] = [50, 975, 900, 975, 900, 990, 50, 990]  # footer at the bottom
        pages_json.append({"fileName": f"{number}.jpg", "content": "\n".join(lines),
                           "pagesMeta": [{"height": height, "width": 1000, "lines": meta_lines}]})
        junk_rows.append({"chart_name": name, "page_name": f"{number}.jpg", "page_number": number,
                          "page_classification": "Cover Page" if item == "cover" else "Main",
                          "page_group": "junk" if item == "cover" else "main", "is_final": "True", "pass_no": 2})
        if info is not None:
            if item == "cover":
                info["cover"] = number
            else:
                info.setdefault("doc_of", {})[number] = doc_index[item]
        if item != "cover":
            truth[number] = item
            footer = lines[-1]
            if footer.startswith("Page "):
                n, _, m = footer[5:].partition(" of ")
                detail_rows.append({"chart_name": name, "page_name": f"{number}.jpg", "page_number": number,
                                    "printed_page_no": n, "printed_page_total": m})
    (chart / "ocr" / f"{name}_final2.json").write_text(json.dumps({"pages": pages_json}), encoding="utf-8")
    for fname, rows in ((f"{name}_junk.csv", junk_rows), (f"{name}_additional_page_details.csv", detail_rows)):
        if rows:
            with open(chart / "imaging" / fname, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return [p for p, _ in sorted(truth.items(), key=lambda kv: kv[1])]


def make_dataset(root: Path, gt_path: Path, n_charts: int = 24) -> dict[str, list[int]]:
    truths, rows = {}, []
    for c in range(n_charts):
        name = f"SYN{c:03d}"
        order = make_chart(root, name, seed=c + 1)
        truths[name] = order
        main = {p: i for i, p in enumerate(order, start=1)}
        n_pages = len(order) + 1
        j = 0
        for page in range(1, n_pages + 1):
            if page in main:
                rows.append([name, f"{page}.jpg", main[page], "No", "No", "No"])
            else:
                j += 1
                rows.append([name, f"{page}.jpg", j, "No", "Yes", "No"])
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Chart_Name", "Id", "Sequence", "Blank Page", "Junk Page", "Is Invoice Page"])
        w.writerows(rows)
    return truths


# --- the training machine's shape: RecordID_DocumentID folders of images + Azure OCR, the pipeline
# outputs CSV (imaging_final export) and the truth CSV (page_ground_truth export) ---

PIPELINE_COLUMNS = (
    "id chart_id page_id member_name member_dob member_id printed_or_handwritten handwritten_area_pct is_visible "
    "quality_tag orientation_angle tilt_angle mirrored blank_junk_flag is_duplicate document_seq page_type "
    "page_subtype specialty model_type codability classification_category page_type_source continuation_rule "
    "needs_review dos_from dos_to dos_source encounter_type provider_name provider_credentials signers "
    "provider_version provider_confidence confidences review_fields signature_present seq duplicate_of_page_id "
    "created_at updated_at chart_name page_number").split()
TRUTH_COLUMNS = (
    "id chart_name page_number source_page_id member_name member_dob member_id dos_from dos_to encounter_type "
    "encounter_label page_type page_subtype codeable blank_page junk_page is_invoice is_duplicate page_sequence "
    "rotation tilt mirrored is_visible quality printed_or_handwritten handwritten_pct rendering_provider "
    "provider_specialty provider_credentials provider_signature deleted_level source_path user_id created_at "
    "updated_at").split()


def _azure_page(page: dict, number: int) -> dict:
    meta = page["pagesMeta"][0]
    words = []
    for line in meta["lines"]:
        x0, y0, x1, _, _, y1, _, _ = line["polygon"]
        tokens = line["content"].split()
        step = (x1 - x0) / max(1, len(tokens))
        for k, token in enumerate(tokens):
            a, b = x0 + k * step, x0 + (k + 1) * step - 2
            words.append({"content": token, "polygon": [a, y0, b, y0, b, y1, a, y1], "confidence": 0.99})
    return {"pageNumber": number, "width": meta["width"], "height": meta["height"], "unit": "pixel",
            "lines": [{"content": l["content"], "polygon": l["polygon"]} for l in meta["lines"]], "words": words}


def _write(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)


def make_raw_dataset(raw_root: Path, ocr_root: Path, pipeline_csv: Path, gt_path: Path, n_docs: int = 12,
                     per_page_ocr: bool = False, extra_docs: int = 2) -> dict[str, list[int]]:
    """Documents as the training machine holds them; returns key -> true Main order.
    ``extra_docs`` more are in the two CSVs only (no images / OCR), like the 50 not yet delivered."""
    import shutil
    import tempfile

    from PIL import Image

    truths, pipe_rows, gt_rows = {}, [], []
    page_id = 313000
    with tempfile.TemporaryDirectory() as tmp:
        for c in range(n_docs + extra_docs):
            key = f"{70082400 + c}_{56511400 + c}"
            info: dict = {}
            order = make_chart(Path(tmp), key, seed=c + 101, info=info)
            pages = json.loads((Path(tmp) / key / "ocr" / f"{key}_final2.json").read_text(encoding="utf-8"))["pages"]
            if c < n_docs:
                (raw_root / key).mkdir(parents=True, exist_ok=True)
                # the pipeline workspace layout: <doc>/ocr/ holds the OCR, beside folders that
                # hold other JSON / text (which must not be read as OCR)
                ocr_dir = ocr_root / key / "ocr"
                ocr_dir.mkdir(parents=True, exist_ok=True)
                for other in ("staging", "imaging"):
                    (ocr_root / key / other).mkdir(exist_ok=True)
                (ocr_root / key / "staging" / "extraction.json").write_text('{"version": 1, "pages": {}}',
                                                                            encoding="utf-8")
                (ocr_root / key / "imaging" / "notes.txt").write_text("not ocr", encoding="utf-8")
                for number in range(1, len(pages) + 1):
                    Image.new("L", (40, 52), 255).save(raw_root / key / f"{number}.png")
                azure = [_azure_page(p, n) for n, p in enumerate(pages, start=1)]
                if per_page_ocr:
                    for n, page in enumerate(azure, start=1):
                        (ocr_dir / f"{key}_{n}.json").write_text(
                            json.dumps({"analyzeResult": {"pages": [page]}}), encoding="utf-8")
                else:
                    (ocr_dir / f"{key}.json").write_text(json.dumps(
                        {"status": "succeeded", "analyzeResult": {"apiVersion": "2024-11-30", "pages": azure}}),
                        encoding="utf-8")
            truths[key] = order
            main = {p: i for i, p in enumerate(order, start=1)}
            seen_docs: set = set()
            scan_rank = 0
            for number in range(1, len(pages) + 1):
                page_id += 1
                cover = number == info.get("cover")
                doc = info.get("doc_of", {}).get(number)
                first_in_scan = doc not in seen_docs
                seen_docs.add(doc)
                if not cover:
                    scan_rank += 1
                pipe = {col: "" for col in PIPELINE_COLUMNS}
                pipe.update({
                    "id": page_id - 200000, "chart_id": 26800 + c, "page_id": page_id, "chart_name": key,
                    "page_number": number, "printed_or_handwritten": "printed", "quality_tag": "high",
                    "blank_junk_flag": "junk" if cover else "not_blank_junk", "is_duplicate": "FALSE",
                    "document_seq": "" if cover else doc + 1,
                    "page_type": "Others" if cover else "Progress Note",
                    "page_subtype": "Cover Page" if cover else "General",
                    "model_type": "Others > Cover Page" if cover else "Progress Note",
                    "page_type_source": "page" if (cover or first_in_scan) else "continuation",
                    "dos_from": "" if cover else f"{1 + doc % 9}/1{doc % 9}/2024",
                    "dos_to": "" if cover else f"{1 + doc % 9}/1{doc % 9}/2024",
                    "dos_source": "" if cover else "document",
                    "encounter_type": "not_applicable" if cover else "outpatient_f2f",
                    "confidences": "{}", "signature_present": "FALSE",
                    "seq": "" if cover else scan_rank,  # v1 as production wrote it: scan order here
                })
                pipe_rows.append(pipe)
                gt = {col: "" for col in TRUTH_COLUMNS}
                gt.update({"id": 5000 + page_id % 10000, "chart_name": key, "page_number": number,
                           "source_page_id": f"{number}.png", "blank_page": "No",
                           "junk_page": "Yes" if cover else "No", "is_invoice": "No", "is_duplicate": "No",
                           "page_sequence": "J1" if cover else main[number]})
                gt_rows.append(gt)
        shutil.rmtree(tmp, ignore_errors=True)
    _write(pipeline_csv, PIPELINE_COLUMNS, pipe_rows)
    _write(gt_path, TRUTH_COLUMNS, gt_rows)
    return truths
