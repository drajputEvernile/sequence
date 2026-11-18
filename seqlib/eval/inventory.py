"""What data there is, per document, across the four inputs:

    raw images      <RAW_DOCS_ROOT>/<RecordID_DocumentID>/...
    OCR             <OCR_DOCS_ROOT>/<RecordID_DocumentID>/...   (or files named after the key)
    pipeline CSV    config.PIPELINE_OUTPUT_CSV  (chart_name, page_number, one row per page)
    truth CSV       config.GROUND_TRUTH_PATH    (chart_name, page_number, page_sequence ...)

A document is *usable* when all four have it; the split and every run use only those.

Nothing here reads page text into the report: the OCR format is recognised from its structure
(JSON keys), and only counts, file types and key names are written out.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from seqlib.io import pipeline_csv
from seqlib.io.ground_truth import ChartTruth

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".pdf"}
OCR_SUFFIXES = {".json", ".txt", ".xml", ".hocr", ".tsv", ".md", ".html"}
MAX_JSON_BYTES = 200 * 1024 * 1024  # do not parse anything larger; it is still counted


@dataclass
class DocInventory:
    doc: str
    image_files: int = 0
    image_types: str = ""
    image_numeric_ids: int = 0  # image file stems that are page numbers (1.png ...)
    ocr_files: int = 0
    ocr_types: str = ""
    ocr_format: str = ""
    ocr_pages: Optional[int] = None
    in_pipeline: bool = False
    pipeline_pages: int = 0
    pipeline_set_aside: int = 0  # blank / junk / duplicate per blank_junk_flag
    pipeline_documents: int = 0  # distinct document_seq
    in_truth: bool = False
    truth_pages: int = 0
    truth_main: int = 0
    truth_junk: int = 0
    truth_unlabelled: int = 0
    truth_duplicates_numbered: int = 0  # is_duplicate Yes but given a Main number
    truth_problems: str = ""
    page_counts: str = ""  # images / ocr / pipeline / truth
    pages_agree: Optional[bool] = None
    truth_in_scan_order: Optional[bool] = None
    usable: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return self.image_files > 0

    @property
    def has_ocr(self) -> bool:
        return self.ocr_files > 0


def doc_key(name: str, key_regex: str) -> Optional[str]:
    m = re.match(key_regex, name)
    return m.group(0) if m else None


def _entries(root: Path, key_regex: str, subfolder: str = "") -> dict[str, list[Path]]:
    """key -> the files that belong to it: everything inside its folder -- only inside
    ``<folder>/<subfolder>`` when a subfolder is given -- or files named after it."""
    out: dict[str, list[Path]] = {}
    if not root or not Path(root).is_dir():
        return out
    for entry in Path(root).iterdir():
        key = doc_key(entry.name, key_regex)
        if not key:
            continue
        if entry.is_dir():
            base = entry / subfolder if subfolder else entry
            files = [p for p in base.rglob("*") if p.is_file()] if base.is_dir() else []
        else:
            files = [entry]
        out.setdefault(key, []).extend(files)
    return out


def _subfolders() -> tuple[str, str]:
    from seqlib.settings import cfg

    return (str(getattr(cfg, "RAW_SUBFOLDER", "") or "").strip("/\\"),
            str(getattr(cfg, "OCR_SUBFOLDER", "") or "").strip("/\\"))


def source_files(raw_root: Path, ocr_root: Path, key_regex: str) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """(images by doc, OCR files by doc). Inside each document folder only config.RAW_SUBFOLDER /
    config.OCR_SUBFOLDER is read when set (the pipeline workspace keeps its OCR in ``<doc>/ocr/``
    beside imaging/, staging/ ... which are not OCR)."""
    raw_sub, ocr_sub = _subfolders()
    raw = _entries(raw_root, key_regex, raw_sub)
    ocr = _entries(ocr_root, key_regex, ocr_sub)
    images = {k: [p for p in v if p.suffix.lower() in IMAGE_SUFFIXES] for k, v in raw.items()}
    ocr = {k: [p for p in v if p.suffix.lower() in OCR_SUFFIXES] for k, v in ocr.items()}
    return {k: v for k, v in images.items() if v}, {k: v for k, v in ocr.items() if v}


def _json_format(path: Path) -> tuple[str, Optional[int], list[str]]:
    """(format name, pages it holds or None, top-level keys) of an OCR JSON file."""
    if path.stat().st_size > MAX_JSON_BYTES:
        return "json (too large to inspect)", None, []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except ValueError:
        return "json (unreadable)", None, []
    if isinstance(data, list):
        return "json list", len(data), []
    if not isinstance(data, dict):
        return "json (other)", None, []
    keys = sorted(data)[:15]
    result = data.get("analyzeResult") if isinstance(data.get("analyzeResult"), dict) else None
    if result is not None:
        pages = result.get("pages") or result.get("readResults") or []
        return "azure document intelligence (analyzeResult)", len(pages), keys
    pages = data.get("pages")
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        first = pages[0]
        if "pagesMeta" in first:
            return "pipeline final2", len(pages), keys
        if "words" in first and "content" in first and "fileName" in first:
            return "pipeline final1", len(pages), keys
        if "fileName" in first and "content" in first:
            return "pipeline final (content only)", len(pages), keys
        if "lines" in first or "words" in first:
            return "azure-style pages[] (lines / words)", len(pages), keys
        return "pages[] (other)", len(pages), keys
    if "readResults" in data:
        return "azure read v3 (readResults)", len(data["readResults"]), keys
    if "lines" in data or "words" in data or "content" in data:
        return "single-page json", 1, keys
    return "json (other)", None, keys


def inspect_ocr(files: list[Path]) -> tuple[str, Optional[int], dict[str, list[str]]]:
    """(format, pages, {format: example top-level keys}) for one document's OCR files."""
    formats: Counter[str] = Counter()
    pages = 0
    keys_by_format: dict[str, list[str]] = {}
    per_page_files = 0
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".json":
            name, n, keys = _json_format(path)
            formats[name] += 1
            keys_by_format.setdefault(name, keys)
            if name == "single-page json":
                per_page_files += 1
            elif n is not None:
                pages = max(pages, n)  # final1 / final2 hold the same pages: count once
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            markers = re.findall(r"^=====\s*.+?\s*=====\s*$", text, re.M)
            if markers:
                formats["pipeline prelim txt"] += 1
                pages = max(pages, len(markers))
            else:
                formats["text"] += 1
                per_page_files += 1
        else:
            formats[suffix.lstrip(".")] += 1
            per_page_files += 1
    if per_page_files and not pages:
        pages = per_page_files
    fmt = " + ".join(f"{k} x{v}" if v > 1 else k for k, v in sorted(formats.items()))
    return fmt, (pages or None), keys_by_format


def build(raw_root: Path, ocr_root: Path, truths: dict[str, ChartTruth], key_regex: str,
          pipeline: Optional[dict[str, dict[int, dict]]] = None) -> tuple[list[DocInventory], dict]:
    images, ocr = source_files(raw_root, ocr_root, key_regex)
    pipeline = pipeline or {}
    keys = sorted(set(images) | set(ocr) | set(truths) | set(pipeline))
    docs: list[DocInventory] = []
    example_keys: dict[str, list[str]] = {}
    for key in keys:
        d = DocInventory(doc=key)
        imgs = images.get(key, [])
        d.image_files = len(imgs)
        d.image_types = " ".join(f"{k}:{v}" for k, v in sorted(Counter(p.suffix.lower() for p in imgs).items()))
        image_ids = {int(p.stem) for p in imgs if p.stem.isdigit()}
        d.image_numeric_ids = len(image_ids)
        if any(p.suffix.lower() == ".pdf" for p in imgs):
            d.notes.append("raw document is a PDF: pages not counted")
        files = ocr.get(key, [])
        d.ocr_files = len(files)
        d.ocr_types = " ".join(f"{k}:{v}" for k, v in sorted(Counter(p.suffix.lower() for p in files).items()))
        if files:
            d.ocr_format, d.ocr_pages, seen = inspect_ocr(files)
            for fmt, ks in seen.items():
                example_keys.setdefault(fmt, ks)
        rows = pipeline.get(key)
        if rows:
            d.in_pipeline = True
            d.pipeline_pages = len(rows)
            d.pipeline_set_aside = sum(1 for r in rows.values() if pipeline_csv.group(r) != "main")
            d.pipeline_documents = len({r.get("document_seq") for r in rows.values() if r.get("document_seq")})
        truth = truths.get(key)
        if truth is not None:
            d.in_truth = True
            d.truth_pages = len(truth.labels)
            d.truth_main = len(truth.main_order)
            d.truth_junk = len(truth.junk_pages)
            d.truth_unlabelled = len(truth.unlabelled)
            d.truth_duplicates_numbered = sum(
                1 for p, v in truth.labels.items()
                if v and v.isdigit() and str(truth.extra.get(p, {}).get("isduplicate", "")).lower().startswith(("y", "t", "1"))
            )
            d.truth_problems = "; ".join(truth.problems())
            if not d.truth_problems:
                d.truth_in_scan_order = truth.main_order == sorted(truth.main_order)
        counts = [d.image_files if imgs and not d.notes else None, d.ocr_pages,
                  d.pipeline_pages or None, d.truth_pages or None]
        d.page_counts = " / ".join("-" if c is None else str(c) for c in counts)
        present = [c for c in counts if c]
        if len(present) >= 2:
            d.pages_agree = len(set(present)) == 1
        d.usable = d.has_images and d.has_ocr and d.in_pipeline and d.in_truth
        docs.append(d)
    return docs, _summary(docs, truths, example_keys, raw_root, ocr_root)


def _summary(docs: list[DocInventory], truths: dict[str, ChartTruth], example_keys: dict, raw_root, ocr_root) -> dict:
    by = {d.doc: d for d in docs}
    sets = {
        "images": {d.doc for d in docs if d.has_images},
        "ocr": {d.doc for d in docs if d.has_ocr},
        "pipeline": {d.doc for d in docs if d.in_pipeline},
        "truth": {d.doc for d in docs if d.in_truth},
    }
    usable = sorted(d.doc for d in docs if d.usable)
    clean = [n for n in usable if not by[n].truth_problems]
    all_four = set(usable)
    missing_from = {
        f"in_{a}_but_no_{b}": len(sets[a] - sets[b])
        for a in ("pipeline", "truth") for b in ("images", "ocr", "pipeline", "truth") if a != b
    }
    return {
        "raw_docs_root": str(raw_root),
        "ocr_docs_root": str(ocr_root),
        "documents_with_raw_images": len(sets["images"]),
        "documents_with_ocr": len(sets["ocr"]),
        "documents_in_pipeline_csv": len(sets["pipeline"]),
        "documents_in_truth_csv": len(sets["truth"]),
        "documents_with_images_and_ocr": len(sets["images"] & sets["ocr"]),
        "USABLE (all four)": len(usable),
        "usable_without_truth_label_problems": len(clean),
        **missing_from,
        "usable_where_page_counts_agree": sum(1 for n in usable if by[n].pages_agree),
        "usable_where_page_counts_differ": sum(1 for n in usable if by[n].pages_agree is False),
        "truth_pages (usable)": sum(by[n].truth_pages for n in usable),
        "truth_main_pages (usable)": sum(by[n].truth_main for n in usable),
        "truth_set_aside_pages (usable)": sum(by[n].truth_junk for n in usable),
        "truth_duplicates_with_a_main_number (usable)": sum(by[n].truth_duplicates_numbered for n in usable),
        "pipeline_set_aside_pages (usable)": sum(by[n].pipeline_set_aside for n in usable),
        "usable_already_in_scan_order": sum(1 for n in clean if by[n].truth_in_scan_order),
        "ocr_formats (documents)": dict(Counter(d.ocr_format for d in docs if d.has_ocr).most_common()),
        "ocr_format_example_keys": example_keys,
        "image_types (documents)": dict(Counter(d.image_types for d in docs if d.has_images).most_common(8)),
        "examples": {
            "truth_but_not_usable": sorted(sets["truth"] - all_four)[:10],
            "images_and_ocr_but_not_in_truth": sorted((sets["images"] & sets["ocr"]) - sets["truth"])[:10],
            "page_counts_differ": [f"{n} ({by[n].page_counts})" for n in usable if by[n].pages_agree is False][:10],
            "truth_label_problems": [f"{n}: {by[n].truth_problems}" for n in usable if by[n].truth_problems][:10],
        },
        "usable_documents": usable,
    }


def rows(docs: list[DocInventory]) -> list[dict]:
    out = []
    for d in docs:
        row = asdict(d)
        row["notes"] = "; ".join(d.notes)
        row["has_images"], row["has_ocr"] = d.has_images, d.has_ocr
        out.append(row)
    return out


def usable_documents(truths: dict[str, ChartTruth], key_regex: str, raw_root: Path, ocr_root: Path,
                     pipeline: dict[str, dict[int, dict]]) -> list[str]:
    images, ocr = source_files(raw_root, ocr_root, key_regex)
    return sorted(set(images) & set(ocr) & set(pipeline) & set(truths))


def write_usable(path: Path, names: Iterable[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(names) + "\n", encoding="utf-8")
