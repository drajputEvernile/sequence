"""One document's OCR into the workspace, whatever its format (ocr_convert), with the raw image
each page came from:

    <RAW_DOCS_ROOT>/<RecordID_DocumentID>/1.png ...
    <OCR_DOCS_ROOT>/<RecordID_DocumentID>/...ocr...
        ->  <CHARTS_ROOT>/<RecordID_DocumentID>/ocr/<doc>_final2.json   (+ source.json)

Everything else the sequencers need comes from the pipeline outputs CSV.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from seqlib.eval.inventory import IMAGE_SUFFIXES, OCR_SUFFIXES
from seqlib.settings import cfg, input_path

from .ocr_convert import convert_document

SOURCES_FILE = "source.json"  # page file name -> raw image path


def _natural(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def image_numbers(images: list[Path]) -> dict[int, Path]:
    """page number -> image: the file stem when every stem is a number (1.png ...), else the
    natural name order (IMG_2 before IMG_10) numbered from 1."""
    images = [p for p in images if p.suffix.lower() in IMAGE_SUFFIXES and p.suffix.lower() != ".pdf"]
    if images and all(p.stem.isdigit() for p in images):
        return {int(p.stem): p for p in images}
    return {i: p for i, p in enumerate(sorted(images, key=_natural), start=1)}


def import_document(key: str, images: list[Path], ocr_files: list[Path]) -> dict[str, Any]:
    chart_dir = Path(cfg.CHARTS_ROOT) / key
    numbered = image_numbers(images)
    converted = convert_document([p for p in ocr_files if p.suffix.lower() in OCR_SUFFIXES], key,
                                 {n: p.name for n, p in numbered.items()})
    if not converted.pages:
        raise ValueError(f"{key}: no OCR page read ({'; '.join(converted.notes) or 'no OCR files'})")
    (chart_dir / "ocr").mkdir(parents=True, exist_ok=True)
    for kind, pages in (("ocr_final2", converted.final2), ("ocr_final1", converted.final1)):
        path = input_path(chart_dir, kind)
        if pages:
            doc = {"recordId": key, "model": converted.source, "pageCount": len(pages),
                   "pages": [pages[n] for n in sorted(pages)]}
            path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        elif path.exists():
            path.unlink()
    if converted.prelim:
        names = {n: (converted.final2.get(n) or {}).get("fileName") or f"{n}.png" for n in converted.prelim}
        input_path(chart_dir, "ocr_prelim").write_text(
            "\n".join(f"===== {names[n]} =====\n{converted.prelim[n]}\n" for n in sorted(converted.prelim)),
            encoding="utf-8")
    sources = {}
    for n in converted.pages:
        page = converted.final2.get(n) or converted.final1.get(n) or {}
        sources[str(page.get("fileName") or f"{n}.png")] = str(numbered[n]) if n in numbered else ""
    (chart_dir / SOURCES_FILE).write_text(json.dumps(
        {"doc": key, "ocr_source": converted.source, "images": sources, "notes": converted.notes}, indent=1),
        encoding="utf-8")
    return {"ocr_pages": len(converted.pages), "images": len(numbered), "ocr_format": converted.source,
            "pages_without_image": len(set(converted.pages) - set(numbered)), "notes": converted.notes}
