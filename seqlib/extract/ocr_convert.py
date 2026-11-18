"""A document's existing OCR, whatever its format, as the pipeline's ``final2`` pages.

The training machine has the raw page images and *their OCR*; the sequencers read the
pipeline's file shape (``ocr/<doc>_final2.json``: ``pages[]`` with
``fileName``, ``content`` and ``pagesMeta[0]`` holding ``lines[]`` and ``words[]``, each with a
``polygon``). This converts, without re-running any OCR:

* the pipeline's own files (``*_final2.json``, ``*_final1.json``, ``*_prelim.txt``): kept as is;
* Azure Document Intelligence output saved to disk (``analyzeResult.pages[]`` with ``lines`` /
  ``words``; the older ``readResults[].lines[]`` with ``boundingBox``): one file for the
  document or one per page;
* any other JSON with ``pages[]`` / ``lines[]`` of ``content`` / ``text`` + ``polygon`` /
  ``boundingBox``;
* one plain-text file per page (text only, no positions).

Page numbers: a page's place in a multi-page file (``pageNumber``), else the number in a
per-page file's name (after the document key), matched to the image of the same number. When a
format is not recognised the error names its top-level keys -- send those, not the file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from seqlib.io.chart import page_number_of
from seqlib.io.ocr_files import read_prelim


class UnknownOcrFormat(ValueError):
    pass


@dataclass
class ConvertedOcr:
    final2: dict[int, dict[str, Any]] = field(default_factory=dict)  # page number -> page dict
    final1: dict[int, dict[str, Any]] = field(default_factory=dict)  # pipeline final1, kept as is
    prelim: dict[int, str] = field(default_factory=dict)
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def pages(self) -> set[int]:
        return set(self.final2) | set(self.final1) | set(self.prelim)


def _flat_polygon(value: Any) -> Optional[list[float]]:
    if not value:
        return None
    if isinstance(value, list) and value and isinstance(value[0], dict):
        out: list[float] = []
        for point in value:
            out += [float(point.get("x", 0)), float(point.get("y", 0))]
        return out
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _boxes(items: list[dict]) -> list[dict[str, Any]]:
    """Lines or words as ``{content, polygon}`` (Azure v3/v4 ``content`` + ``polygon``, older
    ``text`` + ``boundingBox``)."""
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("content") if item.get("content") is not None else item.get("text")
        text = str(raw or "").strip()
        polygon = _flat_polygon(item.get("polygon") or item.get("boundingBox") or item.get("bounding_box"))
        if text:
            out.append({"content": text, "polygon": polygon or []})
    return out


def _page_from_lines(number: int, lines: list[dict], width: Any, height: Any, unit: Any,
                     file_name: str, content: Optional[str] = None,
                     words: Optional[list[dict]] = None) -> dict[str, Any]:
    out_lines = _boxes(lines)
    text = content if content is not None else "\n".join(l["content"] for l in out_lines)
    # Word boxes kept as the pipeline keeps them (pagesMeta[].words), for word-level features.
    out_words = [w for w in _boxes(words or []) if len(w["polygon"]) >= 8]
    meta = {"pageNumber": 1, "width": width, "height": height, "unit": unit,
            "lines": [l for l in out_lines if l["polygon"]], "words": out_words}
    return {"fileName": file_name, "pageNumber": number, "content": text, "pagesMeta": [meta] if height else []}


def _number_from_name(path: Path, doc_key: str) -> Optional[int]:
    stem = path.stem
    if doc_key and stem.startswith(doc_key):
        stem = stem[len(doc_key):]
    found = re.findall(r"\d+", stem)
    return int(found[-1]) if found else None


def _azure_pages(data: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    """Azure-shaped page list (v3/v4 analyzeResult.pages, v2/v3 readResults), else None."""
    result = data.get("analyzeResult") if isinstance(data.get("analyzeResult"), dict) else data
    for key in ("pages", "readResults"):
        pages = result.get(key)
        if isinstance(pages, list) and pages and isinstance(pages[0], dict) and (
            "lines" in pages[0] or "words" in pages[0]
        ):
            return pages
    return None


def _lines_from_words(page: dict[str, Any]) -> list[dict]:
    """A page with words but no lines: one line per word (positions still usable)."""
    return [{"content": w.get("content") or w.get("text"), "polygon": w.get("polygon") or w.get("boundingBox")}
            for w in page.get("words") or [] if isinstance(w, dict)]


def _words_of(page: dict[str, Any]) -> list[dict]:
    """Azure v3/v4 keep words on the page; v2.1 / read v3 keep them inside each line."""
    if page.get("words"):
        return page["words"]
    return [w for line in page.get("lines") or [] if isinstance(line, dict) for w in line.get("words") or []]


def convert_document(ocr_files: list[Path], doc_key: str, image_names: dict[int, str]) -> ConvertedOcr:
    """``image_names``: page number -> the raw image's file name (used as fileName)."""
    out = ConvertedOcr()
    used: list[str] = []

    def name_for(n: int) -> str:
        return image_names.get(n) or f"{n}.jpg"

    def keys_of(data: Any) -> str:
        return ", ".join(sorted(data)[:12]) if isinstance(data, dict) else type(data).__name__

    for path in sorted(ocr_files):
        suffix = path.suffix.lower()
        if suffix == ".txt" and re.search(r"_final[12]$", path.stem, re.I):
            # The pipeline's legacy <doc>_final1.txt (its JSON twin is what is read): not a page.
            out.notes.append(f"{path.name}: legacy pipeline text file, skipped (the .json is read)")
            continue
        if suffix == ".txt":
            prelim = read_prelim(path)
            if prelim:
                for name, text in prelim.items():
                    n = page_number_of(name)
                    if n is not None:
                        out.prelim[n] = text
                used.append("pipeline prelim")
                continue
            n = _number_from_name(path, doc_key)
            if n is None:
                out.notes.append(f"{path.name}: text file without a page number in its name, skipped")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            out.final2.setdefault(n, {"fileName": name_for(n), "pageNumber": n, "content": text, "pagesMeta": []})
            used.append("text per page")
            continue
        if suffix != ".json":
            out.notes.append(f"{path.name}: {suffix} not read")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        except ValueError:
            out.notes.append(f"{path.name}: unreadable JSON, skipped")
            continue
        if not isinstance(data, dict):
            raise UnknownOcrFormat(f"{path.name}: top level is a {type(data).__name__}")
        pages = data.get("pages")
        # 1. the pipeline's own final2 / final1
        if isinstance(pages, list) and pages and isinstance(pages[0], dict) and "fileName" in pages[0]:
            target = out.final1 if "words" in pages[0] and "pagesMeta" not in pages[0] else out.final2
            for page in pages:
                n = page_number_of(str(page.get("fileName") or "")) or page.get("pageNumber")
                if n:
                    target[int(n)] = page
            used.append("pipeline final1" if target is out.final1 else "pipeline final2")
            continue
        # 2. Azure-shaped: a whole document or one page
        az = _azure_pages(data)
        if az is not None:
            n_from_name = _number_from_name(path, doc_key)
            single = len(az) == 1 and n_from_name is not None and path.stem != doc_key
            for k, page in enumerate(az, start=1):
                n = n_from_name if single else int(page.get("pageNumber") or page.get("page") or k)
                lines = page.get("lines") or _lines_from_words(page)
                out.final2[n] = _page_from_lines(n, lines, page.get("width"), page.get("height"),
                                                 page.get("unit"), name_for(n), words=_words_of(page))
            used.append("azure (per page)" if single else "azure (document)")
            continue
        # 3. one page: lines / content at the top level
        if "lines" in data or "content" in data or "text" in data:
            n = _number_from_name(path, doc_key)
            if n is None:
                raise UnknownOcrFormat(f"{path.name}: one page, but no page number in the file name")
            content = data.get("content") if isinstance(data.get("content"), str) else data.get("text")
            out.final2[n] = _page_from_lines(n, data.get("lines") or [], data.get("width"), data.get("height"),
                                             data.get("unit"), name_for(n),
                                             content if isinstance(content, str) else None,
                                             words=_words_of(data))
            used.append("single-page json")
            continue
        raise UnknownOcrFormat(f"{path.name}: unknown OCR JSON (top-level keys: {keys_of(data)})")
    out.source = ", ".join(sorted(set(used)))
    return out
