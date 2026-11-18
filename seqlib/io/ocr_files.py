"""The pipeline's three OCR files, read from disk.

* ``<chart>_prelim.txt``  -- Tesseract text, one ``===== 12.jpg =====`` block per page.
* ``<chart>_final1.json`` -- Docling: ``pages[]`` with ``fileName``, ``content``, ``height`` and
  ``words[]`` (``content`` + ``polygon``), plus ``section_headers`` when the pipeline added them.
* ``<chart>_final2.json`` -- Azure Read output the pipeline already saved: ``pages[]`` with
  ``fileName``, ``content`` and ``pagesMeta[0]`` (``height``, ``lines[]`` with ``polygon``).

Nothing here calls an OCR engine. (The local engine for photographed pages is
``seqlib.local_ocr``; it writes the same ``final2`` shape.)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PAGE_MARKER_RE = re.compile(r"^=====\s*(.+?)\s*=====\s*$", re.MULTILINE)


def read_prelim(path: Path) -> dict[str, str]:
    """``{page file name: text}``; empty when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return {}
    parts = PAGE_MARKER_RE.split(path.read_text(encoding="utf-8", errors="replace"))
    out: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].strip()] = parts[i + 1].strip()
    return out


def read_ocr_json(path: Path) -> dict[str, dict[str, Any]]:
    """``{page file name: page dict}`` from a final1 / final2 JSON; empty when absent or broken."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable OCR JSON %s: %s", path, exc)
        return {}
    out: dict[str, dict[str, Any]] = {}
    for page in data.get("pages") or []:
        if not isinstance(page, dict):
            continue
        name = str(page.get("fileName") or page.get("filename") or f"{page.get('pageNumber')}.jpg")
        out[name] = page
    return out


def page_content(page: dict[str, Any] | None) -> str:
    """A final1 / final2 page's text (``content``, else ``markdown``); a JSON string in it is
    unwrapped the way the pipeline's ``unwrap_ocr_content`` does."""
    if not page:
        return ""
    text = page.get("content")
    if text is None:
        text = page.get("markdown") or ""
    text = str(text or "").strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return str(parsed.get("content") or parsed.get("markdown") or "")
        except ValueError:
            pass
    return text
