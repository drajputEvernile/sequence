"""Extra page facts for v4 to v6, computed once per chart and cached in the chart folder.

* ``dates``  -- each page's own date of service, read from its OCR text with the pipeline's DOS
               rules (seqlib/extract/dos_logic.py: DOS labels at the page's edges, admit /
               discharge, dictation dates). The outputs CSV mostly carries the *document's* date,
               which depends on the page order, so it is not used (design doc FR14).
* ``banner`` -- fax / print banners (design doc 7.1.1): "P.004/012", "Page 4 of 12" on a line with
               a time stamp, "4/12" next to a fax number, in the page's top or bottom lines. The
               counter, the total and an identity (the line with its counter masked) -- pages with
               the same identity came in the same transmission.
* ``hash``   -- a 64-bit difference hash of the top 12% of the page image (design doc 7.1.2):
               the same letterhead / form header gives near-identical hashes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from seqlib.io.chart import Chart

logger = logging.getLogger(__name__)

CACHE_FILE = "ladder_facts.json"
CACHE_VERSION = 1

_FAX_COUNTER = re.compile(r"\bP(?:AGE|G)?\.?\s*0*(\d{1,4})\s*/\s*0*(\d{1,4})\b", re.I)
_PAGE_OF = re.compile(r"\bpage\s+0*(\d{1,4})\s+of\s+0*(\d{1,4})\b", re.I)
_SLASH = re.compile(r"(?<![\d/])0*(\d{1,4})\s*/\s*0*(\d{1,4})(?![\d/])")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?\b", re.I)
_PHONE = re.compile(r"\(?\b\d{3}\)?[-\s.]\d{3}[-.]\d{4}\b")
_FAX_WORD = re.compile(r"\b(?:fax|from|to|sent|received|transmit)", re.I)


@dataclass
class Banner:
    counter: int
    total: Optional[int]
    identity: str


@dataclass
class ChartFacts:
    dos: dict[int, str] = field(default_factory=dict)  # page number -> ISO date read from the page
    dos_source: dict[int, str] = field(default_factory=dict)
    banner: dict[int, Banner] = field(default_factory=dict)
    header_hash: dict[int, int] = field(default_factory=dict)


def _band_lines(page) -> list[str]:
    """The page's top three and bottom three text lines (banners sit there)."""
    lines = [l.strip() for l in (page.text or "").splitlines() if l.strip()]
    return lines[:3] + lines[-3:] if len(lines) > 6 else lines


def parse_banner(lines: Sequence[str]) -> Optional[Banner]:
    for line in lines:
        timed = bool(_TIME.search(line))
        faxy = bool(_PHONE.search(line) or _FAX_WORD.search(line))
        match = _FAX_COUNTER.search(line)
        if match is None and timed:
            match = _PAGE_OF.search(line)
        if match is None and (faxy or timed):
            match = _SLASH.search(line)
            # a date (03/14) is not a counter: needs a fax / time context and counter <= total
        if match is None or not (timed or faxy or match.re is _FAX_COUNTER):
            continue
        n, m = int(match.group(1)), int(match.group(2))
        if not (1 <= n <= m <= 999):
            continue
        identity = line[: match.start()] + "#" + line[match.end():]
        identity = re.sub(r"\d", "0", re.sub(r"\s+", " ", identity.lower())).strip()
        return Banner(n, m, hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12])
    return None


def header_dhash(image_path: Path, top: float = 0.12) -> Optional[int]:
    from PIL import Image

    try:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            w, h = gray.size
            crop = gray.crop((0, 0, w, max(1, int(h * top))))
            ink = sum(1 for v in crop.resize((64, 16)).tobytes() if v < 128) / (64 * 16)
            if ink < 0.005:
                return None
            small = crop.resize((9, 8))
            px = list(small.tobytes())
    except (OSError, ValueError):
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def hash_similarity(a: Optional[int], b: Optional[int]) -> Optional[float]:
    if a is None or b is None:
        return None
    return 1.0 - bin(a ^ b).count("1") / 64.0


def _image_paths(chart: Chart) -> dict[str, Path]:
    source = Path(chart.root) / "source.json"
    out: dict[str, Path] = {}
    if source.is_file():
        data = json.loads(source.read_text(encoding="utf-8"))
        out = {k: Path(v) for k, v in (data.get("images") or {}).items() if v}
    pages_dir = Path(chart.root) / "pages"
    for page in chart.pages:
        if page.page_name not in out and (pages_dir / page.page_name).is_file():
            out[page.page_name] = pages_dir / page.page_name
    return out


def chart_facts(chart: Chart, groups: Sequence[str]) -> ChartFacts:
    """The facts ``groups`` need ("dates", "banner", "hash"), from the cache when it has them."""
    cache_path = Path(chart.root) / CACHE_FILE
    cache: dict = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache.get("version") != CACHE_VERSION:
                cache = {}
        except ValueError:
            cache = {}
    changed = False
    facts = ChartFacts()
    if "dates" in groups:
        if "dates" not in cache:
            from seqlib.extract.dos_logic import detect_dos_per_page

            text = "\n".join(f"===== {p.page_name} =====\n{(p.text or '').rstrip()}\n" for p in chart.pages)
            rows = detect_dos_per_page(text) if chart.pages else []
            cache["dates"] = {str(r.get("page_name")): [r.get("dos_from_iso") or "", r.get("match_type") or ""]
                              for r in rows}
            changed = True
        by_name = {p.page_name: p.page_number for p in chart.pages}
        for name, (iso, source) in cache["dates"].items():
            if iso and name in by_name:
                facts.dos[by_name[name]] = iso
                facts.dos_source[by_name[name]] = source
    if "banner" in groups:
        if "banner" not in cache:
            cache["banner"] = {}
            for p in chart.pages:
                b = parse_banner(_band_lines(p))
                if b is not None:
                    cache["banner"][p.page_name] = [b.counter, b.total, b.identity]
            changed = True
        by_name = {p.page_name: p.page_number for p in chart.pages}
        for name, (n, m, ident) in cache["banner"].items():
            if name in by_name:
                facts.banner[by_name[name]] = Banner(n, m, ident)
    if "hash" in groups:
        if "hash" not in cache:
            paths = _image_paths(chart)
            cache["hash"] = {}
            for p in chart.pages:
                h = header_dhash(paths[p.page_name]) if p.page_name in paths else None
                if h is not None:
                    cache["hash"][p.page_name] = str(h)
            changed = True
        by_name = {p.page_name: p.page_number for p in chart.pages}
        for name, h in cache["hash"].items():
            if name in by_name:
                facts.header_hash[by_name[name]] = int(h)
    if changed:
        cache["version"] = CACHE_VERSION
        try:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError as exc:
            logger.warning("%s: facts cache not written (%s)", chart.name, exc)
    return facts
