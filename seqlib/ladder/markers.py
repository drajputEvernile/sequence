"""Page markers: the page number a page prints, in any of the usual forms (v9).

v1 reads only "Page n of m" with m <= 20. Documents also print:

* ``of``    -- "Page 3 of 41", "Pg. 3/41", "page 3 out of 41", or a line that is only "3 of 41";
* ``page``  -- "Page 3", "Pg. 3", "Page: 3", with no total;
* ``bare``  -- a number alone on the page's first or last line: "3", "- 3 -", "[3]", "(3)", "3.".

Fax counters ("P.004/012" beside a fax number or a time) stay in ``facts.banner``.

Each marker has a **style**: its kind, the band it sits in (top / bottom) and its line with the
digits masked ("printed 00/00/0000 page # of #" ...), plus the total. Pages of one document print
their numbers the same way, so numbers are compared only within a style; two documents that both
number "1, 2, 3" in the same style are told apart by the other evidence.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

BAND = 3  # lines at the top and at the bottom of the page where a page number is looked for
MAX_NUMBER = 999
KIND_RANK = {"of": 3, "page": 2, "bare": 1}

_OF = re.compile(r"\b(?:page|pg|pp)\.?\s*[:#]?\s*0*(\d{1,3})\s*(?:of|/|out\s+of)\s*0*(\d{1,3})\b", re.I)
_OF_LINE = re.compile(r"^\W{0,3}0*(\d{1,3})\s+(?:of|out\s+of)\s+0*(\d{1,3})\W{0,3}$", re.I)
_PAGE = re.compile(r"\b(?:page|pg)\.?\s*[:#]?\s*0*(\d{1,3})\b(?!\s*(?:of\b|/|out\b|-\d|\.\d))", re.I)
_BARE = re.compile(r"^[\s\-–—_\[\(\.\*]*0*(\d{1,3})[\s\-–—_\]\)\.\*]*$")


@dataclass(frozen=True)
class Marker:
    number: int
    total: Optional[int]
    kind: str  # "of" | "page" | "bare"
    band: str  # "top" | "bottom"
    style: str  # same style = numbered the same way (kind, band, masked line, total)


def _style(kind: str, band: str, line: str, total: Optional[int]) -> str:
    masked = re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))
    key = f"{kind}|{band}|{masked}|{total if total is not None else ''}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _lines(text: str) -> list[str]:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def read_marker(text: str) -> Optional[Marker]:
    """The page number ``text`` prints, the most explicit form found ("of" before "page" before
    "bare"); None without one."""
    lines = _lines(text)
    if not lines:
        return None
    k = min(BAND, len(lines))
    bands = [("bottom", i, lines[-1 - i]) for i in range(k)] + [("top", i, lines[i]) for i in range(k)]
    found: list[tuple[int, Marker]] = []
    for band, depth, line in bands:
        m = _OF.search(line) or _OF_LINE.match(line)
        if m:
            n, total = int(m.group(1)), int(m.group(2))
            if 1 <= n <= total <= MAX_NUMBER:
                found.append((KIND_RANK["of"], Marker(n, total, "of", band, _style("of", band, line, total))))
                continue
        m = _PAGE.search(line)
        if m and 1 <= int(m.group(1)) <= MAX_NUMBER:
            found.append((KIND_RANK["page"], Marker(int(m.group(1)), None, "page", band,
                                                    _style("page", band, line, None))))
            continue
        if depth == 0 and len(lines) > 1:  # a bare number only as the very first / last line
            m = _BARE.match(line)
            if m and 1 <= int(m.group(1)) <= MAX_NUMBER:
                found.append((KIND_RANK["bare"], Marker(int(m.group(1)), None, "bare", band,
                                                        _style("bare", band, line, None))))
    if not found:
        return None
    return max(found, key=lambda f: f[0])[1]  # the first found of the best kind (bottom band first)


def page_markers(texts: dict[int, str]) -> dict[int, Marker]:
    """page number -> its marker, for the pages that print one."""
    out = {}
    for page, text in texts.items():
        m = read_marker(text)
        if m is not None:
            out[page] = m
    return out


def marker_successors(markers: dict[int, Marker]) -> list[tuple[int, int]]:
    """Every (a, b) where b prints the next number in a's style: candidates, not forced links."""
    by: dict[tuple[str, int], list[int]] = defaultdict(list)
    for page, m in markers.items():
        by[(m.style, m.number)].append(page)
    out = []
    for page, m in markers.items():
        out += [(page, b) for b in by.get((m.style, m.number + 1), []) if b != page]
    return sorted(out)


def marker_anchors(markers: dict[int, Marker], kinds: Sequence[str] = ("of", "page", "bare")) -> list[tuple[int, int]]:
    """Forced links: number n -> n + 1 in one style, when each is printed by exactly one page."""
    by: dict[tuple[str, int], list[int]] = defaultdict(list)
    for page, m in markers.items():
        if m.kind in kinds:
            by[(m.style, m.number)].append(page)
    return sorted((pages[0], nxt[0]) for (style, n), pages in by.items()
                  for nxt in [by.get((style, n + 1), [])] if len(pages) == 1 and len(nxt) == 1)


def marker_pair(a: Optional[Marker], b: Optional[Marker]) -> dict[str, float]:
    """The ``markers`` pair features for a -> b (NaN-free keys only where they apply)."""
    x: dict[str, float] = {}
    if a is not None:
        x["mk_a_last"] = float(a.total is not None and a.number == a.total)
    if b is not None:
        x["mk_b_first"] = float(b.number == 1)
    if a is not None and b is not None:
        x["mk_both"] = 1.0
        x["mk_kind"] = float(min(KIND_RANK[a.kind], KIND_RANK[b.kind]))
        same = a.style == b.style
        x["mk_same_style"] = float(same)
        if same:
            step = b.number - a.number
            x["mk_step"] = float(max(-20, min(20, step)))
            x["mk_follows"] = float(step == 1)
    elif a is not None or b is not None:
        x["mk_both"] = 0.0
    return x
