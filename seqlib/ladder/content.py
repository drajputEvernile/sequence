"""What a page's own text says about its place, beyond page numbers (v9, group ``content``).

* **Section order** -- a visit note runs chief complaint -> history -> past history / medications
  -> review of systems -> vitals / exam -> results -> assessment / plan -> signature. The last
  section heading on page a and the first on page b: b opening on an earlier section than a ended
  on is unlikely to follow it. (Section list after model-repo's sequencing order clues.)
* **Numbered items** -- a list ending at "14." on a and going on at "15." on b.
* **Encounter id / date** in the running header ("Encounter Id: 2033825", "Visit Date: 03/14/2024"):
  the same id on both pages, or a different one.
* **"Continued"** at the top of b ("(continued)", "continued from previous page", "cont'd").
* **A table carried over** -- a's last lines and b's first lines are both table rows (several
  tokens, many of them numbers / units).
* **Chart pattern** (group ``pattern``) -- whether the chart runs forwards or backwards in time:
  the share of dated scan neighbours whose dates go up minus the share going down. Most of a
  shuffled chart is still in place, so its trend is still visible; a pair stepping along the
  trend fits, one stepping against a strong trend does not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

SECTIONS = (
    (1, r"chief\s+complaint|reason\s+for\s+(?:visit|consult|exam|referral)|\bCC\b|indication"),
    (2, r"history\s+of\s+present\s+illness|\bHPI\b|interval\s+history|clinical\s+history|subjective"),
    (3, r"past\s+medical|past\s+surgical|family\s+history|social\s+history|medications?\b|allergies|problem\s+list"),
    (4, r"review\s+of\s+systems|\bROS\b"),
    (5, r"vital\s*signs?|vitals|physical\s+exam|examination|objective|technique|comparison"),
    (6, r"lab(?:s|oratory)?\s+results?|results|data\s+reviewed|findings|imaging"),
    (7, r"assessment|impression|diagnos[ie]s|\bplan\b|recommendations?|conclusion"),
    (8, r"electronically\s+signed|signed\s+by|attestation|follow[\s-]?up|orders\s+placed|disposition"),
)
_SECTIONS = [(rank, re.compile(p, re.I)) for rank, p in SECTIONS]
_ITEM = re.compile(r"^\s*(\d{1,3})[.)]\s+\S")
_ENC_ID = re.compile(r"(?i)\b(?:encounter|visit|enc|csn|fin|account|acct)\s*(?:#|no\.?|number|num|id)\s*[:#]?\s*"
                     r"([a-z0-9][a-z0-9\-]{3,})\b")
_DATE = r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})"
_ENC_DATE = re.compile(r"(?i)\b(?:encounter|visit|service|appointment|appt|admission|admit)\s*(?:date|dt)\s*"
                       r"(?:of\s+service)?\s*[:#]?\s*" + _DATE)
_CONT_FROM = re.compile(r"(?i)\(\s*cont(?:inued|'d|d)?\.?\s*\)|continued\s+from|\bcont(?:inued|'d)\s*$|^\s*continued\b")
_NUMERIC = re.compile(r"\d")
HEAD, TAIL = 4, 4  # lines read for the running header, "continued", tables


@dataclass
class PageContent:
    sec_first: Optional[int] = None
    sec_last: Optional[int] = None
    item_first: Optional[int] = None
    item_last: Optional[int] = None
    enc_id: Optional[str] = None
    enc_date: Optional[str] = None
    cont_from: bool = False
    table_head: bool = False
    table_tail: bool = False


def _lines(text: str) -> list[str]:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def _iso(m: re.Match) -> Optional[str]:
    mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    y = y + 2000 if y < 50 else y + 1900 if y < 100 else y
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def _table_row(line: str) -> bool:
    tokens = line.split()
    return len(tokens) >= 3 and sum(bool(_NUMERIC.search(t)) for t in tokens) / len(tokens) >= 0.4


def read_content(text: str) -> PageContent:
    lines = _lines(text)
    c = PageContent()
    ranks = []
    for line in lines:
        if len(line) > 70:
            continue
        for rank, rx in _SECTIONS:
            if rx.search(line):
                ranks.append(rank)
                break
    if ranks:
        c.sec_first, c.sec_last = ranks[0], ranks[-1]
    items = [int(m.group(1)) for line in lines for m in [_ITEM.match(line)] if m]
    if items:
        c.item_first, c.item_last = items[0], items[-1]
    band = lines[:HEAD] + lines[-TAIL:] if len(lines) > HEAD + TAIL else lines
    for line in band:
        if c.enc_id is None:
            m = _ENC_ID.search(line)
            if m:
                c.enc_id = m.group(1).lower()
        if c.enc_date is None:
            m = _ENC_DATE.search(line)
            if m:
                c.enc_date = _iso(m)
    c.cont_from = any(_CONT_FROM.search(line) for line in lines[:HEAD])
    body = lines[HEAD:-TAIL] if len(lines) > HEAD + TAIL + 2 else lines
    c.table_head = sum(_table_row(l) for l in body[:3]) >= 2
    c.table_tail = sum(_table_row(l) for l in body[-3:]) >= 2
    return c


def page_contents(texts: dict[int, str]) -> dict[int, PageContent]:
    return {page: read_content(text) for page, text in texts.items()}


def item_successors(contents: dict[int, PageContent]) -> list[tuple[int, int]]:
    """(a, b) where b's numbered list goes on from a's (a ends at k, b starts at k + 1, k >= 2)."""
    starts: dict[int, list[int]] = {}
    for page, c in contents.items():
        if c.item_first is not None:
            starts.setdefault(c.item_first, []).append(page)
    return sorted((a, b) for a, c in contents.items() if c.item_last is not None and c.item_last >= 2
                  for b in starts.get(c.item_last + 1, []) if b != a)


def content_pair(a: Optional[PageContent], b: Optional[PageContent]) -> dict[str, float]:
    if a is None or b is None:
        return {}
    x: dict[str, float] = {"ct_b_cont_from": float(b.cont_from), "ct_table_join": float(a.table_tail and b.table_head)}
    if a.sec_last is not None:
        x["ct_sec_a_last"] = float(a.sec_last)
    if b.sec_first is not None:
        x["ct_sec_b_first"] = float(b.sec_first)
    if a.sec_last is not None and b.sec_first is not None:
        x["ct_sec_delta"] = float(b.sec_first - a.sec_last)
    if a.item_last is not None and b.item_first is not None:
        x["ct_item_delta"] = float(max(-20, min(20, b.item_first - a.item_last)))
        x["ct_item_next"] = float(b.item_first == a.item_last + 1)
    if a.enc_id and b.enc_id:
        x["ct_enc_same"] = float(a.enc_id == b.enc_id)
    if a.enc_date and b.enc_date:
        x["ct_encdate_same"] = float(a.enc_date == b.enc_date)
    return x


def date_trend(dos: dict[int, str], scan: Sequence[int]) -> Optional[float]:
    """(up - down) / (up + down) over scan neighbours among the dated pages; None with no change."""
    dated = [dos[p] for p in scan if dos.get(p)]
    up = sum(b > a for a, b in zip(dated, dated[1:]))
    down = sum(b < a for a, b in zip(dated, dated[1:]))
    return (up - down) / (up + down) if up + down else None


def pattern_pair(da: Optional[str], db: Optional[str], trend: Optional[float]) -> dict[str, float]:
    if trend is None:
        return {}
    x = {"pt_trend": float(trend)}
    if da and db:
        x["pt_with_trend"] = 0.0 if da == db else float(trend if db > da else -trend)
    return x
