"""Why are a document's pages out of scan order? Candidate rules, tested against the truth.

Every rule builds an order of the truth's Main pages from the pipeline's values (no truth):

* ``scan``            -- scan order (what every method does today on this data);
* ``dos_asc``         -- pages sorted by date of service, oldest first (an undated page takes the
                         date of the dated page before it; ties keep scan order);
* ``dos_desc``        -- the same, newest first;
* ``docs_dos_asc``    -- the pipeline's documents (document_seq blocks) moved as whole blocks,
                         oldest DOS first, pages inside a block in scan order;
* ``docs_dos_desc``   -- the same, newest first;
* ``docs_reversed``   -- the pipeline's documents in reverse scan order (a stack flipped);
* ``printed_numbers`` -- v1's engines re-run (they act on printed "page n of m").

For each document the score of each rule (share of pages at the right place) says which one
explains it; over all documents, how many it fixes and how many correct ones it would break.
Nothing here is page text: dates become D1, D2 ... in date order within the document.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from seqlib.eval.analyze import runs
from seqlib.io import pipeline_csv
from seqlib.io.chart import Chart
from seqlib.io.ground_truth import ChartTruth
from seqlib.v1.sequencing.engine.marker_detector import detect_marker

Rule = Callable[[Chart, list[int]], list[int]]


def _dates(chart: Chart, pages: list[int]) -> dict[int, str]:
    """page -> ISO date; an undated page takes the previous dated page's date ('' before any)."""
    by = chart.by_number()
    out, last = {}, ""
    for p in pages:
        d = pipeline_csv.iso_date(by[p].pipeline.get("dos_from")) if by.get(p) and by[p].pipeline else ""
        last = d or last
        out[p] = last
    return out


def _documents(chart: Chart, pages: list[int]) -> list[list[int]]:
    """The pipeline's documents as consecutive blocks (a new block when document_seq changes)."""
    by = chart.by_number()
    blocks: list[list[int]] = []
    last = object()
    for p in pages:
        d = by[p].pipeline.get("document_seq") if by.get(p) and by[p].pipeline else None
        if not blocks or d != last:
            blocks.append([])
        blocks[-1].append(p)
        last = d
    return blocks


def rule_scan(chart: Chart, pages: list[int]) -> list[int]:
    return list(pages)


def rule_dos(desc: bool) -> Rule:
    def rule(chart: Chart, pages: list[int]) -> list[int]:
        dates = _dates(chart, pages)
        dated = [p for p in pages if dates[p]]
        undated_head = [p for p in pages if not dates[p]]
        ordered = sorted(dated, key=lambda p: dates[p], reverse=desc)
        if desc:  # stable within a date: keep scan order inside each date
            groups: dict[str, list[int]] = {}
            for p in dated:
                groups.setdefault(dates[p], []).append(p)
            ordered = [p for d in sorted(groups, reverse=True) for p in groups[d]]
        return undated_head + ordered
    return rule


def rule_docs_dos(desc: bool) -> Rule:
    def rule(chart: Chart, pages: list[int]) -> list[int]:
        dates = _dates(chart, pages)
        blocks = _documents(chart, pages)
        keyed = [(next((dates[p] for p in b if dates[p]), ""), i, b) for i, b in enumerate(blocks)]
        undated = [b for d, _, b in keyed if not d]
        dated = [(d, i, b) for d, i, b in keyed if d]
        dated.sort(key=lambda t: (t[0], t[1]) if not desc else (t[0], -t[1]), reverse=desc)
        return [p for b in undated for p in b] + [p for _, _, b in dated for p in b]
    return rule


def rule_docs_reversed(chart: Chart, pages: list[int]) -> list[int]:
    return [p for b in reversed(_documents(chart, pages)) for p in b]


def rule_printed(chart: Chart, pages: list[int]) -> list[int]:
    from seqlib.v1.runner import run_v1

    keep = set(pages)
    for page in chart.pages:
        page.group_override = "main" if page.page_number in keep else "junk"
    order = [p for p in run_v1(chart).main_order if p in keep]
    for page in chart.pages:
        page.group_override = None
    return order


RULES: dict[str, Rule] = {
    "scan": rule_scan,
    "dos_asc": rule_dos(False),
    "dos_desc": rule_dos(True),
    "docs_dos_asc": rule_docs_dos(False),
    "docs_dos_desc": rule_docs_dos(True),
    "docs_reversed": rule_docs_reversed,
    "printed_numbers": rule_printed,
}


def _position(order: list[int], truth: list[int]) -> float:
    return sum(a == b for a, b in zip(order, truth)) / len(truth) if truth else 1.0


@dataclass
class DocExplanation:
    chart: str
    pages: int
    pages_out_of_place: int
    shape: str  # the truth order as runs of scan pages, e.g. "1-11, 30-36, 12-29"
    scores: dict[str, float]
    best: str
    table: list[dict]


def describe_runs(order: list[int]) -> str:
    parts = []
    for r in runs(order):
        parts.append(str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}")
    return ", ".join(parts)


def explain(chart: Chart, truth: ChartTruth, with_table: bool = True) -> DocExplanation:
    want = truth.main_order
    pages = sorted(want)  # the truth's Main pages in scan order
    scores = {}
    for name, rule in RULES.items():
        try:
            scores[name] = round(_position(rule(chart, pages), want), 4)
        except Exception:  # noqa: BLE001 - a rule that cannot run scores nothing
            scores[name] = 0.0
    best = max(scores, key=lambda k: (scores[k], k == "scan"))
    table = []
    if with_table:
        by = chart.by_number()
        dates = _dates(chart, pages)
        tokens = {d: f"D{i}" for i, d in enumerate(sorted({d for d in dates.values() if d}), start=1)}
        rank = {p: i for i, p in enumerate(want, start=1)}
        for i, p in enumerate(pages, start=1):
            page = by.get(p)
            row = page.pipeline if page and page.pipeline else {}
            marker = detect_marker("\n".join((page.text or "").splitlines()[:5] + (page.text or "").splitlines()[-5:])) \
                if page else None
            path = pipeline_csv.type_path(row)
            own_dos = pipeline_csv.iso_date(row.get("dos_from"))
            table.append({
                "page": p, "scan_rank": i, "truth_rank": rank[p], "moved": rank[p] != i,
                "pipeline_doc": row.get("document_seq") or "",
                "own_type": " > ".join(path[:2]), "doc_type": row.get("page_type") or "",
                "type_source": row.get("page_type_source") or "",
                "dos": tokens.get(own_dos, "-") if own_dos else "-",
                "signed": "y" if pipeline_csv.flag(row.get("signature_present")) else "",
                "printed": f"{marker.page_num}/{marker.total_pages or '?'}" if marker else "",
                "words": len((page.text or "").split()) if page else 0,
            })
    return DocExplanation(truth.chart_name, len(want), sum(a != b for a, b in zip(pages, want)),
                          describe_runs(want), scores, best, table)


def rule_effect(explanations_moved: list[DocExplanation], explanations_scan: list[DocExplanation]) -> dict:
    """Per rule: moved documents it fixes completely / improves, scan-order documents it breaks."""
    out = {}
    for name in RULES:
        fixed = sum(1 for e in explanations_moved if e.scores[name] == 1.0)
        better = sum(1 for e in explanations_moved if e.scores[name] > e.scores["scan"])
        broken = sum(1 for e in explanations_scan if e.scores[name] < 1.0)
        out[name] = {"moved_docs_fixed": fixed, "moved_docs_improved": better,
                     "in_order_docs_broken": broken, "in_order_docs": len(explanations_scan)}
    return out
