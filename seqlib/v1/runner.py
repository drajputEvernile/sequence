"""v1 -- the imaging pipeline's page sequencing, as it ran and as it can be re-run here.

Three ways to score v1 (scripts/evaluate.py):

* ``v1_csv``  -- **what production wrote**: the pipeline outputs CSV's ``seq`` column (v1's
  position for each Main page; empty for set-aside pages). This is the v1 baseline.
* ``v1``      -- the ported engines re-run here on the pipeline's own values: its blank / junk
  verdicts and continuity documents (``document_seq``) from the outputs CSV, the text from the
  OCR. Lets v1's rules be changed and re-measured. ``--documents recompute`` re-runs continuity
  too instead of reading ``document_seq``.
* ``scan``    -- scan order with the pipeline's set-aside pages removed (no reordering at all).

The engines (``continuity/engine.py``, ``sequencing/detect.py`` ...) are the pipeline's code,
copied unchanged except for import paths (``SOURCE_COMMIT``).

Where the re-run cannot match production exactly: the pipeline read the printed "page n of m"
from its key/value extraction (not in the outputs CSV), so here the markers come from the text
patterns v1 falls back to; continuity's recompute uses the outputs CSV's document-level values
(page type, DOS) where the pipeline used each page's own. ``v1_csv`` vs ``v1`` shows the gap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from seqlib.io import pipeline_csv
from seqlib.io.chart import Chart, Page
from seqlib.settings import input_path

from .continuity.engine import PageInput, assign, load_rules, text_lines, trust_points
from .continuity.layout import page_layout
from .continuity.signature import has_signature_section, signed_page_names
from .sequencing.detect import compute_sequence_assignments
from .sequencing.engine.types import ExplicitMarker

logger = logging.getLogger(__name__)

EXCLUDE = frozenset({"blank", "junk", "duplicate"})


@dataclass
class OrderResult:
    """Any method's answer for one chart."""

    chart_name: str
    main_order: list[int]  # Main pages (scan page numbers), first to last
    junk_pages: list[int]  # pages the method set aside (J series, scan order)
    method: str = ""
    # page_number -> anything the method wants to report (confidence, why, document ...)
    detail: dict[int, dict[str, Any]] = field(default_factory=dict)

    def labels(self) -> dict[int, str]:
        """page_number -> "1".."N" / "J1".."Jk", the way the pipeline writes actual_sequence."""
        out = {p: str(i) for i, p in enumerate(self.main_order, start=1)}
        out.update({p: f"J{i}" for i, p in enumerate(sorted(self.junk_pages), start=1)})
        return out


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None or str(value).strip() == "" else float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _printed(page: Page) -> Optional[tuple[int, int]]:
    number, total = _int(page.get("page_details", "printed_page_no")), _int(page.get("page_details", "printed_page_total"))
    return (number, total) if 1 <= number <= total else None


def page_type_of(page: Page) -> tuple[str, str, tuple[str, ...], Optional[float], bool]:
    """(family, subtype, names most specific first, confidence, title hit) of the page itself.

    From the pipeline outputs: ``model_type`` is the page classifier's own path ("Demographics >
    Patient Information"); ``page_type`` is the document's agreed type, used when there is none.
    From a pipeline chart folder: its codeable CSV, as the pipeline's continuity stage read it.
    """
    row = page.pipeline
    if row:
        path = pipeline_csv.type_path(row)
        family = path[0] if path else str(row.get("page_type") or "")
        names = tuple(reversed(path)) or tuple(n for n in (row.get("page_subtype"), row.get("page_type")) if n)
        conf = None
        try:
            import json

            conf = (json.loads(row.get("confidences") or "{}") or {}).get("page_type")
        except ValueError:
            pass
        return family, str(row.get("page_subtype") or ""), names, conf, False
    cls = page.rows.get("classification") or {}
    found = [cls.get("axes_canon_page_type"), cls.get("axes_l3"), cls.get("axes_l2"), cls.get("axes_l1")]
    if not any(found):
        found = [cls.get("page_subtype"), cls.get("page_type")]
    names: list[str] = []
    for name in found:
        name = str(name or "").strip()
        if name and name not in names:
            names.append(name)
    title = bool(str(cls.get("axes_title_l1") or "").strip()) or \
        str(cls.get("keyword_title_hit") or "").strip().lower() in ("true", "1", "y", "yes")
    return (str(cls.get("axes_l1") or cls.get("page_type") or ""), str(cls.get("page_subtype") or ""),
            tuple(names), _float(cls.get("confidence")), title)


def page_dos(page: Page) -> tuple[str, str]:
    """(from, to) ISO dates: the outputs CSV's (the document's DOS), else the DOS CSV's."""
    if page.pipeline:
        start = pipeline_csv.iso_date(page.pipeline.get("dos_from"))
        return start, (pipeline_csv.iso_date(page.pipeline.get("dos_to")) or start) if start else ""
    return str(page.get("dos", "dos_from_iso", "") or ""), str(page.get("dos", "dos_to_iso", "") or "")


def continuity_inputs(chart: Chart) -> list[PageInput]:
    """The PageInputs the pipeline's continuity stage builds, from local files."""
    rules = load_rules()
    signed_names = signed_page_names(input_path(chart.root, "signature"))
    numbers = pipeline_csv.page_ids({p.page_number: p.pipeline for p in chart.pages if p.pipeline})
    inputs: list[PageInput] = []
    for page in chart.pages:
        kind = page.group if page.group in EXCLUDE else "main"
        text = "" if kind in ("blank", "duplicate") else page.text
        lines, headers, _source = page_layout(page.final2, page.final1)
        lines = lines or text_lines(text)
        family, subtype, names, confidence, title = page_type_of(page)
        dos_from, dos_to = page_dos(page)
        tail = "\n".join(l.text for l in lines[-max(1, round(len(lines) * 0.33)):]) if lines else ""
        if page.pipeline:
            dup = numbers.get(str(page.pipeline.get("duplicate_of_page_id") or "").split(".")[0])
            signed_flag = pipeline_csv.flag(page.pipeline.get("signature_present"))
        else:
            dup = _int(page.get("junk", "duplicate_of_page")) or None
            signed_flag = page.page_name in signed_names or str(page.get("signature", "signature_present", "")).lower() == "y"
        inputs.append(
            PageInput(
                page_id=page.page_number,
                page_name=page.page_name,
                page_number=page.page_number,
                text=text,
                lines=lines,
                section_headers=headers,
                page_type=family,
                page_subtype=subtype,
                names=names,
                page_confidence=confidence,
                title_hit=title,
                dos_from=dos_from if kind == "main" else "",
                dos_to=dos_to if kind == "main" else "",
                dos_trust=trust_points(None, rules) if dos_from else None,
                signed=signed_flag or has_signature_section(text),
                signed_in_tail=str(page.get("signature", "region", "")).lower() == "footer"
                or has_signature_section(tail),
                printed=_printed(page),
                visit_ids=frozenset(),
                kind=kind,
                junk_subtype=page.junk_subtype if kind == "junk" else "",
                junk_confidence=_float(page.get("junk", "page_classification_confidence")) if kind == "junk" else None,
                duplicate_of=dup if kind == "duplicate" else None,
                visual=False,
            )
        )
    return inputs


def documents(chart: Chart, source: str = "csv") -> dict[int, Optional[int]]:
    """page_number -> continuity document number: the pipeline's (outputs CSV ``document_seq``, or
    its continuity CSV) with ``source="csv"``, or the ported engine re-run with "recompute"."""
    if source == "csv":
        out = {}
        for p in chart.pages:
            value = p.pipeline.get("document_seq") if p.pipeline else p.get("continuity", "document")
            out[p.page_number] = _int(value) or None
        if any(out.values()):
            return out
    rows = assign(continuity_inputs(chart))
    return {row["page_number"]: row.get("document_seq") for row in rows}


def _marker(page: Page) -> Optional[ExplicitMarker]:
    """The sequencing stage's ``_printed_marker`` (a printed page number the kv extraction read)."""
    number = _int(page.get("page_details", "printed_page_no"))
    total = _int(page.get("page_details", "printed_page_total")) or None
    if number < 1 or (total is not None and number > total):
        return None
    return ExplicitMarker(
        page_num=number, total_pages=total,
        pattern="page_x_of_y" if total else "page_x", confidence=1.0 if total else 0.85,
    )


def run_v1(chart: Chart, documents_source: str = "csv") -> OrderResult:
    """The order the pipeline's sequencing stage would write for ``chart``."""
    docs = documents(chart, documents_source)
    seq_inputs = []
    junk = []
    for page in chart.pages:
        if page.group in EXCLUDE:
            junk.append(page.page_number)
            continue
        seq_inputs.append({
            "page_id": page.page_number,
            "page_number": page.page_number,
            "text": page.text,
            "is_classified": False,
            "page_marker": _marker(page),
            "document": docs.get(page.page_number),
        })
    assignments = compute_sequence_assignments(seq_inputs)
    ordered = sorted(
        (a for a in assignments if a.get("sequence_position") is not None),
        key=lambda a: a["sequence_position"],
    )
    detail = {
        int(a["page_id"]): {
            "method": a["sequence_method"],
            "confidence": a["sequence_confidence"],
            "review": a["sequence_review_flag"],
            "document": docs.get(int(a["page_id"])),
        }
        for a in assignments
    }
    return OrderResult(chart.name, [int(a["page_id"]) for a in ordered], junk, f"v1:{documents_source}", detail)


def run_v1_csv(chart: Chart) -> OrderResult:
    """What production wrote: the outputs CSV's ``seq`` (else the sequencing CSV's actual_sequence)."""
    main: list[tuple[int, int]] = []
    junk: list[int] = []
    for page in chart.pages:
        if page.pipeline:
            value = pipeline_csv.int_or_none(page.pipeline.get("seq"))
            if page.group in EXCLUDE:
                value = None
        else:
            raw = str(page.get("sequencing", "actual_sequence", "") or "").strip()
            value = int(raw) if raw.isdigit() else None
        if value is not None:
            main.append((value, page.page_number))
        else:
            junk.append(page.page_number)
    return OrderResult(chart.name, [p for _, p in sorted(main)], junk, "v1_csv")


def run_scan_order(chart: Chart) -> OrderResult:
    """Baseline: keep the scan order, set the pipeline's blank / junk aside."""
    main = [p.page_number for p in chart.pages if p.group not in EXCLUDE]
    junk = [p.page_number for p in chart.pages if p.group in EXCLUDE]
    return OrderResult(chart.name, main, junk, "scan_order")
