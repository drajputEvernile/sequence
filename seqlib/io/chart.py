"""One chart, loaded from local files only (paths from config.py).

A chart is a folder of page images the imaging pipeline processed; its pages are numbered in
scan order (``1.jpg``, ``2.jpg`` ...). :func:`load_chart` gathers every page's OCR and the
per-page CSV rows the pipeline's other stages wrote, so the sequencers never touch a database,
Azure or an OCR engine.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from seqlib.settings import cfg, input_path

from . import pipeline_csv
from .ocr_files import page_content, read_ocr_json, read_prelim

logger = logging.getLogger(__name__)

PAGE_NUM_RE = re.compile(r"^(\d+)\.(jpe?g|png|webp|tiff?|bmp)$", re.IGNORECASE)
OPTIONAL_CSVS = (
    "junk", "classification", "dos", "page_details", "signature", "quality", "hw_printed",
    "continuity", "sequencing", "final",
)
TRUE = {"true", "t", "1", "y", "yes"}
PRINTED_TYPES = frozenset({"printed", "visual"})  # the pipeline's common.db.PRINTED_TYPES


@dataclass
class Page:
    page_name: str
    page_number: int  # scan order
    prelim: str = ""
    final1: Optional[dict[str, Any]] = None  # the page's final1 JSON object
    final2: Optional[dict[str, Any]] = None  # the page's final2 JSON object
    # One row per optional CSV (config.INPUT_FILES key -> the page's row).
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    # main | blank | junk | duplicate decided by something else than the pipeline (the set-aside
    # model, or the truth for an oracle run); None = the pipeline's verdict.
    group_override: Optional[str] = None

    @property
    def final1_text(self) -> str:
        return page_content(self.final1)

    @property
    def final2_text(self) -> str:
        return page_content(self.final2)

    @property
    def ocr_source(self) -> str:
        if self.final2_text.strip():
            return "final2"
        if self.final1_text.strip():
            return "final1"
        return "prelim" if self.prelim.strip() else ""

    @property
    def text(self) -> str:
        """The pipeline's ``best_page_text``: final2, else final1, else prelim -- prelim never
        for a handwritten / mixed / low-quality page."""
        for value in (self.final2_text, self.final1_text):
            if value.strip():
                return value
        if prelim_blocked(self.rows.get("quality") or self.rows.get("pipeline"),
                          self.rows.get("hw_printed") or self.rows.get("pipeline")):
            return ""
        return self.prelim.strip()

    # --- the pipeline's verdicts, read from its CSVs --------------------------------------

    @property
    def pipeline(self) -> dict[str, Any]:
        """The page's row in the full pipeline outputs CSV ({} without one)."""
        return self.rows.get("pipeline") or {}

    @property
    def group(self) -> str:
        """main | blank | junk | duplicate: the pipeline's final blank / junk verdict (its outputs
        CSV, else its junk CSV); main without one. ``group_override`` wins when set."""
        if self.group_override:
            return self.group_override
        if self.pipeline:
            return pipeline_csv.group(self.pipeline)
        row = self.rows.get("junk") or {}
        value = str(row.get("page_group") or "").strip().lower()
        return value if value in ("blank", "junk", "duplicate") else "main"

    @property
    def junk_subtype(self) -> str:
        row = self.rows.get("junk") or {}
        label = str(row.get("page_classification") or "")
        return "" if label in ("Main", "Blank", "Duplicate") else label

    def get(self, key: str, column: str, default: Any = None) -> Any:
        value = (self.rows.get(key) or {}).get(column)
        return default if value is None or str(value).strip() == "" else value


@dataclass
class Chart:
    name: str
    root: Path
    pages: list[Page]
    missing: list[str] = field(default_factory=list)  # optional inputs not found

    def by_name(self) -> dict[str, Page]:
        return {p.page_name: p for p in self.pages}

    def by_number(self) -> dict[int, Page]:
        return {p.page_number: p for p in self.pages}


def prelim_blocked(quality: Optional[dict[str, Any]], hw: Optional[dict[str, Any]] = None) -> bool:
    """The pipeline's ``page_blocks_prelim``: a low-quality or not-printed page's Tesseract text
    is not used."""
    tag = str((quality or {}).get("quality_tag") or "").strip().lower()
    if tag == "low":
        return True
    kind = str((hw or {}).get("document_type") or (hw or {}).get("handwritten_or_printed")
               or (hw or {}).get("printed_or_handwritten") or "").strip().lower()
    return bool(kind) and kind not in PRINTED_TYPES


def page_number_of(name: str) -> Optional[int]:
    match = PAGE_NUM_RE.match(name.strip())
    if match:
        return int(match.group(1))
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else None


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    return []


def _junk_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The final verdict per page: the ``is_final`` row, else the highest pass."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("page_name") or "").strip()
        if not name:
            continue
        final = str(row.get("is_final") or "").strip().lower() in TRUE
        rank = (final, int(float(row.get("pass_no") or 0)))
        current = best.get(name)
        if current is None or rank >= current["_rank"]:
            best[name] = {**row, "_rank": rank}
    return best


def chart_dirs(root: Optional[Path] = None) -> list[Path]:
    root = Path(root or cfg.CHARTS_ROOT)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def load_chart(name_or_dir: str | Path, root: Optional[Path] = None) -> Chart:
    chart_dir = Path(name_or_dir)
    if not chart_dir.is_dir():
        chart_dir = Path(root or cfg.CHARTS_ROOT) / str(name_or_dir)
    if not chart_dir.is_dir():
        raise FileNotFoundError(f"chart folder not found: {chart_dir}")
    name = chart_dir.name

    prelim = read_prelim(input_path(chart_dir, "ocr_prelim"))
    final1 = read_ocr_json(input_path(chart_dir, "ocr_final1"))
    final2 = read_ocr_json(input_path(chart_dir, "ocr_final2"))
    if not (prelim or final1 or final2):
        raise FileNotFoundError(f"{name}: no OCR file found under {chart_dir}")

    csv_rows: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    for key in OPTIONAL_CSVS:
        if key not in cfg.INPUT_FILES:
            continue
        rows = read_csv_rows(input_path(chart_dir, key))
        if not rows:
            missing.append(key)
            continue
        if key == "junk":
            csv_rows[key] = _junk_rows(rows)
        else:
            csv_rows[key] = {str(r.get("page_name") or "").strip(): r for r in rows if r.get("page_name")}

    names: set[str] = set(prelim) | set(final1) | set(final2)
    for key in ("junk", "classification"):
        names |= set(csv_rows.get(key, {}))
    pages: list[Page] = []
    for page_name in names:
        number = page_number_of(page_name)
        for key in ("junk", "classification", "final"):
            row = csv_rows.get(key, {}).get(page_name)
            if number is None and row and str(row.get("page_number") or "").strip().isdigit():
                number = int(row["page_number"])
        if number is None:
            logger.warning("%s: page %r has no page number; skipped", name, page_name)
            continue
        pages.append(
            Page(
                page_name=page_name,
                page_number=number,
                prelim=prelim.get(page_name, ""),
                final1=final1.get(page_name),
                final2=final2.get(page_name),
                rows={key: rows[page_name] for key, rows in csv_rows.items() if page_name in rows},
            )
        )
    # The full pipeline outputs (config.PIPELINE_OUTPUT_CSV): one row per page, by page number.
    # A page the pipeline has but the OCR lacks is kept (no text), so the page set is the
    # pipeline's -- the same pages the truth numbers.
    pipeline_rows = pipeline_csv.load_pipeline_csv(getattr(cfg, "PIPELINE_OUTPUT_CSV", None)).get(name, {})
    if pipeline_rows:
        missing = []  # the outputs CSV stands in for every imaging/ CSV
        have = {p.page_number for p in pages}
        for number in sorted(set(pipeline_rows) - have):
            pages.append(Page(page_name=f"{number}.png", page_number=number))
        for page in pages:
            row = pipeline_rows.get(page.page_number)
            if row is not None:
                page.rows["pipeline"] = row
        if set(pipeline_rows) - have:
            missing.append(f"ocr for {len(set(pipeline_rows) - have)} page(s)")
    elif getattr(cfg, "PIPELINE_OUTPUT_CSV", None):
        missing.append("pipeline")
    pages.sort(key=lambda p: (p.page_number, p.page_name))
    return Chart(name=name, root=chart_dir, pages=pages, missing=missing)


def iter_charts(names: Iterable[str], root: Optional[Path] = None):
    """Each chart that loads, as (name, Chart or the exception)."""
    for name in names:
        try:
            yield name, load_chart(name, root)
        except Exception as exc:  # noqa: BLE001 - one bad chart must not stop a run
            yield name, exc
