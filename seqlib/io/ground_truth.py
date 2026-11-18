"""Ground-truth page order, read from the truth CSV(s) on disk.

Sheet shapes read:

* **page_ground_truth export** (the one in use): ``chart_name``, ``page_number``,
  ``page_sequence`` (Main pages 1..N in the correct order; blank / junk / invoice pages J1, J2 ...),
  ``blank_page``, ``junk_page``, ``is_invoice``, ``is_duplicate`` ... (``id`` is the row id).
* **Client export**: ``DocumentID``, ``RecordID``, ``Id`` (page file stem), ``Sequence``,
  ``OriginalSequence``, ``DefaultSequence``, ``SortingSequence``, ``Blank Page``, ``Junk Page``,
  ``IsInvoicePage``; the document key is built from config.DOC_NAME_PATTERN. (There ``Page
  Sequence`` is a different column: set GT_SEQUENCE_COLUMN = "Sequence".)

The order comes from config.GT_SEQUENCE_COLUMN. Other columns worth comparing are kept in
``ChartTruth.extra``.

Same rules as the pipeline's loader (db-utilities ``dbutil/ground_truth.py``): page 1, 1.png and
1.jpg are page 1; a page marked Blank / Junk / Invoice (a Yes in any) is in the J series:
"3" -> "J3".
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

# Columns kept beside the order, for comparison (normalised header names).
EXTRA_COLUMNS = ("originalsequence", "defaultsequence", "sortingsequence", "pagesequence", "isvisible", "rotation",
                 "blankpage", "junkpage", "isinvoicepage", "isinvoice", "isduplicate", "pagetype", "pagesubtype")


def norm_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def page_number_from_id(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value == int(value):
        value = int(value)
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip()
    match = re.fullmatch(r"(\d+)(?:\.0+)?(?:\.(?:jpe?g|png|webp|tiff?|bmp))?", text, re.IGNORECASE)
    return int(match.group(1)) if match and int(match.group(1)) > 0 else None


def _yes(value: Any) -> bool:
    return str(value or "").strip().casefold().startswith(("y", "true", "1"))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return str(value).strip()


@dataclass
class ChartTruth:
    chart_name: str
    # page_number -> label as stored ("3", "J2", or None when the cell is empty)
    labels: dict[int, Optional[str]] = field(default_factory=dict)
    # page_number -> the other columns kept (EXTRA_COLUMNS), as text
    extra: dict[int, dict[str, str]] = field(default_factory=dict)
    duplicate_rows: int = 0  # rows naming a page already read
    # Pages marked Blank / Junk / Invoice whose sequence is a plain (Main) number: kept as Main
    # when the flags are not applied (GT_APPLY_JUNK_FLAGS False), moved to J when they are.
    flagged_numbered: list[int] = field(default_factory=list)

    @property
    def main_order(self) -> list[int]:
        """Main pages (scan page numbers) in their correct order."""
        main = [(int(v), p) for p, v in self.labels.items() if v and v.isdigit()]
        return [p for _, p in sorted(main)]

    @property
    def junk_pages(self) -> list[int]:
        return sorted(p for p, v in self.labels.items() if v and v.upper().startswith("J"))

    @property
    def unlabelled(self) -> list[int]:
        return sorted(p for p, v in self.labels.items() if not v)

    def problems(self) -> list[str]:
        """Label problems that make the chart unusable for scoring."""
        out = []
        numbers = [int(v) for v in self.labels.values() if v and v.isdigit()]
        if not numbers:
            out.append("no Main page has a sequence")
        elif sorted(numbers) != list(range(1, len(numbers) + 1)):
            dupes = sorted({n for n in numbers if numbers.count(n) > 1})
            gaps = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
            if dupes:
                out.append(f"sequence numbers used twice: {dupes[:10]}")
            if gaps:
                # A flagged page whose own number is a missing Main number: most likely a Main
                # page with a wrong Blank / Junk / Invoice flag (or a J page numbered by mistake).
                holders = sorted(
                    (int(self.labels[p][1:]), p) for p in self.flagged_numbered
                    if self.labels.get(p, "").upper().startswith("J") and self.labels[p][1:].isdigit()
                    and int(self.labels[p][1:]) in gaps
                )
                why = "".join(f"; page {p} has number {g} but is marked blank / junk / invoice"
                              for g, p in holders[:5])
                out.append(f"sequence numbers missing: {gaps[:10]}{why}")
        bad = [v for v in self.labels.values() if v and not v.isdigit() and not re.fullmatch(r"J\d+", v, re.I)]
        if bad:
            out.append(f"unreadable sequence cells: {bad[:5]}")
        if self.duplicate_rows:
            out.append(f"{self.duplicate_rows} rows repeat a page Id")
        return out

    def order_by(self, column: str) -> Optional[list[int]]:
        """Pages ordered by another sequence column (normalised name), None when it is not
        numeric on every page."""
        values = []
        for page, row in self.extra.items():
            try:
                values.append((float(row.get(column, "")), page))
            except ValueError:
                return None
        return [p for _, p in sorted(values)] if values else None


def _clean_sequence(value: Any) -> Optional[str]:
    text = _cell(value)
    if not text:
        return None
    try:
        number = float(text)
        if number == int(number):
            return str(int(number))
    except ValueError:
        pass
    return text.upper().replace(" ", "") if re.fullmatch(r"[jJ]\s*\d+", text) else text


def _junk_label(value: str) -> str:
    text = value[1:].strip() if value[:1] in ("J", "j") else value
    return f"J{int(text)}" if text.isdigit() and int(text) > 0 else value


def read_table(path: Path) -> tuple[list[Any], list[list[Any]]]:
    path = Path(path)
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        data = path.read_bytes()
        text = ""
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        sample = text[:4096]
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    else:
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        rows = [list(r) for r in book.worksheets[0].iter_rows(values_only=True)]
    rows = [r for r in rows if r and any(c is not None and str(c).strip() for c in r)]
    return (list(rows[0]), rows[1:]) if rows else ([], [])


def iter_files(root: Path) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xlsm") and not path.name.startswith("~$"):
            yield path


def _settings() -> tuple[str, str, Optional[bool]]:
    try:
        from seqlib.settings import cfg

        return (norm_header(getattr(cfg, "GT_SEQUENCE_COLUMN", "page_sequence")),
                getattr(cfg, "DOC_NAME_PATTERN", "{RecordID}_{DocumentID}"),
                getattr(cfg, "GT_APPLY_JUNK_FLAGS", None))
    except Exception:  # noqa: BLE001 - usable without a config
        return "pagesequence", "{RecordID}_{DocumentID}", None


def chart_name_of(row: dict[str, str], pattern: str) -> str:
    """The document key: ``chartname`` when the sheet has it, else ``pattern`` filled from the
    row ({RecordID}, {DocumentID} or any other header, spelled as in the sheet or normalised)."""
    if row.get("chartname"):
        return row["chartname"]
    names = {key: row.get(norm_header(key), "") for key in re.findall(r"{([^}]+)}", pattern)}
    if not all(names.values()):
        return ""
    return pattern.format(**names)


def load_ground_truth(root: Path, sequence_column: Optional[str] = None,
                      name_pattern: Optional[str] = None,
                      apply_junk_flags: Optional[bool] = None) -> dict[str, ChartTruth]:
    """Every document's truth from a file or a folder of CSV / TSV / XLSX sheets.

    ``apply_junk_flags`` (config.GT_APPLY_JUNK_FLAGS; None = decide per sheet): whether a page
    marked Blank / Junk / Invoice whose sequence is a plain number is moved to the J series.
    Both sheets in use need it: blank / junk / invoice pages are numbered in their own series
    (1, 2, 3 beside the Main pages' 1, 2, 3), as the pipeline's loader expects (dbutil
    ground_truth.py ``junk_sequence``). Off, those numbers collide with the Main ones."""
    default_column, default_pattern, default_flags = _settings()
    if apply_junk_flags is None:
        apply_junk_flags = default_flags
    seq_col = norm_header(sequence_column) if sequence_column else default_column
    pattern = name_pattern or default_pattern
    out: dict[str, ChartTruth] = {}
    for path in iter_files(root):
        headers, data = read_table(path)
        keys = [norm_header(h) for h in headers]
        column = seq_col
        if column not in keys and "pagesequence" in keys and "chartname" in keys and "pagenumber" in keys:
            column = "pagesequence"  # a page_ground_truth database export
        if column not in keys:
            continue
        flags = True if apply_junk_flags is None else bool(apply_junk_flags)
        for raw in data:
            row: dict[str, str] = {}
            for i, key in enumerate(keys):
                if key and i < len(raw) and key not in row:  # first of a repeated header wins
                    row[key] = _cell(raw[i])
            chart = chart_name_of(row, pattern)
            # page_number when the sheet has it (in a database export ``id`` is the row id);
            # the client export's ``Id`` is the page file stem.
            page = page_number_from_id(row.get("pagenumber") or row.get("sourcepageid") or row.get("id"))
            if not chart or page is None:
                continue
            value = _clean_sequence(row.get(column))
            flagged = any(_yes(row.get(k)) for k in ("blankpage", "junkpage", "isinvoicepage", "isinvoice"))
            truth = out.setdefault(chart, ChartTruth(chart))
            if page in truth.labels:
                truth.duplicate_rows += 1
                continue
            if value and value.isdigit() and flagged:
                truth.flagged_numbered.append(page)
                if flags:
                    value = _junk_label(value)
            truth.labels[page] = value
            truth.extra[page] = {k: row[k] for k in EXTRA_COLUMNS if k in row}
    return out
