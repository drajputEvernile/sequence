"""The full pipeline outputs: one CSV, one row per page (the imaging_final table export).

Written by the imaging pipeline's last stage (core-pipeline/modules/imaging_final/stage.py).
What its columns mean for sequencing:

| column | meaning |
|---|---|
| chart_name, page_number | the document (RecordID_DocumentID) and the page's scan position |
| page_id | the pipeline's page id (``duplicate_of_page_id`` points at one) |
| blank_junk_flag | the final verdict: not_blank_junk / blank / junk / duplicate. Only blank, junk and duplicate are set aside (the J series); ``is_duplicate`` is also TRUE on a kept "Copy of" page |
| document_seq | continuity's document number (the pipeline's grouping of consecutive pages) |
| seq | **v1's own answer**: the page's position among the Main pages (empty when set aside) |
| page_type | the document's agreed page type (carried from its first pages); ``page_type_source`` says page / continuation / duplicate |
| page_subtype | the page's own sub-type |
| model_type | the page classifier's own path for this page ("Demographics > Patient Information") |
| dos_from, dos_to | the document's date of service (``dos_source`` document / page); 2/22/2222 = a non-visit document |
| signature_present | an electronic signature on the page |
| encounter_type, codability, quality_tag, printed_or_handwritten | as named |

Nothing here was made from the ground truth, so every column is a fair input. (In the pipeline,
imaging_final runs after sequencing; a sequencer built on these values runs after it, or on the
same values from the stages before it.)
"""
from __future__ import annotations

import csv
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

TRUE = {"true", "t", "1", "y", "yes"}
NON_VISIT_YEAR = "2222"  # DOS_NON_VISIT_DATE / DOS_DEFAULT_DATE (dos_canon.json)


def _read_rows(path: Path) -> list[dict[str, str]]:
    data = Path(path).read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:4096]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    import io

    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int) -> dict[str, dict[int, dict[str, str]]]:
    out: dict[str, dict[int, dict[str, str]]] = {}
    for row in _read_rows(Path(path)):
        chart = str(row.get("chart_name") or "").strip()
        try:
            number = int(float(row.get("page_number") or ""))
        except ValueError:
            continue
        if chart:
            out.setdefault(chart, {})[number] = row
    return out


def load_pipeline_csv(path: Optional[Path]) -> dict[str, dict[int, dict[str, str]]]:
    """chart_name -> page_number -> row; empty without the file."""
    if not path or not Path(path).is_file():
        return {}
    return _load(str(path), Path(path).stat().st_mtime_ns)


def flag(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE


def group(row: Optional[dict[str, Any]]) -> str:
    """main | blank | junk | duplicate, from blank_junk_flag."""
    value = str((row or {}).get("blank_junk_flag") or "").strip().lower()
    return value if value in ("blank", "junk", "duplicate") else "main"


def iso_date(value: Any) -> str:
    """'1/6/2025' / '2025-01-06' -> '2025-01-06'; '' for no date or the non-visit default."""
    text = str(value or "").strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        y, mo, d = m.groups()
    else:
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", text)
        if not m:
            return ""
        mo, d, y = m.groups()
    if y == NON_VISIT_YEAR:
        return ""
    try:
        return date(int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return ""


def type_path(row: Optional[dict[str, Any]]) -> tuple[str, ...]:
    """The page's own classifier path, most general first: model_type split on '>'."""
    raw = str((row or {}).get("model_type") or "")
    return tuple(part.strip() for part in raw.split(">") if part.strip())


def page_ids(rows: dict[int, dict[str, str]]) -> dict[str, int]:
    """page_id -> page_number for one chart (to read duplicate_of_page_id)."""
    out = {}
    for number, row in rows.items():
        pid = str(row.get("page_id") or "").strip()
        if pid:
            out[pid.split(".")[0]] = number
    return out


def int_or_none(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
