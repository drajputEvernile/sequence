"""A shuffle test: copies of real documents with their pages put out of order on purpose.

Most real documents are already in scan order, so scoring a sequencer on them says little about
whether it can *fix* an order. Here a sample of documents is copied to a separate folder and each
copy's pages are renamed (1.png -> 7.png ...) so they come in out of order. The renaming is applied
consistently to every input -- the image files, the OCR's ``fileName`` / page markers, the pipeline
outputs' ``page_number`` and the truth's ``page_number`` -- and recorded in a manifest. The truth's
``page_sequence`` is untouched (it belongs to the page's content), so the correct order is still
known exactly. Nothing is written to the original data.

**Order-dependent pipeline values are blanked in the copy.** Some pipeline outputs were computed
from the original page order: ``seq`` (v1's answer), ``document_seq`` / ``page_type_source`` /
``continuation_rule`` (continuity grouped neighbouring pages), ``page_type`` / ``encounter_type`` /
``codability`` / ``classification_category`` / ``needs_review`` (agreed across a document), a DOS
carried from the document (``dos_source`` = document), and the document-level confidences. Left in,
they would label the pages of one document alike after shuffling and give the answer away; in a
real mis-scanned file the pipeline would have computed them on the wrong order. Per-page values
stay: blank / junk verdict, the page's own type (model_type, page_subtype), quality, signature, a
DOS read from the page itself (``dos_source`` = page) and the per-page confidences.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from seqlib.extract.import_ocr import image_numbers
from seqlib.extract.ocr_convert import convert_document
from seqlib.io.ground_truth import ChartTruth

ORDER_DEPENDENT = ("seq", "document_seq", "page_type", "page_type_source", "continuation_rule", "encounter_type",
                   "codability", "classification_category", "needs_review", "review_fields")
PAGE_CONFIDENCES = ("blank_junk", "handwriting", "provider", "specialty")
OPERATIONS = ("swap", "move_page", "move_block", "reverse_block")


@dataclass
class ShuffledDoc:
    doc: str
    pages: int  # every page of the document
    main_pages: int
    operations: list[str] = field(default_factory=list)
    new_of: dict[int, int] = field(default_factory=dict)  # original page number -> new page number
    pages_moved: int = 0

    @property
    def original_of(self) -> dict[int, int]:
        return {new: old for old, new in self.new_of.items()}


MIN_PAGES_FOR = {"swap": 2, "move_page": 3, "move_block": 4, "reverse_block": 3}


def shuffle_order(pages: list[int], rng: random.Random, max_ops: int, kinds=OPERATIONS,
                  min_ops: int = 1) -> tuple[list[int], list[str]]:
    """``pages`` (original page numbers, in scan order) put out of order by ``min_ops`` to ``max_ops``
    operations (a number drawn at random; min_ops = max_ops: exactly that many), each one of the
    kinds the document is long enough for; returns (the new scan order, what was done). The result
    always differs from the input."""
    n = len(pages)
    usable = [k for k in kinds if n >= MIN_PAGES_FOR[k]]
    for _ in range(50 if usable else 0):
        order, done = list(pages), []
        for _ in range(rng.randint(min_ops, max(min_ops, max_ops))):
            kind = rng.choice(usable)
            if kind == "swap":
                i, j = sorted(rng.sample(range(n), 2))
                order[i], order[j] = order[j], order[i]
                done.append(f"swap pages {order[j]} and {order[i]}")
            elif kind == "move_page":
                i = rng.randrange(n)
                page = order.pop(i)
                j = rng.choice([k for k in range(n) if abs(k - i) >= 2] or [0])
                order.insert(j, page)
                done.append(f"move page {page} from position {i + 1} to {j + 1}")
            elif kind == "move_block":
                size = rng.randint(2, max(2, min(8, n // 2)))
                i = rng.randrange(0, n - size + 1)
                block = order[i:i + size]
                rest = order[:i] + order[i + size:]
                j = rng.choice([k for k in range(len(rest) + 1) if k != i] or [0])
                order = rest[:j] + block + rest[j:]
                done.append(f"move block {block[0]}..{block[-1]} ({size} pages) from position {i + 1} to {j + 1}")
            elif kind == "reverse_block":
                size = rng.randint(3, max(3, min(10, n)))
                i = rng.randrange(0, n - size + 1)
                order[i:i + size] = order[i:i + size][::-1]
                done.append(f"reverse block of {size} pages at position {i + 1}")
        if order != pages:
            return order, done
    order = list(pages)
    order[0], order[-1] = order[-1], order[0]
    return order, [f"swap pages {order[0]} and {order[-1]}"]


def choose(truths: dict[str, ChartTruth], candidates: list[str], count: int, min_main: int, seed: int) -> list[str]:
    eligible = sorted(n for n in candidates if n in truths and not truths[n].problems()
                      and len(truths[n].main_order) >= min_main)
    rng = random.Random(seed)
    return sorted(rng.sample(eligible, min(count, len(eligible))))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    data = Path(path).read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    delimiter = "\t" if text[:4096].count("\t") > text[:4096].count(",") else ","
    import io

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return list(reader.fieldnames or []), list(reader), delimiter


def _page_conf(raw: str) -> str:
    try:
        data = json.loads(raw or "{}")
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    return json.dumps({k: v for k, v in data.items() if k in PAGE_CONFIDENCES})


def strip_order_dependent(row: dict[str, str]) -> dict[str, str]:
    row = dict(row)
    for col in ORDER_DEPENDENT:
        if col in row:
            row[col] = ""
    if str(row.get("dos_source") or "").strip().lower() != "page":
        for col in ("dos_from", "dos_to", "dos_source"):
            if col in row:
                row[col] = ""
    if "confidences" in row:
        row["confidences"] = _page_conf(row["confidences"])
    return row


def _new_name(name: str, new_number: int) -> str:
    suffix = Path(name).suffix or ".png"
    return f"{new_number}{suffix}"


def write_ocr(doc: str, ocr_files: list[Path], image_names: dict[int, str], new_of: dict[int, int], out_dir: Path) -> None:
    """The document's OCR, renamed, as the pipeline's three files in ``out_dir``."""
    converted = convert_document(ocr_files, doc, image_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, pages in (("final2", converted.final2), ("final1", converted.final1)):
        if not pages:
            continue
        renamed = []
        for old in sorted(pages, key=lambda n: new_of.get(n, n)):
            page = dict(pages[old])
            new = new_of.get(old, old)
            page["fileName"] = _new_name(str(page.get("fileName") or f"{old}.png"), new)
            page["pageNumber"] = new
            renamed.append(page)
        (out_dir / f"{doc}_{kind}.json").write_text(
            json.dumps({"recordId": doc, "model": converted.source, "pageCount": len(renamed), "pages": renamed},
                       indent=1), encoding="utf-8")
    if converted.prelim:
        blocks = []
        for old in sorted(converted.prelim, key=lambda n: new_of.get(n, n)):
            name = _new_name(image_names.get(old) or f"{old}.png", new_of.get(old, old))
            blocks.append(f"===== {name} =====\n{converted.prelim[old]}\n")
        (out_dir / f"{doc}_prelim.txt").write_text("\n".join(blocks), encoding="utf-8")


def build(
    out_root: Path, docs: list[str], truths: dict[str, ChartTruth], pipeline_rows: dict[str, dict[int, dict]],
    pipeline_csv_path: Path, truth_csv_path: Path, images: dict[str, list[Path]], ocr: dict[str, list[Path]],
    seed: int, max_ops: int, unchanged_share: float = 0.0, splits: Optional[dict[str, str]] = None,
    role: str = "test", min_ops: int = 1,
) -> list[ShuffledDoc]:
    out_root = Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"{out_root} is not empty: pick another --out, or delete it first")
    rng = random.Random(seed)
    made: list[ShuffledDoc] = []
    for doc in docs:
        numbered = image_numbers(images.get(doc, []))
        pages = sorted(set(pipeline_rows.get(doc, {})) | set(truths[doc].labels) | set(numbered))
        item = ShuffledDoc(doc, len(pages), len(truths[doc].main_order))
        if rng.random() < unchanged_share:
            order, item.operations = list(pages), ["unchanged (control)"]
        else:
            order, item.operations = shuffle_order(pages, rng, max_ops, min_ops=min_ops)
        item.new_of = {old: i for i, old in enumerate(order, start=1)}
        item.pages_moved = sum(1 for old, new in item.new_of.items() if old != new)
        # images
        raw_dir = out_root / "raw_images" / doc
        raw_dir.mkdir(parents=True, exist_ok=True)
        for old, path in numbered.items():
            shutil.copy2(path, raw_dir / _new_name(path.name, item.new_of.get(old, old)))
        # OCR
        write_ocr(doc, ocr.get(doc, []), {n: p.name for n, p in numbered.items()}, item.new_of,
                  out_root / "ocr" / doc / "ocr")
        made.append(item)

    by_doc = {m.doc: m for m in made}
    # pipeline outputs: renamed and stripped of order-dependent values
    fields, rows, delimiter = _read_csv(pipeline_csv_path)
    out_rows = []
    for row in rows:
        item = by_doc.get(str(row.get("chart_name") or "").strip())
        if item is None:
            continue
        old = int(float(row["page_number"]))
        row = strip_order_dependent(row)
        row["page_number"] = str(item.new_of.get(old, old))
        out_rows.append(row)
    out_rows.sort(key=lambda r: (r["chart_name"], int(r["page_number"])))
    _write_csv(out_root / "imaging_output_full.csv", fields, out_rows, delimiter)
    # truth: renamed (page_sequence untouched: it belongs to the page's content)
    fields, rows, delimiter = _read_csv(truth_csv_path)
    out_rows = []
    for row in rows:
        item = by_doc.get(str(row.get("chart_name") or "").strip())
        if item is None:
            continue
        old = int(float(row["page_number"]))
        new = item.new_of.get(old, old)
        row = dict(row)
        row["page_number"] = str(new)
        if row.get("source_page_id"):
            row["source_page_id"] = _new_name(row["source_page_id"], new)
        out_rows.append(row)
    out_rows.sort(key=lambda r: (r["chart_name"], int(r["page_number"])))
    _write_csv(out_root / "ground_truth.csv", fields, out_rows, delimiter)

    manifest = {
        "seed": seed, "max_operations": max_ops, "unchanged_share": unchanged_share,
        "blanked_pipeline_columns": list(ORDER_DEPENDENT) + ["dos_from / dos_to unless dos_source = page",
                                                             f"confidences except {list(PAGE_CONFIDENCES)}"],
        "documents": [
            {**{k: v for k, v in asdict(m).items() if k != "new_of"},
             "split": (splits or {}).get(m.doc, ""),
             "renamed (original -> new)": {str(o): n for o, n in sorted(m.new_of.items()) if o != n}}
            for m in made
        ],
    }
    (out_root / "shuffle_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    names = sorted(by_doc)
    (out_root / "split.json").write_text(json.dumps(
        {"train": names if role == "train" else [], "test": names if role == "test" else [], "excluded": {},
         "strata": {"shuffled": len(made)}, "role": role,
         "note": f"shuffle set: every document is a {role} document"}, indent=1), encoding="utf-8")
    return made


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]], delimiter: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


CONFIG_TEMPLATE = '''r"""Config for the shuffle test in this folder (written by scripts/make_shuffle_test.py).

Everything comes from the main config ({main}); only the data paths point here.
Use it with:   $env:SEQ_CONFIG = "{this}"
Stop using it: Remove-Item Env:SEQ_CONFIG
"""
from pathlib import Path

_MAIN = Path(r"{main}")
_g = globals()
_own_file = _g.get("__file__")
_g["__file__"] = str(_MAIN)  # so the main config's REPO is the repo, not this folder
exec(compile(_MAIN.read_text(encoding="utf-8"), str(_MAIN), "exec"), _g)
_g["__file__"] = _own_file

_ROOT = Path(r"{root}")
RAW_DOCS_ROOT = _ROOT / "raw_images"
RAW_SUBFOLDER = ""
OCR_DOCS_ROOT = _ROOT / "ocr"
OCR_SUBFOLDER = "ocr"
PIPELINE_OUTPUT_CSV = _ROOT / "imaging_output_full.csv"
GROUND_TRUTH_PATH = _ROOT / "ground_truth.csv"
CHARTS_ROOT = _ROOT / "workspace"
SPLIT_FILE = _ROOT / "split.json"
OUTPUT_ROOT = _ROOT / "outputs"
SHUFFLE_TEST = True  # the pipeline's order-dependent values are blanked: v1_csv cannot be scored
'''
