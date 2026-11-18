"""Local OCR for photographed pages: Tesseract on this machine, no network call.

Writes a chart folder in the CHARTS_ROOT layout so every sequencer reads it like a pipeline
chart:

    <out>/<chart>/pages/1.jpg, 2.jpg ...          the photos, renamed in the order given
    <out>/<chart>/ocr/<chart>_prelim.txt          Tesseract text
    <out>/<chart>/ocr/<chart>_final2.json         the same text as lines with positions, in the
                                                  pipeline's final2 shape (pagesMeta[0].lines[])

The per-page CSVs (junk, page type, DOS ...) do not exist for such a chart; the sequencers fall
back to what they read from the text, which is what you want when studying a failure.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic")


def _prepare(src: Path, dst: Path, min_side: int = 1800) -> tuple[int, int]:
    """Phone photo -> upright grayscale PNG large enough for Tesseract; returns (width, height)."""
    from PIL import Image, ImageOps

    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        w, h = image.size
        if min(w, h) < min_side:
            scale = min_side / min(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        image = ImageOps.autocontrast(image, cutoff=1)
        image.save(dst)
        return image.size


def tesseract_lines(image: Path, cmd: Path, lang: str = "eng", psm: int = 3) -> list[dict]:
    """Lines with their boxes from ``tesseract <image> stdout tsv``."""
    out = subprocess.run(
        [str(cmd), str(image), "stdout", "-l", lang, "--psm", str(psm), "tsv"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    groups: dict[tuple, dict] = {}
    for row in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text or float(row.get("conf") or -1) < 0:
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        left, top, width, height = (int(row[k]) for k in ("left", "top", "width", "height"))
        line = groups.setdefault(key, {"words": [], "x0": left, "y0": top, "x1": left + width, "y1": top + height})
        line["words"].append(text)
        line["x0"], line["y0"] = min(line["x0"], left), min(line["y0"], top)
        line["x1"], line["y1"] = max(line["x1"], left + width), max(line["y1"], top + height)
    lines = sorted(groups.values(), key=lambda l: (l["y0"], l["x0"]))
    return [
        {
            "content": " ".join(l["words"]),
            "polygon": [l["x0"], l["y0"], l["x1"], l["y0"], l["x1"], l["y1"], l["x0"], l["y1"]],
        }
        for l in lines
    ]


def ocr_chart(photos: list[Path], chart_name: str, out_root: Path, cmd: Path, lang: str = "eng") -> Path:
    """OCR ``photos`` (in page order) into ``out_root/chart_name``; returns the chart folder."""
    if not Path(cmd).is_file():
        raise FileNotFoundError(f"Tesseract not found at {cmd} (config.TESSERACT_CMD)")
    chart_dir = Path(out_root) / chart_name
    (chart_dir / "pages").mkdir(parents=True, exist_ok=True)
    (chart_dir / "ocr").mkdir(parents=True, exist_ok=True)
    final2_pages, prelim_blocks = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for number, photo in enumerate(photos, start=1):
            name = f"{number}{photo.suffix.lower() if photo.suffix.lower() != '.jpeg' else '.jpg'}"
            shutil.copy2(photo, chart_dir / "pages" / name)
            prepared = Path(tmp) / f"{number}.png"
            width, height = _prepare(photo, prepared)
            lines = tesseract_lines(prepared, cmd, lang)
            text = "\n".join(l["content"] for l in lines)
            logger.info("%s page %d (%s): %d lines", chart_name, number, photo.name, len(lines))
            final2_pages.append({
                "fileName": name,
                "pageNumber": number,
                "content": text,
                "pagesMeta": [{"pageNumber": 1, "width": width, "height": height, "unit": "pixel", "lines": lines}],
                "source": f"local tesseract ({photo.name})",
            })
            prelim_blocks.append(f"===== {name} =====\n{text}\n")
    (chart_dir / "ocr" / f"{chart_name}_final2.json").write_text(
        json.dumps({"recordId": chart_name, "model": "local-tesseract", "pageCount": len(final2_pages),
                    "pages": final2_pages}, indent=2),
        encoding="utf-8",
    )
    (chart_dir / "ocr" / f"{chart_name}_prelim.txt").write_text("\n".join(prelim_blocks), encoding="utf-8")
    return chart_dir


def photos_in(folder: Path) -> list[Path]:
    """The folder's images in natural name order (2.jpg before 10.jpg)."""
    import re

    def key(p: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

    return sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=key)
