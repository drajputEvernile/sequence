"""Render docs/SEQUENCING_OVERVIEW.md to SEQUENCING_OVERVIEW.pdf at the repo root.

    pip install reportlab
    python docs/build_pdf.py

Handles the Markdown the document uses: # / ## / ### headings, paragraphs, - and 1. lists,
| tables |, **bold** and `code`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "SEQUENCING_OVERVIEW.md"
OUT = ROOT / "SEQUENCING_OVERVIEW.pdf"

ACCENT = colors.HexColor("#1F4E79")
RULE = colors.HexColor("#C9D3DD")
SHADE = colors.HexColor("#EEF2F6")

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=base["Title"], fontSize=19, leading=23, textColor=ACCENT, alignment=TA_LEFT,
                            spaceAfter=4),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13.5, leading=17, textColor=ACCENT, spaceBefore=12,
                         spaceAfter=5),
    "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5),
    "meta": ParagraphStyle("m", parent=base["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"),
                           spaceAfter=8),
    "cell": ParagraphStyle("c", parent=base["BodyText"], fontSize=8.3, leading=10.4),
    "head": ParagraphStyle("h", parent=base["BodyText"], fontSize=8.3, leading=10.4, fontName="Helvetica-Bold",
                           textColor=colors.white),
}


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def table(rows: list[list[str]], width: float) -> Table:
    data = [[Paragraph(inline(c), S["head"] if r == 0 else S["cell"]) for c in row] for r, row in enumerate(rows)]
    cols = len(rows[0])
    lengths = [max(len(row[k]) for row in rows) for k in range(cols)]
    total = sum(lengths) or 1
    widths = [width * max(0.12, min(0.7, n / total)) for n in lengths]
    scale = width / sum(widths)
    t = Table(data, colWidths=[w * scale for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SHADE]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build() -> Path:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Page Sequencing: v1 and the plan for v2")
    width = A4[0] - 36 * mm
    story: list = []
    para: list[str] = []
    items: list[str] = []
    ordered = False
    rows: list[list[str]] = []
    first_para = True

    def flush():
        nonlocal para, items, rows, first_para
        if para:
            style = S["meta"] if first_para and story and story[-1].style.name == "t" else S["body"]
            story.append(Paragraph(inline(" ".join(para)), style))
            first_para = False
            para = []
        if items:
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(i), S["body"]), leftIndent=12) for i in items],
                bulletType="1" if ordered else "bullet", start="1" if ordered else None, leftIndent=14,
                bulletFormat="%s." if ordered else None,
                bulletFontSize=8 if not ordered else 9.6,
            ))
            items = []
        if rows:
            story.append(table(rows, width))
            story.append(Spacer(1, 6))
            rows = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|"):
            if para or items:
                flush()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                rows.append(cells)
            continue
        if rows:
            flush()
        m_list = re.match(r"^(\s*)(-|\d+\.)\s+(.*)", line)
        if line.startswith("# "):
            flush()
            story.append(Paragraph(inline(line[2:]), S["title"]))
        elif line.startswith("## "):
            flush()
            story.append(Paragraph(inline(line[3:]), S["h2"]))
        elif line.startswith("### "):
            flush()
            story.append(Paragraph(inline(line[4:]), S["h3"]))
        elif m_list:
            if para:
                flush()
            is_ordered = m_list.group(2) != "-"
            if items and is_ordered != ordered:
                flush()
            ordered = is_ordered
            items.append(m_list.group(3))
        elif not line.strip():
            flush()
        elif items:
            items[-1] += " " + line.strip()
        else:
            para.append(line.strip())
    flush()

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(18 * mm, 9 * mm, "Page sequencing: v1 and the plan for v2")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(build())
    sys.exit(0)
