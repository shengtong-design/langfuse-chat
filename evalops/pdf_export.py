"""Markdown -> PDF rendering for EvalOps reports.

Pure ReportLab implementation (no cairo/pango/libxml deps), so it
installs cleanly on Streamlit Cloud without a `packages.txt`.

Handles the subset of Markdown the EvalOps reporter emits:
  - Headings (#, ##, ###)
  - Pipe tables with `|---|---|` separator
  - Fenced code blocks (```)
  - Bullet lists (- )
  - Bold (**...**), inline code (`...`), italics (_..._)
  - Horizontal rules (---)
"""

from __future__ import annotations

import re
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\w)_([^_]+?)_(?!\w)")
_CODE_RE = re.compile(r"`([^`]+)`")


def md_to_pdf_bytes(md_text: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        title="EvalOps report",
    )
    styles = _build_styles()
    story = list(_iter_flowables(md_text, styles))
    doc.build(story)
    return buf.getvalue()


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=10, spaceBefore=6, spaceAfter=2),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=8, leading=10),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            leftIndent=12,
        ),
        "code": ParagraphStyle("code", parent=base["Code"], fontSize=7, leading=8.5),
    }


def _iter_flowables(md_text: str, styles: dict[str, ParagraphStyle]):
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            j = i + 1
            code_lines: list[str] = []
            while j < len(lines) and not lines[j].startswith("```"):
                code_lines.append(lines[j])
                j += 1
            if code_lines:
                yield Preformatted("\n".join(code_lines), styles["code"])
                yield Spacer(1, 4)
            i = j + 1
            continue

        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1].rstrip())
        ):
            j = i
            rows: list[list[str]] = []
            while j < len(lines) and lines[j].rstrip().startswith("|"):
                cells = [c.strip() for c in lines[j].rstrip().strip("|").split("|")]
                rows.append(cells)
                j += 1
            if len(rows) >= 2:
                table_data = [rows[0]] + rows[2:]
                yield _make_table(table_data)
                yield Spacer(1, 4)
            i = j
            continue

        if line.startswith("# "):
            yield Paragraph(_inline(line[2:]), styles["h1"])
        elif line.startswith("## "):
            yield Paragraph(_inline(line[3:]), styles["h2"])
        elif line.startswith("### "):
            yield Paragraph(_inline(line[4:]), styles["h3"])
        elif line.strip() == "---":
            yield Spacer(1, 6)
        elif line.startswith("- "):
            yield Paragraph(f"• {_inline(line[2:])}", styles["bullet"])
        elif line.strip() == "":
            yield Spacer(1, 2)
        else:
            yield Paragraph(_inline(line), styles["body"])
        i += 1


def _make_table(rows: list[list[str]]) -> Table:
    cleaned = [[_inline_for_cell(c) for c in row] for row in rows]
    paragraph_rows = [
        [Paragraph(cell, _cell_style()) for cell in row] for row in cleaned
    ]
    t = Table(paragraph_rows, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return t


def _cell_style() -> ParagraphStyle:
    return ParagraphStyle("cell", fontSize=7, leading=8.5)


def _inline_for_cell(text: str) -> str:
    text = _escape(text)
    text = _CODE_RE.sub(r'<font face="Courier">\1</font>', text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    return text


def _inline(text: str) -> str:
    text = _escape(text)
    text = _CODE_RE.sub(r'<font face="Courier">\1</font>', text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    return text


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
