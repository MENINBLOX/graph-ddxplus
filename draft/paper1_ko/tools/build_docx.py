"""Word file of the manuscript for submission: double spacing, page numbers at the bottom centre, and tables with
horizontal rules only (top, under the header, bottom), as the journal asks.

    uv run --with pypandoc_binary --with python-docx python tools/build_docx.py   # from draft/paper1_ko

The author's memo at the end of draft.md is left out.
"""

import re
import tempfile
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

DRAFT = Path(__file__).resolve().parent.parent
SOURCE, OUT = DRAFT / "draft.md", DRAFT / "paper1_draft_ko.docx"
FONT = "Malgun Gothic"
MEMO = "\n---\n\n*작성 메모"


def rule(parent, edge: str, size: int) -> None:
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "single" if size else "nil")
    el.set(qn("w:sz"), str(size))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    parent.append(el)


def horizontal_rules(table) -> None:
    props = table._tbl.tblPr
    for old in props.findall(qn("w:tblBorders")):
        props.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge, size in (("top", 8), ("left", 0), ("bottom", 8), ("right", 0), ("insideH", 0), ("insideV", 0)):
        rule(borders, edge, size)
    props.append(borders)
    for cell in table.rows[0].cells:
        tc = cell._tc.get_or_add_tcPr()
        cell_borders = OxmlElement("w:tcBorders")
        rule(cell_borders, "bottom", 6)
        tc.append(cell_borders)


def page_number(section) -> None:
    p = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def main() -> None:
    md = SOURCE.read_text(encoding="utf-8").split(MEMO)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as tmp:
        tmp.write(md)
    pypandoc.convert_file(
        tmp.name,
        "docx",
        outputfile=str(OUT),
        extra_args=["--from=markdown+pipe_tables+superscript-implicit_figures", f"--resource-path={DRAFT}"],
    )
    Path(tmp.name).unlink()
    d = Document(str(OUT))
    for style in d.styles:
        if style.type == 1:
            style.font.name = FONT
            rpr = style.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            fonts.set(qn("w:eastAsia"), FONT)
    d.styles["Normal"].font.size = Pt(11)
    for p in d.paragraphs:
        p.paragraph_format.line_spacing = 2.0
    for t in d.tables:
        horizontal_rules(t)
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    for r in p.runs:
                        r.font.size = Pt(8)
    for s in d.sections:
        page_number(s)
    d.save(str(OUT))
    print(f"{OUT.name}: {len(d.paragraphs)} paragraphs, {len(d.tables)} tables, {len(d.inline_shapes)} figures")


main()
