"""
fix_figure_layout.py
Transforms cvgip2025_SGA_english_v1.docx → cvgip2025_SGA_english_v2.docx

## Changelog
### v1.0 — 2026-06-16
Added:
  1. Remove extra empty paragraphs between figure image and caption
  2. Image paragraph: space_before=6pt, space_after=0, CENTER
  3. Caption paragraph: space_before=0, space_after=12pt, CENTER
  4. Wrap Fig. 1, 2, 4, 5, 8 with continuous section breaks (two-col → one-col → two-col)
     so those figures render at full page width instead of within a single column
"""

import sys
import io
import re
import shutil
from copy import deepcopy
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = Path(r"D:\Download\cvgip2025_SGA_english_v1.docx")
DST = Path(r"D:\Download\cvgip2025_SGA_english_v2.docx")
FULLWIDTH_FIGS = {1, 2, 4, 5, 8}

# ── Helpers ───────────────────────────────────────────────────────────────────

def has_drawing(p_elem):
    """Return True if the paragraph element contains an embedded image."""
    return p_elem.find(".//" + qn("w:drawing")) is not None


def get_text(p_elem):
    return "".join(t.text or "" for t in p_elem.findall(".//" + qn("w:t")))


def is_blank(p_elem):
    return get_text(p_elem).strip() == "" and not has_drawing(p_elem)


def fig_num_from(text):
    m = re.match(r"Fig\.\s*(\d+)", text.strip())
    return int(m.group(1)) if m else None


def ensure_pPr(p_elem):
    pPr = p_elem.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        p_elem.insert(0, pPr)
    return pPr


def set_spacing(p_elem, before_twips, after_twips):
    pPr = ensure_pPr(p_elem)
    spacing = pPr.find(qn("w:spacing"))
    if spacing is None:
        spacing = OxmlElement("w:spacing")
        # Insert before sectPr if present
        sectPr = pPr.find(qn("w:sectPr"))
        if sectPr is not None:
            pPr.insert(list(pPr).index(sectPr), spacing)
        else:
            pPr.append(spacing)
    if before_twips is not None:
        spacing.set(qn("w:before"), str(before_twips))
    if after_twips is not None:
        spacing.set(qn("w:after"), str(after_twips))


def set_center(p_elem):
    pPr = ensure_pPr(p_elem)
    jc = pPr.find(qn("w:jc"))
    if jc is None:
        jc = OxmlElement("w:jc")
        sectPr = pPr.find(qn("w:sectPr"))
        if sectPr is not None:
            pPr.insert(list(pPr).index(sectPr), jc)
        else:
            pPr.append(jc)
    jc.set(qn("w:val"), "center")


def make_section_para(end_cols, doc_sectPr):
    """
    Create an empty paragraph whose inline sectPr marks the END of an `end_cols`-column
    section.  The section that FOLLOWS this paragraph is governed by the next sectPr or
    the document-level sectPr (whichever comes next in reading order).

    OOXML rule: sectPr must be the LAST element inside pPr.
    """
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")

    # Zero spacing so this invisible break paragraph takes no vertical space
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "0")
    spacing.set(qn("w:after"), "0")
    pPr.append(spacing)

    # sectPr — must be last inside pPr
    sectPr = OxmlElement("w:sectPr")

    wType = OxmlElement("w:type")
    wType.set(qn("w:val"), "continuous")
    sectPr.append(wType)

    # Columns element
    if end_cols == 2 and doc_sectPr is not None:
        orig_cols = doc_sectPr.find(qn("w:cols"))
        if orig_cols is not None:
            sectPr.append(deepcopy(orig_cols))
        else:
            cols = OxmlElement("w:cols")
            cols.set(qn("w:num"), "2")
            cols.set(qn("w:space"), "720")
            sectPr.append(cols)
    else:
        cols = OxmlElement("w:cols")
        cols.set(qn("w:num"), str(end_cols))
        sectPr.append(cols)

    # Copy page size and margins from document-level sectPr
    if doc_sectPr is not None:
        for tag in [qn("w:pgSz"), qn("w:pgMar")]:
            elem = doc_sectPr.find(tag)
            if elem is not None:
                sectPr.append(deepcopy(elem))

    pPr.append(sectPr)  # sectPr always last in pPr
    p.append(pPr)

    # 1-pt invisible run so Word doesn't inflate the paragraph height
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "2")   # half-points → 1 pt
    szCs = OxmlElement("w:szCs")
    szCs.set(qn("w:val"), "2")
    rPr.append(sz)
    rPr.append(szCs)
    r.append(rPr)
    p.append(r)

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

shutil.copy2(SRC, DST)
doc = Document(DST)
body = doc.element.body

# Get document-level sectPr (direct child of <w:body>, NOT nested inside <w:p>)
doc_sectPr = None
for child in body:
    if child.tag == qn("w:sectPr"):
        doc_sectPr = child
        break

if doc_sectPr is not None:
    cols_elem = doc_sectPr.find(qn("w:cols"))
    cols_num = cols_elem.get(qn("w:num")) if cols_elem is not None else "?"
    print(f"Document-level sectPr found: cols={cols_num}")
else:
    print("WARNING: no document-level sectPr found")

# ── 1. Scan all body paragraphs for figure structures ─────────────────────────
all_p = [c for c in body if c.tag == qn("w:p")]

fig_structures = []
seen_ids = set()

for i, p in enumerate(all_p):
    if not has_drawing(p) or id(p) in seen_ids:
        continue
    seen_ids.add(id(p))

    empty_elems = []
    caption_elem = None
    fig_n = None

    j = i + 1
    while j < len(all_p):
        nxt = all_p[j]
        if is_blank(nxt):
            empty_elems.append(nxt)
            j += 1
        elif get_text(nxt).strip().startswith("Fig."):
            caption_elem = nxt
            fig_n = fig_num_from(get_text(nxt).strip())
            break
        else:
            break

    fig_structures.append({
        "img":     p,
        "empties": empty_elems,
        "caption": caption_elem,
        "fig_num": fig_n,
    })
    print(
        f"  Fig {fig_n}: {len(empty_elems)} empty para(s) removed | "
        f"caption: {get_text(caption_elem).strip()[:50] if caption_elem is not None else 'NOT FOUND'}"
    )

# ── 2. Fix image paragraph spacing ───────────────────────────────────────────
for fs in fig_structures:
    set_spacing(fs["img"], before_twips=120, after_twips=0)   # 6pt before, 0 after
    set_center(fs["img"])

# ── 3. Fix caption paragraph spacing ─────────────────────────────────────────
for fs in fig_structures:
    cap = fs["caption"]
    if cap is None:
        continue
    set_spacing(cap, before_twips=0, after_twips=240)  # 0 before, 12pt after
    set_center(cap)

# ── 4. Remove empty paragraphs between image and caption ─────────────────────
removed = 0
for fs in fig_structures:
    for ep in fs["empties"]:
        parent = ep.getparent()
        if parent is not None:
            parent.remove(ep)
            removed += 1
print(f"\nRemoved {removed} empty paragraph(s).")

# ── 5. Add full-width section breaks for specified figures ────────────────────
fullwidth = [fs for fs in fig_structures if fs["fig_num"] in FULLWIDTH_FIGS]
print(f"Adding full-width section breaks for Fig: {sorted(fs['fig_num'] for fs in fullwidth)}")

# Process in reverse document order so insertions don't invalidate earlier references
for fs in reversed(fullwidth):
    img_p  = fs["img"]
    cap_p  = fs["caption"]
    fig_n  = fs["fig_num"]

    # After caption: paragraph that ENDS the single-column section (col=1)
    if cap_p is not None:
        after_brk = make_section_para(1, doc_sectPr)
        cap_p.addnext(after_brk)

    # Before image: paragraph that ENDS the two-column section (col=2)
    before_brk = make_section_para(2, doc_sectPr)
    img_p.addprevious(before_brk)

    print(f"  Fig {fig_n}: section breaks inserted (two-col → one-col → two-col)")

# ── 6. Save ───────────────────────────────────────────────────────────────────
doc.save(DST)
print(f"\nSaved: {DST}")
