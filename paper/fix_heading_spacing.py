"""
fix_heading_spacing.py
Transforms cvgip2025_SGA_english_v2.docx → cvgip2025_SGA_english_v3.docx

Matches CVGIP2026 template blank-line heading structure:
  - Major headings (1., 2., ..., Abstract, References):
      blank line BEFORE + blank line AFTER, clear paragraph bef/aft spacing
  - Subheadings (2.1., 3.1., ...):
      blank line BEFORE + blank line AFTER, clear bef/aft
  - Sub-subheadings (3.2.1., 4.1.2., ...):
      blank line BEFORE only, NO blank after (text follows directly), clear bef/aft

Processing order: bottom → top (reverse) so later insertions don't shift
earlier sibling checks.

## Changelog
### v1.0 — 2026-06-16
Added: Initial heading spacing script (v2 → v3)
"""

import sys
import io
import re
import shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = Path(r"D:\Download\cvgip2025_SGA_english_v2.docx")
DST = Path(r"D:\Download\cvgip2025_SGA_english_v3.docx")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_text(p_elem):
    return "".join(t.text or "" for t in p_elem.findall(".//" + qn("w:t")))


def heading_level(p_elem):
    """
    Classify paragraph heading level.
    Returns 1 (major: '1.', '2.', 'Abstract', 'References'),
            2 (sub: '2.1.', '3.2.'),
            3 (sub-sub: '3.2.1.', '4.1.2.'),
            0 (not a heading).
    """
    if p_elem.tag != qn("w:p"):
        return 0
    t = get_text(p_elem).strip()
    if not t:
        return 0
    if re.match(r"^\d+\.\d+\.\d+", t):
        return 3
    if re.match(r"^\d+\.\d+", t):
        return 2
    if re.match(r"^\d+\.", t) and not re.match(r"^\d+\.\d+", t):
        return 1
    if t in ("References", "Abstract"):
        return 1
    return 0


def is_structural_blank(elem):
    """
    True if elem is an empty paragraph (no visible text, no image, no section break).
    Section-break paragraphs have w:sectPr and are explicitly excluded — they are
    structural and must not be counted as blank separators.
    """
    if elem is None or elem.tag != qn("w:p"):
        return False
    if get_text(elem).strip():
        return False
    if elem.find(".//" + qn("w:drawing")) is not None:
        return False
    if elem.find(".//" + qn("w:sectPr")) is not None:
        return False
    return True


def clear_spacing(p_elem):
    """
    Remove explicit w:before and w:after from a paragraph's spacing element so that
    the paragraph reverts to its style's default (0 for Normal).
    """
    pPr = p_elem.find(qn("w:pPr"))
    if pPr is None:
        return
    spacing = pPr.find(qn("w:spacing"))
    if spacing is None:
        return
    for attr in [qn("w:before"), qn("w:after"),
                 qn("w:beforeLines"), qn("w:afterLines")]:
        spacing.attrib.pop(attr, None)
    if len(spacing.attrib) == 0 and len(spacing) == 0:
        pPr.remove(spacing)


def make_blank_para():
    """Return a new empty Normal paragraph with no spacing."""
    return OxmlElement("w:p")


# ── Main ──────────────────────────────────────────────────────────────────────

shutil.copy2(SRC, DST)
doc = Document(DST)
body = doc.element.body

# Collect heading elements in document order
heading_elems = []
for child in body:
    lvl = heading_level(child)
    if lvl > 0:
        heading_elems.append((child, lvl))
        print(f"  L{lvl}: {get_text(child).strip()[:60]}")

print(f"\nTotal headings: {len(heading_elems)}")

inserted = 0

# Process BOTTOM → TOP to keep earlier sibling references stable
for h_elem, lvl in reversed(heading_elems):

    # ── Clear existing bef/aft spacing from the heading itself ───────────────
    clear_spacing(h_elem)

    # ── Ensure blank line BEFORE the heading ─────────────────────────────────
    prev = h_elem.getprevious()
    if is_structural_blank(prev):
        # Normalize existing blank — clear its spacing so it's truly empty
        clear_spacing(prev)
    else:
        h_elem.addprevious(make_blank_para())
        inserted += 1

    # ── Ensure blank line AFTER the heading (levels 1 & 2 only) ─────────────
    if lvl <= 2:
        nxt = h_elem.getnext()
        if is_structural_blank(nxt):
            clear_spacing(nxt)
        else:
            h_elem.addnext(make_blank_para())
            inserted += 1

print(f"Inserted {inserted} blank paragraph(s).")

doc.save(DST)
print(f"\nSaved: {DST}")
