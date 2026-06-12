"""
CVGIP 2025 -- Convert cvgip2025_SGA_chinese.docx to PDF via MS Word COM
File   : docx_to_pdf.py
Purpose: cvgip2025_chinese.py only outputs .docx. The user also wants a PDF
         for review. MS Word (WINWORD.EXE) is installed at
         "C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE"
         and pywin32 is available, so use Word's COM automation
         (ExportAsFixedFormat) to produce a PDF with identical layout to
         what the user would see opening the .docx in Word.
Input  : D:\\Contest\\AI GO\\paper\\cvgip2025_SGA_chinese.docx
Output : D:\\Contest\\AI GO\\paper\\cvgip2025_SGA_chinese.pdf
Failure: exits non-zero if the .docx is missing or Word COM automation
         fails (e.g. Word not installed/licensed on this machine).

ENVIRONMENT:
  cd "D:\\Contest\\AI GO\\paper"
  C:\\Users\\bubbl\\anaconda3\\python.exe docx_to_pdf.py

## Changelog
### v1.0.0 -- 2026-06-13
**Added:**
- Initial version.
"""

import sys
from pathlib import Path

import win32com.client

DOCX_PATH = Path(r"D:\Contest\AI GO\paper\cvgip2025_SGA_chinese.docx")
PDF_PATH = Path(r"D:\Contest\AI GO\paper\cvgip2025_SGA_chinese.pdf")

WD_FORMAT_PDF = 17  # wdExportFormatPDF


def main():
    if not DOCX_PATH.is_file():
        sys.exit(f"[FATAL] missing docx: {DOCX_PATH}")

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(DOCX_PATH))
        try:
            doc.ExportAsFixedFormat(str(PDF_PATH), WD_FORMAT_PDF)
        finally:
            doc.Close(False)
    finally:
        word.Quit()

    print(f"Saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
