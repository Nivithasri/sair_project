"""Phase 9 - export docs/REPORT.md to docs/REPORT.pdf (python-markdown -> HTML -> headless Microsoft Edge print-to-PDF).

Usage: python src/export_report.py
"""
import os
import subprocess

import fitz
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "REPORT.md")
HTML = os.path.join(ROOT, "docs", "_report_print.html")             # temporary, next to the .md so relative image paths work
PDF = os.path.join(ROOT, "docs", "REPORT.pdf")
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Cambria", "Georgia", serif; font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 18pt; text-align: center; margin: 0 0 6pt 0; line-height: 1.25; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 12pt; page-break-after: avoid; }
p { margin: 5pt 0; text-align: justify; }
table { border-collapse: collapse; margin: 6pt 0 10pt 0; font-size: 8pt; page-break-inside: avoid; width: 100%; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #eef1f5; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt auto; page-break-inside: avoid; }
em { color: #333; }
code { font-family: Consolas, monospace; font-size: 8.5pt; background: #f4f4f4; padding: 0 2pt; }
hr { border: 0; border-top: 1px solid #ccc; margin: 12pt 0; }
li { margin: 2pt 0; }
"""


def main():
    text = open(SRC, encoding="utf-8").read()
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>REPORT</title><style>{CSS}</style></head><body>{body}</body></html>")
    edge = next(p for p in EDGE if os.path.exists(p))
    url = "file:///" + HTML.replace("\\", "/")
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", url], check=True,
                   capture_output=True, timeout=180)
    os.remove(HTML)
    doc = fitz.open(PDF)
    print(f"wrote {PDF}: {doc.page_count} pages")


if __name__ == "__main__":
    main()
