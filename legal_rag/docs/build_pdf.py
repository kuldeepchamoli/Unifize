"""Render docs/ARCHITECTURE.md to a print-quality PDF.

Run:
    python -m docs.build_pdf
or
    python docs/build_pdf.py

Outputs:
    docs/ARCHITECTURE.html
    docs/ARCHITECTURE.pdf
"""
import os
from pathlib import Path

# WeasyPrint needs Pango/Cairo from Homebrew. Set the dyld search path
# BEFORE importing weasyprint so cffi can dlopen the system libs.
_brew_lib = "/opt/homebrew/lib"
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
    _brew_lib + ":" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
)

import markdown
from weasyprint import HTML, CSS

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "ARCHITECTURE.md"
HTML_OUT = DOCS / "ARCHITECTURE.html"
PDF_OUT = DOCS / "ARCHITECTURE.pdf"

CSS_STYLE = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 4pt;
     margin-top: 18pt; page-break-after: avoid; }
h2 { font-size: 15pt; color: #1a4d80; margin-top: 16pt;
     border-bottom: 1px solid #ccc; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 12pt; color: #333; margin-top: 12pt; page-break-after: avoid; }
h4 { font-size: 11pt; color: #555; margin-top: 10pt; page-break-after: avoid; }
p, li { text-align: justify; }
code { font-family: "SF Mono", Menlo, Consolas, monospace;
       background: #f3f3f5; padding: 1pt 3pt; border-radius: 2pt; font-size: 9.5pt; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 4pt;
      padding: 8pt 10pt; font-size: 9pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0;
        font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 600; }
blockquote { border-left: 3pt solid #1a4d80; margin: 6pt 0; padding: 2pt 10pt;
             color: #444; font-style: italic; }
strong { color: #000; }
hr { border: none; border-top: 1pt solid #999; margin: 14pt 0; }
"""


def main():
    md_text = SRC.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
    html_doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Legal RAG Architecture</title></head>"
        f"<body>{body_html}</body></html>"
    )
    HTML_OUT.write_text(html_doc, encoding="utf-8")
    HTML(string=html_doc, base_url=str(DOCS)).write_pdf(
        PDF_OUT, stylesheets=[CSS(string=CSS_STYLE)]
    )
    print(f"Wrote {HTML_OUT}")
    print(f"Wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
