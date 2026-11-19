"""Step 1: check that your PDF is readable and has 30+ pages of real text.
Usage: python check_pdf.py your.pdf
Writes doc_text.txt (with page markers). COPY your question evidence from that file."""
import sys
from rag import load_pages

pdf = sys.argv[1]
pages = load_pages(pdf)
words = sum(len(p.split()) for p in pages)
print(f"pages: {len(pages)} | total words: {words} | avg words/page: {words // max(1, len(pages))}")
if len(pages) < 30:
    print("WARNING: fewer than 30 pages. The assignment needs at least 30.")
if words // max(1, len(pages)) < 100:
    print("WARNING: very little text per page. The PDF may be scanned images (needs OCR).")
with open("doc_text.txt", "w", encoding="utf-8") as f:
    for i, p in enumerate(pages, 1):
        f.write(f"\n=== PAGE {i} ===\n{p}\n")
print("Wrote doc_text.txt. Open it and check that the text looks clean.")
for i in (2, len(pages) // 2):
    print(f"\n--- sample from page {i + 1} ---\n{pages[i][:500]}")
