"""Search doc_text.txt for a word or phrase and show the text around it with the page number.
Usage: python find.py "thirty days"      (run check_pdf.py first so doc_text.txt exists)
Copy your evidence phrase from what this prints (8 to 20 words)."""
import re
import sys

term = sys.argv[1].lower()
text = open("doc_text.txt", encoding="utf-8").read()
parts = re.split(r"=== PAGE (\d+) ===", text)   # ['', '1', 'page1 text', '2', 'page2 text', ...]
found = 0
for i in range(1, len(parts), 2):
    page, body = parts[i], " ".join(parts[i + 1].split())
    low = body.lower()
    start = 0
    while (j := low.find(term, start)) != -1:
        found += 1
        print(f"[page {page}] ...{body[max(0, j - 120): j + len(term) + 160]}...\n")
        start = j + len(term)
print(f"{found} match(es) for '{term}'")
