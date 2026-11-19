"""Step 3: recall@k for two chunk sizes, for dense, BM25 and hybrid search.
Usage: python eval_rag.py your.pdf questions.json
questions.json: [{"q": "...", "evidence": "exact short phrase from doc_text.txt", "page": 12}, ...]
A retrieved chunk is a HIT if it contains the evidence phrase."""
import json
import sys

import rag

CHUNK_SIZES = [100, 300]      # words per chunk (about 130 and 400 tokens)
KS = [1, 3, 5]

pdf, qfile = sys.argv[1], sys.argv[2]
pages = rag.load_pages(pdf)
full = rag.norm(" ".join(pages))
questions = json.load(open(qfile, encoding="utf-8"))

bad = False
for i, q in enumerate(questions, 1):
    ev = rag.norm(q["evidence"])
    if ev not in full:
        print(f"[Q{i}] evidence NOT found in the document text. Copy it from doc_text.txt: {q['evidence'][:60]}")
        bad = True
    if len(ev.split()) > 20:
        print(f"[Q{i}] evidence is longer than 20 words; shorten it.")
        bad = True
if bad:
    sys.exit("Fix the questions above, then rerun.")
print(f"{len(questions)} questions OK\n")

lines = []
fail_report = []
for size in CHUNK_SIZES:
    chunks = rag.make_chunks(pages, size)
    texts = [c["text"] for c in chunks]
    normed = [rag.norm(t) for t in texts]
    index = rag.build_index(rag.embed(texts))
    bm25 = rag.BM25(texts)
    qvecs = rag.embed([q["q"] for q in questions], query=True)
    hits = {m: {k: 0 for k in KS} for m in ("dense", "bm25", "hybrid")}
    for qi, q in enumerate(questions):
        ev = rag.norm(q["evidence"])
        dense = rag.dense_search(index, qvecs[qi], 10)
        kw = bm25.search(q["q"], 10)
        runs = {"dense": dense, "bm25": kw, "hybrid": rag.rrf([dense, kw], 10)}
        for m, ids in runs.items():
            for k in KS:
                if any(ev in normed[i] for i in ids[:k]):
                    hits[m][k] += 1
        if not any(ev in normed[i] for i in dense[:5]):
            top = chunks[dense[0]]
            fail_report.append(f"[chunk={size}] Q: {q['q']}\n   needed (page {q.get('page','?')}): {q['evidence']}\n"
                               f"   top-1 was page {top['page']}: {top['text'][:200]}...\n")
    n = len(questions)
    lines.append(f"chunk size {size} words | {len(chunks)} chunks")
    for m in hits:
        lines.append(f"  {m:7s} " + "  ".join(f"recall@{k}={hits[m][k]}/{n} ({100*hits[m][k]/n:.0f}%)" for k in KS))
    lines.append("")

out = "\n".join(lines) + "\nDENSE FAILURES (recall@5 missed):\n" + ("\n".join(fail_report) or "none\n")
print(out)
open("results.txt", "w", encoding="utf-8").write(out)
print("Saved results.txt")
