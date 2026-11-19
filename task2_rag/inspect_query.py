"""Look at what the search actually retrieved for ONE question, so you can explain a failure.
Usage: python inspect_query.py cp_act.pdf "your question" 100
(last number = chunk size in words, 100 or 300)"""
import sys

import rag

pdf, question = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 100
pages = rag.load_pages(pdf)
chunks = rag.make_chunks(pages, size)
texts = [c["text"] for c in chunks]
index = rag.build_index(rag.embed(texts))
ids = rag.dense_search(index, rag.embed([question], query=True)[0], 5)
for rank, i in enumerate(ids, 1):
    print(f"#{rank} | page {chunks[i]['page']} | chunk {i}\n{texts[i][:500]}\n")
