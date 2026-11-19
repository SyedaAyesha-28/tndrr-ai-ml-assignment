"""The QA system: hybrid retrieval (dense + BM25) -> top-5 chunks -> a local LLM answers from them.
Needs Ollama running and a model pulled. Check names with:  ollama list
Usage: python -X utf8 ask.py cp_act.pdf "your question" [ollama_model_name]
A bigger model answers better, e.g. pull qwen2.5:1.5b-instruct and pass it as the 3rd argument."""
import sys

import requests

import rag

pdf, question = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:0.5b-instruct-q4_K_M"
SIZE, K = 100, 5      # 100-word chunks, top-5 (hybrid got 12/15 recall@5 in our evaluation)

pages = rag.load_pages(pdf)
chunks = rag.make_chunks(pages, SIZE)
texts = [c["text"] for c in chunks]
index = rag.build_index(rag.embed(texts))
dense = rag.dense_search(index, rag.embed([question], query=True)[0], 10)
keyword = rag.BM25(texts).search(question, 10)
ids = rag.rrf([dense, keyword], K)

context = "\n\n".join(f"[page {chunks[i]['page']}] {texts[i]}" for i in ids)
prompt = ("You answer questions about the Consumer Protection Act 2019 using ONLY the context below.\n"
          "Reply in 1-2 short sentences. Give the number of days / amount / penalty if the context states it, "
          "and mention the page. If the context does not contain the answer, say 'Not found in the retrieved text.'\n\n"
          f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
r = requests.post("http://localhost:11434/api/generate",
                  json={"model": model, "prompt": prompt, "stream": False,
                        "options": {"temperature": 0}}, timeout=600)
data = r.json()
if "error" in data:
    sys.exit(f"Ollama error: {data['error']}\nRun 'ollama list' and pass the exact model name as the 3rd argument.")

print("QUESTION:", question)
print("\nRETRIEVED (hybrid, top 5):")
for i in ids:
    print(f"  - page {chunks[i]['page']}: {texts[i][:160]}...")
print("\nANSWER:", data["response"].strip())
