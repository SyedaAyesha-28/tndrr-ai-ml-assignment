"""RAG from scratch: PDF -> chunks -> embeddings -> FAISS, plus BM25 written in NumPy.
No LangChain / LlamaIndex. Only: pypdf (read PDF), sentence-transformers (embeddings), faiss, numpy."""
import re
from collections import Counter

import faiss
import numpy as np
from pypdf import PdfReader

EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge wants this on queries only
_model = None


def clean(text):
    text = text.replace("-\n", "")          # join words hyphenated at line end
    return re.sub(r"\s+", " ", text).strip()


def norm(text):
    """Used only for matching evidence to chunks."""
    return clean(text).lower()


def load_pages(pdf_path):
    return [clean(p.extract_text() or "") for p in PdfReader(pdf_path).pages]


def make_chunks(pages, size, overlap_frac=0.2):
    """Sliding window over words. size = words per chunk. Each chunk remembers its start page."""
    words, page_of = [], []
    for i, p in enumerate(pages, 1):
        w = p.split()
        words += w
        page_of += [i] * len(w)
    step = max(1, int(size * (1 - overlap_frac)))
    chunks = []
    for s in range(0, len(words), step):
        piece = words[s:s + size]
        if len(piece) < size * 0.3 and chunks:   # skip a tiny tail
            break
        chunks.append({"text": " ".join(piece), "page": page_of[s]})
        if s + size >= len(words):
            break
    return chunks


def embed(texts, query=False, batch=64):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMB_MODEL)
    if query:
        texts = [QUERY_PREFIX + t for t in texts]
    return _model.encode(texts, normalize_embeddings=True, batch_size=batch,
                         show_progress_bar=False).astype("float32")


def build_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])   # inner product on normalised vectors = cosine
    index.add(vectors)
    return index


def dense_search(index, qvec, k=5):
    scores, ids = index.search(qvec.reshape(1, -1), k)
    return ids[0].tolist()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Keyword scoring: rare words count more (idf), repeats help but saturate (k1), long chunks are penalised (b)."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in toks]
        self.len = np.array([len(t) for t in toks], dtype=float)
        self.avg = self.len.mean()
        n = len(docs)
        df = Counter(w for t in toks for w in set(t))
        self.idf = {w: np.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query, k=5):
        scores = np.zeros(len(self.tf))
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    scores[i] += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return np.argsort(-scores)[:k].tolist()


def rrf(rank_lists, k=5, c=60):
    """Reciprocal rank fusion: merge several ranked lists by summing 1/(c+rank)."""
    score = Counter()
    for lst in rank_lists:
        for r, i in enumerate(lst):
            score[i] += 1 / (c + r + 1)
    return [i for i, _ in score.most_common(k)]
