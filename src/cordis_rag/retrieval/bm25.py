import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def tokenize(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in ENGLISH_STOP_WORDS]


class BM25Retriever:
    def __init__(self, chunks):
        self.cids = chunks["chunk_id"].tolist()
        self.pids = chunks["project_id"].tolist()
        self.bm25 = BM25Okapi([tokenize(t) for t in chunks["text"]])

    def search(self, q, k=50):
        s = self.bm25.get_scores(tokenize(q))
        idx = np.argsort(-s)[:k]
        return [(self.cids[i], self.pids[i], float(s[i])) for i in idx]
