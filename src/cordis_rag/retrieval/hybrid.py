import math, numpy as np, pandas as pd
from src.cordis_rag.chunking import build_chunks
from src.cordis_rag.retrieval.bm25 import BM25Retriever
from src.cordis_rag.eval.metrics import project_ranking, gold_ranks 

PREFIX = "Represent this sentence for searching relevant passages: "

def rrf(lists, k, pid_of):
    sc = {}
    for hits in lists:
        for rank, (cid, _, _) in enumerate(hits, 1):
            sc[cid] = sc.get(cid, 0.0) + 1.0 / (k + rank)
    return [(c, pid_of[c], s) for c, s in sorted(sc.items(), key=lambda x: -x[1])]

def rank_all(hit_lists, golds):
    return gold_ranks([project_ranking(h) for h in hit_lists], golds)

class Index:
    def __init__(self, df, size, ov):
        tag = f"bge-small_{size}_{ov}"
        self.ch = build_chunks(df, size, ov)
        self.E = np.load(f"data/processed/emb_{tag}.npy")
        ids = pd.read_parquet(f"data/processed/emb_{tag}_ids.parquet")
        assert ids["chunk_id"].tolist() == self.ch["chunk_id"].tolist(), f"order mismatch {tag}"
        self.cids = self.ch["chunk_id"].tolist()
        self.pids = self.ch["project_id"].tolist()
        self.pid_of = dict(zip(self.cids, self.pids))
        self.text_of = dict(zip(self.cids, self.ch["text"]))
        self.depth = math.ceil(50 * len(self.ch) / len(df))
        self.bm = BM25Retriever(self.ch)

    def bm25_hits(self, questions):
        return [self.bm-search(q, k=self.depth) for q in questions]

    def dense_hits(self, Q):
        out = []
        for s in Q @ self.E.T:
            idx = np.argsort(-s)[:self.depth]
            out.append([(self.cids[i], self.pids[i], float(s[i])) for i in idx])
        return out

    def hybrid_hits(self, bm_h, de_h, k=60):
        return [rrf([b, d], k, self.pid_of) for b, d in zip(bm_h, de_h)]


