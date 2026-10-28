import json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks
from src.cordis_rag.retrieval.bm25 import BM25Retriever
from src.cordis_rag.eval.metrics import project_ranking, gold_ranks, summarize

SIZE, OV, split = 200, 40, "dev"
tag = f"bge-small_{SIZE}_{OV}"
PREFIX = "Represent this sentence for searching relevant passages: "
df = pd.read_parquet("data/processed/projects.parquet")
ch = build_chunks(df, SIZE, OV)
E = np.load(f"data/processed/emb_{tag}.npy")
ids = pd.read_parquet(f"data/processed/emb_{tag}_ids.parquet")
assert ids["chunk_id"].tolist() == ch["chunk_id"].tolist(), "order mismatch"

cids, pids = ch["chunk_id"].tolist(), ch["project_id"].tolist()
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
qs = [json.loads(l) for l in open(f"eval_data/eval_{split}.jsonl")]
golds = [q["project_id"] for q in qs]

def dense_ranks(prefix):
    Q = m.encode([prefix + q["question"] for q in qs], normalize_embeddings=True)
    out = []
    for s in Q @ E.T:
        idx = np.argsort(-s)[:50]
        out.append(project_ranking([(cids[i], pids[i], float(s[i])) for i in idx]))
    return gold_ranks(out, golds)


bm = BM25Retriever(ch)
r_bm = gold_ranks([project_ranking(bm.search(q["question"])) for q in qs], golds)
runs = {"bm25": r_bm, "dense": dense_ranks(""), "dense+prefix": dense_ranks(PREFIX)}
for k, r in runs.items():
    print(f"{k:13s}", {a: round(b, 3) for a, b in summarize(r).items()})
for k in ("dense", "dense+prefix"):
    r = runs[k]
    print(k, "vs bm25 -> better:", int((r < r_bm).sum()), "worse:", int((r > r_bm).sum()),
          "tie:", int((r == r_bm).sum()))  
r_d = runs["dense+prefix"]
for i, q in enumerate(qs):
    if r_bm[i] != r_d[i] or r_bm[i] > 1 or r_d[i] > 1:
        print(f"[{q['style'][:4]}] bm25={r_bm[i]:g} dense={r_d[i]:g} | {q['question'][:90]}")
