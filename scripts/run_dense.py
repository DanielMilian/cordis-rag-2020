import json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks
from src.cordis_rag.retrieval.bm25 import BM25Retriever
from src.cordis_rag.eval.metrics import project_ranking, gold_ranks, summarize, paired_bootstrap
from src.cordis_rag.eval.results import log_result

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

hit5 = lambda r: float((r <= 5).mean())
mrr = lambda r: summarize(r)["mrr@10"]
print("control, bm25 vs itself:", paired_bootstrap(r_bm, r_bm, hit5), "(expect 0, 0, 0)")
for k in ("dense", "dense+prefix"):
    for name, fn in (("hit@5", hit5), ("mrr@10", mrr)):
        d, lo, hi = paired_bootstrap(r_bm, runs[k], fn)
        print(f"{k} - bm25 {name}: {d:+.3f} CI=({lo:+.3f},{hi:+.3f})")
for k, cfg in (("dense", "prefix=no"), ("dense+prefix", "prefix=yes")):
    log_result("dense-bge-small", cfg, SIZE, OV, split, len(qs), summarize(runs[k]))

