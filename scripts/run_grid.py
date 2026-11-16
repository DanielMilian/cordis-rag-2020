import sys, json, math, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks
from src.cordis_rag.retrieval.bm25 import BM25Retriever
from src.cordis_rag.eval.metrics import project_ranking, gold_ranks, summarize, paired_bootstrap
from src.cordis_rag.eval.results import log_result

split, K_RRF, LOGGING = "dev", 60, "--log" in sys.argv
PREFIX = "Represent this sentence for searching relevant passages: "
CONFIGS = [(200, 40), (400, 0), (128, 0), (128, 32), (200, 0), (200, 50), (64, 0), (64, 16)]
df = pd.read_parquet("data/processed/projects.parquet")
qs = [json.loads(l) for l in open(f"eval_data/eval_{split}.jsonl")]
golds = [q["project_id"] for q in qs]
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
Q = m.encode([PREFIX + q["question"] for q in qs], normalize_embeddings=True)

def rrf(lists, k, pid_of):
    sc = {}
    for hits in lists:
        for rank, (cid, _, _) in enumerate(hits, 1):
            sc[cid] = sc.get(cid, 0.0) + 1.0 / (k + rank)
    return [(c, pid_of[c], s) for c, s in sorted(sc.items(), key=lambda x: -x[1])]

def run_config(size, ov):
    tag = f"bge-small_{size}_{ov}"
    ch = build_chunks(df, size, ov)
    E = np.load(f"data/processed/emb_{tag}.npy")
    ids = pd.read_parquet(f"data/processed/emb_{tag}_ids.parquet")
    assert ids["chunk_id"].tolist() == ch["chunk_id"].tolist(), f"order mismatch {tag}"
    cids, pids = ids["chunk_id"].tolist(), ch["project_id"].tolist()
    pid_of = dict(zip(cids, pids))
    depth = math.ceil(50 * len(ch) / len(df))
    bm = BM25Retriever(ch)
    bm_h = [bm.search(q["question"], k=depth) for q in qs]
    de_h = []
    for s in Q @ E.T:
        idx = np.argsort(-s)[:depth]
        de_h.append([(cids[i], pids[i], float(s[i])) for i in idx])
    hy_h = [rrf([b, d], K_RRF, pid_of) for b, d in zip(bm_h, de_h)]
    rk = lambda hl: gold_ranks([project_ranking(h) for h in hl], golds)
    return depth, len(ch), {"bm25": rk(bm_h), "dense": rk(de_h), "hybrid": rk(hy_h)}

LOG = {"bm25": ("bm25", ""), "dense": ("dense-bge-small", "prefix=yes"),
       "hybrid": ("hybrid-rrf", f"bge-small+bm25 prefix=yes k={K_RRF}")}
res = {}
for size, ov in CONFIGS:
    depth, n_ch, rr = run_config(size, ov)
    res[(size, ov)] = (n_ch, rr)
    print(f"\n{size}/{ov} chunks={n_ch} depth={depth}")
    for name, r in rr.items():
        print(f" {name:7s}", {a: round(b, 3) for a, b in summarize(r).items()})
        if LOGGING and (size, ov) != (200, 40):
            meth, cfg = LOG[name]
            log_result(meth, f"{cfg} depth={depth}".strip(), size, ov, split, len(qs), summarize(r))


mrr = lambda r: summarize(r)["mrr@10"]
ref = res[(200, 40)][1]["hybrid"]
cands = [c for c in CONFIGS if c != (200, 40)]
order = sorted(cands, key=lambda c: -mrr(res[c][1]["hybrid"]))
print("\nHYBRID mrr@10 by config (paired vs the 200/40 reference):")
for c in order:
    r = res[c][1]["hybrid"]
    d, lo, hi = paired_bootstrap(ref, r, mrr)
    print(c, "chunks", res[c][0], "mrr", round(mrr(r), 3), f"vs 200/40: {d:+.3f} ({lo:+.3f},{hi:+.3f})")
best, second = order[0], order[1]
d, lo, hi = paired_bootstrap(res[second][1]["hybrid"], res[best][1]["hybrid"], mrr)
print("best vs runner-up:", best, second, f"{d:+.3f} ({lo:+.3f},{hi:+.3f})")
tie = lo <= 0 <= hi
pick = min((best, second), key=lambda c: res[c][0]) if tie else best
print("selection rule ->", pick, "(CI includes 0: fewer chunks wins)" if tie else "(clear winner)")

