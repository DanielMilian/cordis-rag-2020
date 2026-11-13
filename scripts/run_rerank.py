import json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks
from src.cordis_rag.retrieval.bm25 import BM25Retriever
from src.cordis_rag.eval.metrics import project_ranking, gold_ranks, summarize, paired_bootstrap
from src.cordis_rag.eval.results import log_result

SIZE, OV, split, K_RRF, DEPTH = 200, 40, "dev", 60, 100
tag = f"bge-small_{SIZE}_{OV}"
PREFIX = "Represent this sentence for searching relevant passages: "
df = pd.read_parquet("data/processed/projects.parquet")
ch = build_chunks(df, SIZE, OV)
E = np.load(f"data/processed/emb_{tag}.npy")
ids = pd.read_parquet(f"data/processed/emb_{tag}_ids.parquet")
assert ids["chunk_id"].tolist() == ch["chunk_id"].tolist(), "order mismatch"
cids, pids = ch["chunk_id"].tolist(), ch["project_id"].tolist()
pid_of = dict(zip(cids, pids))

m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
qs = [json.loads(l) for l in open(f"eval_data/eval_{split}.jsonl")]
golds = [q["project_id"] for q in qs]
bm = BM25Retriever(ch)
Q = m.encode([PREFIX + q["question"] for q in qs], normalize_embeddings=True)

def dense_hits(qv):
    s = E @ qv
    idx = np.argsort(-s)[:DEPTH]
    return [(cids[i], pids[i], float(s[i])) for i in idx]

def rrf(lists, k):
    sc = {}
    for hits in lists:
        for rank, (cid, _, _) in enumerate(hits, 1):
            sc[cid] = sc.get(cid, 0.0) + 1.0 / (k + rank)
    return [(c, pid_of[c], s) for c, s in sorted(sc.items(), key=lambda x: -x[1])]

bm_h = [bm.search(q["question"], k=DEPTH) for q in qs]
de_h = [dense_hits(Q[i]) for i in range(len(qs))]
ranks_for = lambda h1: gold_ranks([project_ranking(h) for h in h1], golds)
r_bm, r_de = ranks_for(bm_h), ranks_for(de_h)
r_hy = ranks_for([rrf([b, d], K_RRF) for b, d in zip(bm_h, de_h)])
for name, r in (("bm25", r_bm), ("dense+prefix", r_de), ("hybrid", r_hy)):
    print(f"{name:13s}", {a: round(b, 3) for a, b in summarize(r).items()})

from sentence_transformers import CrossEncoder
CE, N = "cross-encoder/ms-marco-MiniLM-L-6-v2", 50
ce = CrossEncoder(CE, device="cpu", max_length=512)
text_of = dict(zip(cids, ch["text"]))
cand = [rrf([b, d], K_RRF)[:N] for b, d in zip(bm_h, de_h)]
rec = np.mean([g in project_ranking(c) for c, g in zip(cand, golds)])
print("candidate recall (gold project in top-N hybrid chunks):", round(float(rec), 3))
pairs = [(q["question"], text_of[c[0]]) for q, cs in zip(qs, cand) for c in cs]
lens = [len(ce.tokenizer(a, b)["input_ids"]) for a,b in pairs[:300]]
print("pair tokens max", max(lens), "share > 512", round(float(np.mean(np.array(lens) > 512)), 3))

sc = ce.predict(pairs, batch_size=32, show_progress_bar=False)
re_h, k0 = [], 0
for cs in cand:
    s = sc[k0:k0 + len(cs)]
    k0 += len(cs)
    re_h.append([(cs[j][0], cs[j][1], float(s[j])) for j in np.argsort(-s)])
rng = np.random.default_rng(0)
rnd_h = [[cs[j] for j in rng.permutation(len(cs))] for cs in cand]

r_re, r_rn = ranks_for(re_h), ranks_for(rnd_h)
for name, r in (("hybrid", r_hy), ("hybrid+rerank", r_re), ("random order (control)", r_rn)):
    print(f"{name:22s}", {a: round(b, 3) for a,b in summarize(r).items()})
hit5 = lambda r: float((r <= 5).mean())
mrr = lambda r: summarize(r)["mrr@10"]
for name, fn in (("hit@5", hit5), ("mrr@10", mrr)):
    d, lo, hi = paired_bootstrap(r_hy, r_re, fn)
    print(f"rerank - hybrid {name}: {d:+.3f} CI=({lo:+.3f}, {hi:+.3f})")
for i, q in enumerate(qs):
    if r_re[i] > 1 or r_hy[i] > 1:
        print(f"rerank={r_re[i]:g} hybrid={r_hy[i]:g} | {q['question'][:80]}")
log_result("hybrid-rrf+rerank", f"{CE} N={N} k={K_RRF}", SIZE, OV, split, len(qs), summarize(r_re))

