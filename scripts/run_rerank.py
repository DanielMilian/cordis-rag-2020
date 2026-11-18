import sys, json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.cordis_rag.eval.metrics import summarize, paired_bootstrap
from src.cordis_rag.eval.results import log_result
from src.cordis_rag.retrieval.hybrid import Index, PREFIX, rank_all

SIZE, OV, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
split, K_RRF, LOGGING = "dev", 60, "--log" in sys.argv
CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
df = pd.read_parquet("data/processed/projects.parquet")
qs = [json.loads(l) for l in open(f"eval_data/eval_{split}.jsonl")]
questions, golds = [q["question"] for q in qs], [q["project_id"] for q in qs]
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
Q = m.encode([PREFIX + q for q in questions], normalize_embeddings=True)
ix = Index(df, SIZE, OV)
bm_h, de_h = ix.bm25_hits(questions), ix.dense_hits(Q)
hy_h = ix.hybrid_hits(bm_h, de_h, K_RRF)


ce = CrossEncoder(CE, device="cpu", max_length=512)
cand = [h[:N] for h in hy_h]
pr = lambda hl: [list(dict.fromkeys(p for _, p, _ in h)) for h in hl]
rec = np.mean([g in r for r, g in zip(pr(cand), golds)])
print(f"{SIZE}/{OV} N={N} depth={ix.depth} candidate recall:", round(float(rec), 3))
pairs = [(q, ix.text_of[c[0]]) for q, cs in zip(questions, cand) for c in cs]
lens = np.array([len(ce.tokenizer(a, b)["input_ids"]) for a,b in pairs[:300]])
print("pair tokens max", lens.max(), "share > 512:", round(float((lens > 512).mean()), 3))
sc = ce.predict(pairs, batch_size=32, show_progress_bar=False)
re_h, k0 = [], 0
for cs in cand:
    s = sc[k0:k0 + len(cs)]
    k0 += len(cs)
    re_h.append([(cs[j][0], cs[j][1], float(s[j])) for j in np.argsort(-s)])
rng = np.random.default_rng(0)
rnd_h = [[cs[j] for j in rng.permutation(len(cs))] for cs in cand]


r_hy, r_re, r_rn = rank_all(hy_h, golds), rank_all(re_h, golds), rank_all(rnd_h, golds)
for name, r in (("hybrid", r_hy), ("hybrid+rerank", r_re), ("random order (control)", r_rn)):
    print(f"{name:22s}", {a: round(b, 3) for a, b in summarize(r).items()})
hit3 = lambda r: float((r <= 3).mean())
mrr = lambda r: summarize(r)["mrr@10"]
for name, fn in (("hit@3", hit3), ("mrr@10", mrr)):
    d, lo, hi = paired_bootstrap(r_hy, r_re, fn)
    print(f"rerank - hybrid {name}: {d:+.3f} CI=({lo:+.3f}, {hi:+.3f})")
for i, q in enumerate(qs):
    if r_re[i] > 1 or r_hy[i] > 1:
        print(f"rerank={r_re[i]:g} hybrid={r_hy[i]:g} | {q['question'][:80]}")
if LOGGING:
    log_result("hybrid-rrf+rerank", f"{CE} N={N} k={K_RRF}", SIZE, OV, split, len(qs), summarize(r_re))
