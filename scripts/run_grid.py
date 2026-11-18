import sys, json, pandas as pd
from sentence_transformers import SentenceTransformer
from src.cordis_rag.eval.metrics import summarize, paired_bootstrap
from src.cordis_rag.eval.results import log_result
from src.cordis_rag.retrieval.hybrid import Index, PREFIX, rank_all

split, K_RRF, LOGGING = "dev", 60, "--log" in sys.argv
PREFIX = "Represent this sentence for searching relevant passages: "
CONFIGS = [(200, 40), (400, 0), (128, 0), (128, 32), (200, 0), (200, 50), (64, 0), (64, 16)]


def run_config(size, ov):
    ix = Index(df, size, ov)
    bm_h = ix.bm25_hits[(q["question"] for q in qs])
    de_h = ix.dense_hits(Q)
    hy_h = ix.hybrid_hits(bm_h, de_h, K_RRF)
    rr = {"bm25": rank:all(bm_h, golds), "dense": rank_all(de_h, golds), "hybrid": rank_all(hy_h, golds)} 
    return ix.depth, len(ix.ch), rr

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

