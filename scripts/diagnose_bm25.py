import json
import numpy as np
import pandas as pd
from src.cordis_rag.chunking import build_chunks
from src.cordis_rag.retrieval.bm25 import BM25Retriever

df = pd.read_parquet("data/processed/projects.parquet")
ret = BM25Retriever(build_chunks(df, 200, 40))
obj = df.set_index("id")["objective"]
tcol = "title" if "title" in df.columns else None
title = df.set_index("id")[tcol] if tcol else None

def ranked(q, k=200):
    out = []
    for _, pid, _ in ret.search(q, k=k):
        if pid not in out:
            out.append(pid)
    return out

def grank(rp, gold):
    return rp.index(gold) + 1 if gold in rp else None

def show(pid):
    t = f" | {title[pid]}" if title is not None else ""
    return f"{pid}{t}\n     {obj[pid][:200]}"

print("=== SANITY FAILURES ===")
s = df.sample(200, random_state=1)
for pid, text in zip(s["id"], s["objective"]):
    q = " ".join(text.split()[:20])
    rp = ranked(q)
    if not rp or rp[0] != pid:
        print("query:", q)
        print("gold rank:", grank(rp, pid))
        print("gold:", show(pid))
        print("top1:", show(rp[0]))


qs = [json.loads(l) for l in open("eval_data/eval_dev.jsonl")]
rps = [ranked(q["question"]) for q in qs]
golds = [q["project_id"] for q in qs]
ranks = [grank(rp, g) for rp, g in zip(rps, golds)]

print("\n=== DEV MISSES (rank != 1) ===")
for q, rp, g, r in zip(qs, rps, golds, ranks):
    if r != 1:
        print(f"\n[{q['style']}] gold rank: {r}")
        print("Q:", q["question"])
        print("gold:", show(g))
        print("top1:", show(rp[0]))

print("\n=== NEGATIVE CONTROL ===")
n = len(qs)
h = [np.mean([(grank(rps[i], golds[(i + sh) % n]) or 999) <= 10 for i in range(n)])
     for sh in range(1, n)]
print("shuffled-gold hit@10, mean over shifts:", round(float(np.mean(h)), 4), "(expect ~0)")
