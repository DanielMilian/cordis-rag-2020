import argparse, csv, datetime as dt, json
from pathlib import Path
import pandas as pd
from src.cordis_rag.chunking import build_chunks
from src.cordis_rag.retrieval.bm25 import BM25Retriever
from src.cordis_rag.eval.metrics import project_ranking, gold_ranks, summarize, bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", choices=["dev", "test"])
ap.add_argument("--size", type=int, default=200)
ap.add_argument("--overlap", type=int, default=40)
ap.add_argument("--sanity", action="store_true")
a = ap.parse_args()

df = pd.read_parquet("data/processed/projects.parquet")
chunks = build_chunks(df, a.size, a.overlap)
ret = BM25Retriever(chunks)

if a.sanity:
    s = df.sample(200, random_state=1)
    rk = [project_ranking(ret.search(" ".join(t.split()[:20]))) for t in s["objective"]]
    r = gold_ranks(rk, s["id"].tolist())
    print("SANITY self-retrieval hit@1:", summarize(r)["hit@1"], "(expect ~1.0)")

qs = [json.loads(l) for l in open(f"eval_data/eval_{a.split}.jsonl")]
rankings = [project_ranking(ret.search(q["question"])) for q in qs]
ranks = gold_ranks(rankings, [q["project_id"] for q in qs])
m = summarize(ranks)
lo, hi = bootstrap_ci(ranks, lambda r: (r <= 5).mean())
print(f"BM25 size={a.size} ov={a.overlap} split={a.split} n={len(qs)}")
print({k: round(v, 3) for k, v in m.items()}, f"hit@5 95% CI=({lo:.2f},{hi:.2f})")
for st in ("specific", "paraphrase"):
    ix = [i for i, q in enumerate(qs) if q["style"] == st]
    print(f"  {st:10s} n={len(ix)}", {k: round(v, 3) for k, v in summarize(ranks[ix]).items()})

Path("results").mkdir(exist_ok=True)
f = Path("results/results.csv")
row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "method": "bm25",
       "size": a.size, "overlap": a.overlap, "split": a.split, "n": len(qs), **m}
new = not f.exists()
with open(f, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=row.keys())
    if new:
        w.writeheader()
    w.writerow(row)
