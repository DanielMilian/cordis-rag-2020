import json, numpy as np, psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from src.cordis_rag.retrieval.hybrid import PREFIX

SIZE, OV, K = 400, 0, 51
tag, table = f"bge-small_{SIZE}_{OV}", f"chunks_{SIZE}_{OV}"
E = np.load(f"data/processed/emb_{tag}.npy")
import pandas as pd
cids = pd.read_parquet(f"data/processed/emb_{tag}_ids.parquet").chunk_id.astype(str).tolist()
qs = [json.loads(l) for l in open("eval_data/eval_dev.jsonl")]
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
Q = m.encode([PREFIX + q["question"] for q in qs], normalize_embeddings=True).astype(np.float32)
env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)
conn = psycopg.connect(host="localhost", port=5433, dbname=env["POSTGRES_DB"],
                       user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"])
register_vector(conn)
cur = conn.cursor()

def pg_top(Qm):
    out = []
    for v in Qm:
        cur.execute(f"SELECT chunk_id FROM {table} ORDER BY emb <=> %s, pos LIMIT {K}", (v,))
        out.append([r[0] for r in cur.fetchall()])
    return out

def np_top(Qm):
    return [[cids[i] for i in np.argsort(-s, kind="stable")[:K]] for s in Qm @ E.T]

def compare(a, b, label):
    top1 = sum(x[0] == y[0] for x, y in zip(a, b))
    setk = sum(set(x) == set(y) for x, y in zip(a, b))
    exact = sum(x == y for x, y in zip(a, b))
    print(f"{label}: top1 {top1}/{len(a)}  same set@{K} {setk}/{len(a)}  same order {exact}/{len(a)}")

npr = np_top(Q)
compare(pg_top(Q), npr, "real queries   ")
rng = np.random.default_rng(0)
compare(pg_top(Q[rng.permutation(len(Q))]), npr, "shuffled (ctrl)")
