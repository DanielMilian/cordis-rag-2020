import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks


m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
df = pd.read_parquet("data/processed/projects.parquet")
print("max_seq_length:", m.max_seq_length, "| projects:", len(df))
for size, ov in [(64, 0), (64, 16), (128,0), (128, 32), (200, 0), (200,50), (400, 0), (400, 100)]:
    ch = build_chunks(df, size, ov)
    s = ch["text"].sample(min(2000, len(ch)), random_state=0)
    n = np.array([len(m.tokenizer(t)["input_ids"]) for t in s])
    print(size, ov, "chunks", len(ch), "per-project", round(len(ch) / len(df), 2), "| tokens median", int(np.median(n)), "max", n.max(), "share>512", round(float((n > 512).mean()), 3))

