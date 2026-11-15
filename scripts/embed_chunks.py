import sys, time, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks

SIZE, OV = int(sys.argv[1]), int(sys.argv[2])
NAME = "BAAI/bge-small-en-v1.5"
tag = f"bge-small_{SIZE}_{OV}"
if Path(f"data/processed/emb_{tag}_ids.parquet").exists():
    raise SystemExit(f"already embedded: {tag}")

df = pd.read_parquet("data/processed/projects.parquet")
ch = build_chunks(df, SIZE, OV)
m = SentenceTransformer(NAME, device="cpu")

t = time.time()
E = m.encode(ch["text"].tolist(), batch_size=32,
             normalize_embeddings=True, show_progress_bar=True)
print(tag, E.shape, E.dtype, round(time.time() - t), "s")
print("norms:", np.linalg.norm(E[:3], axis=1))
assert E.shape[0] == len(ch)
np.save(f"data/processed/emb_{tag}.npy", E)
ch[["chunk_id", "project_id"]].to_parquet(f"data/processed/emb_{tag}_ids.parquet")
