import time, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks

SIZE, OV = 200, 40
NAME = "BAAI/bge-small-en-v1.5"
df = pd.read_parquet("data/processed/projects.parquet")
ch = build_chunks(df, SIZE, OV)
m = SentenceTransformer(NAME, device="cpu")

t = time.time()
E = m.encode(ch["text"].tolist(), batch_size=32,
             normalize_embeddings=True, show_progress_bar=True)
print(E.shape, E.dtype, round(time.time() - t), "s")
print("norms:", np.linalg.norm(E[:3], axis=1))
tag = f"bge-small_{SIZE}_{OV}"
np.save(f"data/processed/emb_{tag}.npy", E)
ch[["chunk_id", "project_id"]].to_parquet(f"data/processed/emb_{tag}_ids.parquet")
