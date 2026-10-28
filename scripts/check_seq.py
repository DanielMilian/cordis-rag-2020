import torch, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from src.cordis_rag.chunking import build_chunks

print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

NAME = "BAAI/bge-small-en-v1.5"
m = SentenceTransformer(NAME, device="cpu")
print(NAME, "max_seq_length:", m.max_seq_length, "dim:", m.get_sentence_embedding_dimension())

df = pd.read_parquet("data/processed/projects.parquet")
ch = build_chunks(df, 200, 40).sample(500, random_state=0)
n = np.array([len(m.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in ch["text"]])
print("tokens/chunk: median", int(np.median(n)), "p90", int(np.percentile(n, 90)), "max", n.max())
print("share truncated:", round(float((n > m.max_seq_length).mean()), 3))
