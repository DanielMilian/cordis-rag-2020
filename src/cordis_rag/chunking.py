import pandas as pd, yaml


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Word-based sliding window; size/overlap in words."""
    assert 0 <= overlap < size
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    out = []
    for i in range(0, len(words), step):
        w = words[i:i + size]
        if len(w) > overlap:        # drop tail fully contained in previous chunk
            out.append(" ".join(w))
    return out


def build_chunks(df: pd.DataFrame, size: int, overlap: int) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        for j, ch in enumerate(chunk_text(r.objective, size, overlap)):
            rows.append({"chunk_id": f"{r.id}_{j}", "project_id": r.id,
                         "chunk_idx": j, "text": ch, "n_words": len(ch.split())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))["chunking"]
    df = pd.read_parquet("data/processed/projects.parquet")
    ch = build_chunks(df, cfg["size"], cfg["overlap"])
    ch.to_parquet("data/processed/chunks.parquet", index=False)
    print(f"{len(df)} projects -> {len(ch)} chunks (size={cfg['size']}, overlap={cfg['overlap']})")
    print("\nchunks per project:\n", ch.groupby("project_id").size().value_counts().sort_index())
    print("\ngrid preview:")
    for size, ov in [(64, 0), (64, 16), (128, 0), (128, 32), (200, 40)]:
        print(f"  size={size:3d} overlap={ov:2d} -> {len(build_chunks(df, size, ov))} chunks")
