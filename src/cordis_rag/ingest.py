import hashlib, json, re, datetime as dt
from pathlib import Path
import pandas as pd, yaml
from .csv_io import read_cordis_csv


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def load_projects(cfg_path="configs/default.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(open(cfg_path))
    raw = Path(cfg["raw_path"])
    df = read_cordis_csv(raw)

    cols = cfg["columns"]
    missing = [v for v in cols.values() if v not in df.columns]
    assert not missing, f"columns not in CSV: {missing}"
    df = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})

    n0 = len(df)
    df = df.dropna(subset=["id", "objective"])
    df["objective"] = df["objective"].map(norm)
    df["title"] = df["title"].fillna("").map(norm)
    df = df[df["objective"].str.split().str.len() >= cfg["min_words"]]
    df = df.drop_duplicates(subset="id")
    df["text_hash"] = df["objective"].str.lower().map(
        lambda t: hashlib.md5(t.encode()).hexdigest())
    n_dup = int(df.duplicated("text_hash").sum())
    df = df.drop_duplicates(subset="text_hash")

    if cfg.get("sample_n"):
        df = df.sample(min(cfg["sample_n"], len(df)), random_state=cfg["seed"])
    df = df.sort_values("id").reset_index(drop=True)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/processed/projects.parquet", index=False)
    json.dump({
        "raw_file": str(raw), "sha256": sha256(raw),
        "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rows_raw": n0, "rows_final": len(df),
        "exact_dup_abstracts_dropped": n_dup,
        "seed": cfg["seed"], "sample_n": cfg.get("sample_n"),
    }, open("data/raw/MANIFEST.json", "w"), indent=2)
    return df


if __name__ == "__main__":
    d = load_projects()
    print(d.shape)
