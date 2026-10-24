import json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

THRESH = 0.4


def near_dup_scores(df, batch=1000):
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(df["objective"])
    best, nn = np.zeros(len(df)), np.zeros(len(df), dtype=int)
    for i in range(0, len(df), batch):
        S = (X[i:i + batch] @ X.T).toarray()
        for r in range(S.shape[0]):
            S[r, i + r] = 0.0
        nn[i:i + batch], best[i:i + batch] = S.argmax(1), S.max(1)
    return best, nn


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/projects.parquet")
    best, nn = near_dup_scores(df)
    df["max_sim"], df["nn_id"] = best, df["id"].values[nn]
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"max_sim >= {t}: {(best >= t).sum()} projects")
    top = df.sort_values("max_sim", ascending=False).head(12)
    print(top[["id", "acronym", "nn_id", "max_sim"]].to_string())
    elig = df.loc[df["max_sim"] < THRESH, "id"].tolist()
    json.dump(elig, open("data/processed/eligible_ids.json", "w"))
    print(f"\neligible at THRESH={THRESH}: {len(elig)} / {len(df)}")
